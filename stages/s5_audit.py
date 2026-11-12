"""Stage 5: Span audit & quality filtering.

Cross-validates evidence spans against abstracts using similarity metrics,
then flags papers whose spans are not grounded in the abstract (excluding
them only on request).  No LLM calls — pure similarity computation.

Optional packages (graceful degradation):
  - sentence_transformers  → semantic similarity   (else no-semantic weights)
  - rapidfuzz              → deep fuzzy analysis    (else no fuzzy rescues)

Token Jaccard, containment and TF-IDF are pure Python (``s5_audit_text``);
record validation and paper identity live in ``s5_audit_records``.
"""
import csv
import json
import logging
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from core.config import DEFAULT_AUDIT_SIMILARITY_THRESHOLD, DEFAULT_EMBEDDING_MODEL
from core.io import atomic_write_json
from core.models import REMOVAL_LOG_FILENAME
from stages.s5_audit_records import (  # noqa: F401 — re-exported for tests/callers
    _as_dict,
    _as_list,
    _derive_paper_id,
    _normalize_paper_ids,
    _profile_lookup_ids,
    _text,
    _validate_records,
)
from stages.s5_audit_text import (  # noqa: F401 — re-exported for tests/callers
    _best_window_wer,
    _containment,
    _tfidf_counts,
    _tfidf_similarity,
    _tfidf_similarity_counts,
    _token_similarity_sets,
    _tokenize,
    _word_tokens,
)

logger = logging.getLogger(__name__)

_AUDIT_FLAGS_FILENAME = "audit_flags.json"
_FILTERED_FILENAME = "claims_filtered.json"
_AUDIT_CSV_NAME = "span_similarity_audit.csv"
_DEEP_CSV_NAME = "p2_deep_analysis.csv"
_UPDATED_CSV_NAME = "span_similarity_audit_updated.csv"

# ── Optional dependency probes ────────────────────────────────────────────
# These probes deliberately catch ``Exception`` rather than ``ImportError``.
# A *broken* optional dependency (wrong version, missing distribution
# metadata, failed native extension) raises something other than ImportError
# during import, and that would otherwise propagate and make this module —
# and therefore the whole ``stages`` package — unimportable, defeating the
# graceful degradation promised in the module docstring.
_HAS_SENTENCE_TRANSFORMERS = False
SentenceTransformer = None
st_util = None
try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers import util as st_util
    _HAS_SENTENCE_TRANSFORMERS = True
except Exception as _exc:  # pragma: no cover - depends on local environment
    logger.warning(
        "sentence_transformers unavailable (%s: %s) — semantic similarity disabled",
        type(_exc).__name__, _exc,
    )

_HAS_RAPIDFUZZ = False
rfuzz = None
try:
    from rapidfuzz import fuzz as rfuzz
    _HAS_RAPIDFUZZ = True
except Exception as _exc:  # pragma: no cover - depends on local environment
    logger.warning(
        "rapidfuzz unavailable (%s: %s) — Phase 2 deep analysis will be skipped",
        type(_exc).__name__, _exc,
    )


# ── CSV writers ───────────────────────────────────────────────────────────

_AUDIT_CSV_FIELDS = [
    "paper_id", "title", "span_field", "semantic_sim", "semantic_available",
    "token_sim", "tfidf_sim", "containment", "composite", "priority",
]
_DEEP_CSV_FIELDS = [
    "paper_id", "title", "span_field", "composite", "priority",
    "fuzzy_partial", "fuzzy_sort", "wer", "fuzzy_error", "reclassified",
    "reclassification_reason",
]
_UPDATED_CSV_FIELDS = [*_AUDIT_CSV_FIELDS, "reclassified", "reclassification_reason"]


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """Write audit rows to CSV with a fixed column set."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    logger.info("  Wrote %s (%d rows)", path, len(rows))


def _write_audit_csv(path: Path, rows: list[dict]) -> None:
    """Write Phase 1 span similarity audit CSV."""
    _write_csv(path, rows, _AUDIT_CSV_FIELDS)


def _write_deep_csv(path: Path, rows: list[dict]) -> None:
    """Write Phase 2 deep analysis CSV."""
    _write_csv(path, rows, _DEEP_CSV_FIELDS)


def _write_updated_csv(path: Path, rows: list[dict]) -> None:
    """Write Phase 3 merged/updated audit CSV."""
    _write_csv(path, rows, _UPDATED_CSV_FIELDS)


# ── Phase 1: Span similarity scoring ─────────────────────────────────────

_SPAN_MODEL_NAME = DEFAULT_EMBEDDING_MODEL
# Sentinel distinguishes "never attempted" from "attempted and failed", so a
# failed load is not retried on every call.
_span_model_cache: list = [None, False]  # [model, attempted]


def _get_span_model():
    """Return the cached span-similarity model, loading it at most once.

    The model was previously constructed on every ``_compute_span_similarities``
    call. Loading ``all-MiniLM-L6-v2`` costs seconds and hundreds of megabytes,
    so repeated Stage 5 runs in a single GUI session paid that cost each time.
    """
    model, attempted = _span_model_cache
    if attempted:
        return model
    _span_model_cache[1] = True
    if not _HAS_SENTENCE_TRANSFORMERS:
        return None
    assert SentenceTransformer is not None  # guaranteed by the _HAS_SENTENCE_TRANSFORMERS check above
    try:
        model = SentenceTransformer(_SPAN_MODEL_NAME)
        _span_model_cache[0] = model
        logger.info("  Loaded sentence-transformers model: %s", _SPAN_MODEL_NAME)
        return model
    except Exception as exc:
        logger.warning("  Failed to load sentence-transformers model: %s", exc)
        return None


def _is_cancelled(cancel_event: Any) -> bool:
    if cancel_event is None:
        return False
    return cancel_event.is_set() if hasattr(cancel_event, "is_set") else bool(cancel_event)


# Stage 2 records where its relevance_quote came from; a full-text quote cannot
# be grounded in the abstract, so it is not audited against it.
_FULL_TEXT_SOURCE = "full_text"


def _relevance_quote_from_full_text(paper: dict, nested: dict) -> bool:
    source = _text(paper.get("relevance_quote_source")) or _text(
        nested.get("relevance_quote_source"))
    return source == _FULL_TEXT_SOURCE


def _spans_to_check(paper: dict, profile: dict) -> list[tuple[str, str]]:
    """(span_field, text) pairs for a paper: its quotes plus method evidence spans.

    Stage 3 nests extracted fields under paper["claims"], so fall back to it
    when the top level is blank. Null or non-string values are ignored.
    """
    nested = _as_dict(paper.get("claims"))
    spans: list[tuple[str, str]] = []
    for field in ("key_quote", "relevance_quote"):
        if field == "relevance_quote" and _relevance_quote_from_full_text(paper, nested):
            continue
        val = _text(paper.get(field)) or _text(nested.get(field))
        if val:
            spans.append((field, val))
    method_profile = _as_dict(profile.get("method_profile"))
    for es in _as_list(method_profile.get("evidence_spans")):
        txt = _text(es.get("text") or es.get("quote")) if isinstance(es, dict) else _text(es)
        if txt:
            spans.append(("evidence_span", txt))
    return spans


def _encode_paper(st_model, spans: list[tuple[str, str]], abstract: str, pid: str):
    """Batch-encode a paper's spans and abstract; (None, None) if unavailable.

    One batched encode per paper instead of one 2-item encode per span. A
    failure is logged and the paper is scored without the semantic component.
    """
    if st_model is None:
        return None, None
    try:
        encoded = st_model.encode([t for _, t in spans] + [abstract], convert_to_tensor=True)
    except Exception as exc:
        logger.warning("  Embedding failed for paper %s (%s: %s); scoring it without "
                       "semantic similarity", pid, type(exc).__name__, exc)
        return None, None
    return encoded[:-1], encoded[-1]


def _semantic_similarity(span_embeddings, abstract_embedding, span_idx: int, pid: str):
    """Cosine similarity of one span to the abstract, or None if unavailable."""
    if span_embeddings is None or abstract_embedding is None:
        return None
    assert st_util is not None  # embeddings only get set when sentence_transformers loaded
    try:
        return float(st_util.cos_sim(span_embeddings[span_idx], abstract_embedding).item())
    except Exception as exc:
        logger.warning("  Cosine similarity failed for paper %s span %d (%s: %s)",
                       pid, span_idx, type(exc).__name__, exc)
        return None


def _score_span(span_text: str, sem_sim: float | None, abstract_tokens: set[str],
                abstract_counts: Counter) -> dict:
    """Similarity metrics, composite and priority for one span."""
    span_tokens = _tokenize(span_text)
    tok_sim = _token_similarity_sets(span_tokens, abstract_tokens)
    tfidf_sim = _tfidf_similarity_counts(_tfidf_counts(span_text), abstract_counts)
    semantic_available = sem_sim is not None
    sem = sem_sim if semantic_available else 0.0
    # Composite — no-semantic weights whenever this span has no embedding score.
    if semantic_available:
        composite = 0.4 * sem + 0.3 * tok_sim + 0.3 * tfidf_sim
    else:
        composite = 0.5 * tok_sim + 0.5 * tfidf_sim

    if (sem < 0.5 and tok_sim < 0.3) or composite < 0.35:
        priority = "P1"
    elif composite < 0.55:
        priority = "P2"
    else:
        priority = "P3"
    return {
        "semantic_sim": round(sem, 4),
        "semantic_available": semantic_available,
        "token_sim": round(tok_sim, 4),
        "tfidf_sim": round(tfidf_sim, 4),
        "containment": round(_containment(span_tokens, abstract_tokens), 4),
        "composite": round(composite, 4),
        "priority": priority,
    }


def _score_paper(paper: dict, profile: dict, st_model, first_row_id: int) -> list[dict]:
    """Audit rows for every span of one paper (empty without abstract or spans)."""
    pid = paper.get("paper_id", paper.get("id", ""))
    abstract = _text(paper.get("abstract"))
    spans = _spans_to_check(paper, profile) if abstract else []
    if not spans:
        return []

    # Precompute the abstract side of every metric once per paper.
    abstract_tokens = _tokenize(abstract)
    abstract_counts = _tfidf_counts(abstract)
    span_embeddings, abstract_embedding = _encode_paper(st_model, spans, abstract, pid)
    title = (_text(paper.get("title")) or "?")[:80]
    rows: list[dict] = []
    for span_idx, (span_field, span_text) in enumerate(spans):
        sem_sim = _semantic_similarity(span_embeddings, abstract_embedding, span_idx, pid)
        rows.append({
            "row_id": first_row_id + span_idx,
            "paper_id": pid,
            "title": title,
            "span_field": span_field,
            "span_text": span_text,
            "abstract": abstract,
            **_score_span(span_text, sem_sim, abstract_tokens, abstract_counts),
        })
    return rows


def _log_priority_distribution(audit_rows: list[dict]) -> None:
    p_counts = Counter(r["priority"] for r in audit_rows)
    composites = sorted(r["composite"] for r in audit_rows)
    if composites:
        logger.info("  Similarity distribution: min=%.4f median=%.4f max=%.4f",
                    composites[0], composites[len(composites) // 2], composites[-1])
    logger.info("  Priority counts: P1=%d, P2=%d, P3=%d",
                p_counts["P1"], p_counts["P2"], p_counts["P3"])


def _profile_for(profile_map: dict[str, dict], pid: str) -> dict:
    """The claim's profile, also for a disambiguated duplicate id (``pid#2``)."""
    for key in _profile_lookup_ids(pid):
        if key in profile_map:
            return profile_map[key]
    return {}


def _report_progress(phase: str, done: int, total: int, noun: str,
                     progress_callback: Callable[[str], None] | None) -> None:
    if done % 50 == 0 or done == total:
        _announce(f"{phase} {done}/{total} {noun}", progress_callback)


def _compute_span_similarities(
    claims: list[dict],
    profiles: list[dict],
    progress_callback: Callable[[str], None] | None = None,
    cancel_event: Any = None,
) -> list[dict]:
    """Compute similarity between evidence spans and abstract text.

    For each paper, compare key_quote, relevance_quote (unless it was taken
    from the full text) and method evidence spans against the abstract.
    Returns one audit row per span, each with a unique ``row_id`` that later
    phases merge on.
    """
    profile_map = {pid: p for p in profiles if (pid := p.get("paper_id", p.get("id", "")))}

    # Load sentence-transformers model if available (process-wide cache)
    st_model = _get_span_model()

    audit_rows: list[dict] = []
    total = len(claims)
    for idx, paper in enumerate(claims):
        if _is_cancelled(cancel_event):
            logger.info("Phase 1 cancelled after %d/%d papers", idx, total)
            break
        pid = paper.get("paper_id", paper.get("id", ""))
        audit_rows.extend(
            _score_paper(paper, _profile_for(profile_map, pid), st_model, len(audit_rows)))
        _report_progress("Phase 1: Scored", idx + 1, total, "papers", progress_callback)

    _log_priority_distribution(audit_rows)
    return audit_rows


# ── Phase 2: Deep fuzzy analysis ─────────────────────────────────────────

# Rescue thresholds. WER and token_sort_ratio are measured against the
# best-aligned abstract window, not the whole abstract (whose surrounding words
# would count as insertions and make the rule unreachable).
_NEAR_SUBSTRING_PARTIAL = 95
_FUZZY_SORT_MIN = 85
_WER_MAX = 0.3


def _fuzzy_scores(row: dict, span_text: str, abstract: str, window: str):
    """(partial_ratio vs abstract, token_sort_ratio vs aligned window, error)."""
    if not (_HAS_RAPIDFUZZ and span_text and abstract):
        return 0.0, 0.0, ""
    assert rfuzz is not None  # guaranteed by the _HAS_RAPIDFUZZ check above
    try:
        partial = rfuzz.partial_ratio(span_text, abstract)
        sort = rfuzz.token_sort_ratio(" ".join(_word_tokens(span_text)), window)
    except (TypeError, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.warning("  rapidfuzz failed for paper %s (%s); fuzzy rescue skipped",
                       row.get("paper_id"), error)
        return 0.0, 0.0, error
    return partial, sort, ""


def _rescue_reason(fuzzy_partial: float, fuzzy_sort: float, wer: float,
                   semantic_sim: float, semantic_threshold: float) -> str:
    """Why a flagged span counts as likely valid, or "" if it does not."""
    if _HAS_RAPIDFUZZ and fuzzy_partial >= _NEAR_SUBSTRING_PARTIAL:
        return "near_substring"
    if _HAS_RAPIDFUZZ and fuzzy_sort >= _FUZZY_SORT_MIN and wer < _WER_MAX:
        return "high_fuzzy_low_wer"
    if semantic_sim >= semantic_threshold:
        return "high_semantic_sim"
    return ""


def _deep_row(row: dict, semantic_threshold: float) -> dict:
    """Phase 2 metrics and rescue verdict for one flagged span."""
    span_text = row.get("span_text", "")
    abstract = row.get("abstract", "")
    abstract_tokens = _word_tokens(abstract)
    wer, start, end = _best_window_wer(_word_tokens(span_text), abstract_tokens)
    window = " ".join(abstract_tokens[start:end])
    fuzzy_partial, fuzzy_sort, fuzzy_error = _fuzzy_scores(row, span_text, abstract, window)
    reason = _rescue_reason(fuzzy_partial, fuzzy_sort, wer,
                            row.get("semantic_sim") or 0, semantic_threshold)
    return {
        "row_id": row.get("row_id"),
        "paper_id": row["paper_id"],
        "title": row["title"],
        "span_field": row["span_field"],
        "composite": row["composite"],
        "priority": row["priority"],
        "fuzzy_partial": round(fuzzy_partial, 2),
        "fuzzy_sort": round(fuzzy_sort, 2),
        "wer": round(wer, 4),
        "fuzzy_error": fuzzy_error,
        "reclassified": bool(reason),
        "reclassification_reason": reason,
    }


def _deep_analysis(
    audit_rows: list[dict],
    progress_callback: Callable[[str], None] | None = None,
    semantic_threshold: float = DEFAULT_AUDIT_SIMILARITY_THRESHOLD,
    cancel_event: Any = None,
) -> list[dict]:
    """Run deeper analysis on P1/P2 flagged spans.

    Uses rapidfuzz (partial_ratio, token_sort_ratio) + best-window WER.
    Reclassifies spans that pass thresholds as likely_valid. Stops early when
    *cancel_event* is set and returns the rows analysed so far.
    """
    flagged = [r for r in audit_rows if r["priority"] in ("P1", "P2")]
    if not flagged:
        logger.info("Phase 2: No P1/P2 spans to analyse")
        return []

    logger.info("Phase 2: Deep analysis on %d flagged spans "
                "(rapidfuzz=%s)", len(flagged), _HAS_RAPIDFUZZ)

    deep_rows: list[dict] = []
    for idx, row in enumerate(flagged):
        if _is_cancelled(cancel_event):
            logger.info("Phase 2 cancelled after %d/%d spans", idx, len(flagged))
            return deep_rows
        deep_rows.append(_deep_row(row, semantic_threshold))
        _report_progress("Phase 2: Analysed", idx + 1, len(flagged), "flagged spans",
                         progress_callback)

    rescued = sum(1 for d in deep_rows if d["reclassified"])
    logger.info("  Phase 2 complete: %d/%d flagged spans reclassified as likely_valid",
                rescued, len(deep_rows))
    return deep_rows


# ── Phase 3: Merge reclassifications ──────────────────────────────────────

def _merge_key(row: dict) -> Any:
    """Phase 1 row id; (paper_id, span_field) only for rows written without one."""
    row_id = row.get("row_id")
    return row_id if row_id is not None else (row.get("paper_id"), row.get("span_field"))


def _merge_reclassifications(
    audit_rows: list[dict],
    deep_rows: list[dict],
) -> list[dict]:
    """Update audit rows with deep analysis reclassifications.

    Rows are matched on their Phase 1 ``row_id``: a paper can have several
    evidence spans, so ``(paper_id, span_field)`` is not unique.
    """
    deep_map = {_merge_key(dr): dr for dr in deep_rows}

    updated: list[dict] = []
    for row in audit_rows:
        merged = dict(row)
        dr = deep_map.get(_merge_key(row))
        if dr and dr.get("reclassified"):
            merged["reclassified"] = True
            merged["reclassification_reason"] = dr["reclassification_reason"]
        else:
            merged["reclassified"] = False
            merged["reclassification_reason"] = ""
        updated.append(merged)

    reclassified_count = sum(1 for u in updated if u["reclassified"])
    logger.info("Phase 3: Merged reclassifications — %d spans marked likely_valid",
                reclassified_count)

    return updated


# ── Phase 4: Grounding flags (optional exclusion) ────────────────────────

# A span is grounded when Phase 2 rescued it or when at least this share of its
# word tokens occurs in the abstract. Containment does not shrink as the
# abstract grows, unlike Jaccard/TF-IDF against the whole abstract, so the test
# is absolute: it never removes a fixed share of the corpus.
_GROUNDED_CONTAINMENT = 0.5


def _span_is_grounded(row: dict) -> bool:
    return bool(row.get("reclassified")) or (row.get("containment") or 0.0) >= _GROUNDED_CONTAINMENT


def _paper_grounding(audit_rows: list[dict]) -> dict[str, dict]:
    """Per paper: span count, grounded-span count, best containment, mean composite."""
    by_paper: dict[str, list[dict]] = {}
    for row in audit_rows:
        by_paper.setdefault(row["paper_id"], []).append(row)
    return {
        pid: {
            "n_spans": len(rows),
            "n_grounded": sum(1 for r in rows if _span_is_grounded(r)),
            "best_containment": max((r.get("containment") or 0.0) for r in rows),
            "avg_composite": round(sum(r["composite"] for r in rows) / len(rows), 4),
        }
        for pid, rows in by_paper.items()
    }


def _claim_pid(claim: dict) -> str:
    return claim.get("paper_id", claim.get("id", ""))


def _audit_flags(claims: list[dict], grounding: dict[str, dict]) -> list[dict]:
    """Papers needing manual review: no auditable span, or no grounded span."""
    flags: list[dict] = []
    for claim in claims:
        pid = _claim_pid(claim)
        info = grounding.get(pid)
        if info is None:
            flags.append({"paper_id": pid, "reason": "no_audit_rows", "removed": False})
        elif info["n_grounded"] == 0:
            flags.append({"paper_id": pid, "reason": "ungrounded_spans", "removed": False,
                          **info})
    return flags


def _filter_claims(
    claims: list[dict],
    audit_rows: list[dict],
    exclude_ungrounded: bool = False,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Flag papers whose spans are not grounded in the abstract.

    Flag-only by default. With *exclude_ungrounded*, papers none of whose spans
    is grounded are removed; papers without auditable spans (no abstract, no
    quotes) are always kept and flagged, never excluded.
    Returns (kept_claims, removal_log, flags); the removal log lists only
    papers that were actually removed.
    """
    flags = _audit_flags(claims, _paper_grounding(audit_rows))
    if not audit_rows:
        logger.warning("Phase 4: no audit rows produced — keeping all %d papers. Check that "
                       "claims carry key_quote/evidence spans.", len(claims))
    n_ungrounded = sum(1 for f in flags if f["reason"] == "ungrounded_spans")
    logger.info("Phase 4: %d papers flagged (%d ungrounded, %d without audit rows); "
                "exclusion %s", len(flags), n_ungrounded, len(flags) - n_ungrounded,
                "on" if exclude_ungrounded else "off (flag only)")
    if not exclude_ungrounded:
        return list(claims), [], flags

    excluded = {f["paper_id"] for f in flags if f["reason"] == "ungrounded_spans"}
    removed = [{**f, "removed": True} for f in flags if f["paper_id"] in excluded]
    flags = [{**f, "removed": f["paper_id"] in excluded} for f in flags]
    kept = [c for c in claims if _claim_pid(c) not in excluded]
    return kept, removed, flags


# ── Main entry point ──────────────────────────────────────────────────────

def _announce(message: str, progress_callback: Callable[[str], None] | None) -> None:
    logger.info(message)
    if progress_callback:
        progress_callback(message)


def _write_json_output(path: Path, data: list[dict], label: str) -> None:
    atomic_write_json(path, data)
    logger.info("  Wrote %s (%d %s)", path, len(data), label)


def _resolve_paths(config, claims_path: str | None) -> dict[str, Path]:
    """Input files and output directories (created) for one Stage 5 run."""
    claims_file = Path(claims_path) if claims_path else (
        config.claims_dir / "claims.json" if config else Path("data/claims/claims.json"))
    profiles_dir = (config.data_dir / "method_review" if config
                    else Path("data/method_review"))
    review_dir = profiles_dir / "review"
    claims_out_dir = config.claims_dir if config else Path("data/claims")
    review_dir.mkdir(parents=True, exist_ok=True)
    claims_out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "claims": claims_file,
        "profiles": profiles_dir / "method_profiles_validated.json",
        "filtered": claims_out_dir / _FILTERED_FILENAME,
        "removal_log": claims_out_dir / REMOVAL_LOG_FILENAME,
        "flags": claims_out_dir / _AUDIT_FLAGS_FILENAME,
        "audit_csv": review_dir / _AUDIT_CSV_NAME,
        "deep_csv": review_dir / _DEEP_CSV_NAME,
        "updated_csv": review_dir / _UPDATED_CSV_NAME,
    }
    logger.info("  Claims input:  %s", paths["claims"])
    logger.info("  Profiles input: %s", paths["profiles"])
    logger.info("  Review dir:    %s", review_dir)
    return paths


def _load_inputs(paths: dict[str, Path]) -> tuple[list[dict], list[dict]]:
    """Validated claims and method-profile records."""
    if not paths["claims"].exists():
        msg = f"Claims file not found: {paths['claims']}"
        logger.error(msg)
        raise FileNotFoundError(msg)
    with open(paths["claims"], encoding="utf-8") as f:
        claims = _validate_records(json.load(f), "claims")
    logger.info("  Loaded %d claims", len(claims))

    if not paths["profiles"].exists():
        logger.warning("  Method profiles not found at %s — proceeding without",
                       paths["profiles"])
        return claims, []
    with open(paths["profiles"], encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        logger.warning("  Method profiles file is not a JSON list — proceeding without")
        return claims, []
    profiles = _validate_records(data, "method profile")
    logger.info("  Loaded %d method profiles", len(profiles))
    return claims, profiles


# Every file a run writes; all are removed when a run is cancelled so neither
# the Pipeline Board nor later stages read a previous run's results.
_OUTPUT_KEYS = ("filtered", "removal_log", "flags", "audit_csv", "deep_csv", "updated_csv")


def _cancelled(paths: dict[str, Path], n_claims: int) -> dict:
    """Remove previous-run outputs after a cancel and report the cancellation."""
    removed = []
    for key in _OUTPUT_KEYS:
        if paths[key].exists():
            paths[key].unlink()
            removed.append(paths[key].name)
    logger.info("Stage 5 cancelled; removed stale outputs: %s", ", ".join(removed) or "none")
    return {"total": n_claims, "cancelled": True}


def _run_phases(claims, profiles, config, paths, cancel_event, progress_callback):
    """Phases 1-3; returns (audit_rows, deep_rows, updated_rows) or None if cancelled."""
    _announce(f"Phase 1: Computing similarities for {len(claims)} papers...", progress_callback)
    audit_rows = _compute_span_similarities(
        claims, profiles, progress_callback=progress_callback, cancel_event=cancel_event)
    logger.info("Phase 1 complete: %d span audit rows", len(audit_rows))
    if _is_cancelled(cancel_event):
        return None
    _write_audit_csv(paths["audit_csv"], audit_rows)

    _announce("Phase 2: Deep fuzzy analysis on flagged spans...", progress_callback)
    semantic_threshold = float(getattr(
        config, "audit_similarity_threshold", DEFAULT_AUDIT_SIMILARITY_THRESHOLD))
    logger.info("  Semantic reclassification threshold: %.2f", semantic_threshold)
    deep_rows = _deep_analysis(audit_rows, progress_callback=progress_callback,
                               semantic_threshold=semantic_threshold,
                               cancel_event=cancel_event)
    if _is_cancelled(cancel_event):
        return None
    _write_deep_csv(paths["deep_csv"], deep_rows)

    _announce("Phase 3: Merging reclassifications...", progress_callback)
    updated_rows = _merge_reclassifications(audit_rows, deep_rows)
    _write_updated_csv(paths["updated_csv"], updated_rows)
    return audit_rows, deep_rows, updated_rows


def _write_outputs(paths, filtered_claims, removal_log, audit_flags) -> None:
    _write_json_output(paths["filtered"], filtered_claims, "papers")
    _write_json_output(paths["removal_log"], removal_log, "removed papers")
    _write_json_output(paths["flags"], audit_flags, "papers flagged for review")


def _summary(claims, filtered_claims, audit_flags, rows, paths) -> dict:
    audit_rows, deep_rows, updated_rows = rows
    return {
        "total": len(claims),
        "filtered": len(filtered_claims),
        "removed": len(claims) - len(filtered_claims),
        "audit_spans": len(audit_rows),
        "deep_analysed": len(deep_rows),
        "reclassified": sum(1 for r in updated_rows if r.get("reclassified")),
        "output_path": str(paths["filtered"]),
        "removal_log_path": str(paths["removal_log"]),
        "flagged": len(audit_flags),
        "flags_path": str(paths["flags"]),
        "audit_csv_path": str(paths["audit_csv"]),
        "deep_csv_path": str(paths["deep_csv"]),
        "updated_csv_path": str(paths["updated_csv"]),
    }


def run(
    config=None,
    claims_path: str | None = None,
    cancel_event: Any = None,
    progress_callback: Callable[[str], None] | None = None,
    **kw,
) -> dict:
    """Run 4-phase span audit.

    Phase 1: Compute span similarities (semantic + token Jaccard + TF-IDF)
    Phase 2: Deep fuzzy analysis on P1/P2 rows (rapidfuzz if available)
    Phase 3: Merge reclassifications (rescue likely_valid spans)
    Phase 4: Flag papers with no grounded span (absolute containment test);
             remove them only when ``exclude_ungrounded=True`` is passed

    Input : claims.json + method_profiles_validated.json
    Output: claims_filtered.json + removal_log.json (removed papers only)
            + audit_flags.json (papers for manual review) + 3 audit CSVs.
    A cancelled run writes none of these and deletes any left by earlier runs.
    """
    logger.info("═" * 60)
    logger.info("STAGE 5: SPAN AUDIT & QUALITY FILTERING")
    logger.info("═" * 60)
    logger.info("  Optional deps: sentence_transformers=%s, rapidfuzz=%s",
                _HAS_SENTENCE_TRANSFORMERS, _HAS_RAPIDFUZZ)

    paths = _resolve_paths(config, claims_path)
    claims, profiles = _load_inputs(paths)

    if not claims:
        logger.warning("No claims to audit — writing empty outputs")
        _write_outputs(paths, [], [], [])
        return {"total": 0, "filtered": 0, "removed": 0, "output_path": str(paths["filtered"])}

    _announce(f"Stage 5: Auditing {len(claims)} papers", progress_callback)
    # Guarantee unique paper ids before anything keys off them.
    _normalize_paper_ids(claims, profiles)

    rows = _run_phases(claims, profiles, config, paths, cancel_event, progress_callback)
    if rows is None or _is_cancelled(cancel_event):
        return _cancelled(paths, len(claims))

    exclude_ungrounded = bool(kw.get(
        "exclude_ungrounded", getattr(config, "audit_exclude_ungrounded", False)))
    _announce("Phase 4: Checking that every paper has a grounded span...", progress_callback)
    filtered_claims, removal_log, audit_flags = _filter_claims(
        claims, rows[2], exclude_ungrounded=exclude_ungrounded)
    _write_outputs(paths, filtered_claims, removal_log, audit_flags)

    summary = _summary(claims, filtered_claims, audit_flags, rows, paths)
    logger.info("Stage 5 summary: %s", summary)
    if progress_callback:
        progress_callback(
            f"Stage 5 complete: {summary['filtered']} papers kept, "
            f"{summary['removed']} removed, {summary['flagged']} flagged for review, "
            f"{summary['reclassified']} rescued")
    return summary
