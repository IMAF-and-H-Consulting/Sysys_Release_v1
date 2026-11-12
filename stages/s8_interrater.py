"""Legacy cross-model comparison API, retired from the active pipeline.

Retained for historical callers and agreement regression tests. Stage 2 now
uses combined Laya and BRT-BERT screening. Shared-model agreement does not
establish independent validation or screening effectiveness.

Agreement metrics and the report live in ``stages.s8_interrater_metrics``
and are re-exported here.
"""
import copy
import logging
import random
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

from core.agreement import format_adjacent_agreement
from core.checkpoint import CheckpointManager, input_fingerprint, run_sliding_window
from core.claims import claim_list
from core.config import (
    EMBEDDING_RATERS,
    PROVIDER_MODEL_FIELD,
    Config,
    normalize_provider_name,
)
from core.doi import normalize_doi
from core.io import atomic_write_json, atomic_write_text
from process.paths import load_paper_list, resolve_claims_corpus, resolve_validation_dir
from process.relevance_cascade import MODES as CASCADE_MODES
from stages._common import is_cancelled, make_progress
from stages.s8_interrater_metrics import (  # noqa: F401 — re-exported API
    _fmt,
    _interpret_kappa,
    _is_rated,
    _record_values,
    compute_agreement,
    fleiss_kappa,
    generate_report,
    normalize_label,
    pearson_correlation,
)

logger = logging.getLogger(__name__)

# Providers that score with a local cross-encoder instead of an LLM.  These
# emit a relevance logit only — no free-text generation — so claim extraction
# is skipped and IRR runs in score-only mode.  Both spellings are accepted
# because the GUI uses "brt-bert"/"brt-irr" while configs may use underscores.
# brt-bert  = brt-bert-relevance (original scorer, used as independent rater)
# brt-irr   = brt-bert-irr-buddy-mlm (model fine-tuned specifically for IRR)
EMBEDDING_SECOND_RATERS = EMBEDDING_RATERS

SUPPORTED_SECOND_RATERS = frozenset(PROVIDER_MODEL_FIELD) | EMBEDDING_RATERS | frozenset(CASCADE_MODES)

# brt-bert-relevance/final and the default primary screener's BRT model are the
# same checkpoint (identical SHA-256), so their agreement is not IRR.
_SHARED_WEIGHT_RATERS = frozenset({"brt-bert", "brt_bert"})
_BRT_BERT_NOTE = (
    "The brt-bert rater shares its weights with the BRT model of the default primary "
    "screener (laya-bert), so this agreement measures that model's consistency and is "
    "not independent validation."
)
_CASCADE_NOTE = (
    "Shared Laya/BRT checkpoints measure screening consistency, not independent validation. "
    "0/5 scores encode reject/keep; ordinal agreement with other score scales is descriptive only."
)
# Bumped when the checkpoint row identity changes, so older checkpoints are discarded.
_IDENTITY_SCHEME = "paper_key/v1"
_ID_FIELDS = ("pmid", "paper_id", "s2_id", "id")
CHECKPOINT_EVERY = 10
DEFAULT_WORKERS = 4
DEFAULT_STAGGER_SECONDS = 0.5
DEFAULT_WORKER_TIMEOUT_SECONDS = 300.0
_LOG_TITLE_CHARS = 60


# ── Stratified sampling ─────────────────────────────────────────────────

def stratified_sample(papers: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Sample papers stratified by relevance_score.

    Ensure representation across score bins (0-2, 3-4, 5).
    If n >= len(papers), return all.
    """
    if n >= len(papers):
        return list(papers)

    bins: dict[str, list[dict]] = {"low": [], "mid": [], "high": []}
    for p in papers:
        # Unscored papers (None) sample with the low bin; compute_agreement
        # then excludes them because they have no original rating.
        score = p.get("relevance_score") or 0
        if score <= 2:
            bins["low"].append(p)
        elif score <= 4:
            bins["mid"].append(p)
        else:
            bins["high"].append(p)

    rng = random.Random(seed)
    sampled: list[dict] = []
    remaining = n

    # Proportional allocation
    non_empty_bins = {k: v for k, v in bins.items() if v}
    if not non_empty_bins:
        return []

    for bin_papers in non_empty_bins.values():
        proportion = len(bin_papers) / len(papers)
        alloc = max(1, round(proportion * n))
        alloc = min(alloc, len(bin_papers), remaining)
        chosen = rng.sample(bin_papers, alloc)
        sampled.extend(chosen)
        remaining -= len(chosen)
        if remaining <= 0:
            break

    # Fill any remaining slots from the full pool (excluding already sampled)
    if remaining > 0:
        sampled_ids = {id(p) for p in sampled}
        pool = [p for p in papers if id(p) not in sampled_ids]
        if pool:
            extra = rng.sample(pool, min(remaining, len(pool)))
            sampled.extend(extra)

    return sampled


# ── Second-rater LLM calls ──────────────────────────────────────────────

def provider_score_relevance(model_name: str, paper: dict,
                             hypothesis: str = "") -> dict:
    """Score a paper using the second rater's LLM."""
    from process.llm import chat as llm_chat
    from process.relevance_filter import (
        RELEVANCE_PROMPT,
        _fit_single_relevance_input,
        _paper_text_for_scoring,
        _parse_relevance_response,
    )

    title = paper.get("title", "Unknown")
    paper_text, text_source = _paper_text_for_scoring(paper)

    if not paper_text.strip():
        logger.warning("Paper '%s' has no text to score", title[:60])
        return {"relevance_score": 0, "relevance_quote": "", "error": "no text"}

    # Fall back to a neutral hypothesis if none supplied.  (The
    # relevance_filter prompt itself substitutes "the research topic" when
    # the hypothesis is blank, so an empty string here is also safe.)
    if not hypothesis:
        hypothesis = "the research topic"

    prompt = RELEVANCE_PROMPT.format(
        hypothesis=hypothesis,
        title=title,
        text_source=text_source,
        paper_text=paper_text,
    )
    messages = [{"role": "user", "content": prompt}]

    paper_text, coverage, budget = _fit_single_relevance_input(
        paper,
        paper_text,
        text_source,
        hypothesis=hypothesis,
        context_tokens=None,
        max_tokens=None,
        task="interrater",
    )
    prompt = RELEVANCE_PROMPT.format(
        hypothesis=hypothesis,
        title=title,
        text_source=text_source,
        paper_text=paper_text,
    )
    messages = [{"role": "user", "content": prompt}]

    t0 = time.time()
    response = llm_chat(
        model_name, messages, temperature=0, max_tokens=None,
        task="interrater", expected_items=1,
        context_tokens=budget.context_window_tokens,
        token_budget=budget,
    )
    elapsed = time.time() - t0

    result = _parse_relevance_response(response)
    result["relevance_input_coverage"] = coverage
    logger.info(
        "[IRR-Score] %s → score=%s (%.1fs)",
        title[:60], result.get("relevance_score"), elapsed,
    )
    return result


def provider_extract_claims(model_name: str, paper: dict,
                            research_config: dict | None = None) -> dict:
    """Extract claims using the second rater's LLM.

    ``research_config`` selects the extraction schema so the second rater
    measures agreement against the same fields the primary extractor used.
    """
    from process.claim_extractor import extract_claims

    title = paper.get("title", "Unknown")
    abstract = paper.get("abstract", "")
    if not abstract.strip():
        logger.warning("[IRR-Extract] No abstract for '%s'", title[:60])
        return {"error": "no_abstract"}

    t0 = time.time()
    result = extract_claims(
        model_name,
        paper,
        think=None,
        max_tokens=None,
        research_config=research_config,
        context_tokens=None,
    )
    elapsed = time.time() - t0
    n_findings = len(result.get("findings", []))
    logger.info(
        "[IRR-Extract] %s → %d findings (%.1fs)",
        title[:60], n_findings, elapsed,
    )
    return result



# ── Run setup ────────────────────────────────────────────────────────────

def _no_release() -> None:
    """Nothing to release (LLM raters hold no local model)."""


@dataclass(frozen=True)
class _RunSpec:
    """What one Stage 8 run compares."""
    config: Any
    provider: str
    model_name: str
    model_path: str  # cross-encoder checkpoint override ("" for the preset)
    hypothesis: str
    sample_size: int
    seed: int

    @property
    def use_bert(self) -> bool:
        return self.provider in EMBEDDING_SECOND_RATERS

    @property
    def use_cascade(self) -> bool:
        return self.provider in CASCADE_MODES


@dataclass(frozen=True)
class _Rater:
    """The second rater's backend: scoring, checkpoint, model release."""
    score: Callable[[dict], dict]
    checkpoint: CheckpointManager
    doc_chars: Callable[[dict], int] | None = None
    release: Callable[[], None] = _no_release
    summary_extras: dict = field(default_factory=dict)


def _second_rater_name(provider: str) -> str | None:
    """Canonical second-rater name, or None when unsupported."""
    if provider in EMBEDDING_SECOND_RATERS:
        return provider
    name = normalize_provider_name(provider)
    return name if name in SUPPORTED_SECOND_RATERS else None


def _research_question(config: Any) -> str:
    """The primary (Stage 2) question; ``hypothesis_text`` is the fallback."""
    if config is None:
        return ""
    getter = getattr(config, "get_natural_language_question", None)
    question = str(getter() or "").strip() if callable(getter) else ""
    return question or (getattr(config, "hypothesis_text", "") or "").strip()


@contextmanager
def _second_rater_llm(config: Any, provider: str,
                      second_model: str | None) -> Iterator[str | None]:
    """Point ``process.llm`` at an LLM second rater, then restore the previous config.

    Yields the second rater's model name (None for a cross-encoder, which
    leaves the LLM settings alone).
    """
    if provider in EMBEDDING_SECOND_RATERS:
        yield None
        return
    from process import llm_clients
    from process.llm import set_config

    second_cfg = copy.deepcopy(config) if config is not None else Config()
    second_cfg.llm_provider = provider
    if second_model:
        setattr(second_cfg, PROVIDER_MODEL_FIELD[provider], second_model)
    previous = llm_clients._config
    set_config(second_cfg)
    try:
        yield second_cfg.llm_model
    finally:
        set_config(previous)  # type: ignore[arg-type]  # None restores "unconfigured"


def _output_paths(config: Any, output_dir: str | None) -> dict[str, Path]:
    validation_dir = resolve_validation_dir(config)
    default_out_dir = config.output_dir if config is not None else Path.cwd() / "output"
    out_dir = Path(output_dir) if output_dir else default_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "validation": validation_dir,
        "results": validation_dir / "interrater_results.json",
        "summary": validation_dir / "interrater_summary.json",
        "report": out_dir / "interrater_report.md",
    }


def _load_corpus(config: Any, claims_path: str | None,
                 progress: Callable[[str], None]) -> list[dict] | dict:
    """The corpus papers, or a stage error dict."""
    corpus_file = resolve_claims_corpus(config, claims_path)
    if corpus_file is None:
        progress("ERROR: No claims corpus found. Run prior stages first.")
        return {"error": "no_claims_corpus"}
    progress(f"Loading corpus from {corpus_file}")
    papers = load_paper_list(corpus_file)
    if not papers:
        progress("ERROR: Corpus is empty.")
        return {"error": "empty_corpus"}
    progress(f"Corpus loaded: {len(papers)} papers")
    return papers


def _cascade_rater(spec: _RunSpec, papers: list[dict], validation_dir: Path,
                   progress: Callable[[str], None]) -> _Rater:
    from process.relevance_cascade import (
        CascadeScreener,
        fingerprint,
        model_signature,
        paper_passage,
        relevance_result,
        settings,
    )
    options = settings(spec.config)
    if spec.model_path:
        options["package"] = str(Path(spec.model_path).expanduser().resolve())
    screener = CascadeScreener(**options)
    screener.prepare(spec.provider)
    key = fingerprint({"mode": spec.provider, "options": options, "question": spec.hypothesis,
                       "papers": papers, "sample_size": spec.sample_size, "seed": spec.seed,
                       "models": model_signature(options["package"]),
                       "identity": _IDENTITY_SCHEME})
    progress("pw4 screening comparison: binary 0/5 decisions; shared models are not independent raters.")

    def score(paper: dict) -> dict:
        passage, source = paper_passage(paper)
        return relevance_result(screener.screen(spec.hypothesis, [passage], spec.provider)[0], source)

    return _Rater(
        score=score,
        checkpoint=CheckpointManager(validation_dir, f"interrater_cascade_{key}.json"),
        summary_extras={"screening_mode": spec.provider, "screening_settings": options,
                        "score_kind": "binary_screening_decision",
                        "independence_note": _CASCADE_NOTE},
    )


def _cross_encoder_rater(spec: _RunSpec, checkpoint: CheckpointManager,
                         progress: Callable[[str], None]) -> _Rater | None:
    """A brt-bert/brt-irr rater, or None when its model is unavailable."""
    from process import relevance_embedder as embedder

    backend = "embedding" if spec.model_path else spec.provider
    if not embedder.model_available(backend, spec.model_path):
        progress(
            f"ERROR: {spec.provider} relevance model not found. "
            "brt-bert expects brt-bert-relevance/final; "
            "brt-irr expects brt-bert-irr-buddy-mlm/relevance/final. "
            "Pass a custom model path/HF id in the Model field to override."
        )
        return None
    rewritten = embedder.question_to_claim(spec.hypothesis)
    progress(f"{spec.provider} backend={backend} | score-only IRR | "
             f"hypothesis chars={len(spec.hypothesis)} -> query={rewritten[:80]!r}")

    def score(paper: dict) -> dict:
        return embedder.score_relevance_embedding(
            paper, hypothesis=spec.hypothesis, backend=backend,
            model_path=spec.model_path, device="auto")

    def release() -> None:
        # Cross-encoder checkpoints are hundreds of MB; evict them once
        # scoring is done so they are not pinned for the process lifetime.
        try:
            embedder.clear_model_cache()
        except Exception as exc:
            logger.warning("Could not release the %s model cache: %s", spec.provider, exc)

    shared = spec.provider in _SHARED_WEIGHT_RATERS and not spec.model_path
    return _Rater(score=score, checkpoint=checkpoint,
                  doc_chars=lambda paper: len(embedder._paper_text(paper)),
                  release=release,
                  summary_extras={"independence_note": _BRT_BERT_NOTE} if shared else {})


def _build_rater(spec: _RunSpec, papers: list[dict], validation_dir: Path,
                 progress: Callable[[str], None]) -> _Rater | None:
    if spec.use_cascade:
        return _cascade_rater(spec, papers, validation_dir, progress)
    checkpoint = CheckpointManager(
        validation_dir, "interrater_checkpoint.json",
        fingerprint=input_fingerprint(
            papers, sample_size=spec.sample_size, seed=spec.seed, provider=spec.provider,
            model=spec.model_name, model_path=spec.model_path, question=spec.hypothesis,
            identity=_IDENTITY_SCHEME))
    if spec.use_bert:
        return _cross_encoder_rater(spec, checkpoint, progress)
    return _Rater(
        score=lambda paper: provider_score_relevance(
            spec.model_name, paper, hypothesis=spec.hypothesis),
        checkpoint=checkpoint,
    )


# ── Per-paper rating ─────────────────────────────────────────────────────

def _paper_key(paper: dict, index: int) -> str:
    """Stable identity: DOI, then a source ID, else the sample position.

    The checkpoint fingerprint pins corpus, sample size and seed, so a sample
    position names the same paper on resume. Titles are not identities:
    untitled papers and different papers sharing a title stay distinct.
    """
    doi = normalize_doi(paper.get("doi"))
    if doi:
        return f"doi:{doi}"
    for name in _ID_FIELDS:
        value = str(paper.get(name) or "").strip()
        if value:
            return f"{name}:{value}"
    return f"sample:{index}"


def _pending_papers(sample: list[dict], completed: list[dict]
                    ) -> tuple[list[tuple[int, dict, str]], int]:
    """Papers still to rate as (sample index, paper, key), and duplicates skipped."""
    done = {r.get("paper_key") for r in completed}
    seen: set[str] = set()
    pending: list[tuple[int, dict, str]] = []
    duplicates = 0
    for i, paper in enumerate(sample):
        key = _paper_key(paper, i)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        if key not in done:
            pending.append((i, paper, key))
    return pending, duplicates


def _paper_title(paper: dict) -> str:
    return str(paper.get("title") or "Unknown")


def _claim_fields(paper: dict, extracted: dict) -> dict:
    """``original_*``/``second_*`` claim lists (Stage 3 nests them under "claims")."""
    return {
        f"{side}_{name}": claim_list(source, name)
        for name in ("findings", "mechanisms", "conditions", "key_entities")
        for side, source in (("original", paper), ("second", extracted))
    }


def _second_score(spec: _RunSpec, rater: _Rater, paper: dict
                  ) -> tuple[int | None, str | None, dict]:
    """(score, error, raw result) from the second rater."""
    try:
        result = rater.score(paper)
    except Exception as exc:
        if spec.use_cascade:
            raise  # A failed model call is not a negative relevance judgment.
        logger.error("Score failed for '%s': %s", _paper_title(paper)[:_LOG_TITLE_CHARS], exc)
        return None, f"score_failed: {type(exc).__name__}: {exc}", {}
    # An unparseable reply (relevance_score=None) or an error is a failed
    # rating, kept out of the agreement metrics.
    score, error = _second_rating(result)
    return score, error, result


def _second_claims(spec: _RunSpec, paper: dict) -> tuple[dict, str | None]:
    """Second-rater claims; cross-encoders cannot extract, so they run score-only."""
    if spec.use_bert:
        return {}, None
    try:
        result = provider_extract_claims(
            spec.model_name, paper,
            research_config=getattr(spec.config, "research_config", None))
    except Exception as exc:
        logger.error("Extract failed for '%s': %s", _paper_title(paper)[:_LOG_TITLE_CHARS], exc)
        return {}, f"extract_failed: {type(exc).__name__}: {exc}"
    error = result.get("error")
    return result, str(error) if error else None


def _rate_paper(spec: _RunSpec, rater: _Rater, total: int,
                item: tuple[int, dict, str]) -> dict:
    """Worker: re-score and re-extract one paper with the second rater."""
    i, paper, key = item
    title = _paper_title(paper)
    logger.info("[%d/%d] Re-scoring: %s", i + 1, total, title[:_LOG_TITLE_CHARS])
    second_score, score_error, score_result = _second_score(spec, rater, paper)
    # Show paper-text length so an all-failed cross-encoder run is diagnosable.
    doc_chars = rater.doc_chars(paper) if rater.doc_chars else None
    extracted, extract_error = _second_claims(spec, paper)
    record = {
        "paper_key": key,
        "title": title,
        "original_score": paper.get("relevance_score", 0),
        "second_score": second_score,
        **_claim_fields(paper, extracted),
    }
    if score_error:
        record["error"] = score_error
    if extract_error:
        record["extract_error"] = extract_error
    if spec.use_cascade:
        record["screening"] = score_result.get("screening")
        record["second_score_kind"] = "binary_screening_decision"
    return {"record": record, "i": i, "doc_chars": doc_chars}


def _failed_out(item: tuple[int, dict, str], error: str) -> dict:
    """Worker output for a paper whose worker failed or timed out."""
    i, paper, key = item
    record = {"paper_key": key, "title": _paper_title(paper),
              "original_score": paper.get("relevance_score"), "second_score": None,
              "error": error, **_claim_fields({}, {})}
    return {"record": record, "i": i, "doc_chars": None}


def _progress_line(worker_out: dict, total: int) -> str:
    record = worker_out["record"]
    line = (f"[{worker_out['i'] + 1}/{total}] Re-scoring: {record['title'][:_LOG_TITLE_CHARS]} "
            f"→ original={record['original_score']}, second={record['second_score']}")
    if worker_out["doc_chars"] is not None:
        line += f" (doc={worker_out['doc_chars']} chars)"
    return line


# ── Scoring loop ─────────────────────────────────────────────────────────

def _concurrency(config: Any, kw: dict, use_bert: bool) -> tuple[int, float, float]:
    """(workers, stagger seconds, per-item timeout). Cross-encoders run alone."""
    workers = kw.get("concurrent_workers")
    if workers is None and config is not None:
        workers = getattr(config, "concurrent_workers", None)
    workers = 1 if use_bert else max(1, int(workers or DEFAULT_WORKERS))
    stagger = max(0.0, float(kw.get("stagger_delay", DEFAULT_STAGGER_SECONDS)))
    if "stagger_delay" not in kw and config is not None:
        stagger = max(0.0, float(getattr(config, "stagger_delay", stagger)))
    timeout = float(getattr(config, "worker_timeout_seconds", None)
                    or DEFAULT_WORKER_TIMEOUT_SECONDS)
    return workers, stagger, timeout


def _window_cancel_event(cancel_event: Any) -> Any:
    """run_sliding_window polls ``is_set()``; adapt a plain bool / truthy flag."""
    if cancel_event is None or hasattr(cancel_event, "is_set"):
        return cancel_event
    flag = threading.Event()
    if bool(cancel_event):
        flag.set()
    return flag


def _save_checkpoint(ckpt: CheckpointManager, rows: list[dict], total: int,
                     progress: Callable[[str], None]) -> bool:
    saved = ckpt.force_save(_checkpointable(rows), total)
    if not saved:
        progress("WARNING: Could not save interrater checkpoint.")
    return saved


def _rate_pending(spec: _RunSpec, rater: _Rater, pending: list[tuple[int, dict, str]],
                  rows: tuple[list[dict], list[dict]], total: int,
                  progress: Callable[[str], None], cancel_event: Any, kw: dict
                  ) -> tuple[list[dict], bool]:
    """Rate ``pending`` in a sliding window; ``rows`` = (completed, new records)."""
    completed, new_records = rows
    workers, stagger, timeout = _concurrency(spec.config, kw, spec.use_bert)
    progress(f"Interrater concurrency: workers={workers}")
    failures: dict[int, str] = {}  # pending index -> "failed" | "timed_out"

    def on_failure(idx: int, kind: str, _error: BaseException | None) -> None:
        failures[idx] = kind

    def build(idx: int, worker_out: dict | None) -> dict:
        if worker_out is None:  # worker failed or timed out and was skipped
            timed_out = failures.get(idx) == "timed_out"
            worker_out = _failed_out(pending[idx], "worker_timeout" if timed_out else "worker_failed")
            logger.error("Interrater worker %s for '%s'", "timed out" if timed_out else "failed",
                         worker_out["record"]["title"][:_LOG_TITLE_CHARS])
        new_records.append(worker_out["record"])
        progress(_progress_line(worker_out, total))
        return worker_out["record"]

    def on_checkpoint() -> None:
        if len(new_records) % CHECKPOINT_EVERY == 0:
            done = completed + new_records
            if _save_checkpoint(rater.checkpoint, done, total, progress):
                progress(f"Checkpoint saved: {len(done)} results")

    return run_sliding_window(
        items=pending, worker_fn=partial(_rate_paper, spec, rater, total), result_fn=build,
        workers=workers, stagger=stagger, checkpoint_fn=on_checkpoint,
        cancel_event=_window_cancel_event(cancel_event), future_timeout=timeout,
        on_failure=on_failure,
    )


def _finalize(spec: _RunSpec, rater: _Rater, results: list[dict], duplicates: int,
              paths: dict[str, Path], total: int, progress: Callable[[str], None]) -> dict:
    """Compute agreement, write results/summary/report, settle the checkpoint."""
    progress(f"Computing agreement metrics for {len(results)} papers…")
    summary = {**compute_agreement(results, score_only=spec.use_bert),
               "n_duplicates_skipped": duplicates, **rater.summary_extras}
    progress(
        f"Agreement: weighted κ={_fmt(summary['score_weighted_kappa'])} "
        f"({summary['interpretation']}), κ={_fmt(summary['score_kappa'])}, "
        f"r={_fmt(summary['score_correlation'])}, "
        f"adjacent={format_adjacent_agreement(summary)}, "
        f"claim overlap={_fmt(summary['claim_jaccard_mean'])}"
    )
    report_text = generate_report(summary, results)
    if spec.use_cascade:
        report_text += f"\nScreening mode: {spec.provider}.\n"

    atomic_write_json(paths["results"], results)
    progress(f"Results saved: {paths['results']}")
    atomic_write_json(paths["summary"], summary)
    progress(f"Summary saved: {paths['summary']}")
    atomic_write_text(paths["report"], report_text)
    progress(f"Report saved: {paths['report']}")
    _finish_checkpoint(rater.checkpoint, results, total, progress)
    progress("Stage 8 complete.")
    return summary


def _rate_sample(spec: _RunSpec, rater: _Rater, papers: list[dict], paths: dict[str, Path],
                 progress: Callable[[str], None], cancel_event: Any, kw: dict) -> dict:
    sample = stratified_sample(papers, spec.sample_size, seed=spec.seed)
    total = len(sample)
    progress(f"Sampled {total} papers (requested {spec.sample_size})")
    loaded, start_idx = rater.checkpoint.load()
    completed = [r for r in loaded if r.get("paper_key")]
    if start_idx > 0:
        progress(f"Resuming from checkpoint: {start_idx} already completed")
    pending, duplicates = _pending_papers(sample, completed)
    if duplicates:
        progress(f"Skipped {duplicates} duplicate paper(s) in the sample (same DOI or ID).")

    new_records: list[dict] = []
    try:
        new_results, cancelled = _rate_pending(spec, rater, pending, (completed, new_records),
                                               total, progress, cancel_event, kw)
        results = completed + new_results
        if cancelled or is_cancelled(cancel_event):
            progress("Cancelled by user.")
            _save_checkpoint(rater.checkpoint, results, total, progress)
            return {"cancelled": True, "partial_results": len(results)}
        summary = _finalize(spec, rater, results, duplicates, paths, total, progress)
    except BaseException:
        # Keep completed ratings so a re-run resumes instead of starting over.
        _save_checkpoint(rater.checkpoint, completed + new_records, total, progress)
        raise
    finally:
        rater.release()
    return _run_result(summary, paths)


def _run_result(summary: dict, paths: dict[str, Path]) -> dict:
    keys = ("n_papers", "n_failed", "n_duplicates_skipped", "score_kappa",
            "score_weighted_kappa", "score_correlation", "adjacent_agreement_pct",
            "adjacent_agreement_count", "adjacent_agreement_total", "claim_jaccard_mean",
            "interpretation")
    return {**{k: summary[k] for k in keys},
            "results_path": str(paths["results"]),
            "summary_path": str(paths["summary"]),
            "report_path": str(paths["report"])}


# ── Main entry point ─────────────────────────────────────────────────────

def run(
    claims_path: str | None = None,
    output_dir: str | None = None,
    config: Any = None,
    second_provider: str = "ollama",
    second_model: str | None = None,
    sample_size: int = 100,
    seed: int = 42,
    cancel_event: Any = None,
    progress_callback: Callable[[str], None] | None = None,
    **kw,
) -> dict:
    """Run inter-rater reliability analysis.

    1. Load claims
    2. Stratified sample of papers
    3. Re-score each with second LLM provider
    4. Re-extract claims with second LLM
    5. Compute agreement metrics
    6. Generate report
    7. Save results + summary + report

    An LLM second rater is made the active ``process.llm`` config for the
    run only; the previous config is restored afterwards, even on error.
    """
    progress = make_progress(logger, progress_callback)
    paths = _output_paths(config, output_dir)
    papers = _load_corpus(config, claims_path, progress)
    if isinstance(papers, dict):
        return papers

    provider = _second_rater_name(second_provider)
    if provider is None:
        progress(f"ERROR: Unsupported second-rater provider {second_provider!r}. "
                 f"Supported: {sorted(SUPPORTED_SECOND_RATERS)}.")
        return {"error": "unsupported_second_provider"}
    hypothesis = _research_question(config)
    if provider in EMBEDDING_SECOND_RATERS and not hypothesis:
        progress("ERROR: no research question found in config (natural_language_question / "
                 "hypothesis_text both empty); a cross-encoder rater cannot score without it. "
                 "Set the research question in the Research config tab.")
        return {"error": "no_research_question"}
    if hypothesis:
        progress(f"Research question: {hypothesis[:120]!r}")
    else:
        progress("WARNING: no research question found in config "
                 "(natural_language_question / hypothesis_text both empty). "
                 "The LLM will judge relevance against a generic placeholder. "
                 "Set the research question in the Research config tab.")

    with _second_rater_llm(config, provider, second_model) as llm_model:
        model_name = (llm_model or second_model
                      or (config.llm_model if config is not None else "") or "default")
        progress(f"Second rater: provider={provider}, model={model_name}")
        model_path = (second_model or "").strip() if provider in EMBEDDING_SECOND_RATERS else ""
        spec = _RunSpec(config=config, provider=provider, model_name=model_name,
                        model_path=model_path, hypothesis=hypothesis,
                        sample_size=sample_size, seed=seed)
        rater = _build_rater(spec, papers, paths["validation"], progress)
        if rater is None:
            return {"error": "embedding_model_unavailable"}
        return _rate_sample(spec, rater, papers, paths, progress, cancel_event, kw)


def _checkpointable(records: list[dict]) -> list[dict]:
    """Rows worth resuming; failed second ratings are retried on the next run."""
    return [r for r in records if not r.get("error")]


def _second_rating(score_result: dict) -> tuple[int | None, str | None]:
    """(score, error) from a second-rater result; a missing score is a failure."""
    score = score_result.get("relevance_score")
    error = score_result.get("error")
    if score is None or error:
        return None, str(error or "unparseable_reply")
    return score, None


def _finish_checkpoint(ckpt: CheckpointManager, results: list[dict], total: int,
                       progress: Callable[[str], None]) -> None:
    """Clear the checkpoint, or keep the rated rows so a re-run retries failures."""
    failed = len(results) - len(_checkpointable(results))
    if not failed:
        ckpt.cleanup()
        return
    progress(
        f"WARNING: {failed} second rating(s) failed and were excluded from the "
        "agreement metrics. Re-run Stage 8 to retry them."
    )
    _save_checkpoint(ckpt, results, total, progress)
