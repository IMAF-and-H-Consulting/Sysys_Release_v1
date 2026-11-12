"""Stage 3: Structured claim extraction.

Reads relevant.json from Stage 2, extracts structured claims from each paper
using LLM + heuristic fallback, outputs claims.json.
"""
import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.claims import DIRECTION_FIELD, UNKNOWN_DIRECTION, claim_direction, claim_list
from core.io import atomic_write_json
from process.domain_patterns import contains_term, extraction_patterns, paper_text
from process.domain_patterns import research_config as _research_config
from process.term_normalization import (
    normalize_claim_records,
    normalization_audit_path,
)

if TYPE_CHECKING:
    from core.config import Config

logger = logging.getLogger(__name__)

DEFAULT_INPUT = Path("data/claims/relevant.json")
DEFAULT_OUTPUT = Path("data/claims/claims.json")

# A sentence ends at a terminator followed by whitespace and a capital, digit,
# quote or bracket, so decimals ("2.5 fold") stay inside their sentence.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")

# Fields the heuristic fills when an LLM record leaves them empty.
_LIST_FILL_FIELDS = ("findings", "mechanisms", "conditions", "key_entities")
_STRING_FILL_FIELDS = ("methodology", "relationships", "key_quote")

PARSE_FAILED = "parse_failed"


def _clean_terms(values: list[Any]) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for value in values:
        term = " ".join(str(value or "").split()).strip()
        if not term:
            continue
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        terms.append(term)
    return terms


def _search_terms_from_config(config: Any) -> list[str]:
    research = _research_config(config)
    search_raw = research.get("search")
    search = search_raw if isinstance(search_raw, dict) else {}
    values: list[Any] = []
    values.extend(search.get("base_terms") or [])
    for domain in search.get("domains") or []:
        if not isinstance(domain, dict):
            continue
        values.append(domain.get("name"))
        values.extend(domain.get("terms") or [])
        values.extend(domain.get("require_also") or [])
    values.extend(getattr(config, "co_occurring_conditions", []) or [])
    return _clean_terms(values)


# ---------------------------------------------------------------------------
# Heuristic extraction (pure regex/keyword, no LLM)
# ---------------------------------------------------------------------------

def _heuristic_extract(paper: dict, config: Any = None, *, patterns=None,
                       search_terms=None) -> dict:
    """Regex/keyword-based extraction as fallback when LLM is unavailable."""
    abstract = paper_text(paper, "abstract")
    original_text = abstract + " " + paper_text(paper, "title")
    text = original_text.lower()
    patterns = extraction_patterns(config) if patterns is None else patterns
    condition_patterns = patterns["conditions"]
    configured_terms = (search_terms if search_terms is not None
                        else _search_terms_from_config(config))
    mechanism_terms = patterns["mechanisms"]

    conditions = [
        c for c, patterns in condition_patterns.items()
        if any(re.search(p, original_text, re.IGNORECASE) for p in patterns)
    ]
    mechanisms = [m for m in mechanism_terms if contains_term(text, m)]
    key_entities = [e for e in patterns["key_entities"] if contains_term(text, e)]
    methodology = next((m for m in patterns["methodology_hints"] if contains_term(text, m)), "")
    network_effects = any(contains_term(text, t) for t in patterns["network_terms"])

    # Extract sentences that look like findings
    findings = [
        s for s in _sentences(abstract)
        if any(contains_term(s, h) for h in patterns["finding_hints"])
    ]
    # Hint words and search terms only show the paper is on topic (it was
    # retrieved with those terms); they are not evidence that it supports
    # the question, so the outcome stays unknown.
    topic_match = bool(
        findings
        or any(contains_term(text, term) for term in configured_terms)
    )

    claims = {
        "findings": findings,
        "mechanisms": mechanisms,
        "conditions": conditions,
        "key_entities": key_entities,
        "methodology": methodology,
        "supports_primary_question": None,
        DIRECTION_FIELD: UNKNOWN_DIRECTION,
        "heuristic_topic_match": topic_match,
        "network_effects_described": network_effects,
        "relationships": "",
        "key_quote": findings[0] if findings else "",
        "error": None,
    }
    # Opt-in legacy flag: only a pack or config that defines
    # criticality_terms produces it.
    if patterns["criticality_terms"]:
        claims["supports_criticality"] = any(
            contains_term(text, t) for t in patterns["criticality_terms"]
        )
    return claims


def _sentences(text: str) -> list[str]:
    """Split text into sentences without their final terminator."""
    stripped = (part.strip().rstrip(".!?").strip() for part in _SENTENCE_BREAK.split(text))
    return [s for s in stripped if s]


def _merge_heuristic_into_claims(claims: dict, heuristic: dict) -> dict:
    """Fill any empty fields in LLM claims with heuristic results."""
    merged = dict(claims)
    for field in _LIST_FILL_FIELDS:
        # claim_list also sees legacy names (brain_regions) from older schemas.
        if not claim_list(merged, field):
            merged[field] = heuristic.get(field, [])
    for field in _STRING_FILL_FIELDS:
        if not merged.get(field):
            merged[field] = heuristic.get(field, "")
    return merged


def _heuristic_fields(claims: dict, merged: dict) -> list[str]:
    """Fields whose value in ``merged`` came from the heuristic, not the LLM."""
    return [f for f in (*_LIST_FILL_FIELDS, *_STRING_FILL_FIELDS)
            if merged.get(f) and merged.get(f) != claims.get(f)]


def _with_schema_fields(claims: dict, field_names: list[str]) -> dict:
    """Configured fields the heuristic cannot extract are present as None."""
    return {**{name: None for name in field_names}, **claims}


def _has_recovered_values(claims: dict, field_names: list[str]) -> bool:
    return any(claims.get(name) not in (None, "", [], False) for name in field_names)


def _finalize_llm_paper(paper: dict, config: Any, patterns: dict,
                        search_terms: list[str],
                        field_names: list[str] | None = None) -> dict:
    """Merge heuristics into one LLM result and record its evidence direction.

    A failed LLM call falls back to the heuristic claims, keeping the error
    under ``llm_error`` so the fallback stays visible downstream. A reply
    that could not be parsed as JSON but yielded fields through regex
    recovery keeps those fields, and the heuristic only fills the gaps.
    """
    field_names = field_names or []
    claims = paper.get("claims") or {}
    heuristic = _heuristic_extract(paper, config, patterns=patterns,
                                   search_terms=search_terms)
    error = claims.get("error")
    is_partial = bool(error) and _has_recovered_values(claims, field_names)
    if error and not is_partial:
        logger.info("[Extract] LLM failed for '%s' — using heuristic",
                    (paper_text(paper, "title") or "?")[:60])
        fallback = {**_with_schema_fields(heuristic, field_names),
                    "llm_error": str(error), "extraction_method": "heuristic_fallback"}
        return {**paper, "claims": fallback}
    merged = _merge_heuristic_into_claims(claims, heuristic)
    merged[DIRECTION_FIELD] = claim_direction(merged)
    merged["heuristic_fields"] = _heuristic_fields(claims, merged)
    merged["extraction_method"] = "llm+heuristic"
    if is_partial:
        merged.update(error=None, llm_error=str(error),
                      extraction_method="llm_partial+heuristic")
    return {**paper, "claims": merged}


def _method_count(papers: list[dict], method: str) -> int:
    return sum(1 for p in papers
               if (p.get("claims") or {}).get("extraction_method") == method)


def _schema_field_names(research: Any) -> list[str]:
    from process.claim_extractor import _field_specs
    return [name for name, _, _ in _field_specs(research)]


def _is_cancelled(cancel_event: Any) -> bool:
    if cancel_event is None:
        return False
    return cancel_event.is_set() if hasattr(cancel_event, "is_set") else bool(cancel_event)


def _run_llm_extraction(papers: list[dict], config: Any, model: str, *,
                        cancel_event: Any, progress_callback: Callable[[str], None] | None,
                        use_thinking: bool, max_tokens: int | None,
                        context_tokens: int, research: Any,
                        ) -> tuple[list[dict], bool]:
    """LLM extraction with checkpoint/resume; the checkpoint outlives the call."""
    from core.config import DEFAULT_CHECKPOINT_EVERY
    from process.claim_extractor import extract_all_claims
    from process.llm import set_config

    set_config(config)
    extracted, was_cancelled = extract_all_claims(
        papers,
        model=model,
        checkpoint_dir=config.claims_dir,
        checkpoint_every=getattr(config, "checkpoint_every", DEFAULT_CHECKPOINT_EVERY),
        cancel_event=cancel_event,
        progress_callback=progress_callback,
        use_thinking=use_thinking,
        max_tokens=max_tokens,
        context_tokens=context_tokens,
        research_config=research,
        config=config,
        keep_checkpoint=True,
    )
    if was_cancelled:
        logger.info("LLM extraction was cancelled")
    return extracted, was_cancelled


def _report(message: str, progress_callback: Callable[[str], None] | None) -> None:
    logger.info(message)
    if progress_callback:
        progress_callback(message)


def _run_heuristic_extraction(papers: list[dict], config: Any, patterns: dict,
                              search_terms: list[str], cancel_event: Any,
                              progress_callback: Callable[[str], None] | None,
                              ) -> tuple[list[dict], bool]:
    """Pure heuristic extraction (no LLM configured)."""
    total = len(papers)
    extracted: list[dict] = []
    for i, paper in enumerate(papers):
        if _is_cancelled(cancel_event):
            logger.info("Heuristic extraction cancelled at %d/%d", i + 1, total)
            return extracted, True
        title = (paper_text(paper, "title") or "?")[:80]
        _report(f"[{i + 1}/{total}] Extracting (heuristic): {title}", progress_callback)
        claims = {**_heuristic_extract(paper, config, patterns=patterns,
                                       search_terms=search_terms),
                  "extraction_method": "heuristic_only"}
        extracted.append({**paper, "claims": claims})
        _report(f"[{i + 1}/{total}] {title} → {len(claims['findings'])} findings, "
                f"{len(claims['mechanisms'])} mechanisms, "
                f"{len(claim_list(claims, 'key_entities'))} entities", progress_callback)
    return extracted, False


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------

def run(
    config: "Config | None" = None,
    cancel_event: Any = None,
    progress_callback: Callable[[str], None] | None = None,
    **kwargs,
) -> dict:
    """Run Stage 3 extraction.

    Strategy:
    1. Try LLM extraction via process.claim_extractor (if LLM is configured)
    2. Fall back to heuristic extraction for any papers that fail
    3. Always run heuristic as a supplement to fill in any empty fields

    Returns a summary dict with paper count and output path.
    """
    patterns = extraction_patterns(config)  # Validate before LLM calls or output writes.
    search_terms = _search_terms_from_config(config)

    # Resolve paths
    input_path = kwargs.get("input_path") or kwargs.get("relevant_path") or (
        config.claims_dir / "relevant.json" if config else DEFAULT_INPUT)
    output_path = kwargs.get("output_path") or (
        config.claims_dir / "claims.json" if config else DEFAULT_OUTPUT)
    input_path = Path(input_path)
    output_path = Path(output_path)

    logger.info("Stage 3 — Claim Extraction")
    logger.info("  Input:  %s", input_path)
    logger.info("  Output: %s", output_path)

    # Load relevant papers
    if not input_path.exists():
        msg = f"Input file not found: {input_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    with open(input_path, encoding="utf-8") as f:
        papers = json.load(f)

    total = len(papers)
    logger.info("  Loaded %d relevant papers", total)
    if progress_callback:
        progress_callback(f"Loaded {total} relevant papers")

    if total == 0:
        logger.warning("No papers to extract — writing empty claims.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output_path, [])
        normalized, normalization_audit = normalize_claim_records([], config)
        assert normalized == []
        term_audit_path = normalization_audit_path(output_path)
        atomic_write_json(term_audit_path, normalization_audit)
        return {
            "total": 0,
            "extracted": 0,
            "output_path": str(output_path),
            "term_normalization_path": str(term_audit_path),
        }

    # Determine if LLM extraction is available
    use_llm = False
    model = ""
    max_tokens = int(kwargs.get("max_tokens", 0) or 0)
    use_thinking = kwargs.get("use_thinking", False)

    if config and getattr(config, "llm_provider", ""):
        model = getattr(config, "llm_model", "")
        if model:
            use_llm = True
            logger.info("  LLM extraction: provider=%s, model=%s",
                        config.llm_provider, model)
        else:
            logger.info("  No LLM model configured — heuristic-only mode")
    else:
        logger.info("  No LLM config — heuristic-only mode")

    research = getattr(config, "research_config", None)
    if use_llm:
        assert config is not None  # use_llm is only set True when config is truthy
        from process.llm_budget import resolve_model_capacity
        capacity = resolve_model_capacity(
            config,
            requested_context_tokens=kwargs.get("context_tokens"),
            model=model,
        )
        logger.info(
            "  Token action: context=%d (%s), response=%s; complete abstracts are used "
            "when they fit and any context-derived reduction is saved per paper",
            capacity.context_window_tokens, capacity.source,
            max_tokens if max_tokens > 0 else "auto",
        )
        extracted_papers, was_cancelled = _run_llm_extraction(
            papers, config, model, cancel_event=cancel_event,
            progress_callback=progress_callback, use_thinking=use_thinking,
            max_tokens=max_tokens, context_tokens=capacity.context_window_tokens,
            research=research)
    else:
        extracted_papers, was_cancelled = _run_heuristic_extraction(
            papers, config, patterns, search_terms, cancel_event, progress_callback)
    if was_cancelled:
        return {"total": total, "extracted": len(extracted_papers),
                "cancelled": True, "output_path": str(output_path)}

    field_names = _schema_field_names(research)
    if use_llm:
        # Heuristics fill gaps in LLM results and replace failed extractions.
        extracted_papers = [
            _finalize_llm_paper(paper, config, patterns, search_terms, field_names)
            for paper in extracted_papers
        ]
    else:
        extracted_papers = [
            {**paper, "claims": _with_schema_fields(paper["claims"], field_names)}
            for paper in extracted_papers
        ]

    # Structured categorical values become canonical at their producer boundary.
    # The paid/raw model checkpoint remains untouched; claims.json receives the
    # canonical values plus per-record raw mappings, and the corpus mapping is a
    # sibling audit.  Synthesis and later stages therefore never normalize data.
    extracted_papers, normalization_audit = normalize_claim_records(
        extracted_papers, config,
    )
    term_audit_path = normalization_audit_path(output_path)
    for field, details in normalization_audit.get("fields", {}).items():
        _report(
            "Structured-term action: "
            f"{field}: {details.get('input_variants', 0)} raw variants -> "
            f"{details.get('canonical_terms', 0)} canonical labels; "
            f"{details.get('changed_occurrences', 0)} occurrence(s) changed; "
            f"{details.get('within_record_duplicates_removed', 0)} within-paper "
            "duplicate(s) removed. Raw mappings are retained.",
            progress_callback,
        )

    # Save claims.json and its normalization audit; only then drop the checkpoint
    # holding the raw LLM results.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_path, extracted_papers, default=str)
    atomic_write_json(term_audit_path, normalization_audit, default=str)
    logger.info("Saved %d extracted papers to %s", len(extracted_papers),
                output_path)
    logger.info("Saved structured-term normalization audit to %s", term_audit_path)
    retryable_model_errors = sum(
        1 for paper in extracted_papers
        if (paper.get("claims") or {}).get("llm_error")
        and (paper.get("claims") or {}).get("llm_error") != "no_abstract"
    )
    if use_llm and retryable_model_errors == 0:
        from process.claim_extractor import clear_extraction_checkpoint
        clear_extraction_checkpoint(config.claims_dir)

    # Summary
    n_with_findings = sum(
        1 for p in extracted_papers
        if p.get("claims", {}).get("findings"))
    n_with_mechanisms = sum(
        1 for p in extracted_papers
        if p.get("claims", {}).get("mechanisms"))
    n_with_entities = sum(
        1 for p in extracted_papers if claim_list(p, "key_entities"))

    summary = {
        "total": total,
        "extracted": len(extracted_papers),
        "with_findings": n_with_findings,
        "with_mechanisms": n_with_mechanisms,
        "with_key_entities": n_with_entities,
        "heuristic_fallback": _method_count(extracted_papers, "heuristic_fallback"),
        "heuristic_only": _method_count(extracted_papers, "heuristic_only"),
        "llm_partial": _method_count(extracted_papers, "llm_partial+heuristic"),
        "unresolved_model_extractions": retryable_model_errors,
        "status": "partial" if retryable_model_errors else "complete",
        "with_field_warnings": sum(
            1 for p in extracted_papers if (p.get("claims") or {}).get("field_warnings")),
        "output_path": str(output_path),
        "term_normalization_path": str(term_audit_path),
        "term_normalization": {
            "policy_id": normalization_audit["policy_id"],
            "fields": {
                field: {
                    "input_variants": details.get("input_variants", 0),
                    "canonical_terms": details.get("canonical_terms", 0),
                    "changed_occurrences": details.get("changed_occurrences", 0),
                    "within_record_duplicates_removed": details.get(
                        "within_record_duplicates_removed", 0,
                    ),
                }
                for field, details in normalization_audit.get("fields", {}).items()
            },
        },
    }
    logger.info("Extraction summary: %s", summary)
    if progress_callback:
        progress_callback(
            f"Extraction complete: {len(extracted_papers)} papers, "
            f"{n_with_findings} with findings, "
            f"{n_with_mechanisms} with mechanisms")
    if summary["heuristic_fallback"]:
        msg = (f"WARNING: LLM extraction failed for {summary['heuristic_fallback']} "
               "paper(s); their claims are heuristic and their outcome is unknown")
        logger.warning(msg)
        if progress_callback:
            progress_callback(msg)
    if retryable_model_errors:
        msg = (
            f"PARTIAL: {retryable_model_errors} model extraction(s) remain unresolved. "
            "Completed fields and heuristic supplements were saved, and the checkpoint "
            "was retained before the first unresolved paper so a rerun retries it."
        )
        logger.warning(msg)
        if progress_callback:
            progress_callback(msg)

    return summary
