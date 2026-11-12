"""LLM-based claim extraction for the Sysys pipeline."""
import logging
import re
import time
from copy import deepcopy
from collections.abc import Callable
from pathlib import Path
from typing import Any

from core.checkpoint import CheckpointManager, input_fingerprint
from core.claims import claim_flag, claim_list
from process.domain_patterns import paper_text
from process.llm_budget import (
    COMPATIBILITY_FALLBACK_CONTEXT_TOKENS,
    ESTIMATOR_CHARS_PER_TOKEN,
    BudgetPlanningError,
    ModelCapacity,
    TokenBudget,
    estimate_messages_tokens,
    plan_token_budget,
    resolve_model_capacity,
)
from process.llm_context import ContextWindowExceededError, OutputLimitExceededError

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_EVERY = 1
_TOKEN_POLICY_VERSION = "context-aware/v1"
_EXPLICIT_OUTPUT_REASON = (
    "The user selected this Stage 3 response-token override. It replaces automatic "
    "context-relative extraction output scaling for this request."
)
_CONTEXT_REDUCTION_REASON = (
    "The complete abstract did not fit after reserving the task-scaled structured response "
    "and safety allowance, so a context-derived opening prefix was used. Coverage is saved "
    "with the claims for audit and a larger model context will include more text."
)

EXTRACTION_PROMPT = """You are extracting structured research claims from an academic paper for a systematic evidence review.

Research question:
{research_question}

Paper title: {title}
Abstract: {abstract}

Extract the following structured information as a JSON object:
{schema}

Configured extraction guidance:
{extraction_guidance}

Use supports_primary_question for the configured research question above.
If the schema includes evidence_direction, set it to exactly one of "supports", "contradicts" or "neutral"; use "contradicts" when the findings argue against the research question.
Respond with ONLY the JSON object, no other text.
"""

# Legacy default schema, used when no research_config (or no
# extraction.fields list) is supplied. Preserved for backward compatibility.
_LEGACY_FIELDS: list[tuple[str, str, str]] = [
    ("findings", "list", "list of key findings/results stated in the paper"),
    ("mechanisms", "list", "list of mechanisms, constructs, or explanatory factors discussed"),
    ("conditions", "list", "list of populations, conditions, constructs, or contextual factors mentioned"),
    ("key_entities", "list", "list of the specific entities, systems, variables, or constructs the findings concern"),
    ("methodology", "string", "brief description of study design"),
    ("supports_primary_question", "boolean", "whether the findings support the primary research question"),
    ("evidence_direction", "string", "one of: supports, contradicts, neutral — whether the findings support, contradict, or are neutral toward the primary research question"),
    ("network_effects_described", "boolean", "whether network-level effects are described"),
    ("relationships", "string", "description of relationships among the primary topic, populations, constructs, or contextual factors"),
    ("key_quote", "string", "the single most relevant sentence from the abstract"),
]

_FIELD_TYPES = ("list", "string", "boolean")


def _field_specs(research_config: dict | None) -> list[tuple[str, str, str]]:
    """Return (name, type, description) specs for the extraction schema.

    When ``research_config`` defines ``extraction.fields``, the schema is
    built from exactly those fields, so any research question can define its
    own extraction targets. Otherwise the legacy default schema is used.
    """
    return _configured_specs(research_config) or list(_LEGACY_FIELDS)


def _configured_specs(research_config: dict | None) -> list[tuple[str, str, str]]:
    """Valid specs from ``extraction.fields``; fields of unknown type are dropped."""
    fields = _configured_fields(research_config) or []
    specs: list[tuple[str, str, str]] = []
    for field in fields:
        if not isinstance(field, dict):
            continue
        name = str(field.get("name") or "").strip()
        ftype = str(field.get("type") or "string").strip().lower()
        desc = str(field.get("description") or "").strip()
        if name and ftype in _FIELD_TYPES:
            specs.append((name, ftype, desc or name))
    return specs


def _configured_fields(research_config: dict | None) -> list | None:
    """The raw ``extraction.fields`` list, or None when none is configured."""
    if not isinstance(research_config, dict):
        return None
    extraction = research_config.get("extraction")
    if not isinstance(extraction, dict):
        return None
    fields = extraction.get("fields")
    return fields if isinstance(fields, list) else None


def _schema_block(specs: list[tuple[str, str, str]]) -> str:
    """Render the JSON schema block for the extraction prompt."""
    # The block is passed to str.format as a value, so braces are literal.
    lines = ["{"]
    for i, (name, ftype, desc) in enumerate(specs):
        if ftype == "list":
            rendered = f'["{desc}"]'
        elif ftype == "boolean":
            rendered = "true/false"
        else:
            rendered = f'"{desc}"'
        comma = "," if i < len(specs) - 1 else ""
        lines.append(f'    "{name}": {rendered}{comma}')
    lines.append("}")
    return "\n".join(lines)


def _empty_claims(specs: list[tuple[str, str, str]] | None = None) -> dict:
    """Type-appropriate empty claims dict for the given schema."""
    result: dict = {}
    for name, ftype, _ in specs or _LEGACY_FIELDS:
        result[name] = [] if ftype == "list" else (False if ftype == "boolean" else "")
    result["error"] = None
    return result


_EMPTY_CLAIMS = _empty_claims()

_CHECKPOINT_FILENAME = "extract_checkpoint.json"


def _claim_think_mode(model: str, enabled: bool = True) -> bool | str | None:
    """Prefer lightweight exposed reasoning for models that support it."""
    if not enabled:
        return None
    base = model.split(":", 1)[0].strip().lower()
    if base.startswith("gpt-oss"):
        return "low"
    return True


# ---------------------------------------------------------------------------
# Single-paper extraction
# ---------------------------------------------------------------------------

def _research_question_from_config(research_config: dict | None) -> str:
    if not isinstance(research_config, dict):
        return "What does the scholarly literature report for this research question?"
    relevance = research_config.get("relevance")
    if not isinstance(relevance, dict):
        relevance = {}
    for value in (
        research_config.get("natural_language_question"),
        relevance.get("research_question"),
        research_config.get("hypothesis_text"),
        research_config.get("hypothesis"),
        research_config.get("question"),
        research_config.get("description"),
    ):
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    return "What does the scholarly literature report for this research question?"


def _format_extraction_guidance(research_config: dict | None) -> str:
    """Guidance lines for the configured fields the schema actually uses."""
    lines = [f"- {name}: {desc}" for name, _, desc in _configured_specs(research_config)
             if desc != name]
    return "\n".join(lines) if lines else "- Use the generic schema descriptions above."


def _render_extraction_prompt(
    title: str,
    abstract: str,
    research_config: dict | None = None,
) -> str:
    specs = _field_specs(research_config)
    return EXTRACTION_PROMPT.format(
        research_question=_research_question_from_config(research_config),
        title=title,
        abstract=abstract,
        schema=_schema_block(specs),
        extraction_guidance=_format_extraction_guidance(research_config),
    )


def _extraction_capacity(context_tokens: int | None) -> ModelCapacity:
    value = int(context_tokens or 0)
    from process import llm_clients

    if llm_clients._config is not None:
        # Keep the provider/model output ceiling separate from, and alongside,
        # the already resolved context size passed down by the stage runner.
        return resolve_model_capacity(
            llm_clients._config,
            requested_context_tokens=value if value > 0 else None,
        )
    if value > 0:
        return ModelCapacity(value, source="Stage 3 resolved model context")
    logger.warning(
        "Stage 3 token action: no active LLM config or explicit context; using the "
        "unconfirmed %d-token compatibility fallback for this legacy call",
        COMPATIBILITY_FALLBACK_CONTEXT_TOKENS,
    )
    return ModelCapacity(
        COMPATIBILITY_FALLBACK_CONTEXT_TOKENS,
        source="unconfirmed compatibility fallback",
        confirmed=False,
        detail="No Stage 3 context value was supplied by this legacy caller.",
    )


def _clip_utf8_prefix(text: str, byte_budget: int) -> str:
    raw = text.encode("utf-8")
    if len(raw) <= max(0, byte_budget):
        return text
    return raw[:max(0, byte_budget)].decode("utf-8", errors="ignore").rstrip()


def _extraction_prompt_and_coverage(
    title: str,
    abstract: str,
    research_config: dict | None,
    *,
    context_tokens: int | None,
    max_tokens: int | None,
) -> tuple[str, dict[str, Any], TokenBudget]:
    from process.llm import compress_prompt_text

    capacity = _extraction_capacity(context_tokens)
    original = str(abstract or "").strip()
    candidate = compress_prompt_text(original)
    explicit = int(max_tokens) if max_tokens and int(max_tokens) > 0 else None
    full_prompt = _render_extraction_prompt(title, candidate, research_config)
    try:
        budget = plan_token_budget(
            "claim_extraction", capacity,
            messages=[{"role": "user", "content": full_prompt}],
            expected_items=1,
            explicit_output_tokens=explicit,
            explicit_reason=_EXPLICIT_OUTPUT_REASON,
        )
        fitted = candidate
        prompt = full_prompt
    except BudgetPlanningError:
        empty_prompt = _render_extraction_prompt(title, "", research_config)
        base_budget = plan_token_budget(
            "claim_extraction", capacity,
            messages=[{"role": "user", "content": empty_prompt}],
            expected_items=1,
            explicit_output_tokens=explicit,
            explicit_reason=_EXPLICIT_OUTPUT_REASON,
        )
        available_tokens = max(
            0,
            base_budget.input_capacity_tokens
            - estimate_messages_tokens([{"role": "user", "content": empty_prompt}]),
        )
        fitted = _clip_utf8_prefix(
            candidate, int(available_tokens * ESTIMATOR_CHARS_PER_TOKEN))
        while fitted:
            prompt = _render_extraction_prompt(title, fitted, research_config)
            try:
                budget = plan_token_budget(
                    "claim_extraction", capacity,
                    messages=[{"role": "user", "content": prompt}],
                    expected_items=1,
                    explicit_output_tokens=explicit,
                    explicit_reason=_EXPLICIT_OUTPUT_REASON,
                )
                break
            except BudgetPlanningError:
                fitted = fitted[:max(0, int(len(fitted) * 0.95))].rstrip()
        else:
            prompt = empty_prompt
            budget = base_budget

    source_chars = len(original)
    compressed_chars = len(candidate)
    submitted_chars = len(fitted)
    compression_applied = candidate != original
    context_truncated = submitted_chars < compressed_chars
    truncated = compression_applied or context_truncated
    reasons = []
    if compression_applied:
        reasons.append(
            "Prompt compression removed trailing references/boilerplate or redundant PDF whitespace."
        )
    if context_truncated:
        reasons.append(_CONTEXT_REDUCTION_REASON)
    if not reasons:
        reasons.append("The complete abstract fit the model context.")
    coverage = {
        "source_chars": source_chars,
        "compressed_chars": compressed_chars,
        "submitted_chars": submitted_chars,
        "coverage_ratio": round(submitted_chars / source_chars, 6) if source_chars else 1.0,
        "truncated": truncated,
        "compression_applied": compression_applied,
        "context_truncated": context_truncated,
        "policy": "context-derived-prefix" if context_truncated else (
            "compressed-input" if compression_applied else "complete-abstract"),
        "reason": " ".join(reasons),
        "context_window_tokens": capacity.context_window_tokens,
        "token_policy": _TOKEN_POLICY_VERSION,
    }
    if truncated:
        logger.warning(
            "Stage 3 abstract coverage for %r: %d/%d chars (%.1f%%). %s",
            title[:80], submitted_chars, source_chars, 100.0 * coverage["coverage_ratio"],
            coverage["reason"],
        )
    return prompt, coverage, budget


def build_extraction_prompt(
    title: str,
    abstract: str,
    research_config: dict | None = None,
    *,
    context_tokens: int | None = None,
    max_tokens: int | None = None,
) -> str:
    """Build a context-fitted extraction prompt; complete abstracts are preferred."""
    prompt, _, _ = _extraction_prompt_and_coverage(
        title, abstract, research_config,
        context_tokens=context_tokens, max_tokens=max_tokens)
    return prompt


def extract_claims(model: str, paper: dict, think=None,
                   max_tokens: int | None = None,
                   research_config: dict | None = None,
                   context_tokens: int | None = None) -> dict:
    """Extract structured claims from a single paper using the LLM."""
    from process.llm import chat as llm_chat

    title = paper_text(paper, "title") or "Unknown"
    abstract = paper_text(paper, "abstract")
    if not abstract.strip():
        logger.warning("[Extract] No abstract for '%s' — using heuristic fallback",
                       title[:60])
        return {**_EMPTY_CLAIMS, "error": "no_abstract"}

    prompt, coverage, budget = _extraction_prompt_and_coverage(
        title, abstract, research_config,
        context_tokens=context_tokens, max_tokens=max_tokens)
    specs = _field_specs(research_config)
    messages = [{"role": "user", "content": prompt}]

    logger.info("[Extract] Extracting claims: %s", title[:80])
    response = llm_chat(
        model, messages, temperature=0,
        max_tokens=max_tokens if max_tokens and max_tokens > 0 else None,
        think=think,
        task="claim_extraction",
        expected_items=1,
        context_tokens=budget.context_window_tokens,
        token_budget=budget,
        explicit_limit_reason=_EXPLICIT_OUTPUT_REASON,
    )
    result = _parse_extraction_response(response, specs)
    result["input_coverage"] = coverage

    n_findings = len(result.get("findings", []))
    n_mechanisms = len(result.get("mechanisms", []))
    n_entities = len(claim_list(result, "key_entities"))
    logger.info("[Extract] Result: %d findings, %d mechanisms, %d entities",
                n_findings, n_mechanisms, n_entities)
    return result


def _config_for_extraction_specs(
    research_config: dict | None,
    specs: list[tuple[str, str, str]],
) -> dict:
    """Copy the research config while narrowing only the extraction schema."""
    narrowed = deepcopy(research_config) if isinstance(research_config, dict) else {}
    extraction = dict(narrowed.get("extraction") or {})
    extraction["fields"] = [
        {"name": name, "type": field_type, "description": description}
        for name, field_type, description in specs
    ]
    narrowed["extraction"] = extraction
    return narrowed


def _merge_recovered_claims(
    specs: list[tuple[str, str, str]],
    parts: list[dict],
    errors: list[str],
) -> dict:
    """Merge disjoint schema responses without turning a partial into success."""
    merged = _empty_claims(specs)
    coverage = []
    field_warnings: dict[str, str] = {}
    for part in parts:
        if part.get("error"):
            errors.append(str(part["error"]))
        if isinstance(part.get("input_coverage"), dict):
            coverage.append(part["input_coverage"])
        field_warnings.update(part.get("field_warnings") or {})
        for name, field_type, _description in specs:
            if name not in part:
                continue
            value = part.get(name)
            if field_type == "list":
                for item in value or []:
                    if item not in merged[name]:
                        merged[name].append(item)
            elif value not in (None, ""):
                merged[name] = value
    if field_warnings:
        merged["field_warnings"] = field_warnings
    merged["input_coverage"] = {
        "policy": "schema-split-recovery",
        "requests": coverage,
        "reason": (
            "The original complete-schema response hit a model capacity/response boundary, "
            "so disjoint schema sections were retried and merged without using a partial reply."
        ),
    }
    merged["recovery"] = {
        "action": "split-extraction-schema-and-merge-complete-replies",
        "requests_completed": len(parts),
        "terminal_errors": list(errors),
    }
    merged["error"] = (
        "partial_schema_recovery: " + " | ".join(errors)
        if errors else None
    )
    return merged


def extract_claims_with_recovery(
    model: str,
    paper: dict,
    *,
    think=None,
    max_tokens: int | None = None,
    research_config: dict | None = None,
    context_tokens: int | None = None,
    _specs: list[tuple[str, str, str]] | None = None,
) -> dict:
    """Extract claims, splitting the schema after a typed capacity failure.

    Provider-truncated text is never parsed. Each child request must return a
    complete response; completed child fields are retained even if another
    child remains unresolved, and the resulting error keeps the paper
    checkpoint-retryable.
    """
    specs = list(_specs or _field_specs(research_config))
    narrowed = _config_for_extraction_specs(research_config, specs)
    try:
        return extract_claims(
            model,
            paper,
            think=think,
            max_tokens=max_tokens,
            research_config=narrowed,
            context_tokens=context_tokens,
        )
    except (OutputLimitExceededError, ContextWindowExceededError, BudgetPlanningError) as exc:
        if len(specs) <= 1:
            raise
        midpoint = max(1, len(specs) // 2)
        parts: list[dict] = []
        errors: list[str] = []
        for child_specs in (specs[:midpoint], specs[midpoint:]):
            try:
                parts.append(extract_claims_with_recovery(
                    model,
                    paper,
                    think=think,
                    max_tokens=max_tokens,
                    research_config=research_config,
                    context_tokens=context_tokens,
                    _specs=child_specs,
                ))
            except (OutputLimitExceededError, ContextWindowExceededError,
                    BudgetPlanningError) as child_exc:
                errors.append(
                    f"{','.join(name for name, _kind, _desc in child_specs)}: "
                    f"{type(child_exc).__name__}: {child_exc}"
                )
        if not parts:
            raise exc
        logger.warning(
            "[Extract] Recovered %d/%d schema section(s) for %r after %s; %d remain unresolved",
            len(parts), 2, (paper_text(paper, "title") or "?")[:80],
            type(exc).__name__, len(errors),
        )
        return _merge_recovered_claims(specs, parts, errors)


def _parse_extraction_response(
    text: str,
    specs: list[tuple[str, str, str]] | None = None,
) -> dict:
    """Parse LLM response into claims dict."""
    from process.llm import parse_llm_json

    specs = specs or _LEGACY_FIELDS
    data = parse_llm_json(text)
    if data is not None:
        return _normalize_claims(data, specs)

    # Last resort: try to extract individual fields with regex
    logger.warning("[Extract] Could not parse JSON from LLM response, "
                   "attempting field-level regex extraction")
    result = _empty_claims(specs)
    result["error"] = "parse_failed"

    # Extract list fields
    for name, ftype, _ in specs:
        if ftype != "list":
            continue
        pattern = rf'"{re.escape(name)}"\s*:\s*\[(.*?)\]'
        m = re.search(pattern, text, re.DOTALL)
        if m:
            items = re.findall(r'"([^"]+)"', m.group(1))
            result[name] = items

    # Extract string fields
    for name, ftype, _ in specs:
        if ftype != "string":
            continue
        pattern = rf'"{re.escape(name)}"\s*:\s*"((?:[^"\\]|\\.)*)"'
        m = re.search(pattern, text, re.DOTALL)
        if m:
            result[name] = m.group(1)

    # Extract boolean fields
    for name, ftype, _ in specs:
        if ftype != "boolean":
            continue
        pattern = rf'"{re.escape(name)}"\s*:\s*(true|false)'
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            result[name] = m.group(1).lower() == "true"

    return result


def _normalize_claims(
    data: dict,
    specs: list[tuple[str, str, str]] | None = None,
) -> dict:
    """Coerce each schema field to its type; note every coercion or gap.

    ``field_warnings`` (present only when needed) maps a field name to
    "missing", "unrecognised" or "coerced from <type>", so an omitted or
    mistyped value is never mistaken for a genuine empty or ``false`` answer.
    """
    specs = specs or _LEGACY_FIELDS
    result = _empty_claims(specs)
    warnings: dict[str, str] = {}
    for name, ftype, _ in specs:
        value, warning = _coerce_field(data.get(name), ftype)
        if value is not None:
            result[name] = value
        if warning:
            warnings[name] = warning
    if warnings:
        result["field_warnings"] = warnings
    result["error"] = None
    return result


def _coerce_field(value: Any, ftype: str) -> tuple[Any, str | None]:
    """(coerced value or None, warning or None) for one extracted field."""
    if value is None:
        return None, "missing"
    coerce = {"list": _coerce_list, "boolean": _coerce_boolean}.get(ftype, _coerce_string)
    coerced = coerce(value)
    if coerced is None:
        return None, "unrecognised"
    expected = {"list": list, "boolean": bool}.get(ftype, str)
    if isinstance(value, expected):
        return coerced, None
    return coerced, f"coerced from {type(value).__name__}"


def _coerce_list(value: Any) -> list[str] | None:
    if isinstance(value, list):
        return [str(v) for v in value if v]
    if isinstance(value, dict):
        return None
    text = str(value).strip()
    return [text] if text else []


def _coerce_string(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "; ".join(str(v) for v in value if v)
    if isinstance(value, dict):
        return None
    return str(value)


_FALSE_STRINGS = {"false", "no", "0"}


def _coerce_boolean(value: Any) -> bool | None:
    """claim_flag's true values; explicit false values; anything else unknown."""
    if isinstance(value, bool):
        return value
    if claim_flag({"v": value}, "v"):
        return True
    if isinstance(value, (int, float)) and value == 0:
        return False
    if isinstance(value, str) and value.strip().lower() in _FALSE_STRINGS:
        return False
    return None


# ---------------------------------------------------------------------------
# Batch extraction with checkpoint/resume
# ---------------------------------------------------------------------------

_CHECKPOINT_VERSION = 3  # v3 records context-derived input coverage and token policy


def extraction_checkpoint_fingerprint(
    papers: list[dict],
    model: str,
    *,
    think: bool | str | None = None,
    max_tokens: int | None = None,
    context_tokens: int | None = None,
    research_config: dict | None = None,
) -> str:
    """Identify an extraction run: ordered papers, model and rendered prompt parts.

    A Stage 3 checkpoint resumes only when this matches, so a checkpoint left
    by a cancelled run is never merged into a run over different papers
    (for example after Stage 2 is re-run) or with a different model/prompt.
    """
    return input_fingerprint(
        papers,
        version=_CHECKPOINT_VERSION,
        model=model,
        think=think,
        max_tokens=int(max_tokens or 0),
        context_tokens=int(context_tokens or COMPATIBILITY_FALLBACK_CONTEXT_TOKENS),
        token_policy=_TOKEN_POLICY_VERSION,
        prompt=EXTRACTION_PROMPT,
        schema=_field_specs(research_config),
        question=_research_question_from_config(research_config),
        guidance=_format_extraction_guidance(research_config),
    )


def _load_resume_state(ckpt: CheckpointManager, total: int,
                       progress_callback: Callable[[str], None] | None,
                       ) -> tuple[list[dict], int]:
    extracted, start_idx = ckpt.load()
    if start_idx > total:
        logger.warning(
            "Discarding extraction checkpoint: %d records for %d papers.",
            start_idx, total)
        return [], 0
    if start_idx > 0:
        logger.info("Resuming extraction from paper %d/%d", start_idx + 1, total)
        if progress_callback:
            progress_callback(
                f"Resuming extraction from paper {start_idx + 1}/{total}")
    return extracted, start_idx


def clear_extraction_checkpoint(checkpoint_dir: Path | None) -> None:
    """Remove the Stage 3 checkpoint once claims.json has been written."""
    CheckpointManager(checkpoint_dir, _CHECKPOINT_FILENAME).cleanup()


def _resumable_prefix(records: list[dict], failed_indices: dict[int, str]) -> list[dict]:
    """Records up to the first worker failure, so a resume retries it."""
    if not failed_indices:
        return records
    return records[:min(failed_indices)]


def extract_all_claims(
    papers: list[dict],
    model: str,
    checkpoint_dir: Path | None = None,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    cancel_event: Any = None,
    progress_callback: Callable[[str], None] | None = None,
    use_thinking: bool = False,
    max_tokens: int | None = None,
    context_tokens: int | None = None,
    research_config: dict | None = None,
    config: Any = None,
    concurrent_workers: int | None = None,
    keep_checkpoint: bool = False,
    **kw,
) -> tuple[list[dict], bool]:
    """Extract claims from all papers with checkpoint/resume.

    With ``keep_checkpoint=True`` a completed run leaves its (complete)
    checkpoint in place; the caller removes it with
    ``clear_extraction_checkpoint`` once its own output is safely written,
    so a crash in post-processing never loses the paid LLM results.

    Papers are processed concurrently through
    ``core.checkpoint.run_sliding_window``; results are collected in input
    order, so output records are identical to a sequential run. Set
    ``concurrent_workers=1`` for local inference servers (Ollama / LM
    Studio) to avoid model contention.

    Returns (papers_with_claims, was_cancelled).
    """
    from core.checkpoint import run_sliding_window

    total = len(papers)
    logger.info("extract_all_claims: %d papers, model=%s", total, model)

    think = _claim_think_mode(model, use_thinking)
    if context_tokens is None and config is not None:
        context_tokens = resolve_model_capacity(config, model=model).context_window_tokens
    effective_context = _extraction_capacity(context_tokens).context_window_tokens
    logger.info(
        "Stage 3 token action: context=%d; max output=%s; policy=%s",
        effective_context,
        int(max_tokens) if max_tokens and int(max_tokens) > 0 else "auto",
        _TOKEN_POLICY_VERSION,
    )

    # Resume from checkpoint (only one written for these exact inputs)
    fingerprint = extraction_checkpoint_fingerprint(
        papers, model, think=think, max_tokens=max_tokens,
        context_tokens=effective_context,
        research_config=research_config)
    ckpt = CheckpointManager(
        checkpoint_dir, _CHECKPOINT_FILENAME, checkpoint_every,
        fingerprint=fingerprint)
    extracted, start_idx = _load_resume_state(ckpt, total, progress_callback)

    # Spacing between submission of concurrent calls. The old hard-coded
    # 2.0s sleep after *every* call was pure overhead; real rate-limit
    # protection is the 429-with-backoff retry in process.llm.
    stagger_delay = float(kw.get("stagger_delay", 0.5))
    if config is not None and "stagger_delay" not in kw:
        stagger_delay = float(getattr(config, "stagger_delay", stagger_delay))
    if stagger_delay < 0:
        stagger_delay = 0.0

    _workers = concurrent_workers
    if _workers is None and config is not None:
        _workers = getattr(config, "concurrent_workers", None)
    if _workers is None:
        _workers = 4
    _workers = max(1, int(_workers))

    _timeout = getattr(config, "worker_timeout_seconds", None) or 300.0

    logger.info(
        "Extraction concurrency: workers=%d, stagger=%.1fs",
        _workers, stagger_delay,
    )

    new_records: list[dict] = []
    # Absolute paper index -> "failed" | "timed_out" from run_sliding_window.
    failed_indices: dict[int, str] = {}

    def _on_failure(idx: int, kind: str, _error: BaseException | None) -> None:
        failed_indices[idx] = kind

    def _worker(paper: dict) -> tuple[dict, float]:
        title = (paper_text(paper, "title") or "?")[:80]
        logger.info("[Extract] Extracting claims: %s", title)
        t0 = time.time()
        try:
            claims = extract_claims_with_recovery(
                model,
                paper,
                think=think,
                max_tokens=max_tokens,
                research_config=research_config,
                context_tokens=effective_context,
            )
        except Exception as exc:
            logger.error("[Extract] Error extracting '%s': %s", title, exc)
            claims = {**_EMPTY_CLAIMS, "error": str(exc)}
        return claims, time.time() - t0

    def _build_result(idx: int, worker_out: tuple[dict, float] | None) -> dict:
        if worker_out is None:
            logger.error("Worker timed out or failed for paper %d/%d, using empty claims", idx + 1, total)
            error = "worker_failed" if failed_indices.get(idx) == "failed" else "worker_timeout"
            claims, elapsed = {**_EMPTY_CLAIMS, "error": error}, 0.0
        else:
            claims, elapsed = worker_out
        if claims.get("error") and claims.get("error") != "no_abstract":
            # Model failures and partial schema recoveries remain retryable.
            # Do not let an error-shaped record advance the resumable prefix.
            failed_indices[idx] = "model_error"
        record = {**papers[idx], "claims": claims}
        new_records.append(record)
        n_f = len(claims.get("findings", []))
        n_m = len(claims.get("mechanisms", []))
        n_e = len(claim_list(claims, "key_entities"))
        title = (paper_text(papers[idx], "title") or "?")[:80]
        result_msg = (
            f"[{idx + 1}/{total}] {title} → "
            f"{n_f} findings, {n_m} mechanisms, {n_e} entities ({elapsed:.1f}s)"
        )
        logger.info(result_msg)
        if progress_callback:
            progress_callback(result_msg)
        return record

    def _on_checkpoint() -> None:
        ckpt.save(_resumable_prefix(extracted + new_records, failed_indices), total)

    new_results, was_cancelled = run_sliding_window(
        items=papers,
        worker_fn=_worker,
        result_fn=_build_result,
        start_index=start_idx,
        workers=_workers,
        stagger=stagger_delay,
        checkpoint_fn=_on_checkpoint,
        cancel_event=cancel_event,
        future_timeout=_timeout,
        on_failure=_on_failure,
    )
    extracted.extend(new_results)

    # Final checkpoint if cancelled
    if was_cancelled:
        if not ckpt.force_save(_resumable_prefix(extracted, failed_indices), total):
            logger.warning("Extraction cancelled but checkpoint save failed.")
        logger.info("Extraction cancelled: %d/%d papers processed",
                    len(extracted), total)
    elif failed_indices:
        if not ckpt.force_save(_resumable_prefix(extracted, failed_indices), total):
            logger.warning("Extraction had unresolved model results but checkpoint save failed.")
        logger.warning(
            "Extraction finished with %d unresolved paper(s); the checkpoint stops before "
            "the first unresolved paper so a rerun retries it.",
            len(failed_indices),
        )
    elif keep_checkpoint:
        if not ckpt.force_save(_resumable_prefix(extracted, failed_indices), total):
            logger.warning("Extraction finished but the final checkpoint save failed.")
    else:
        ckpt.cleanup()

    logger.info("Extraction complete: %d/%d papers processed%s",
                len(extracted), total,
                " (cancelled)" if was_cancelled else "")
    return extracted, was_cancelled
