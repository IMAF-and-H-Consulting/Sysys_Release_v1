"""Stage runner — async wrappers for each pipeline stage.

Every stage in ``_STAGE_REGISTRY`` is wired to its module in stages/. Banner
numbers follow the GUI's stage numbering, which is not the module number
after the retired interrater stage (GUI 8/9/10 = s9/s10/s11).
Stage 1 (literature search) runs live through ``gui.workers.PipelineSearchWorker``
→ ``gui.search_runner.run_gui_search`` directly and has no registry entry here
by design.
"""
import asyncio
import copy
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from core.config import (
    EMBEDDING_RATERS as EMBEDDING_SECOND_RATERS,
)
from core.config import (
    PROVIDER_MODEL_FIELD,
    normalize_provider_name,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> list[dict]:
    """Load a JSON file and return its contents."""
    logger.info("Loading JSON: %s", path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    logger.info("  Loaded %d items from %s (%d bytes)", len(data), path, path.stat().st_size)
    return data


def _stage_banner(stage_num: int, stage_name: str) -> None:
    """Log a prominent stage boundary banner."""
    logger.info("═" * 60)
    logger.info("STAGE %d: %s", stage_num, stage_name.upper())
    logger.info("═" * 60)


# ---------------------------------------------------------------------------
# Stage 2: Score
# ---------------------------------------------------------------------------
async def run_score(
    config,
    threshold: int | None = None,
    corpus_path: Path | None = None,
    workload: str = "abstract",
    cancel_event: asyncio.Event | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> list[dict]:
    """Stage 2: relevance filtering with the selected screening strategy.

    *threshold* overrides ``config.relevance_threshold`` on a copy of the
    config; the caller's config is never changed. Unknown keyword arguments
    raise ``TypeError`` instead of being silently ignored.
    """
    import stages.s2_score as s2

    _stage_banner(2, "RELEVANCE FILTERING")
    config = copy.copy(config)
    if threshold is not None:
        config.relevance_threshold = threshold
    mode = getattr(config, "relevance_screening_mode", "off")
    logger.info("  Screening strategy: %s", mode)
    if mode == "off":
        logger.info("  Provider: %s", config.llm_provider)
        logger.info("  Model: %s", config.llm_model)
        logger.info("  Threshold: %d", config.relevance_threshold)
    logger.info("  Workload: %s", workload)
    logger.info("  Workers: %d", getattr(config, "concurrent_workers", 1))

    corpus = None
    if corpus_path:
        corpus = _load_json(corpus_path)

    result = s2.run(
        config,
        corpus=corpus,
        workload=workload,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
    )
    return result


# ---------------------------------------------------------------------------
# Stage 3: Extract
# ---------------------------------------------------------------------------
async def run_extract(
    config,
    cancel_event: asyncio.Event | None = None,
    progress_callback: Callable[[str], None] | None = None,
    **kw,
) -> dict:
    """Stage 3: Claim extraction."""
    import stages.s3_extract as s3
    from process.llm import set_config

    _stage_banner(3, "CLAIM EXTRACTION")
    logger.info("  Provider: %s", config.llm_provider)
    logger.info("  Model: %s", config.llm_model)
    output_override = int(kw.get("max_tokens") or 0)
    logger.info(
        "  Token action: %s",
        f"explicit synthetic response override={output_override:,}"
        if output_override else "automatic context-aware response planning",
    )
    logger.info("  Use thinking: %s", kw.get("use_thinking", False))

    set_config(config)

    result = s3.run(
        config=config,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
        **kw,
    )
    return result


# ---------------------------------------------------------------------------
# Stage 4: Validate
# ---------------------------------------------------------------------------
async def run_validate(
    config,
    cancel_event: asyncio.Event | None = None,
    progress_callback: Callable[[str], None] | None = None,
    **kw,
) -> dict:
    """Stage 4: Method review & span validation."""
    import stages.s4_validate as s4
    from process.llm import set_config

    _stage_banner(4, "METHOD REVIEW & SPAN VALIDATION")
    logger.info("  Provider: %s", config.llm_provider)
    logger.info("  Model: %s", config.llm_model)

    set_config(config)

    result = s4.run(
        config=config,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
        **kw,
    )
    return result


# ---------------------------------------------------------------------------
# Stage 5: Audit
# ---------------------------------------------------------------------------
async def run_audit(
    config,
    cancel_event: asyncio.Event | None = None,
    progress_callback: Callable[[str], None] | None = None,
    **kw,
) -> dict:
    """Stage 5: Span audit & quality filtering."""
    import stages.s5_audit as s5

    _stage_banner(5, "SPAN AUDIT & QUALITY FILTERING")
    logger.info("  Claims dir: %s", config.claims_dir)
    logger.info("  Data dir:   %s", config.data_dir)

    result = s5.run(
        config=config,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
        **kw,
    )
    return result


# ---------------------------------------------------------------------------
# Stage 6: Synthesize
# ---------------------------------------------------------------------------
async def run_synthesize(
    config,
    cancel_event: asyncio.Event | None = None,
    progress_callback: Callable[[str], None] | None = None,
    **kw,
) -> dict:
    """Stage 6: Cluster-level synthesis — cluster, narrate, draft abstract."""
    import stages.s6_synthesize as s6
    from process.llm import set_config

    _stage_banner(6, "SYNTHESIS (CLUSTER · NARRATE · ABSTRACT)")
    logger.info("  Provider: %s", config.llm_provider)
    logger.info("  Model: %s", config.llm_model)
    logger.info("  Extra kwargs: %s", list(kw.keys()))

    set_config(config)

    result = s6.run(
        config=config,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
        **kw,
    )
    return result


# ---------------------------------------------------------------------------
# Stage 7: Fulltext retrieval
# ---------------------------------------------------------------------------
async def run_fulltext(
    config,
    cancel_event: asyncio.Event | None = None,
    progress_callback: Callable[[str], None] | None = None,
    **kw,
) -> dict:
    """Stage 7: Full-text retrieval & metric extraction."""
    import stages.s7_fulltext as s7

    _stage_banner(7, "FULL-TEXT RETRIEVAL & METRIC EXTRACTION")
    logger.info("  Data dir: %s", config.data_dir)
    logger.info("  Extra kwargs: %s", list(kw.keys()))

    # s7.run blocks (downloads, PDF parsing): keep the event loop free.
    result = await asyncio.to_thread(
        lambda: s7.run(
            config=config,
            cancel_event=cancel_event,
            progress_callback=progress_callback,
            **kw,
        )
    )
    return result


# ---------------------------------------------------------------------------
# Stage 2b / 8: Corpus full-text re-score
# ---------------------------------------------------------------------------
async def run_corpus_fulltext(
    config,
    cancel_event: asyncio.Event | None = None,
    **kw,
) -> dict:
    """Stage 2b: Corpus full-text enrichment before scoring.

    Errors propagate (like every other wrapper) so the worker emits ``error``
    and the GUI reports a failure instead of a successful 0/0 enrichment.
    """
    import stages.s2b_fulltext as s2b

    logger.info("═" * 60)
    logger.info("STAGE 2b: CORPUS FULL-TEXT ENRICHMENT")
    logger.info("═" * 60)
    result = await asyncio.to_thread(
        lambda: s2b.run(config=config, cancel_event=cancel_event, **kw)
    )
    return result


# ---------------------------------------------------------------------------
# Stage 2c: Statistical extraction
# ---------------------------------------------------------------------------
async def run_statistical_extract(
    config,
    cancel_event: asyncio.Event | None = None,
    progress_callback: Callable[[str], None] | None = None,
    **kw,
) -> dict:
    """Stage 2c: deterministic statistical evidence extraction."""
    import stages.s2c_statistical_extract as s2c

    logger.info("═" * 60)
    logger.info("STAGE 2c: STATISTICAL EVIDENCE EXTRACTION")
    logger.info("═" * 60)
    result = await asyncio.to_thread(
        lambda: s2c.run(
            config=config,
            cancel_event=cancel_event,
            progress_callback=progress_callback,
            **kw,
        )
    )
    return result


# ---------------------------------------------------------------------------
# Retired interrater API (not registered as a pipeline stage)
# ---------------------------------------------------------------------------
def resolve_second_rater(config, second_provider: str,
                         second_model: str | None) -> tuple[Any, str, str]:
    """Build the second rater's config, provider and model.

    ``config`` must carry the PRIMARY rater's settings. Returns
    ``(second_cfg, provider, model)`` where ``model`` is always concrete —
    never ``None`` — so the caller and ``s8.run`` agree on what ran.

    Raises ValueError for an unsupported provider rather than letting
    ``llm.chat`` fall through to its Ollama default, which would silently
    rate with a model nobody asked for.
    """
    import copy

    second_cfg = copy.deepcopy(config)
    provider = normalize_provider_name(second_provider)
    model = (second_model or "").strip()

    # Cross-encoders bypass the LLM stack entirely: the model string is a
    # checkpoint path, so leave the deep-copied LLM settings alone.
    if provider in EMBEDDING_SECOND_RATERS:
        return second_cfg, provider, model

    field = PROVIDER_MODEL_FIELD.get(provider)
    if field is None:
        raise ValueError(
            f"Unsupported second-rater provider: {second_provider!r}. "
            f"Supported: {sorted(PROVIDER_MODEL_FIELD)} "
            f"or a cross-encoder ({sorted(EMBEDDING_SECOND_RATERS)})."
        )

    second_cfg.llm_provider = provider
    if model:
        setattr(second_cfg, field, model)
    else:
        # Fall back to whatever that provider is configured to use, so the
        # resolved name is reported instead of a bare "default".
        model = getattr(second_cfg, field, "") or ""

    return second_cfg, provider, model


def resolve_primary_rater(config, primary_provider: str | None = None,
                          primary_model: str | None = None) -> tuple[str, str]:
    """Identify the rater that produced the scores already in the corpus.

    Callers that know it explicitly (the GUI reads it off the Stage 2
    widgets) should pass it; otherwise fall back to the config's LLM
    settings. A blank model is resolved to the provider's configured model
    so it compares equal to a second rater that names that model outright.
    """
    provider = (primary_provider or "").strip()
    model = (primary_model or "").strip()

    if not provider:
        screening_mode = getattr(config, "relevance_screening_mode", "off")
        if screening_mode != "off":
            return screening_mode, getattr(config, "laya_screener_dir", "")
        return _off_mode_primary_rater(config, model)

    if provider in EMBEDDING_SECOND_RATERS:
        return provider, model

    provider = normalize_provider_name(provider)
    if not model:
        field = PROVIDER_MODEL_FIELD.get(provider)
        if field:
            model = getattr(config, field, "") or ""
    return provider, model


def _off_mode_primary_rater(config, model: str) -> tuple[str, str]:
    """The rater Stage 2 used in mode "off": ``relevance_model``, else the LLM.

    Mirrors ``stages.s2_score``: a BERT ``relevance_model`` scores locally;
    any other name is an LLM model on ``llm_provider``.
    """
    from process.relevance_bert import _is_bert_relevance_model

    relevance_model = (getattr(config, "relevance_model", "") or "").strip()
    if relevance_model and _is_bert_relevance_model(relevance_model):
        name = normalize_provider_name(relevance_model)
        return (name if name in EMBEDDING_SECOND_RATERS else "brt-bert"), model
    provider = getattr(config, "llm_provider", "") or ""
    return provider, model or relevance_model or (getattr(config, "llm_model", "") or "")


def _same_rater(primary: tuple[str, str], second: tuple[str, str]) -> bool:
    """True when the two raters are indistinguishable.

    Both sides arrive with their models already resolved to concrete names,
    so this is a plain comparison; an unknown side never claims a match.
    """
    p_provider, p_model = primary
    s_provider, s_model = second
    if not p_provider or not s_provider:
        return False
    return (p_provider, p_model or "") == (s_provider, s_model or "")


async def run_interrater(
    config,
    cancel_event: asyncio.Event | None = None,
    progress_callback: Callable[[str], None] | None = None,
    sample_size: int = 100,
    second_provider: str = "ollama",
    second_model: str | None = None,
    primary_provider: str | None = None,
    primary_model: str | None = None,
    **kw,
) -> dict:
    """Legacy interrater API, retained for historical callers only.

    The primary rater is whatever produced the scores already stored in the
    corpus; pass it as ``primary_provider``/``primary_model`` when known,
    otherwise ``config``'s LLM settings are used. ``second_provider``/
    ``second_model`` select an independent second rater. If the two resolve
    to the same provider and model the run measures one model's
    self-consistency, not inter-rater reliability, so that case is reported
    loudly instead of being passed off as agreement.
    """
    import stages.s8_interrater as s8
    from process.llm import set_config

    _stage_banner(8, "INTER-RATER RELIABILITY")

    def _emit(msg: str) -> None:
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    second_cfg, provider, model = resolve_second_rater(
        config, second_provider, second_model)
    primary = resolve_primary_rater(config, primary_provider, primary_model)
    same_rater = _same_rater(primary, (provider, model))

    logger.info("  Primary provider: %s / %s", primary[0], primary[1] or "default")
    logger.info("  Second provider: %s / %s", provider, model or "default")
    logger.info("  Sample size: %d", sample_size)

    if same_rater:
        _emit(
            f"WARNING: the second rater ({provider} / {model or 'default'}) is "
            f"identical to the primary rater. Agreement will reflect one "
            f"model's self-consistency, not inter-rater reliability. Pick a "
            f"different provider or model in the comparison arguments."
        )

    set_config(second_cfg)
    try:
        result = s8.run(
            config=config,
            sample_size=sample_size,
            second_provider=provider,
            second_model=model,
            cancel_event=cancel_event,
            progress_callback=progress_callback,
            **kw,
        )
    finally:
        # Always restore the primary config. The GUI process is long-lived, so
        # leaking the second rater's provider would silently redirect every
        # later stage to the wrong model.
        set_config(config)

    if isinstance(result, dict):
        result.setdefault("second_provider", provider)
        result.setdefault("second_model", model)
        result.setdefault("primary_provider", primary[0])
        result.setdefault("primary_model", primary[1])
        result.setdefault("same_rater", same_rater)
    return result


# ---------------------------------------------------------------------------
# Stage 8: Robustness
# ---------------------------------------------------------------------------
async def run_robustness(
    config,
    cancel_event: asyncio.Event | None = None,
    progress_callback: Callable[[str], None] | None = None,
    **kw,
) -> dict:
    """Stage 8: Robustness analyses — bootstrap, split-half, sensitivity,
    permutation test, condition–entity network analysis, and condition sensitivity."""
    import stages.s9_robustness as s9

    # The stage emits its own banner through the progress callback.
    logger.info("  Data dir: %s", config.data_dir)
    logger.info("  Extra kwargs: %s", list(kw.keys()))

    result = s9.run(
        config=config,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
        **kw,
    )
    return result


# ---------------------------------------------------------------------------
# Stage 9: Verify
# ---------------------------------------------------------------------------
async def run_verify(
    config,
    cancel_event: asyncio.Event | None = None,
    progress_callback: Callable[[str], None] | None = None,
    **kw,
) -> dict:
    """Stage 9: Final source-linked review with internal numerical checks."""
    from process.final_review import run_review

    _stage_banner(9, "FINAL SOURCE-LINKED REVIEW")
    logger.info("  Data dir: %s", config.data_dir)
    logger.info("  Extra kwargs: %s", list(kw.keys()))

    result = run_review(
        config=config,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
        **kw,
    )
    return result


# ---------------------------------------------------------------------------
# Stage 10: Compare Runs
# ---------------------------------------------------------------------------
async def run_compare_runs(
    config,
    cancel_event: asyncio.Event | None = None,
    progress_callback: Callable[[str], None] | None = None,
    claims_paths: list | None = None,
    comparison_name: str = "run_compare",
    **kw,
) -> dict:
    """Stage 10: Compare pipeline runs."""
    import stages.s11_compare_runs as s11

    _stage_banner(10, "COMPARE RUNS")
    logger.info("  Claims paths: %s", claims_paths)
    logger.info("  Comparison name: %s", comparison_name)

    result = s11.run(
        config=config,
        claims_paths=claims_paths,
        comparison_name=comparison_name,
        cancel_event=cancel_event,
        progress_callback=progress_callback,
        **kw,
    )
    return result


# ---------------------------------------------------------------------------
# Stage Registry
# ---------------------------------------------------------------------------
# Each entry: stage_name -> (async_func, is_async_flag)
_STAGE_REGISTRY: dict[str, tuple[Callable, bool]] = {}


def _register_stages() -> None:
    """Populate the global _STAGE_REGISTRY dict."""
    global _STAGE_REGISTRY
    _STAGE_REGISTRY = {}
    # Each entry: stage_name -> (async_func, is_async_flag). All stage
    # runners share the (config, cancel_event, **kw) signature.
    for _name, _func in (
        ("score", run_score),
        ("extract", run_extract),
        ("validate", run_validate),
        ("audit", run_audit),
        ("synthesize", run_synthesize),
        ("fulltext", run_fulltext),
        ("corpus_fulltext", run_corpus_fulltext),
        ("statistical_extract", run_statistical_extract),
        ("robustness", run_robustness),
        ("verify", run_verify),
        ("compare_runs", run_compare_runs),
    ):
        _STAGE_REGISTRY[_name] = (
            lambda config, cancel_event=None, _f=_func, **kw: _f(
                config, cancel_event=cancel_event, **kw
            ),
            True,
        )


# Auto-register on import
_register_stages()
