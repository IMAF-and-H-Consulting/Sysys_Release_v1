"""Stage 2: combined Laya and BRT-BERT relevance filtering.

Reads the Stage 1 corpus and keeps papers that pass both calibrated gates.
Explicit legacy configurations can still select the BERT/LLM scorer.
"""
import json
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


def run(
    config,
    corpus: list[dict] | None = None,
    workload: str = "abstract",
    cancel_event: Any = None,
    progress_callback: Callable[[str], None] | None = None,
) -> list[dict]:
    """Run Stage 2 scoring.

    Parameters
    ----------
    config : Config
        Pipeline configuration (must have ``corpus_dir``, ``claims_dir``,
        ``relevance_threshold``, ``llm_model``, etc.).
    corpus : list[dict], optional
        Pre-loaded corpus.  If *None*, loads ``corpus.json`` from
        ``config.corpus_dir``.
    workload : str
        ``"abstract"`` or ``"fulltext"`` — controls which text field is
        sent to the selected screener.
    cancel_event : threading.Event, optional
        Set this to signal cancellation.
    progress_callback : callable, optional
        Called with progress strings for the GUI.

    Returns
    -------
    list[dict]
        Papers that passed screening, enriched with scoring metadata. When
        the run is cancelled this is the partial result only: ``relevant.json``
        and ``scored_all.json`` from the previous run are left untouched
        (progress stays in the stage checkpoints for a resume).
    """
    from process.llm import set_config
    from process.relevance_filter import _is_bert_relevance_model, filter_corpus

    set_config(config)

    # ── Load corpus ──
    if corpus is None:
        corpus_path = config.corpus_dir / "corpus.json"
        logger.info("Loading corpus from %s", corpus_path)
        with open(corpus_path, encoding="utf-8") as f:
            loaded_corpus: list[dict] = json.load(f)
        corpus = loaded_corpus

    relevance_model = getattr(config, "relevance_model", "") or config.llm_model
    screening_mode = getattr(config, "relevance_screening_mode", "off")
    if screening_mode == "off":
        logger.info(
            "Stage 2: Scoring %d papers (threshold=%d, model=%s, workload=%s)",
            len(corpus), config.relevance_threshold, relevance_model, workload,
        )
    else:
        logger.info(
            "Stage 2: Screening %d papers (strategy=%s, workload=%s)",
            len(corpus), screening_mode, workload,
        )

    # ── Score ──
    hypothesis = _research_question(config)
    if screening_mode != "off":
        from process.relevance_cascade import filter_cascade
        relevant, cancelled, scored_all = filter_cascade(
            corpus, config, hypothesis=hypothesis, workload=workload,
            cancel_event=cancel_event, progress_callback=progress_callback,
        )
    else:
        context_tokens = 0
        if not _is_bert_relevance_model(relevance_model):
            from process.llm_budget import resolve_model_capacity
            capacity = resolve_model_capacity(config, model=relevance_model)
            context_tokens = capacity.context_window_tokens
            logger.info(
                "Stage 2 token action: context=%d (%s); zero output/character settings "
                "mean automatic context-relative planning",
                context_tokens, capacity.source,
            )
        abstract_limit = int(getattr(config, "score_abstract_chars", 0) or 0) or None
        configured_fulltext_limit = int(getattr(config, "score_fulltext_chars", 0) or 0)
        relevant, cancelled, scored_all = filter_corpus(
            corpus,
            relevance_model,
            threshold=config.relevance_threshold,
            checkpoint_dir=config.claims_dir,
            checkpoint_every=getattr(config, "checkpoint_every", 1),
            stagger_delay=getattr(config, "stagger_delay", None),
            concurrent_workers=getattr(config, "concurrent_workers", None),
            future_timeout=getattr(config, "worker_timeout_seconds", None),
            cancel_event=cancel_event,
            progress_callback=progress_callback,
            hypothesis=hypothesis,
            use_thinking=getattr(config, "score_use_thinking", True),
            score_max_tokens=getattr(config, "score_max_tokens", 0),
            score_batch_max_tokens=getattr(config, "score_batch_max_tokens", 0),
            batch_size=getattr(config, "score_batch_size", 1),
            abstract_chars=abstract_limit,
            fulltext_chars=(
                0 if str(workload).lower().strip() == "abstract"
                else configured_fulltext_limit or None
            ),
            context_tokens=context_tokens,
            return_scored=True,
        )

    if cancelled:
        _report_cancelled(len(scored_all), len(corpus), progress_callback)
        return relevant

    # ── Save relevance artifacts ──
    out_path = config.claims_dir / "relevant.json"
    scored_path = config.claims_dir / "scored_all.json"
    config.ensure_dirs()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(relevant, f, indent=2, default=str)
    with open(scored_path, "w", encoding="utf-8") as f:
        json.dump(scored_all, f, indent=2, default=str)
    logger.info(
        "Stage 2 complete: %d/%d papers relevant (saved to %s; all scores to %s)",
        len(relevant), len(corpus), out_path, scored_path,
    )

    return relevant


def _research_question(config) -> str:
    """The question to screen against; ValueError when there is none.

    A failure reading the natural-language question is logged, then the
    hypothesis text is used. Scoring never falls back to a placeholder.
    """
    nlq = ""
    if hasattr(config, "get_natural_language_question"):
        try:
            nlq = config.get_natural_language_question() or ""
        except Exception as exc:  # logged; hypothesis_text is used instead
            logger.warning("Could not read the natural-language question: %s", exc,
                           exc_info=True)
    question = (nlq or getattr(config, "hypothesis_text", "") or "").strip()
    if not question:
        raise ValueError(
            "Stage 2 needs a research question: set the natural-language question "
            "or hypothesis before scoring."
        )
    return question


def _report_cancelled(n_scored: int, n_total: int,
                      progress_callback: Callable[[str], None] | None) -> None:
    """Explain that a cancelled run did not replace the final outputs."""
    message = (
        f"Stage 2 cancelled after {n_scored}/{n_total} papers; relevant.json and "
        "scored_all.json were not overwritten. Run Stage 2 again to resume."
    )
    logger.warning(message)
    if progress_callback:
        progress_callback(message)
