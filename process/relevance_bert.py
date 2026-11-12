"""Legacy BRT-BERT sequence-classification relevance scorer.

Used by ``process.relevance_filter`` when ``relevance_model`` names a local
BERT checkpoint (``relevance_screening_mode="off"``). The default Stage 2
path is the Laya + BRT-BERT cascade in ``process.relevance_cascade``.
"""
from __future__ import annotations

import json
import logging
import math
import re
import threading
from pathlib import Path
from typing import Any

from process import relevance_embedder as _embedder

logger = logging.getLogger(__name__)

_BERT_RELEVANCE_MODEL_ALIASES = {
    # Canonical name: BRT-BERT (BERT Relevance Trained)
    "brt-bert",
    "brt-bert-relevance",
    "brtbert",
    "brt",
    # Legacy aliases — kept so existing saved configs and project files keep working.
    "housecatbert-scibert-releevance",
    "housecatbert-scibert-relevance",
    "housecatbert-scibert",
    "housecatbert/scibert",
}
# Exact (normalised) label names meaning "relevant". Substring matching
# would pick "not_relevant"/"irrelevant" (audit 05-M2).
_BERT_RELEVANT_LABELS = frozenset({
    "relevant", "relevance", "positive", "pos", "entailment", "yes", "true",
    "support", "supports", "include", "included",
})
_GENERIC_LABEL_RE = re.compile(r"label_\d+")
# Used only when a model has no calibrated threshold: the old p > 0.5 cut.
_UNCALIBRATED_THRESHOLD = 0.5
_TRAINER_STATE = "trainer_state.json"
_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
_BERT_SCORER_CACHE: dict[str, tuple[Any, Any, Any]] = {}
_BERT_SCORER_LOAD_ERRORS: dict[str, str] = {}
_BERT_THRESHOLDS: dict[str, float] = {}
_BERT_SCORER_LOCK = threading.Lock()
# One tokenizer/model pair is shared by every worker thread, including a
# timed-out worker the sliding window abandoned; HF fast tokenizers are not
# safe to call concurrently, so inference is serialised.
_BERT_INFERENCE_LOCK = threading.Lock()
_LOCAL_BERT_RELEVANCE_DIR = (
    Path(__file__).resolve().parent.parent / "brt-bert-relevance"
)


def _is_bert_relevance_model(model: str) -> bool:
    """Return True when relevance scoring should use a local BERT classifier."""
    name = (model or "").strip().lower()
    if name in _BERT_RELEVANCE_MODEL_ALIASES:
        return True
    if not model:
        return False
    path = Path(model).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    return path.exists()


def _resolve_bert_model_name(model: str) -> str:
    """Map shorthand aliases to the local relevance model path."""
    name = (model or "").strip()
    lower = name.lower()
    if lower in _BERT_RELEVANCE_MODEL_ALIASES:
        return str(_resolve_local_bert_relevance_dir())
    path = Path(name).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    if path.exists():
        return str(_resolve_local_bert_relevance_dir(path))
    return str(_resolve_local_bert_relevance_dir())


def _is_loadable_model_dir(path: Path) -> bool:
    return (path / "config.json").is_file() and any(
        (path / name).is_file() for name in _WEIGHT_FILES)


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def _trainer_best_checkpoint(checkpoints: list[Path]) -> Path | None:
    """The trainer's ``best_model_checkpoint`` if it is one of ``checkpoints``."""
    by_name = {path.name: path for path in checkpoints}
    for checkpoint in sorted(checkpoints, key=_checkpoint_step, reverse=True):
        state_path = checkpoint / _TRAINER_STATE
        if not state_path.is_file():
            continue
        try:
            best = json.loads(state_path.read_text(encoding="utf-8")).get("best_model_checkpoint")
        except (OSError, ValueError, AttributeError) as exc:
            logger.warning("Could not read %s: %s", state_path, exc)
            continue
        if isinstance(best, str) and best:
            name = re.split(r"[\\/]", best.rstrip("\\/"))[-1]
            if name in by_name:
                return by_name[name]
    return None


def _resolve_local_bert_relevance_dir(base_dir: Path = _LOCAL_BERT_RELEVANCE_DIR) -> Path:
    """The weights to score with: ``base_dir`` itself, ``final/``, or the best checkpoint.

    A training run's last ``checkpoint-N`` is not its best model (audit
    05-M1): prefer ``final/``, then the trainer's ``best_model_checkpoint``,
    and only then the highest step, with a warning.
    """
    base_dir = base_dir.expanduser().resolve()
    if (base_dir / "config.json").exists():
        return base_dir
    if _is_loadable_model_dir(base_dir / "final"):
        return base_dir / "final"

    checkpoints = [
        child for child in (base_dir.iterdir() if base_dir.is_dir() else [])
        if child.is_dir() and child.name.startswith("checkpoint-")
        and _is_loadable_model_dir(child)
    ]
    if not checkpoints:
        return base_dir
    best = _trainer_best_checkpoint(checkpoints)
    if best is not None:
        return best
    latest = max(checkpoints, key=_checkpoint_step)
    logger.warning(
        "No final/ model or trainer best checkpoint in %s; using the last checkpoint %s",
        base_dir, latest.name,
    )
    return latest


def bert_decision_threshold(model_name: str) -> float:
    """The model's calibrated relevance threshold (cached), or 0.5 if it has none."""
    cached = _BERT_THRESHOLDS.get(model_name)
    if cached is not None:
        return cached
    path = Path(model_name)
    threshold = (_embedder.read_metrics_threshold(str(path / "final"))
                 or _embedder.read_metrics_threshold(model_name))
    if threshold is None:
        logger.warning(
            "No calibrated threshold (final_metrics.json) for BERT model %s; using p >= %.2f",
            model_name, _UNCALIBRATED_THRESHOLD,
        )
        threshold = _UNCALIBRATED_THRESHOLD
    _BERT_THRESHOLDS[model_name] = threshold
    return threshold


def bert_scoring_identity(model: str) -> dict:
    """Checkpoint-fingerprint settings for a BERT model: weights and threshold."""
    model_name = _resolve_bert_model_name(model)
    return {"bert_model": model_name, "bert_threshold": bert_decision_threshold(model_name),
            "bert_scoring": "calibrated-ordinal"}


def reset_bert_load_errors() -> None:
    """Forget earlier load failures and thresholds so a new run retries them."""
    with _BERT_SCORER_LOCK:
        _BERT_SCORER_LOAD_ERRORS.clear()
        _BERT_THRESHOLDS.clear()


def _bert_device() -> str:
    """The same device the cross-encoder embedder would pick (MPS, CUDA, CPU)."""
    return _embedder._resolve_device("auto")


def _load_bert_scorer(model_name: str) -> tuple[Any, Any, Any]:
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model_obj = AutoModelForSequenceClassification.from_pretrained(
            model_name, local_files_only=True)
    except Exception as exc:
        _BERT_SCORER_LOAD_ERRORS[model_name] = f"{type(exc).__name__}: {exc}"
        raise RuntimeError(
            f"Unable to load BERT relevance model '{model_name}'. "
            "Ensure 'torch' and 'transformers' are installed and the model id is valid."
        ) from exc
    device = _bert_device()
    model_obj.to(device)
    model_obj.eval()
    logger.info("Loaded BERT relevance model: %s (%s)", model_name, device)
    logger.info(
        "BRT-BERT token action: tokenizer truncation remains at 512 tokens because "
        "this is the cross-encoder architecture boundary, not a synthetic SystS LLM cap. "
        "The research question and paper text are encoded together within that boundary."
    )
    return tokenizer, model_obj, device


def _get_cached_bert_scorer(model: str) -> tuple[Any, Any, Any]:
    """Load and cache tokenizer/model/device for BERT relevance scoring."""
    model_name = _resolve_bert_model_name(model)
    with _BERT_SCORER_LOCK:
        prior_error = _BERT_SCORER_LOAD_ERRORS.get(model_name)
        if prior_error:
            raise RuntimeError(
                f"BERT relevance model '{model_name}' failed to load earlier: {prior_error}"
            )
        cached = _BERT_SCORER_CACHE.get(model_name)
        if cached is None:
            cached = _load_bert_scorer(model_name)
            _BERT_SCORER_CACHE[model_name] = cached
        return cached


def _normalise_label(raw_label: Any) -> str:
    return re.sub(r"[\s-]+", "_", str(raw_label).strip().lower())


def _bert_relevant_label_index(model_obj: Any) -> int:
    """The classifier label index meaning 'relevant'.

    Labels are matched exactly, so "not_relevant" or "irrelevant" never
    count. Generic ``LABEL_0``/``LABEL_1`` binary heads use index 1. Any other
    head without exactly one relevant label raises: guessing would invert
    every decision.
    """
    config = getattr(model_obj, "config", None)
    num_labels = int(getattr(config, "num_labels", 0) or 0)
    if num_labels <= 1:
        return 0
    id2label = getattr(config, "id2label", {}) or {}
    labels = {int(idx): _normalise_label(label) for idx, label in id2label.items()}
    relevant = [idx for idx, label in labels.items() if label in _BERT_RELEVANT_LABELS]
    if len(relevant) == 1:
        return relevant[0]
    if num_labels == 2 and all(_GENERIC_LABEL_RE.fullmatch(label) for label in labels.values()):
        return 1
    raise ValueError(
        f"Cannot tell which BERT label means relevant from id2label={dict(id2label)!r}; "
        "name exactly one label 'relevant'."
    )


def _bert_logits(model: str, question: str, paper_text: str) -> tuple[Any, Any]:
    """Run the classifier once; serialised across worker threads."""
    import torch

    tokenizer, model_obj, device = _get_cached_bert_scorer(model)
    with _BERT_INFERENCE_LOCK:
        inputs = tokenizer(question, paper_text, return_tensors="pt",
                           truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model_obj(**inputs).logits[0].detach().cpu()
    return logits, model_obj


def _relevance_probability(logits: Any, model_obj: Any) -> tuple[float, float]:
    """(logit, probability) of the relevant label."""
    import torch

    if int(logits.shape[0]) <= 1:
        logit = float(logits[0].item())
        return logit, 1.0 / (1.0 + math.exp(-logit))
    rel_idx = _bert_relevant_label_index(model_obj)
    rel_idx = max(0, min(rel_idx, int(logits.shape[0]) - 1))
    probabilities = torch.softmax(logits, dim=0)
    return float(logits[rel_idx].item()), float(probabilities[rel_idx].item())


def _score_relevance_with_bert(
    model: str,
    paper_text: str,
    *,
    hypothesis: str = "",
) -> dict:
    """Score relevance with a sequence-classification BERT model.

    The probability maps to 0-5 anchored on the model's calibrated threshold
    (pinned to 2.5), so p >= threshold is a score of at least 3.
    """
    question = (hypothesis or "").strip()
    if not question:
        raise ValueError("BERT relevance scoring needs a research question")
    logits, model_obj = _bert_logits(model, question, paper_text)
    relevance_logit, relevance_probability = _relevance_probability(logits, model_obj)
    threshold = bert_decision_threshold(_resolve_bert_model_name(model))
    return {
        "relevance_score": _embedder._prob_to_ordinal(relevance_probability, threshold),
        "relevance_quote": "",
        "relevance_quote_source": "bert_logits",
        "relevance_quote_valid": False,
        "relevance_logit": relevance_logit,
        "relevance_confidence": relevance_probability,
        "relevance_threshold": threshold,
    }


def _bert_overrides(use_thinking: bool, workers: int, batch_size: int,
                    max_tokens: int, batch_max_tokens: int) -> tuple[int, int]:
    """Log ignored LLM settings and force one worker and per-paper scoring."""
    if use_thinking:
        logger.info("BERT relevance scoring ignores use_thinking setting.")
    if max_tokens > 0 or batch_max_tokens > 0:
        logger.info(
            "BERT relevance scoring ignores the explicit LLM response overrides "
            "score_max_tokens=%d and score_batch_max_tokens=%d because this path "
            "uses a sequence classifier, not text generation.",
            max_tokens,
            batch_max_tokens,
        )
    if workers > 1:
        logger.info(
            "BERT relevance scoring runs with concurrent_workers=1 to avoid model contention; "
            "ignoring concurrent_workers=%d",
            workers,
        )
    if batch_size > 1:
        logger.info(
            "BERT relevance scoring ignores batch prompts; falling back to per-paper scoring "
            "(requested batch_size=%d)",
            batch_size,
        )
    return 1, 1
