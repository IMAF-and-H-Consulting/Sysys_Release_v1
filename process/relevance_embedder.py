"""Local cross-encoder relevance scorer (brt-bert / BioBERT / SciBERT / custom).

Alternative to the LLM scorer in ``process.relevance_filter``.  A
fine-tuned cross-encoder reads the (hypothesis, paper_text) pair and
emits a 0-5 integer score matching the shape used downstream. The active
Stage 2 defaults to the packaged Laya + BRT-BERT cascade in relevance_cascade.
This module remains available for legacy scoring and comparison callers.

Quote extraction is **intentionally not done here** — a cross-encoder
emits a single relevance logit, not interpretable spans.

Backends (passed as ``backend=`` or via ``Config.relevance_model``)
-------------------------------------------------------------------
``"brt_bert"`` — load ``<repo_root>/brt-bert-irr-buddy-mlm/relevance/final``
                 (the model produced by scripts/train_relevance_crossencoder.py).
``"biobert"``  — load ``<repo_root>/biobert`` if present.
``"scibert"``  — load ``<repo_root>/scibert-ce-scifact (2)`` if present.
``"embedding"``— load whatever ``model_path`` points at (HF id or local dir).

Device selection
----------------
``device`` accepts ``"auto"``, ``"mps"``, ``"cuda"``, or ``"cpu"``.
``"auto"`` prefers Apple Silicon MPS, then CUDA, then CPU.

Hypothesis normalisation
------------------------
Cross-encoders fine-tuned on declarative claims score better when the
research *question* is rewritten as a claim.  :func:`question_to_claim`
performs a lightweight, dependency-free rewrite:

    "Does soil salinity reduce wheat yield?"
    → "soil salinity reduce wheat yield"
"""
from __future__ import annotations

import gc
import json
import logging
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Cache of loaded models, keyed by (resolved_path, device).  Module-level
# so a single stage run loads each checkpoint exactly once.
_model_cache: dict[tuple[str, str], CrossEncoder] = {}

# Cache of F1-optimal thresholds read from each model's adjacent
# final_metrics.json (sibling of the model's final/ dir).  Keyed by the
# resolved model_ref so each preset gets its own calibration without a
# hardcoded module-level constant.
_threshold_cache: dict[str, float] = {}

# Fallback HF hub id when no preset or path resolves.
_DEFAULT_CHECKPOINT = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# Preset alias → directory (resolved against the repo root).
_PRESET_DIRS = {
    "brt_bert": "brt-bert-relevance/final",
    "brt-bert": "brt-bert-relevance/final",
    "brt_irr": "brt-bert-irr-buddy-mlm/relevance/final",
    "brt-irr": "brt-bert-irr-buddy-mlm/relevance/final",
    "biobert": "biobert",
    "scibert": "scibert-ce-scifact (2)",
}


# ---------------------------------------------------------------------------
# Path / device resolution
# ---------------------------------------------------------------------------
def _repo_root() -> Path:
    # process/relevance_embedder.py → repo root is the parent's parent.
    return Path(__file__).resolve().parent.parent


def resolve_model_ref(backend: str, model_path: str = "") -> str:
    """Turn a (backend, model_path) pair into something CrossEncoder can load.

    Preset backends resolve to a local directory; ``"embedding"`` uses
    ``model_path`` verbatim (HF hub id or filesystem path); unknown
    backends fall back to the default checkpoint so we still produce
    *something* rather than crashing.
    """
    b = (backend or "").strip().lower()
    if b in _PRESET_DIRS:
        candidate = _repo_root() / _PRESET_DIRS[b]
        if candidate.exists():
            return str(candidate)
        logger.warning(
            "Preset backend %r resolved to %s, but that path does not exist. "
            "Falling back to HF default %r.",
            b, candidate, _DEFAULT_CHECKPOINT,
        )
        return _DEFAULT_CHECKPOINT
    ref = (model_path or "").strip()
    return ref or _DEFAULT_CHECKPOINT


def _resolve_device(preference: str) -> str:
    pref = (preference or "auto").strip().lower()
    if pref != "auto":
        return pref
    try:
        import torch
    except ImportError:
        return "cpu"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"




# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
_THRESHOLD_KEYS = (
    "relevance_f1_threshold",
    "eval_relevance_f1_threshold",
    "relevance_accuracy_threshold",
    "eval_relevance_accuracy_threshold",
)


def _valid_threshold(value: object) -> float | None:
    """A finite probability strictly between 0 and 1, else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and 0.0 < value < 1.0 else None


def read_metrics_threshold(model_ref: str) -> float | None:
    """The calibrated decision threshold saved next to a model, or None.

    Reads ``final_metrics.json`` beside the model's directory (the trainer
    writes it next to ``final/``). A missing file, unreadable JSON, a
    missing key or a value that is not a probability in (0, 1) gives None.
    """
    if not model_ref:
        return None
    metrics_path = Path(model_ref).parent / "final_metrics.json"
    if not metrics_path.is_file():
        return None
    try:
        with metrics_path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        logger.warning("Could not read threshold from %s: %s", metrics_path, exc)
        return None
    if not isinstance(data, dict):
        logger.warning("Ignoring %s: expected a JSON object", metrics_path)
        return None
    for key in _THRESHOLD_KEYS:
        if key in data:
            threshold = _valid_threshold(data[key])
            if threshold is None:
                logger.warning("Ignoring invalid %s=%r in %s", key, data[key], metrics_path)
                return None
            logger.info("Loaded threshold %.4f from %s (key=%s)", threshold, metrics_path, key)
            return threshold
    return None


def _load_threshold(model_ref: str) -> float:
    """The model's calibrated threshold, cached; the module default otherwise."""
    if model_ref not in _threshold_cache:
        threshold = read_metrics_threshold(model_ref)
        _threshold_cache[model_ref] = (
            BRT_BERT_RELEVANCE_THRESHOLD if threshold is None else threshold)
    return _threshold_cache[model_ref]


def _load_model(model_ref: str, device: str) -> CrossEncoder | None:
    """Load (and cache) a ``sentence_transformers.CrossEncoder``.

    Also populates :data:`_threshold_cache` from the model's adjacent
    ``final_metrics.json`` so each preset uses its own calibrated threshold.

    Returns ``None`` if sentence-transformers is missing or the checkpoint
    fails to load; :func:`score_relevance_embedding` then reports an error
    instead of a score.
    """
    key = (model_ref, device)
    if key in _model_cache:
        return _model_cache[key]

    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        logger.warning(
            "sentence-transformers not installed — cannot load cross-encoder. "
            "Install with: pip install sentence-transformers"
        )
        return None

    try:
        model = CrossEncoder(model_ref, device=device)
    except Exception as exc:
        logger.error("Failed to load cross-encoder %r on %s: %s", model_ref, device, exc)
        return None

    _model_cache[key] = model
    _load_threshold(model_ref)  # populate _threshold_cache as a side-effect
    logger.info("Loaded cross-encoder %s on %s", model_ref, device)
    return model


def clear_model_cache() -> None:
    """Evict cached cross-encoder models and thresholds to release RAM/VRAM.

    Callers that use the embedder should call
    this when scoring is done: the checkpoints are hundreds of MB each and
    are otherwise pinned for the lifetime of the process. Clearing the
    threshold cache also picks up a retrained model's ``final_metrics.json``.
    """
    _threshold_cache.clear()
    if not _model_cache:
        return
    _model_cache.clear()
    gc.collect()
    _release_accelerator_memory()
    logger.info("Cleared cross-encoder model cache")


def _release_accelerator_memory() -> None:
    """Return freed CUDA/MPS memory to the device; log, never raise."""
    try:
        import torch
    except ImportError:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            torch.mps.empty_cache()
    except (RuntimeError, AttributeError) as exc:
        logger.warning("Could not release accelerator memory: %s", exc)


# ---------------------------------------------------------------------------
# Hypothesis → claim rewriting
# ---------------------------------------------------------------------------
_LEADING_DROP = (
    r"how|why|what|when|where|which|whether|who|whom|whose|"
    r"can|could|should|would|will|may|might|must|shall"
)
# Only determiners — keep prepositions ("in", "of", "with") because they
# carry relational meaning cross-encoders weigh heavily.
_DISTIL_STOPWORDS = {
    "a", "an", "the",
    "that", "this", "these", "those",
}

_DO_DOES_DID_RE = re.compile(r"^\s*(do|does|did)\s+", re.IGNORECASE)
_IS_ARE_RE = re.compile(r"^\s*(is|are|was|were)\s+", re.IGNORECASE)
_LEADING_DROP_RE = re.compile(rf"^\s*(?:{_LEADING_DROP})\s+", re.IGNORECASE)


def _distil(text: str) -> str:
    """Remove a small set of filler stopwords to concentrate meaning.

    Preserves word order and never touches the first or last token.
    """
    tokens = text.split()
    if len(tokens) <= 2:
        return text
    keep = [tokens[0]]
    for tok in tokens[1:-1]:
        if tok.lower().strip(".,;:!?") in _DISTIL_STOPWORDS:
            continue
        keep.append(tok)
    keep.append(tokens[-1])
    return " ".join(keep)


def question_to_claim(text: str, *, distil: bool = True) -> str:
    """Rewrite a natural-language research question as a declarative claim.

    Transparent, deterministic, no dependencies.  Nudges the user's input
    toward the declarative-claim distribution the cross-encoder was
    fine-tuned on — not to produce grammatically perfect English.
    """
    if not text or not text.strip():
        return ""
    s = text.strip().rstrip("?.!")

    s = _LEADING_DROP_RE.sub("", s).strip()

    # Drop a leading auxiliary or copula rather than moving it: the subject's
    # extent is unknown without parsing, so "Is autism spectrum disorder X"
    # must not become "autism is spectrum disorder X" (audit 05-L5).
    if _DO_DOES_DID_RE.match(s):
        s = _DO_DOES_DID_RE.sub("", s).strip()
    elif _IS_ARE_RE.match(s):
        s = _IS_ARE_RE.sub("", s).strip()

    if distil:
        s = _distil(s)

    return s.strip()


# ---------------------------------------------------------------------------
# Document-side text construction
# ---------------------------------------------------------------------------
def _paper_text(paper: dict, abstract_chars: int = 8000) -> str:
    """Build the document side of the (hypothesis, paper) pair.

    Matches the LLM scorer's text selection so side-by-side comparisons
    stay meaningful: prefer full text, else title + abstract + tldr.
    """
    fulltext = paper.get("full_text") or paper.get("fulltext") or ""
    if isinstance(fulltext, str) and fulltext.strip():
        return fulltext.strip()[: max(1, abstract_chars)]
    parts = [
        str(paper.get("title") or "").strip(),
        str(paper.get("abstract") or "").strip(),
        str(paper.get("tldr") or "").strip(),
    ]
    text = "\n\n".join(p for p in parts if p)
    return text[: max(1, abstract_chars)]


# ---------------------------------------------------------------------------
# Score conversion
# ---------------------------------------------------------------------------
def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# F1-optimal decision threshold reported in the brt-bert relevance model's
# final_metrics.json.  The model's probabilities are compressed (a clearly
# relevant paper scores ~0.52, just above this boundary), so it is exposed
# here for any threshold-anchored mapping.
BRT_BERT_RELEVANCE_THRESHOLD = 0.368


def _outputs_probability(model: object) -> bool | None:
    """Whether ``model.predict()`` returns probabilities, from its activation.

    True for a Sigmoid/Softmax ``activation_fn`` (our brt-bert checkpoints),
    False for Identity (logit output), None when the model does not say.
    """
    activation = getattr(model, "activation_fn", None)
    if activation is None:
        activation = getattr(model, "default_activation_function", None)
    if activation is None:
        return None
    name = getattr(activation, "__name__", None) or type(activation).__name__
    name = name.lower()
    if name in {"sigmoid", "softmax"}:
        return True
    if name == "identity":
        return False
    return None


def _to_probability(raw: float, is_probability: bool | None = None) -> float:
    """Coerce a CrossEncoder ``predict()`` output to a probability in [0, 1].

    ``is_probability`` comes from the model's configured activation (see
    :func:`_outputs_probability`): a probability is used as-is and a logit
    goes through the sigmoid, so the mapping is monotonic. Only when the
    model does not declare its activation is the range used to guess, which
    never applies the sigmoid twice to a probability.
    """
    if is_probability is True:
        return min(1.0, max(0.0, raw))
    if is_probability is False:
        return _sigmoid(raw)
    if 0.0 <= raw <= 1.0:
        return raw
    return _sigmoid(raw)


def _prob_to_ordinal(
    prob: float,
    threshold: float = BRT_BERT_RELEVANCE_THRESHOLD,
) -> int:
    """Map a relevance probability in [0, 1] to the app's 0-5 ordinal.

    Threshold-anchored piecewise-linear stretch: the model's F1-optimal
    decision boundary (``threshold``) is pinned to the score midpoint (2.5),
    and each side is stretched to fill its half of the 0-5 range.  Because
    fine-tuned CrossEncoders here emit probabilities compressed around the
    low 0.2-0.5 band, a plain ``prob * 5`` would park confident matches at
    3; this stretch lets high-confidence papers reach 4-5 and aligns the
    rater's distribution with a full-range LLM rater so weighted-κ reflects
    true rank agreement rather than a constant offset.

    The threshold is per-model — see :func:`_load_threshold`.
    """
    t = threshold
    midpoint = 2.5
    if prob <= t:
        # [0, t] -> [0, 2.5]
        scaled = (prob / t) * midpoint if t > 0 else midpoint
    else:
        # [t, 1] -> [2.5, 5]
        scaled = midpoint + ((prob - t) / (1.0 - t)) * (5.0 - midpoint)
    ordinal = int(scaled + 0.5)  # round half up (scaled is always >= 0)
    return max(0, min(5, ordinal))


def _raw_to_ordinal(
    raw: float,
    threshold: float = BRT_BERT_RELEVANCE_THRESHOLD,
    is_probability: bool | None = None,
) -> int:
    """Map a cross-encoder ``predict()`` output to the 0-5 ordinal.

    ``is_probability`` is the model's declared output kind (see
    :func:`_to_probability`). ``threshold`` should be the loaded model's
    per-model F1 threshold.
    """
    return _prob_to_ordinal(_to_probability(raw, is_probability), threshold=threshold)


def _empty_for(paper: dict) -> dict:
    return {
        "relevance_score": 0,
        "relevance_quote": "",
        "relevance_quote_source": "full_text" if (paper.get("full_text") or paper.get("fulltext")) else "abstract",
        "relevance_quote_valid": False,
    }


def _failed_for(paper: dict, error: str) -> dict:
    """A paper the cross-encoder could not rate: no score, never 0."""
    return {**_empty_for(paper), "relevance_score": None, "error": error}


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def score_relevance_embedding(
    paper: dict,
    *,
    hypothesis: str,
    backend: str = "brt_bert",
    model_path: str = "",
    device: str = "auto",
    abstract_chars: int = 8000,
    rewrite_hypothesis: bool = True,
) -> dict:
    """Score a single paper against *hypothesis* using a cross-encoder.

    Returns the LLM scorer's relevance shape. When the model cannot be loaded,
    ``predict()`` fails, or there is no text or question to compare,
    ``relevance_score`` is None and ``error`` says why: a failure is never
    reported as a score of 0.
    """
    empty = _empty_for(paper)
    model_ref = resolve_model_ref(backend, model_path)
    model = _load_model(model_ref, _resolve_device(device))
    if model is None:
        return _failed_for(paper, f"model_unavailable: cross-encoder {model_ref!r} could not be loaded")

    query = question_to_claim(hypothesis) if rewrite_hypothesis else hypothesis
    doc = _paper_text(paper, abstract_chars=abstract_chars)
    if not doc:
        return _failed_for(paper, "no_text")
    if not query:
        return _failed_for(paper, "no_question")

    try:
        raw = float(model.predict([(query, doc)])[0])
    except Exception as exc:
        logger.warning("Cross-encoder predict() failed for %r: %s",
                       str(paper.get("title") or "?")[:80], exc)
        return _failed_for(paper, f"predict_failed: {type(exc).__name__}: {exc}")

    threshold = _threshold_cache.get(model_ref, BRT_BERT_RELEVANCE_THRESHOLD)
    score = _raw_to_ordinal(raw, threshold=threshold, is_probability=_outputs_probability(model))
    return {**empty, "relevance_score": score}


def model_available(backend: str = "brt_bert", model_path: str = "") -> bool:
    """True if the resolved checkpoint exists locally (no HF-default fallback).

    Lets callers fail loudly for IRR rather than silently scoring with the
    wrong fallback model.
    """
    b = (backend or "").strip().lower()
    if b in _PRESET_DIRS:
        return (_repo_root() / _PRESET_DIRS[b]).exists()
    ref = (model_path or "").strip()
    if not ref:
        return False
    return Path(ref).exists()

