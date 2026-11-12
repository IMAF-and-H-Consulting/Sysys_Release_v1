"""Claim clustering and narrative synthesis.

The deterministic Stage 6 artifacts live in sibling modules and are
re-exported here so ``process.synthesizer`` stays the single import point:

- ``process.synthesis_papers``: shared paper accessors
- ``process.synthesis_explanation``: hypothesis explanation
- ``process.synthesis_sources``: source citations and evidence network
"""
import logging
import math
import re
import warnings
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, cast

import numpy as np

from core.claims import claim_flag, claim_list
from core.config import DEFAULT_EMBEDDING_MODEL
from process.claim_extractor import _claim_think_mode as _synthesis_think_mode
from process.claim_extractor import _research_question_from_config
from process.synthesis_explanation import (  # noqa: F401  (re-exported)
    _grade_summary,
    build_hypothesis_explanation,
    render_hypothesis_explanation_markdown,
)
from process.synthesis_papers import (  # noqa: F401  (re-exported)
    _as_list,
    _claim_value,
    _paper_direction,
    _paper_id,
    _paper_quote,
    _paper_rank_key,
    _paper_score,
    _paper_title,
    _rank_papers,
    _source_identity_key,
    _top_terms,
)
from process.synthesis_sources import (  # noqa: F401  (re-exported)
    build_evidence_network,
    build_synthesis_source_citations,
    render_synthesis_source_citations_markdown,
)
from process.synthesis_narrative import (  # noqa: F401  (re-exported)
    NARRATIVE_COMBINE_PROMPT,
    NARRATIVE_PROMPT,
    _fair_clip_texts,
    _synthesis_capacity,
    generate_cluster_narrative,
)
from process.llm_budget import (
    ESTIMATOR_CHARS_PER_TOKEN,
    BudgetPlanningError,
    ModelCapacity,
    TokenBudget,
    estimate_messages_tokens,
    plan_token_budget,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedding model cache
# ---------------------------------------------------------------------------
_model_cache: dict = {}

# ---------------------------------------------------------------------------
# Prompt budget and cluster-quality policy
# ---------------------------------------------------------------------------
WEAK_SILHOUETTE_THRESHOLD = 0.25  # below this, clusters have little real structure
_EXPLICIT_ABSTRACT_OUTPUT_REASON = (
    "The user selected this Stage 6 abstract-response override. The 350-word publication "
    "format remains an intentional content constraint."
)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
ABSTRACT_PROMPT = """You are drafting a structured abstract for a systematic review.

Research question:
{research_question}

Total papers analyzed: {total_papers}

Pipeline facts (the only basis for the Methods section):
{pipeline_facts}

Cluster summaries:
{cluster_summaries}

Write a structured abstract with these sections:
- Background (1-2 sentences)
- Objective (1 sentence)
- Methods (2-3 sentences stating only the pipeline facts above; do not name databases, dates, or criteria that are not listed)
- Results (3-5 sentences, the key findings across all clusters)
- Conclusions (2-3 sentences)

Grounding rules:
- Use only the facts and cluster summaries above.
- Keep the bracketed citation keys from the cluster summaries when stating a finding; do not invent keys.
- Match certainty to the evidence: follow the GRADE note in the pipeline facts and do not say the evidence "confirms" or "establishes" anything without a GRADE assessment supporting it.

Keep it under 350 words."""


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------
def _get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Load sentence-transformers model (cached).  Returns None if unavailable."""
    if model_name in _model_cache:
        return _model_cache[model_name]
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning(
            "sentence_transformers not installed — using TF-IDF fallback for clustering")
        return None
    try:
        model = SentenceTransformer(model_name)
    except (OSError, RuntimeError, ValueError) as exc:
        # e.g. offline with no cached weights; fall back instead of aborting Stage 6.
        logger.warning(
            "Could not load embedding model %s (%s) — using TF-IDF fallback", model_name, exc)
        return None
    _model_cache[model_name] = model
    logger.info("Loaded sentence-transformers model: %s", model_name)
    return model


def _embedding_backend(model_name: str) -> str:
    """Name the encoder ``_get_embeddings`` used, mirroring its fallback order."""
    if _model_cache.get(model_name) is not None:
        return f"sentence-transformers:{model_name}"
    try:
        import sklearn  # noqa: F401
    except ImportError:
        return "hashing"
    return "tfidf"


def _paper_to_text(paper: dict) -> str:
    """Build text representation for embedding."""
    parts = [paper.get("title", "")]
    parts.append(paper.get("abstract", ""))
    parts.extend(claim_list(paper, "findings"))
    parts.extend(claim_list(paper, "mechanisms"))
    return " ".join(p for p in parts if p)



_HASHING_STOP_WORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for", "from", "had", "has", "have", "he", "her", "his", "if", "in", "into", "is", "it", "its", "of", "on", "or", "she", "that", "the", "their", "then", "there", "these", "they", "this", "to", "was", "were", "what", "when", "which", "who", "will", "with", "would", "you", "your", "our", "we", "us", "not", "no", "than", "them"]
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _hashing_embeddings(texts: list[str], n_features: int = 512) -> np.ndarray:
    """Embed texts with deterministic feature hashing (no third-party deps).

    This is the last-resort encoder used when neither ``sentence-transformers``
    nor ``sklearn`` is installed. It is a sublinear-tf, L2-normalized bag of
    hashed unigrams — far weaker than a real sentence embedding, but it is a
    genuine function of the text, so clustering still groups papers by shared
    vocabulary instead of by noise.

    Token buckets come from :func:`zlib.crc32` rather than the builtin
    :func:`hash`, because ``hash()`` on ``str`` is salted per interpreter
    process (``PYTHONHASHSEED``) and would make clusters irreproducible
    across runs.
    """
    matrix = np.zeros((len(texts), n_features), dtype=np.float64)
    for row, text in enumerate(texts):
        counts: Counter = Counter(
            tok
            for tok in _TOKEN_RE.findall(text.lower())
            if len(tok) > 2 and tok not in _HASHING_STOP_WORDS
        )
        for token, count in counts.items():
            bucket = zlib.crc32(token.encode("utf-8")) % n_features
            # Signed hashing cancels collision bias in expectation.
            sign = 1.0 if (zlib.crc32(b"s" + token.encode("utf-8")) & 1) else -1.0
            matrix[row, bucket] += sign * (1.0 + np.log(count))
        norm = np.linalg.norm(matrix[row])
        if norm > 0:
            matrix[row] /= norm
    return matrix


def _get_embeddings(
    papers: list[dict], model_name: str = DEFAULT_EMBEDDING_MODEL
) -> np.ndarray:
    """Embed papers.  Uses sentence-transformers if available, else TF-IDF."""
    model = _get_embedding_model(model_name)
    texts = [_paper_to_text(p) for p in papers]

    if model is not None:
        logger.info("Encoding %d papers with sentence-transformers", len(texts))
        return np.asarray(model.encode(texts, show_progress_bar=False))

    # TF-IDF fallback
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        logger.info("Encoding %d papers with TF-IDF fallback", len(texts))
        vec = TfidfVectorizer(max_features=5000, stop_words="english")
        tfidf_matrix: Any = vec.fit_transform(texts)
        return tfidf_matrix.toarray()
    except ImportError:
        logger.warning(
            "Neither sentence_transformers nor sklearn is installed — falling back to "
            "deterministic feature hashing for %d papers. Clusters will be based on "
            "shared vocabulary only, not semantics; install sentence-transformers or "
            "scikit-learn before relying on these clusters in a published review.",
            len(texts),
        )
        return _hashing_embeddings(texts)


# ---------------------------------------------------------------------------
# Simple k-means fallback (no sklearn)
# ---------------------------------------------------------------------------
def _simple_kmeans(
    X: np.ndarray, k: int, max_iter: int = 50, seed: int = 42
) -> np.ndarray:
    """Minimal k-means via random partition.

    Returns an array of cluster labels (length = number of rows in X).

    Uses a locally seeded generator so repeated runs over the same corpus
    produce identical clusters. The global ``np.random`` state is never
    touched, so callers cannot perturb clustering by seeding elsewhere.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    # Random initial assignment
    labels = np.array([i % k for i in range(n)])
    rng.shuffle(labels)

    for _iteration in range(max_iter):
        # Compute centroids
        centroids = np.zeros((k, X.shape[1]))
        for ci in range(k):
            members = X[labels == ci]
            if len(members) > 0:
                centroids[ci] = members.mean(axis=0)
            else:
                centroids[ci] = X[rng.integers(n)]

        # Reassign
        new_labels = np.argmin(
            np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2),
            axis=1,
        )
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

    return labels


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def _select_k_by_silhouette(
    embeddings: np.ndarray,
    k_min: int = 2,
    k_max: int = 15,
    seed: int = 42,
    selection_info: dict | None = None,
) -> tuple[int, list[tuple[int, float]]]:
    """Evaluate every K through the requested maximum (limited only by N-1).

    Article-count heuristics and small clusters are advisory. Return the
    silhouette maximum as an automatic recommendation, plus the full sweep.
    """
    n = embeddings.shape[0]
    starting_cap = math.isqrt(n // 2)
    upper = min(k_max, n - 1)
    minimum_size = max(1, math.ceil(n * 0.02))
    info = selection_info if selection_info is not None else {}
    info.update(article_count=n, square_root_cap=starting_cap,
                requested_max_k=k_max, search_max_k=max(0, upper),
                min_cluster_fraction=0.02, min_cluster_size=minimum_size,
                stop_reason="search_cap", candidates=[], inertia_scores=[])
    k_min = max(2, k_min)
    if upper < k_min:
        info["stop_reason"] = "too_few_articles"
        return 1, []
    try:
        from sklearn.metrics import silhouette_score
    except ImportError:
        logger.warning("sklearn not available — cannot sweep silhouette scores; using k=1")
        info["stop_reason"] = "sklearn_unavailable"
        return 1, []

    scores: list[tuple[int, float]] = []
    for k in range(k_min, upper + 1):
        labels = _sklearn_kmeans_labels(embeddings, k, seed)
        _, counts = np.unique(labels, return_counts=True)
        candidate = {"k": k, "cluster_sizes": counts.tolist()}
        info["candidates"].append(candidate)
        if len(counts) != k:
            candidate["status"] = "collapsed_clusters"
            continue
        candidate["small_cluster_warning"] = int(counts.min()) < minimum_size
        inertia = sum(float(np.sum((members - members.mean(axis=0)) ** 2))
                      for label in np.unique(labels)
                      for members in [embeddings[labels == label]])
        candidate["inertia"] = inertia
        info["inertia_scores"].append({"k": k, "inertia": inertia})
        try:
            score = float(silhouette_score(embeddings, labels))
        except ValueError as exc:
            candidate["status"] = "unscorable"
            logger.warning("  silhouette sweep: k=%d cannot be scored (%s)", k, exc)
            continue
        if not math.isfinite(score):
            candidate["status"] = "unscorable"
            continue
        candidate.update(status="eligible", score=score)
        scores.append((k, score))
        logger.info("  silhouette sweep: k=%d score=%.4f", k, score)

    if not scores:
        if info["stop_reason"] == "search_cap":
            info["stop_reason"] = "no_valid_candidates"
        logger.warning("No eligible k could be scored; using k=1")
        return 1, []
    info["declining_from_k2"] = (len(scores) >= 4 and scores[0][0] == 2
        and all(k2 == k1 + 1 and s2 < s1
                for (k1, s1), (k2, s2) in zip(scores, scores[1:])))
    best_k, best_score = max(scores, key=lambda t: t[1])
    info["recommended_k"] = best_k
    logger.info("  Selected k=%d (silhouette %.4f); stopped: %s",
                best_k, best_score, info["stop_reason"])
    return best_k, scores


def _auto_k_info(chosen_k: int, sweep: list[tuple[int, float]]) -> dict:
    """Describe an auto-k selection, flagging weak or missing cluster structure."""
    if not sweep:
        return {
            "mode": "fallback",
            "chosen_k": chosen_k,
            "silhouette_scores": [],
            "structure_warning": (
                "No eligible silhouette score was available (too few papers, sklearn unavailable, or no "
                "valid clustering under the selection rules), so k was not selected by silhouette score."),
        }
    best = max(score for _, score in sweep)
    info = {
        "mode": "auto",
        "chosen_k": chosen_k,
        "silhouette_scores": [{"k": k, "score": round(s, 4)} for k, s in sweep],
        "best_silhouette": round(best, 4),
        "weak_structure": best < WEAK_SILHOUETTE_THRESHOLD,
    }
    if info["weak_structure"]:
        info["structure_warning"] = (
            f"Best silhouette score {best:.3f} is below {WEAK_SILHOUETTE_THRESHOLD:.2f}: "
            "the papers show little separable structure, so the clusters are "
            "descriptive groupings rather than distinct themes.")
    return info


def _fixed_k(n_clusters: int, n_papers: int) -> int:
    actual_k = min(n_clusters, n_papers)
    if actual_k < n_clusters:
        logger.info(
            "Reducing clusters from %d to %d (only %d papers)",
            n_clusters, actual_k, n_papers,
        )
    return max(actual_k, 1)


def _sklearn_kmeans_labels(embeddings: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    """sklearn k-means labels (raises ImportError without sklearn).

    Duplicate rows can yield fewer distinct clusters than k. sklearn's
    ConvergenceWarning is silenced because callers handle that case:
    the silhouette sweep skips the k, and empty clusters are dropped and
    logged.
    """
    from sklearn.cluster import KMeans
    from sklearn.exceptions import ConvergenceWarning

    # sklearn's own signature leaves n_init unannotated (default "auto"),
    # so pyright infers it as `str`; an int is a documented-valid value.
    km = KMeans(n_clusters=k, n_init=cast(str, 10), random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return km.fit_predict(embeddings)


def _fit_labels(embeddings: np.ndarray, k: int) -> np.ndarray:
    try:
        logger.info("Running sklearn KMeans with k=%d on %d rows", k, embeddings.shape[0])
        return _sklearn_kmeans_labels(embeddings, k)
    except ImportError:
        logger.info("sklearn not available — running simple k-means fallback with k=%d", k)
        return _simple_kmeans(embeddings, k)


def _ranked_terms(papers: list[dict], field: str) -> list[str]:
    """Distinct claim terms, most frequent first (ties alphabetical)."""
    counts = Counter(term for p in papers for term in claim_list(p, field))
    return [term for term, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def _build_clusters(papers: list[dict], labels, k: int) -> list[dict]:
    """One summary per non-empty label, renumbered 0..n-1."""
    clusters: list[dict] = []
    for ci in range(k):
        cluster_papers = [papers[i] for i, lb in enumerate(labels) if lb == ci]
        if not cluster_papers:
            continue
        clusters.append({
            "cluster_id": len(clusters),
            "papers": cluster_papers,
            "paper_count": len(cluster_papers),
            "mechanisms": _ranked_terms(cluster_papers, "mechanisms"),
            "conditions": _ranked_terms(cluster_papers, "conditions"),
            "key_entities": _ranked_terms(cluster_papers, "key_entities"),
            "primary_support_count": sum(1 for p in cluster_papers if claim_flag(p)),
        })
    return clusters


def cluster_claims(
    papers: list[dict],
    n_clusters: int = 8,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    k_info: dict | None = None,
    k_max: int = 15,
) -> list[dict]:
    """Cluster papers thematically using k-means on embeddings.

    Pass ``n_clusters=0`` (or less) to auto-select k by silhouette score
    (all K through k_max; article-count heuristics are advisory).
    When *k_info* is given as a dict it
    is filled with ``mode`` (``auto``/``fixed``/``fallback``), ``chosen_k``,
    ``silhouette_scores``, ``embedding_backend`` and
    ``empty_clusters_dropped``, plus ``best_silhouette``/``weak_structure``/
    ``structure_warning`` when relevant.

    Empty clusters are dropped and the rest renumbered. Term lists are
    ordered by frequency within the cluster.

    Returns list of cluster dicts::

        [{"cluster_id": 0, "papers": [...], "mechanisms": [...],
          "conditions": [...], "key_entities": [...],
          "primary_support_count": int}, ...]
    """
    if not papers:
        logger.warning("No papers to cluster — returning empty list")
        return []

    embeddings = _get_embeddings(papers, model_name=embedding_model)
    if n_clusters <= 0:
        selection_info: dict = {}
        actual_k, sweep = _select_k_by_silhouette(embeddings, k_max=k_max, selection_info=selection_info)
        info = _auto_k_info(actual_k, sweep)
        info.update(selection_info)
    else:
        actual_k = _fixed_k(n_clusters, len(papers))
        info = {"mode": "fixed", "chosen_k": actual_k, "silhouette_scores": []}

    clusters = _build_clusters(papers, _fit_labels(embeddings, actual_k), actual_k)
    info["empty_clusters_dropped"] = actual_k - len(clusters)
    info["embedding_backend"] = _embedding_backend(embedding_model)
    if info["empty_clusters_dropped"]:
        logger.warning("Dropped %d empty cluster(s) out of k=%d",
                       info["empty_clusters_dropped"], actual_k)
    if k_info is not None:
        k_info.update(info)

    logger.info(
        "Clustered %d papers into %d clusters — sizes: %s",
        len(papers), len(clusters), [c["paper_count"] for c in clusters],
    )
    return clusters


# ---------------------------------------------------------------------------
# Narrative generation
# ---------------------------------------------------------------------------
def _clustering_fact(clusters: list[dict], k_info: dict) -> str:
    sizes = ", ".join(str(c.get("paper_count", 0)) for c in clusters)
    backend = k_info.get("embedding_backend") or "text embeddings"
    mode = k_info.get("mode")
    if mode == "auto":
        how = (f"k chosen by silhouette sweep (best silhouette "
               f"{k_info.get('best_silhouette', 0):.2f})")
    elif mode == "fixed":
        how = f"k fixed at {k_info.get('chosen_k')} by the user"
    else:
        how = "no eligible silhouette score was available to select k"
    return (f"Papers were embedded ({backend}) and grouped by k-means into "
            f"{len(clusters)} clusters ({how}); cluster sizes: {sizes}.")


def _grade_fact(grade_summary: dict | None) -> str:
    counts = (grade_summary or {}).get("certainty_counts") or {}
    if not counts:
        return ("GRADE certainty was not assessed for this synthesis: use calibrated, "
                "hedged language and do not state that the evidence is certain.")
    listed = ", ".join(f"{label}: {n}" for label, n in counts.items())
    return f"GRADE certainty ratings across outcomes: {listed}."


def build_pipeline_facts(
    *,
    total_papers: int,
    clusters: list[dict],
    k_info: dict | None = None,
    k_max: int = 15,
    claims_path: str | Path | None = None,
    grade_summary: dict | None = None,
) -> list[str]:
    """Facts about how this synthesis was produced, for the abstract's Methods."""
    k_info = k_info or {}
    source = Path(claims_path).name if claims_path else "the Stage 3 claims output"
    facts = [
        f"Input: {total_papers} papers with LLM-extracted claims, read from {source}.",
        _clustering_fact(clusters, k_info),
        ("Each cluster narrative was written by an LLM from paper titles, abstract text and "
         "extracted findings; any context-derived evidence selection is recorded in the "
         "narrative input-coverage metadata."),
        "No meta-analysis or risk-of-bias results were supplied to this draft.",
        _grade_fact(grade_summary),
    ]
    if k_info.get("structure_warning"):
        facts.append(str(k_info["structure_warning"]))
    return facts


def _abstract_cluster_block(cluster_narratives: list[dict]) -> str:
    parts: list[str] = []
    for cn in cluster_narratives:
        mechs = ", ".join(claim_list(cn, "mechanisms"))
        snippet = cn.get("narrative", "") or ""
        parts.append(
            f"Cluster {cn.get('cluster_id', '?')} ({cn.get('paper_count', 0)} papers, "
            f"mechanisms: {mechs}):\n{snippet}"
        )
    return "\n\n".join(parts) if parts else "No cluster summaries."


def _abstract_prompt_text(
    cluster_block: str,
    total_papers: int,
    research_config: dict | None,
    pipeline_facts: list[str],
) -> str:
    return ABSTRACT_PROMPT.format(
        research_question=_research_question_from_config(research_config),
        total_papers=total_papers,
        pipeline_facts="\n".join(f"- {fact}" for fact in pipeline_facts),
        cluster_summaries=cluster_block,
    )


def _fit_abstract_prompt(
    cluster_narratives: list[dict],
    total_papers: int,
    research_config: dict | None,
    pipeline_facts: list[str],
    capacity: ModelCapacity,
    max_tokens: int | None,
) -> tuple[str, dict[str, Any], TokenBudget]:
    expected = max(1, len(cluster_narratives))
    explicit = int(max_tokens) if max_tokens and int(max_tokens) > 0 else None

    def plan(prompt: str) -> TokenBudget:
        return plan_token_budget(
            "synthesis_abstract", capacity,
            messages=[{"role": "user", "content": prompt}],
            expected_items=expected,
            explicit_output_tokens=explicit,
            explicit_reason=_EXPLICIT_ABSTRACT_OUTPUT_REASON,
        )

    full_block = _abstract_cluster_block(cluster_narratives)
    full_prompt = _abstract_prompt_text(
        full_block, total_papers, research_config, pipeline_facts)
    source_chars = sum(
        len(cn.get("narrative", "") or "")
        + len(", ".join(claim_list(cn, "mechanisms")))
        for cn in cluster_narratives
    )
    try:
        budget = plan(full_prompt)
        return full_prompt, {
            "context_window_tokens": capacity.context_window_tokens,
            "clusters_available": len(cluster_narratives),
            "clusters_submitted": len(cluster_narratives),
            "source_chars": source_chars,
            "submitted_chars": source_chars,
            "coverage_ratio": 1.0,
            "truncated": False,
            "policy": "complete-cluster-narratives",
            "reason": "All cluster narratives and mechanism terms fit the model context.",
            "format_constraint": "Structured abstract under 350 words.",
        }, budget
    except BudgetPlanningError:
        pass

    headers = [
        f"Cluster {cn.get('cluster_id', '?')} ({cn.get('paper_count', 0)} papers):"
        for cn in cluster_narratives
    ]
    variables = [
        (f" mechanisms: {', '.join(claim_list(cn, 'mechanisms'))}\n"
         f"{cn.get('narrative', '') or ''}")
        for cn in cluster_narratives
    ]
    base_block = "\n\n".join(headers) if headers else "No cluster summaries."
    base_prompt = _abstract_prompt_text(
        base_block, total_papers, research_config, pipeline_facts)
    base_budget = plan(base_prompt)
    available_tokens = max(
        0,
        base_budget.input_capacity_tokens
        - estimate_messages_tokens([{"role": "user", "content": base_prompt}]),
    )
    fitted = _fair_clip_texts(
        variables, int(available_tokens * ESTIMATOR_CHARS_PER_TOKEN))
    while True:
        block = "\n\n".join(
            header + value for header, value in zip(headers, fitted)
        ) if headers else "No cluster summaries."
        prompt = _abstract_prompt_text(
            block, total_papers, research_config, pipeline_facts)
        try:
            budget = plan(prompt)
            break
        except BudgetPlanningError:
            if not any(fitted):
                raise
            fitted = [value[:max(0, int(len(value) * 0.95))].rstrip()
                      for value in fitted]
    submitted_chars = sum(len(value) for value in fitted)
    coverage = {
        "context_window_tokens": capacity.context_window_tokens,
        "clusters_available": len(cluster_narratives),
        "clusters_submitted": len(cluster_narratives),
        "source_chars": source_chars,
        "submitted_chars": submitted_chars,
        "coverage_ratio": round(submitted_chars / source_chars, 6) if source_chars else 1.0,
        "truncated": submitted_chars < source_chars,
        "policy": "context-derived-fair-cluster-allocation",
        "reason": (
            "All clusters remain represented, but their narrative text was fairly reduced to "
            "fit after reserving the context-relative structured-abstract response."
        ),
        "format_constraint": "Structured abstract under 350 words.",
    }
    logger.warning(
        "Stage 6 abstract input coverage: %d/%d chars (%.1f%%). %s",
        submitted_chars, source_chars, 100.0 * coverage["coverage_ratio"],
        coverage["reason"],
    )
    return prompt, coverage, budget


def generate_abstract_draft(
    model: str,
    cluster_narratives: list[dict],
    total_papers: int,
    *,
    research_config: dict | None = None,
    pipeline_facts: list[str] | None = None,
    use_thinking: bool = False,
    max_tokens: int | None = None,
    model_capacity: ModelCapacity | int | None = None,
    budget_metadata: dict[str, Any] | None = None,
    cancel_event: Any = None,
) -> str:
    """Generate a structured abstract from all cluster narratives.

    *pipeline_facts* (see :func:`build_pipeline_facts`) are the only basis
    the prompt allows for the Methods section.
    """
    from process.llm import chat as llm_chat

    if pipeline_facts is None:
        pipeline_facts = build_pipeline_facts(
            total_papers=total_papers, clusters=cluster_narratives)
    capacity = _synthesis_capacity(model_capacity)
    prompt_text, coverage, budget = _fit_abstract_prompt(
        cluster_narratives, total_papers, research_config, pipeline_facts,
        capacity, max_tokens,
    )
    if budget_metadata is not None:
        budget_metadata.update(input_coverage=coverage, token_budget=budget.as_dict())
    logger.info(
        "Generating abstract draft from %d cluster narratives (%d total papers, model=%s)",
        len(cluster_narratives), total_papers, model,
    )
    abstract = llm_chat(
        model=model,
        messages=[{"role": "user", "content": prompt_text}],
        max_tokens=max_tokens if max_tokens and max_tokens > 0 else None,
        think=_synthesis_think_mode(model, enabled=use_thinking),
        cancel_event=cancel_event,
        task="synthesis_abstract",
        expected_items=max(1, len(cluster_narratives)),
        context_tokens=capacity.context_window_tokens,
        token_budget=budget,
        explicit_limit_reason=_EXPLICIT_ABSTRACT_OUTPUT_REASON,
    )
    logger.info("Abstract draft generated: %d chars", len(abstract))
    return abstract
