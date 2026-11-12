"""Human calibration queue for relevance-screening agreement checks."""
from __future__ import annotations

import hashlib
import math
import random
from datetime import UTC, datetime
from typing import Any

from core.agreement import adjacent_agreement
from process.relevance_cascade import probability_bin

ARTICLE_ID_KEYS = (
    "doi",
    "pmid",
    "s2_id",
    "s2_paper_id",
    "paperId",
    "oa_id",
    "openalex_id",
    "id",
)


BINARY_SCORE_KIND = "binary_screening_decision"
# Each non-empty stratum gets at least this many papers when n allows, so a
# rare stratum (e.g. the few kept papers) is never starved (audit 05-M5).
MIN_PER_STRATUM = 2
_CONTENT_ID_CHARS = 16


def _content_id(paper: dict[str, Any]) -> str:
    """A stable id from the paper's text, or "" when it has none."""
    text = " ".join(str(
        paper.get("abstract") or paper.get("full_text") or paper.get("tldr") or ""
    ).split())
    if not text:
        return ""
    return "text:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:_CONTENT_ID_CHARS]


def _identity(paper: dict[str, Any]) -> str:
    """The paper's id, title or text hash; "" when it has none of them."""
    for key in ARTICLE_ID_KEYS:
        value = str(paper.get(key, "") or "").strip()
        if value:
            return value
    title = str(paper.get("title", "") or "").strip()
    return title[:120] if title else _content_id(paper)


def article_id(paper: dict[str, Any], index: int = 0) -> str:
    """The paper's identity; ``paper_<index+1>`` only for a record with no content.

    A positional id differs between a sample and the scored list, so it is
    never used to pair a review row with a scored paper (audit 05-L1).
    """
    return _identity(paper) or f"paper_{index + 1}"


def recommended_sample_size(corpus_size: int) -> int:
    """Pick a small human-review queue size that scales from 20 to 50."""
    if corpus_size <= 0:
        return 0
    if corpus_size < 20:
        return corpus_size
    if corpus_size <= 150:
        return 20
    if corpus_size <= 500:
        return 30
    if corpus_size <= 1500:
        return 40
    return 50


def _abstract_snippet(paper: dict[str, Any], limit: int = 420) -> str:
    text = str(
        paper.get("abstract")
        or paper.get("tldr")
        or paper.get("full_text")
        or paper.get("text")
        or ""
    ).strip()
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _score_bin(paper: dict[str, Any]) -> str:
    score = paper.get("relevance_score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return "unscored"
    if score <= 2:
        return "low"
    if score <= 4:
        return "mid"
    return "high"


def _screening(paper: dict[str, Any]) -> dict[str, Any] | None:
    screening = paper.get("screening")
    return screening if isinstance(screening, dict) and "passed" in screening else None


def _stratum(paper: dict[str, Any]) -> str:
    """Cascade keep/reject decision and probability bin, else the score bin."""
    screening = _screening(paper)
    if screening is None:
        return _score_bin(paper)
    decision = "keep" if screening["passed"] else "reject"
    probability = screening.get("laya_p")
    if probability is None:
        probability = screening.get("brt_p")
    band = f"bin{probability_bin(probability)}" if probability is not None else "no_p"
    return f"{decision}:{band}"


def _stratified_on(papers: list[dict[str, Any]]) -> str:
    if any(_screening(paper) is not None for paper in papers):
        return "screening_decision"
    if any(_score_bin(paper) != "unscored" for paper in papers):
        return "relevance_score"
    return "none"


def _strata(papers: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    strata: dict[str, list[dict[str, Any]]] = {}
    for paper in papers:
        strata.setdefault(_stratum(paper), []).append(paper)
    return strata


def _allocate(sizes: dict[str, int], n: int) -> dict[str, int]:
    """Per-stratum sample sizes: a minimum each (round-robin, rarest first),
    then the rest proportional to stratum size (largest remainder)."""
    alloc = dict.fromkeys(sizes, 0)
    rarest_first = sorted(sizes, key=lambda k: (sizes[k], k))
    for _ in range(MIN_PER_STRATUM):
        for name in rarest_first:
            if sum(alloc.values()) < n and alloc[name] < sizes[name]:
                alloc[name] += 1
    remaining = n - sum(alloc.values())
    total = sum(sizes.values())
    quotas = {name: remaining * size / total for name, size in sizes.items()}
    for name, size in sizes.items():
        alloc[name] += min(int(quotas[name]), size - alloc[name])
    by_remainder = sorted(sizes, key=lambda k: (int(quotas[k]) - quotas[k], k))
    left = n - sum(alloc.values())
    while left > 0:
        for name in [k for k in by_remainder if alloc[k] < sizes[k]][:left]:
            alloc[name] += 1
        left = n - sum(alloc.values())
    return alloc


def _stratified_sample(papers: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """A sample of n papers stratified on the Stage 2 decision or score.

    Every stratum gets at least ``MIN_PER_STRATUM`` papers when n allows,
    the rest is proportional to stratum size, and the result is shuffled so
    the queue order does not reveal the model's decision.
    """
    if n >= len(papers):
        return list(papers)
    if n <= 0:
        return []
    rng = random.Random(seed)
    strata = _strata(papers)
    alloc = _allocate({name: len(rows) for name, rows in strata.items()}, n)
    selected = [paper for name in sorted(strata)
                for paper in rng.sample(strata[name], alloc[name])]
    rng.shuffle(selected)
    return selected


def build_review_queue(
    papers: list[dict[str, Any]],
    *,
    sample_size: int | None = None,
    seed: int = 42,
    source_path: str = "",
) -> dict[str, Any]:
    size = recommended_sample_size(len(papers)) if not sample_size else int(sample_size)
    size = max(0, min(size, len(papers)))
    sample = _stratified_sample(papers, size, seed)

    records = []
    for index, paper in enumerate(sample):
        records.append({
            "review_id": f"human_{index + 1:03d}",
            "article_id": article_id(paper, index),
            "title": str(paper.get("title") or ""),
            "year": paper.get("year"),
            "doi": paper.get("doi"),
            "pmid": paper.get("pmid"),
            "source": paper.get("source"),
            "abstract": _abstract_snippet(paper),
            "model_score": paper.get("relevance_score"),
            "model_quote": paper.get("relevance_quote", ""),
            "human_score": paper.get("human_score"),
            "human_verified": bool(paper.get("human_verified", False)),
            "human_notes": paper.get("human_notes", ""),
        })

    return {
        "stage": "human_relevance_review",
        "created_at": datetime.now(UTC).isoformat(),
        "source_path": source_path,
        "corpus_size": len(papers),
        "sample_size": len(records),
        "seed": seed,
        "stratified_on": _stratified_on(papers),
        "strata": {name: len(rows) for name, rows in sorted(_strata(papers).items())},
        "score_scale": "0=not relevant, 5=highly relevant",
        "records": records,
    }


def _score_from_row(row: dict[str, Any], key: str) -> int | None:
    """An integral 0-5 score (int, whole float or digit string), else None."""
    value = row.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        text = value.strip()
        value = int(text) if text.isdigit() else None
    if isinstance(value, float):
        if not (math.isfinite(value) and value.is_integer()):
            return None
        value = int(value)
    if isinstance(value, int) and 0 <= value <= 5:
        return value
    return None


def _weighted_kappa(scores_a: list[int], scores_b: list[int], max_score: int = 5) -> float:
    if not scores_a or len(scores_a) != len(scores_b):
        return 0.0
    n = len(scores_a)
    observed = sum(abs(a - b) / max_score for a, b in zip(scores_a, scores_b)) / n
    counts_a = [scores_a.count(i) for i in range(max_score + 1)]
    counts_b = [scores_b.count(i) for i in range(max_score + 1)]
    expected = 0.0
    for i, count_a in enumerate(counts_a):
        for j, count_b in enumerate(counts_b):
            expected += (count_a / n) * (count_b / n) * (abs(i - j) / max_score)
    if expected == 0:
        return 1.0 if observed == 0 else 0.0
    return round(1.0 - (observed / expected), 4)


def _cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    if not labels_a or len(labels_a) != len(labels_b):
        return 0.0
    labels = sorted(set(labels_a) | set(labels_b))
    n = len(labels_a)
    observed = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    expected = sum((labels_a.count(label) / n) * (labels_b.count(label) / n) for label in labels)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return round((observed - expected) / (1.0 - expected), 4)


def _index_scored_papers(scored_papers: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Scored papers by identity and title; papers with neither are not indexed."""
    indexed: dict[str, dict[str, Any]] = {}
    for paper in scored_papers:
        key = _identity(paper)
        if not key:
            continue
        indexed[key] = paper
        title = str(paper.get("title") or "").strip()
        if title:
            indexed.setdefault(title[:120], paper)
    return indexed


def _lookup(scored_index: dict[str, dict[str, Any]], row: dict[str, Any]) -> dict | None:
    matched = scored_index.get(str(row.get("article_id") or "").strip())
    if matched is None:
        title = str(row.get("title") or "").strip()
        matched = scored_index.get(title[:120]) if title else None
    return matched


def _is_binary_decision(paper: dict[str, Any] | None) -> bool:
    return bool(paper) and (
        paper.get("relevance_score_kind") == BINARY_SCORE_KIND or _screening(paper) is not None)


def _model_score_for(row: dict[str, Any], matched: dict | None) -> tuple[int | None, str]:
    if row.get("model_score_human_corrected"):
        return _score_from_row(row, "model_score"), "human_correction"
    if matched is not None:
        return _score_from_row(matched, "relevance_score"), "model_output"
    # A score copied into the queue earlier, with no scored paper to confirm it.
    return _score_from_row(row, "model_score"), "review_queue"


def _model_relevant(score: int, matched: dict | None, binary: bool, threshold: int) -> bool:
    """The model's keep decision; a binary cascade decision ignores ``threshold``."""
    if not binary:
        return score >= threshold
    screening = _screening(matched or {})
    return bool(screening["passed"]) if screening is not None else score > 0


def _compare_row(row: dict[str, Any], scored_index: dict, threshold: int) -> dict | None:
    human_score = _score_from_row(row, "human_score")
    if human_score is None:
        return None
    matched = _lookup(scored_index, row)
    model_score, source = _model_score_for(row, matched)
    comparison = {
        "article_id": str(row.get("article_id") or "").strip(),
        "title": row.get("title", ""),
        "human_score": human_score,
        "model_score": model_score,
        "model_score_source": source,
        "human_verified": bool(row.get("human_verified", False)),
        "matched": matched is not None,
        "paired": model_score is not None and source != "review_queue",
    }
    if not comparison["paired"]:
        return comparison
    binary = source == "model_output" and _is_binary_decision(matched)
    return {
        **comparison,
        "binary_model_decision": binary,
        "human_relevant": human_score >= threshold,
        "model_relevant": _model_relevant(model_score, matched, binary, threshold),
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _decision_metrics(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Agreement on the relevant / not-relevant decision (human as reference)."""
    human = [row["human_relevant"] for row in pairs]
    model = [row["model_relevant"] for row in pairs]
    agree = sum(h == m for h, m in zip(human, model))
    true_pos = sum(h and m for h, m in zip(human, model))
    true_neg = sum(not h and not m for h, m in zip(human, model))
    labels = {True: "relevant", False: "not_relevant"}
    return {
        "threshold_agreement_pct": round(100 * agree / len(pairs), 2) if pairs else 0.0,
        "threshold_kappa": _cohens_kappa([labels[h] for h in human], [labels[m] for m in model]),
        "sensitivity": _ratio(true_pos, sum(human)),
        "specificity": _ratio(true_neg, len(human) - sum(human)),
    }


def _ordinal_metrics(pairs: list[dict[str, Any]], binary: bool) -> dict[str, Any]:
    """Score-level agreement; not applicable to the cascade's 0/5 decision (05-M6)."""
    if binary:
        return {**adjacent_agreement([], []), "exact_agreement_pct": None,
                "score_weighted_kappa": None, "ordinal_metrics": "not_applicable"}
    human = [row["human_score"] for row in pairs]
    model = [row["model_score"] for row in pairs]
    exact = sum(h == m for h, m in zip(human, model))
    return {
        **adjacent_agreement(human, model),
        "exact_agreement_pct": round(100 * exact / len(pairs), 2) if pairs else 0.0,
        "score_weighted_kappa": _weighted_kappa(human, model),
        "ordinal_metrics": "applicable",
    }


def compare_human_to_model(
    review_records: list[dict[str, Any]],
    scored_papers: list[dict[str, Any]],
    *,
    threshold: int = 3,
) -> dict[str, Any]:
    """Agreement between human review scores and the Stage 2 model output.

    A row is paired when its paper is found in ``scored_papers`` (or carries
    a human-corrected model score). ``matched`` counts pairs; ``unmatched``
    counts reviewed rows without one. When the model scores are the
    cascade's binary decision, only decision metrics are reported.
    """
    scored_index = _index_scored_papers(scored_papers)
    comparisons = [c for c in (_compare_row(row, scored_index, threshold)
                               for row in review_records) if c is not None]
    pairs = [row for row in comparisons if row["paired"]]
    binary = any(row["binary_model_decision"] for row in pairs)
    return {
        "stage": "human_relevance_agreement",
        "created_at": datetime.now(UTC).isoformat(),
        "threshold": threshold,
        "score_kind": BINARY_SCORE_KIND if binary else "ordinal",
        "reviewed": len(comparisons),
        "human_verified": sum(bool(row.get("human_verified", False)) for row in review_records),
        "matched": len(pairs),
        "unmatched": len(comparisons) - len(pairs),
        **_ordinal_metrics(pairs, binary),
        **_decision_metrics(pairs),
        "comparisons": comparisons,
    }
