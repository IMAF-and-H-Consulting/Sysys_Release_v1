"""Stage 5 text-similarity metrics (pure Python, no optional dependencies).

Token Jaccard, containment, an exact two-document TF-IDF cosine, and a
best-window word error rate used by the Phase 2 rescue rules.
"""
import math
import re
from collections import Counter

_WORD_RE = re.compile(r"\w+")


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokens (alpha-only, len >= 2)."""
    return {w for w in re.findall(r"[a-z]{2,}", text.lower())}


def _word_tokens(text: str) -> list[str]:
    """Lowercase ``\\w+`` tokens in order, as used for word error rate."""
    return _WORD_RE.findall(text.lower())


def _token_similarity_sets(ta: set[str], tb: set[str]) -> float:
    """Jaccard similarity between two pre-tokenized sets."""
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _containment(span_tokens: set[str], abstract_tokens: set[str]) -> float:
    """Share of the span's word tokens found in the abstract (length-robust)."""
    if not span_tokens:
        return 0.0
    return len(span_tokens & abstract_tokens) / len(span_tokens)


# TF-IDF term pattern — matches scikit-learn's default ``token_pattern``.
_TFIDF_TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")

# Inverse document frequency over a two-document collection with
# ``smooth_idf=True``: idf(t) = ln((1 + n) / (1 + df)) + 1 for n = 2.
_IDF_IN_BOTH = 1.0                       # df = 2 → ln(3/3) + 1
_IDF_IN_ONE = math.log(3.0 / 2.0) + 1.0  # df = 1 → ln(3/2) + 1


def _tfidf_counts(text: str) -> Counter:
    """Term-frequency counts using scikit-learn's default tokenization."""
    return Counter(_TFIDF_TOKEN_RE.findall(text.lower()))


def _tfidf_similarity_counts(ca: Counter, cb: Counter) -> float:
    """Exact TF-IDF cosine between two documents, from precomputed counts.

    Reproduces ``TfidfVectorizer().fit_transform([a, b])`` followed by cosine
    similarity, but without constructing a vectorizer per call. Because the
    collection is exactly the two input documents, every term's document
    frequency is either 1 or 2, so the idf values collapse to two constants
    and the whole score can be computed in a single pass over the smaller
    counter. Scores therefore do not depend on whether scikit-learn is
    installed.
    """
    if not ca or not cb:
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for term, tf in ca.items():
        weight = tf * (_IDF_IN_BOTH if term in cb else _IDF_IN_ONE)
        norm_a += weight * weight
        if term in cb:
            dot += weight * (cb[term] * _IDF_IN_BOTH)
    for term, tf in cb.items():
        weight = tf * (_IDF_IN_BOTH if term in ca else _IDF_IN_ONE)
        norm_b += weight * weight
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _tfidf_similarity(a: str, b: str) -> float:
    """TF-IDF cosine similarity between two strings."""
    if not a.strip() or not b.strip():
        return 0.0
    return _tfidf_similarity_counts(_tfidf_counts(a), _tfidf_counts(b))


def _best_window_wer(ref: list[str], hyp: list[str]) -> tuple[float, int, int]:
    """Word error rate of *ref* against its best-aligned window of *hyp*.

    Semi-global alignment: the window may start and end anywhere in *hyp*, so
    abstract words outside the quoted passage are not counted as insertions.
    Same O(len(ref) * len(hyp)) cost as a plain WER. Returns
    ``(wer, start, end)`` with ``hyp[start:end]`` the aligned window.
    """
    if not ref:
        return (1.0 if hyp else 0.0), 0, 0
    # prev[j]: edits aligning ref[:i] to a window ending at hyp[j]; prev_start[j]: its start.
    prev = [0] * (len(hyp) + 1)
    prev_start = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        cur, cur_start = [i], [0]
        for j, hyp_word in enumerate(hyp, start=1):
            best, start = prev[j - 1] + (ref_word != hyp_word), prev_start[j - 1]
            if prev[j] + 1 < best:          # ref word deleted
                best, start = prev[j] + 1, prev_start[j]
            if cur[j - 1] + 1 < best:       # hyp word inserted
                best, start = cur[j - 1] + 1, cur_start[j - 1]
            cur.append(best)
            cur_start.append(start)
        prev, prev_start = cur, cur_start
    end = min(range(len(prev)), key=prev.__getitem__)
    return prev[end] / len(ref), prev_start[end], end
