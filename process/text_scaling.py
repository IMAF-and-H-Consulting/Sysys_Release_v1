"""Wordfish text scaling and lightweight stylometric profiling.

The stylometric counts are heuristics: passive voice is approximated from
"be + participle" patterns and sentences are split on terminal punctuation
outside common abbreviations, so treat both as approximate rates.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

_WORD_RE = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*")
_SENTENCE_BREAK_RE = re.compile(r"(?<=[.!?])\s+")
# Tokens whose trailing period does not end a sentence ("et al.", "e.g.", "Fig.").
_ABBREVIATION_END_RE = re.compile(
    r"(?:\bal|\be\.g|\bi\.e|\bcf|\bvs|\bfigs?|\beqs?|\btabs?|\brefs?|\bapprox"
    r"|\bca|\bno|\bnos|\bdr|\bprof|\bresp|\bsuppl|\bvol|\bpp?|(?-i:\b[A-Z]))\.$",
    re.IGNORECASE,
)
_VOWEL_GROUP_RE = re.compile(r"[aeiouy]+")
# "be" + up to two adverbs + a candidate participle (checked in _is_participle).
_PASSIVE_RE = re.compile(
    r"\b(?:am|is|are|was|were|be|been|being)\s+"
    r"(?:(?:not|also|then|often|further|being|been|\w+ly)\s+){0,2}(\w+)\b",
    re.IGNORECASE,
)
_IRREGULAR_PARTICIPLES = {
    "beaten", "become", "begun", "bitten", "born", "bought", "brought", "built",
    "caught", "chosen", "done", "drawn", "driven", "eaten", "fallen", "felt",
    "forgotten", "found", "given", "gotten", "grown", "held", "hidden", "kept",
    "known", "laid", "led", "left", "lost", "made", "meant", "met", "paid",
    "put", "read", "run", "said", "seen", "sent", "set", "shown", "sold",
    "spoken", "spent", "taken", "taught", "thought", "told", "understood",
    "undertaken", "won", "written",
}
_NOT_PARTICIPLES = {"indeed", "need", "seed", "speed", "bed", "red", "shed", "feed", "embed"}

_MATTR_WINDOW = 50
TEXT_SOURCES = ("auto", "full_text", "abstract")

_HEDGE_WORDS = {
    "apparently", "approximately", "arguably", "could", "generally",
    "likely", "may", "might", "perhaps", "possibly", "potentially",
    "presumably", "seem", "seemed", "seems", "suggest", "suggested",
    "suggesting", "suggests", "typically", "unlikely",
}
_CERTAINTY_WORDS = {
    "certainly", "clearly", "confirm", "confirmed", "confirms",
    "conclusive", "definite", "definitely", "demonstrate", "demonstrated",
    "demonstrates", "establish", "established", "establishes", "evident",
    "indeed", "must", "show", "showed", "showing", "shown", "shows",
    "strongly", "undoubtedly",
}


class TextScalingCancelled(RuntimeError):
    """Raised when a text-scaling run is cancelled."""


def _field(article: Any, name: str, default: Any = "") -> Any:
    if isinstance(article, Mapping):
        return article.get(name, default)
    return getattr(article, name, default)


def _source_body(article: Any, source: str) -> str:
    """The article's full text or abstract (empty when it has none)."""
    if source == "full_text":
        value = _field(article, "full_text", "") or _field(article, "fulltext", "")
    else:
        value = _field(article, "abstract", "")
    return str(value or "").strip()


def _available_source(article: Any) -> str:
    """Full text when the article has it, otherwise the abstract."""
    return "full_text" if _source_body(article, "full_text") else "abstract"


def _article_text(article: Any, source: str = "auto") -> str:
    """Title plus the body of one text type ("auto" prefers full text)."""
    source = _available_source(article) if source == "auto" else source
    title = str(_field(article, "title", "") or "").strip()
    body = _source_body(article, source)
    if title and body:
        return f"{title}. {body}"
    return body or title


def _choose_text_source(articles: list[Any], requested: str) -> str:
    """Pick one text type for the whole corpus so scaling never mixes them.

    "auto" uses full text only when at least as many articles have full text
    as have abstracts; otherwise every document is scaled on its abstract.
    """
    if requested not in TEXT_SOURCES:
        raise ValueError(f"text_source must be one of {TEXT_SOURCES}, got {requested!r}")
    if requested != "auto":
        return requested
    n_full = sum(1 for a in articles if _source_body(a, "full_text"))
    n_abstract = sum(1 for a in articles if _source_body(a, "abstract"))
    return "full_text" if n_full and n_full >= n_abstract else "abstract"


def _split_sentences(text: str) -> list[str]:
    """Split on terminal punctuation, keeping abbreviations inside sentences."""
    sentences: list[str] = []
    start = 0
    for match in _SENTENCE_BREAK_RE.finditer(text):
        if _ABBREVIATION_END_RE.search(text[start:match.start()]):
            continue
        sentences.append(text[start:match.start()])
        start = match.end()
    sentences.append(text[start:])
    return [part.strip() for part in sentences if part.strip()]


def _is_participle(word: str) -> bool:
    word = word.lower()
    if word in _IRREGULAR_PARTICIPLES:
        return True
    return len(word) > 3 and word.endswith("ed") and word not in _NOT_PARTICIPLES


def _passive_count(text: str) -> int:
    return sum(1 for m in _PASSIVE_RE.finditer(text) if _is_participle(m.group(1)))


def _moving_average_ttr(words: list[str], window: int = _MATTR_WINDOW) -> float:
    """Moving-average type-token ratio (Covington & McFall 2010), length-robust."""
    if len(words) <= window:
        return len(set(words)) / len(words) if words else 0.0
    counts = Counter(words[:window])
    total = len(counts)
    for index in range(window, len(words)):
        dropped = words[index - window]
        counts[dropped] -= 1
        if not counts[dropped]:
            del counts[dropped]
        counts[words[index]] += 1
        total += len(counts)
    return total / (len(words) - window + 1)


def _syllable_count(word: str) -> int:
    normalized = re.sub(r"[^a-z]", "", word.lower())
    if not normalized:
        return 0
    count = len(_VOWEL_GROUP_RE.findall(normalized))
    if normalized.endswith("e") and not normalized.endswith(("le", "ye")) and count > 1:
        count -= 1
    return max(1, count)


def build_stylometric_profile(
    article_id: str, article: Any, text: str | None = None, text_source: str = "",
) -> dict:
    """Compute interpretable writing-style measures for one article.

    ``text_source`` records which text type was profiled; when ``text`` is
    omitted it is the article's full text if present, else its abstract.
    """
    if text is None:
        text_source = text_source or _available_source(article)
        text = _article_text(article, text_source)
    words = [word.lower() for word in _WORD_RE.findall(text)]
    word_count = len(words)
    sentence_count = max(1, len(_split_sentences(text)))
    syllables = sum(_syllable_count(word) for word in words)

    hedge_count = sum(word in _HEDGE_WORDS for word in words)
    certainty_count = sum(word in _CERTAINTY_WORDS for word in words)
    passive_count = _passive_count(text)
    per_100_words = 100.0 / word_count if word_count else 0.0
    avg_sentence_length = word_count / sentence_count
    syllables_per_word = syllables / word_count if word_count else 0.0
    fk_grade = 0.39 * avg_sentence_length + 11.8 * syllables_per_word - 15.59

    return {
        "article_id": str(article_id),
        "title": str(_field(article, "title", "") or article_id),
        "text_source": text_source,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "average_sentence_length": round(avg_sentence_length, 4),
        "hedge_count": hedge_count,
        "hedge_density": round(hedge_count * per_100_words, 4),
        "certainty_count": certainty_count,
        "certainty_density": round(certainty_count * per_100_words, 4),
        "flesch_kincaid_grade": round(max(0.0, fk_grade), 4),
        "type_token_ratio": round(len(set(words)) / word_count, 4) if word_count else 0.0,
        "moving_average_type_token_ratio": round(_moving_average_ttr(words), 4),
        "passive_voice_count": passive_count,
        "passive_voice_density": round(100.0 * passive_count / sentence_count, 4),
    }


def _check_cancel(cancel_cb: Callable[[], bool] | None) -> None:
    if cancel_cb and cancel_cb():
        raise TextScalingCancelled("Text scaling cancelled")


# ── Wordfish model ────────────────────────────────────────────────────

_EPSILON = 1e-12
_CHUNK_SIZE = 256
_DAMPING = 0.5
DEFAULT_MAX_ITERATIONS = 500
DEFAULT_TOLERANCE = 1e-6          # relative log-likelihood change
DEFAULT_THETA_TOLERANCE = 1e-3    # largest per-iteration change in a position


@dataclass(frozen=True)
class _WordfishParams:
    alpha: Any   # document fixed effects
    psi: Any     # word fixed effects
    theta: Any   # document positions
    beta: Any    # word discrimination


@dataclass(frozen=True)
class _Moments:
    predicted_rows: Any
    predicted_columns: Any
    theta_mu_beta: Any
    theta_information: Any
    beta_mu_theta: Any
    beta_information: Any


def _initial_parameters(counts, row_totals, column_totals) -> _WordfishParams:
    """SVD start for theta; marginal log-rates for alpha and psi."""
    import numpy as np
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfTransformer

    tfidf = TfidfTransformer(sublinear_tf=True).fit_transform(counts)
    theta = TruncatedSVD(n_components=1, random_state=0).fit_transform(tfidf).ravel()
    if float(np.std(theta)) < 1e-8:
        theta = np.linspace(-1.0, 1.0, counts.shape[0])
    theta = (theta - float(np.mean(theta))) / max(float(np.std(theta)), 1e-8)

    weights = np.maximum(column_totals, 1.0)
    beta = np.asarray(counts.T.dot(theta)).ravel() / weights
    beta = beta - float(np.average(beta, weights=weights))
    beta = np.clip(beta / max(float(np.std(beta)), 1e-8) * 0.1, -0.5, 0.5)
    return _WordfishParams(
        alpha=np.log(np.maximum(row_totals, _EPSILON)),
        psi=np.log(np.maximum(column_totals / float(row_totals.sum()), _EPSILON)),
        theta=theta,
        beta=beta,
    )


def _expected_moments(
    params: _WordfishParams, cancel_cb: Callable[[], bool] | None,
) -> _Moments:
    """Poisson means and their theta/beta moments, in feature chunks."""
    import numpy as np

    n_documents, n_features = len(params.theta), len(params.beta)
    moments = _Moments(*(np.zeros(size) for size in (
        n_documents, n_features, n_documents, n_documents, n_features, n_features,
    )))
    theta_squared = params.theta * params.theta
    for start in range(0, n_features, _CHUNK_SIZE):
        _check_cancel(cancel_cb)
        stop = min(start + _CHUNK_SIZE, n_features)
        beta_block = params.beta[start:stop]
        eta = (
            params.alpha[:, None]
            + params.psi[start:stop][None, :]
            + params.theta[:, None] * beta_block[None, :]
        )
        mu = np.exp(np.clip(eta, -30.0, 30.0))
        moments.predicted_rows[:] += np.sum(mu, axis=1)
        moments.predicted_columns[start:stop] = np.sum(mu, axis=0)
        moments.theta_mu_beta[:] += mu.dot(beta_block)
        moments.theta_information[:] += mu.dot(beta_block * beta_block)
        moments.beta_mu_theta[start:stop] = params.theta.dot(mu)
        moments.beta_information[start:stop] = theta_squared.dot(mu)
    return moments


def _log_likelihood(counts_coo, params: _WordfishParams, moments: _Moments) -> float:
    import numpy as np

    rows, cols = counts_coo.row, counts_coo.col
    eta = params.alpha[rows] + params.psi[cols] + params.theta[rows] * params.beta[cols]
    return float(np.dot(counts_coo.data, eta) - np.sum(moments.predicted_rows))


def _damped_log_ratio(observed, predicted):
    import numpy as np

    ratio = np.maximum(observed, _EPSILON) / np.maximum(predicted, _EPSILON)
    return _DAMPING * np.clip(np.log(ratio), -1.0, 1.0)


def _damped_newton(score, information):
    import numpy as np

    return _DAMPING * np.clip(score / np.maximum(information, _EPSILON), -0.5, 0.5)


def _newton_step(
    counts, params: _WordfishParams, moments: _Moments, row_totals, column_totals,
) -> _WordfishParams:
    """One damped Newton update of every parameter block."""
    import numpy as np

    y_beta = np.asarray(counts.dot(params.beta)).ravel()
    y_theta = np.asarray(counts.T.dot(params.theta)).ravel()
    beta = params.beta + _damped_newton(
        y_theta - moments.beta_mu_theta, moments.beta_information,
    )
    return _WordfishParams(
        alpha=params.alpha + _damped_log_ratio(row_totals, moments.predicted_rows),
        psi=params.psi + _damped_log_ratio(column_totals, moments.predicted_columns),
        theta=params.theta + _damped_newton(
            y_beta - moments.theta_mu_beta, moments.theta_information,
        ),
        beta=np.clip(beta, -5.0, 5.0),
    )


def _identify(params: _WordfishParams, initial_theta) -> _WordfishParams:
    """Enforce mean-0/sd-1 theta, mean-0 alpha and the start's sign, keeping eta."""
    import numpy as np

    theta_mean = float(np.mean(params.theta))
    theta_std = max(float(np.std(params.theta)), 1e-8)
    alpha_shift = float(np.mean(params.alpha))
    theta = (params.theta - theta_mean) / theta_std
    beta = params.beta * theta_std
    sign = -1.0 if float(np.dot(theta, initial_theta)) < 0.0 else 1.0
    return _WordfishParams(
        alpha=params.alpha - alpha_shift,
        psi=params.psi + theta_mean * params.beta + alpha_shift,
        theta=theta * sign,
        beta=beta * sign,
    )


def _relative_change(current: float, previous: float | None) -> float:
    if previous is None:
        return math.inf
    return abs((current - previous) / max(abs(previous), 1.0))


def _fit_wordfish(
    counts,
    *,
    progress_cb: Callable[[int, int], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    tolerance: float = DEFAULT_TOLERANCE,
    theta_tolerance: float = DEFAULT_THETA_TOLERANCE,
) -> dict:
    """Fit the one-dimensional Poisson Wordfish model with damped Newton steps.

    Convergence needs both a relative log-likelihood change below
    ``tolerance`` and a largest position change below ``theta_tolerance``.
    """
    import numpy as np

    counts = counts.astype(np.float64).tocsr()
    counts_coo = counts.tocoo()
    row_totals = np.asarray(counts.sum(axis=1)).ravel()
    column_totals = np.asarray(counts.sum(axis=0)).ravel()
    params = _initial_parameters(counts, row_totals, column_totals)
    initial_theta = params.theta

    converged = False
    max_delta = math.inf
    log_likelihood = -math.inf
    relative_change = math.inf
    previous_log_likelihood: float | None = None
    completed_iterations = 0

    for iteration in range(1, max_iterations + 1):
        _check_cancel(cancel_cb)
        moments = _expected_moments(params, cancel_cb)
        log_likelihood = _log_likelihood(counts_coo, params, moments)
        relative_change = _relative_change(log_likelihood, previous_log_likelihood)
        previous_log_likelihood = log_likelihood
        if iteration >= 3 and relative_change < tolerance and max_delta < theta_tolerance:
            converged = True
        else:
            updated = _identify(
                _newton_step(counts, params, moments, row_totals, column_totals),
                initial_theta,
            )
            max_delta = float(np.max(np.abs(updated.theta - params.theta)))
            params = updated
        completed_iterations = iteration
        if progress_cb:
            progress_cb(iteration, max_iterations)
        if converged:
            break

    return {
        "theta": params.theta,
        "beta": params.beta,
        "alpha": params.alpha,
        "psi": params.psi,
        "converged": converged,
        "iterations": completed_iterations,
        "max_delta": max_delta,
        "log_likelihood": log_likelihood,
        "relative_log_likelihood_change": relative_change,
    }


# ── Corpus-level scaling ──────────────────────────────────────────────

@dataclass(frozen=True)
class _EligibleCorpus:
    document_ids: list[str]
    texts: list[str]
    profiles: dict[str, dict]
    excluded: int


def _eligible_documents(
    articles: Mapping[str, Any],
    text_source: str,
    min_doc_words: int,
    progress_cb: Callable[[int, int], None] | None,
    cancel_cb: Callable[[], bool] | None,
) -> _EligibleCorpus:
    """Profile every article on ``text_source``; keep those long enough to scale."""
    document_ids: list[str] = []
    texts: list[str] = []
    profiles: dict[str, dict] = {}
    excluded = 0
    items = list((articles or {}).items())
    for index, (article_id, article) in enumerate(items, start=1):
        _check_cancel(cancel_cb)
        has_source = bool(_source_body(article, text_source))
        text = _article_text(article, text_source)
        profile = build_stylometric_profile(str(article_id), article, text, text_source)
        if not has_source or profile["word_count"] < max(1, int(min_doc_words)):
            excluded += 1
        else:
            document_ids.append(str(article_id))
            texts.append(text)
            profiles[str(article_id)] = profile
        if progress_cb and (index % 100 == 0 or index == len(items)):
            progress_cb(index, max(1, len(items)))
    return _EligibleCorpus(document_ids, texts, profiles, excluded)


def _document_term_matrix(corpus: _EligibleCorpus, max_features: int):
    """Count matrix plus vectorizer, dropping documents with no kept terms."""
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=max(2, int(max_features)),
        min_df=2 if len(corpus.document_ids) >= 3 else 1,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z'-]{1,}\b",
    )
    # CountVectorizer.fit_transform ships no type stubs; sklearn's own return-type
    # inference conflates it with an internal helper's tuple return, so pyright
    # infers nonsense (tuple[Unknown, set[Unknown]] | None) instead of the actual
    # scipy sparse matrix. Annotate explicitly rather than let that noise cascade.
    counts: Any = vectorizer.fit_transform(corpus.texts)
    kept = np.flatnonzero(np.asarray(counts.sum(axis=1)).ravel() > 0)
    if len(kept) < len(corpus.document_ids):
        kept_ids = [corpus.document_ids[int(index)] for index in kept]
        corpus = _EligibleCorpus(
            document_ids=kept_ids,
            texts=[corpus.texts[int(index)] for index in kept],
            profiles={key: corpus.profiles[key] for key in kept_ids},
            excluded=corpus.excluded + len(corpus.document_ids) - len(kept),
        )
        counts = counts[kept]
    if counts.shape[0] < 2 or counts.shape[1] < 2:
        raise ValueError(
            "The eligible articles do not contain enough shared vocabulary for text scaling."
        )
    return counts, vectorizer, corpus


def _scaling_outputs(
    fitted: dict, corpus: _EligibleCorpus, terms, counts_shape, text_source: str,
) -> tuple[dict, dict[str, dict], dict]:
    """Assemble the model, per-article profiles and corpus statistics."""
    import numpy as np

    theta, beta = fitted["theta"], fitted["beta"]
    summary = {k: v for k, v in fitted.items() if k not in ("theta", "beta", "alpha", "psi")}
    positions = {doc_id: float(theta[i]) for i, doc_id in enumerate(corpus.document_ids)}
    profiles = {
        doc_id: {**corpus.profiles[doc_id], "position": round(position, 6)}
        for doc_id, position in positions.items()
    }
    top_indices = np.argsort(np.abs(beta))[::-1][: min(50, len(beta))]
    model = {
        "algorithm": "Wordfish Poisson text scaling",
        "n_documents": counts_shape[0],
        "n_features": counts_shape[1],
        "text_source": text_source,
        "positions": positions,
        "discriminating_terms": [
            {"term": str(terms[i]), "beta": round(float(beta[i]), 6)} for i in top_indices
        ],
        **summary,
    }
    statistics = {
        "n_documents": counts_shape[0],
        "n_features": counts_shape[1],
        "text_source": text_source,
        "excluded_documents": corpus.excluded,
        "converged": bool(model["converged"]),
        "iterations": int(model["iterations"]),
        "max_delta": round(float(model["max_delta"]), 6),
        "relative_log_likelihood_change": round(
            float(model["relative_log_likelihood_change"]), 8
        ),
        "average_hedge_density": round(
            float(np.mean([p["hedge_density"] for p in profiles.values()])), 4
        ),
        "average_certainty_density": round(
            float(np.mean([p["certainty_density"] for p in profiles.values()])), 4
        ),
    }
    return model, profiles, statistics


def run_text_scaling(
    articles: Mapping[str, Any],
    *,
    min_doc_words: int = 50,
    max_features: int = 5000,
    progress_cb: Callable[[int, int], None] | None = None,
    status_cb: Callable[[str], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    tolerance: float = DEFAULT_TOLERANCE,
    text_source: str = "auto",
) -> tuple[dict, dict[str, dict], dict]:
    """Scale eligible articles and return a model, profiles, and corpus statistics.

    Every document is scaled on the same text type (``text_source``: "auto",
    "full_text" or "abstract"); articles without it are excluded, so the
    main dimension cannot separate full texts from abstracts.
    """
    try:
        import numpy  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Text scaling requires numpy and scikit-learn. Install the project "
            "requirements and try again."
        ) from exc

    source = _choose_text_source(list((articles or {}).values()), text_source)
    if status_cb:
        status_cb(f"Preparing article text ({source}) and stylometric profiles...")
    corpus = _eligible_documents(articles, source, min_doc_words, progress_cb, cancel_cb)
    if len(corpus.document_ids) < 2:
        raise ValueError(
            "Text scaling needs at least two articles with "
            f"{min_doc_words} or more words of {source}; found {len(corpus.document_ids)}."
        )

    if status_cb:
        status_cb(f"Building document-term matrix for {len(corpus.document_ids)} articles...")
    counts, vectorizer, corpus = _document_term_matrix(corpus, max_features)

    if status_cb:
        status_cb(
            f"Fitting Wordfish model ({counts.shape[0]} documents, "
            f"{counts.shape[1]} features)..."
        )
    fitted = _fit_wordfish(
        counts,
        progress_cb=progress_cb,
        cancel_cb=cancel_cb,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    model, profiles, statistics = _scaling_outputs(
        fitted, corpus, vectorizer.get_feature_names_out(), counts.shape, source,
    )
    if status_cb:
        convergence = "converged" if model["converged"] else "reached iteration limit"
        status_cb(
            f"Text scaling {convergence}: {counts.shape[0]} profiles, "
            f"{counts.shape[1]} features ({source})"
        )
    return model, profiles, statistics
