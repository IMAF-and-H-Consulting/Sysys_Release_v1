"""Bridge between the GUI and the async search pipeline."""

import asyncio
import json
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ProviderChoice(Enum):
    ALL = "All Providers"
    ELICIT = "Elicit"
    S2 = "Semantic Scholar"
    OPENALEX = "OpenAlex"
    PUBMED = "PubMed"
    WOS = "Web of Science"
    SCOPUS = "Scopus"


_PROVIDER_ORDER = (
    ProviderChoice.PUBMED,
    ProviderChoice.S2,
    ProviderChoice.OPENALEX,
    ProviderChoice.ELICIT,
    ProviderChoice.SCOPUS,
    ProviderChoice.WOS,
)


def normalize_provider_choices(provider) -> set[ProviderChoice]:
    """Return concrete search engines, expanding ALL to every provider."""
    if provider is None:
        raw = {ProviderChoice.ALL}
    elif isinstance(provider, (list, tuple, set)):
        raw = set(provider)
    else:
        raw = {provider}

    if not raw or ProviderChoice.ALL in raw:
        return set(_PROVIDER_ORDER)

    selected = {choice for choice in raw if choice in _PROVIDER_ORDER}
    return selected or set(_PROVIDER_ORDER)


def selected_search_provider_count(provider) -> int:
    """Count concrete search engines selected for a Stage 1 search."""
    return len(normalize_provider_choices(provider))


def provider_result_limits(max_results: int, provider) -> dict[ProviderChoice, int]:
    """Split a total result cap across selected search engines.

    The returned budgets sum to ``max_results``. Remainders follow the UI order
    so common default searches give the first selected engines the extra slots.
    """
    selected = normalize_provider_choices(provider)
    total_limit = max(1, int(max_results or 1))
    base, remainder = divmod(total_limit, len(selected))

    budgets: dict[ProviderChoice, int] = {}
    for choice in _PROVIDER_ORDER:
        if choice not in selected:
            continue
        budgets[choice] = base + (1 if remainder > 0 else 0)
        remainder = max(0, remainder - 1)
    return budgets


BRIDGE_SHUTDOWN_TIMEOUT_SECONDS = 5.0


class AsyncBridge:
    """Runs an asyncio event loop on a background daemon thread.

    Allows the Qt main thread to submit coroutines without blocking.
    """

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="async-bridge",
        )
        self._thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            self._finish_pending_tasks()
            self._loop.close()

    def _finish_pending_tasks(self) -> None:
        """Cancel unfinished tasks and let them unwind before the loop closes.

        Like ``asyncio.run``: a cancelled search still runs its ``finally``
        blocks (``await client.close()``), so HTTP sessions are not leaked.
        """
        pending = asyncio.all_tasks(self._loop)
        for task in pending:
            if not task.cancelling():
                task.cancel()
        results = self._loop.run_until_complete(
            asyncio.gather(*pending, return_exceptions=True)
        ) if pending else []
        for failure in (r for r in results if isinstance(r, Exception)):
            logger.warning("Async bridge task failed while unwinding: %r", failure)
        self._loop.run_until_complete(self._loop.shutdown_asyncgens())

    def submit(self, coro):
        """Submit a coroutine; returns a concurrent.futures.Future."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def shutdown(self, timeout: float = BRIDGE_SHUTDOWN_TIMEOUT_SECONDS) -> None:
        """Stop the loop, unwind pending tasks and close the loop (idempotent)."""
        if not self._thread.is_alive():
            return
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except RuntimeError:
            pass  # the loop closed between the liveness check and this call
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logger.warning(
                "Async bridge still unwinding cancelled tasks after %.0f s", timeout,
            )


# ── Query builders ──────────────────────────────────────────
#
# Each provider issues exactly ONE query. An explicit Boolean query keeps its
# AND/OR structure and is wrapped in the provider's field code unless it already
# carries one; otherwise the term list is ORed into one combined query.
#
# The SAME Boolean query string is sent to PubMed, WoS and Scopus alike, so it
# must stay field-tag-free (plain terms/AND/OR/quoted phrases only) — a query
# hand-written with one provider's field tags (e.g. PubMed's "[Title/Abstract]")
# will not match another provider's field codes and gets wrapped whole inside
# that provider's own field template, producing a nonsense query for it (Stage
# 1 review round 1, item 2). ``_run_providers`` logs the literal query sent to
# each provider so a mismatch like this is visible while the search runs.

_PUBMED_FIELD = ("({})[Title/Abstract]", (
    "[TITLE/ABSTRACT]", "[TIAB]", "[TW]", "[TITLE]", "[MESH", "[ALL FIELDS]"))
_WOS_FIELD = ("TS=({})", ("TS=", "TI=", "AU=", "DO=", "AB=", "AK=", "ALL="))
_SCOPUS_FIELD = ("TITLE-ABS-KEY({})", (
    "TITLE-ABS-KEY(", "TITLE(", "ABS(", "KEY(", "AUTH(", "DOI(", "ALL("))


def _field_query(field: tuple[str, tuple[str, ...]], raw_boolean_query: str,
                 terms: list[str]) -> str:
    """One provider query, preferring the explicit Boolean query."""
    template, field_codes = field
    raw = (raw_boolean_query or "").strip()
    if raw:
        upper = raw.upper()
        return raw if any(code in upper for code in field_codes) else template.format(raw)
    return template.format(" OR ".join(f'"{t}"' if " " in t else t for t in terms))


def build_elicit_query(question: str, semantic_terms: list[str]) -> str:
    """Build an Elicit query, preferring the explicit natural-language question."""
    clean_question = " ".join((question or "").split())
    if clean_question:
        return clean_question
    return " ".join(t.strip() for t in semantic_terms if t.strip())


# ── Search-tab filters (year range, full text only) ────────

PUBMED_FREE_FULL_TEXT_FILTER = "free full text[sb]"
_OPEN_FULL_TEXT_FIELDS = (
    "pmc",
    "s2_open_access_pdf_url",
    "openalex_oa_url",
    "openalex_best_oa_url",
    "openalex_best_oa_pdf_url",
    "full_text",
    "full_text_path",
)
_OPEN_FULL_TEXT_FLAGS = (
    "openalex_is_oa",
    "pubmed_free_full_text",
    "has_full_text",
    "open_access",
)


def validate_year_range(min_year: int | None, max_year: int | None) -> None:
    """Reject an inverted Search-tab year range before any provider is queried."""
    if min_year is not None and max_year is not None and min_year > max_year:
        raise ValueError(f"Min Year ({min_year}) is later than Max Year ({max_year}).")


def _paper_year(paper: dict) -> int | None:
    try:
        return int(str(paper.get("year") or "").strip()[:4])
    except ValueError:
        return None


def _in_year_range(paper: dict, min_year: int | None, max_year: int | None) -> bool:
    """True when the paper's year is inside the range; unknown years are kept."""
    year = _paper_year(paper)
    if year is None:
        return True
    if min_year is not None and year < min_year:
        return False
    return max_year is None or year <= max_year


def has_open_full_text(paper: dict) -> bool:
    """True when a provider reported freely available full text for the paper."""
    if any(str(paper.get(field) or "").strip() for field in _OPEN_FULL_TEXT_FIELDS):
        return True
    return any(paper.get(flag) is True for flag in _OPEN_FULL_TEXT_FLAGS)


def split_search_filter_removals(
    papers: list[dict],
    min_year: int | None,
    max_year: int | None,
    fulltext_only: bool,
) -> tuple[list[dict], dict[str, int]]:
    """Return the kept papers and how many each Search-tab filter removed."""
    in_years = [p for p in papers if _in_year_range(p, min_year, max_year)]
    kept = [p for p in in_years if has_open_full_text(p)] if fulltext_only else in_years
    removals = {
        "year_range": len(papers) - len(in_years),
        "fulltext_only": len(in_years) - len(kept),
    }
    return kept, removals


def apply_search_filters(
    papers: list[dict],
    *,
    min_year: int | None = None,
    max_year: int | None = None,
    fulltext_only: bool = False,
) -> list[dict]:
    """Return the papers that satisfy the Search tab's year and access filters."""
    return split_search_filter_removals(papers, min_year, max_year, fulltext_only)[0]


def _pubmed_free_full_text_query(query: str) -> str:
    return f"({query}) AND {PUBMED_FREE_FULL_TEXT_FILTER}"


def _normalize_pm_free_full_text(normalize: Callable[[dict], dict]) -> Callable[[dict], dict]:
    """Tag PubMed records fetched with the free-full-text filter."""
    return lambda paper: {**normalize(paper), "pubmed_free_full_text": True}


# ── Provider isolation ─────────────────────────────────────


class SearchResult(list):
    """Deduplicated corpus (a plain list) plus per-provider failures.

    ``provider_errors`` maps a provider label to its error message for every
    attempted provider that failed during this search.
    """

    def __init__(self, papers=(), provider_errors: dict[str, str] | None = None):
        super().__init__(papers)
        self.provider_errors: dict[str, str] = dict(provider_errors or {})


def _truncation(hits: int | None, raw_count: int, retrieved: int, limit: int) -> bool | None:
    """True when the provider had more records than were kept; None if unknown."""
    if raw_count > retrieved or (hits is not None and hits > retrieved):
        return True
    if hits is not None or retrieved < limit:
        return False
    return None


def _provider_record(label: str, query: str, client, retrieved: int,
                     error: str | None, truncated: bool | None) -> dict:
    """One provider/query entry for the search manifest."""
    hits = getattr(client, "last_total_hits", None)
    return {
        "source": label,
        "query": query,
        "hits": hits if isinstance(hits, int) else None,
        "retrieved": retrieved,
        "error": error,
        "truncated": truncated,
    }


async def _search_one_provider(
    label: str, client, run_query: Callable, normalize: Callable[[dict], dict],
    limit: int, raw_path, save_raw: Callable,
    query: str = "",
) -> tuple[list[dict], dict]:
    """Run one provider search in isolation.

    Returns ``(papers, record)`` where *record* is the provider's search-manifest
    entry; on failure the papers are ``[]`` and ``record["error"]`` is set. The
    raw file is written only on success so a failed rerun never overwrites the
    audit trail of an earlier good search. ``_run_providers`` reports each
    provider's paper count once the gathered searches return.
    """
    from acquire.http_retry import describe_error, redacted_traceback

    try:
        raw_papers = await run_query(client)
        papers = [normalize(p) for p in raw_papers][:limit]
    except Exception as exc:
        # Redacted: provider URLs in exception text can carry api_key/email.
        error = describe_error(exc)
        logger.error("%s search failed: %s", label, error)
        logger.debug("%s search traceback:\n%s", label, redacted_traceback(exc))
        return [], _provider_record(label, query, client, 0, error, None)
    finally:
        await client.close()

    save_raw(raw_path, papers)
    hits = getattr(client, "last_total_hits", None)
    # A client-reported hard cap (e.g. Elicit's 100-result ceiling) always
    # means truncation, even when the caller's own limit was never reached —
    # Stage 1 review round 1, item 1.
    truncated = (
        True if getattr(client, "hit_hard_cap", False)
        else _truncation(hits if isinstance(hits, int) else None,
                         len(raw_papers), len(papers), limit)
    )
    return papers, _provider_record(label, query, client, len(papers), None, truncated)


def _check_provider_outcomes(outcomes: dict[str, str | None]) -> dict[str, str]:
    """Return the failed providers; raise if every attempted provider failed."""
    failed = {label: error for label, error in outcomes.items() if error}
    if outcomes and len(failed) == len(outcomes):
        details = "; ".join(f"{label}: {error}" for label, error in failed.items())
        raise RuntimeError(
            f"All {len(outcomes)} selected search providers failed ({details}). "
            "Refusing to save the corpus: this is a provider outage, not a "
            "zero-result search."
        )
    for label, error in failed.items():
        logger.error("Search provider %s failed; corpus is incomplete: %s", label, error)
    return failed


def _filter_new_papers(
    papers: list[dict], min_year: int | None, max_year: int | None,
    fulltext_only: bool, progress: Callable[[str], None],
) -> tuple[list[dict], dict[str, int]]:
    """Apply the Search tab filters to this run's papers and report the effect.

    Returns the kept papers and the per-filter removal counts.
    """
    kept, removals = split_search_filter_removals(papers, min_year, max_year, fulltext_only)
    if min_year is None and max_year is None and not fulltext_only:
        return kept, removals
    undated = sum(1 for p in kept if _paper_year(p) is None)
    years = f"{min_year or 'any'}-{max_year or 'any'}"
    message = (
        f"Search filters (years {years}, full text only={'yes' if fulltext_only else 'no'}): "
        f"kept {len(kept)} of {len(papers)} new papers"
    )
    if undated:
        message += f" ({undated} without a year kept)"
    logger.info(message)
    progress(message)
    return kept, removals


# ── Search manifest (PRISMA identification provenance) ─────


def _appendable_manifest(path, corpus_before: int) -> dict | None:
    """The saved manifest when it still describes the corpus being extended.

    A manifest whose last run does not end at *corpus_before* records was
    written for a different corpus (edited, replaced or loaded elsewhere), so
    its runs no longer add up to this corpus and are not extended.
    """
    from core.io import load_json

    manifest = load_json(path)
    if not isinstance(manifest, dict):
        return None
    runs = manifest.get("runs")
    if not isinstance(runs, list) or not runs or not isinstance(runs[-1], dict):
        return None
    if runs[-1].get("corpus_after") != corpus_before:
        logger.warning(
            "Search manifest %s ends at %s records but corpus.json has %d; "
            "starting a new manifest with the existing corpus as its baseline",
            path, runs[-1].get("corpus_after"), corpus_before,
        )
        return None
    return manifest


def _record_search_run(config, run) -> None:
    """Append *run* (a ``SearchRunRecord``) to the corpus's search manifest."""
    from core.io import atomic_write_json
    from core.models import SEARCH_MANIFEST_FILENAME, new_search_manifest

    path = config.corpus_dir / SEARCH_MANIFEST_FILENAME
    manifest = (
        _appendable_manifest(path, run.corpus_before)
        if run.corpus_before is not None else None
    )
    if manifest is None:
        manifest = new_search_manifest(run, baseline_records=run.corpus_before)
    else:
        manifest = {**manifest, "runs": [*manifest["runs"], run.to_dict()]}
    atomic_write_json(path, manifest)
    logger.info("Search manifest updated: %s (%d runs)", path, len(manifest["runs"]))


# ── Existing corpus (searches are additive) ────────────────


class CorpusLoadError(RuntimeError):
    """corpus.json exists but cannot be used, so a search must not replace it."""


def _read_corpus_file(path) -> object:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _corpus_shape_problem(papers: object) -> str | None:
    """Why *papers* is not a list of paper objects, or None when it is."""
    if not isinstance(papers, list):
        return f"expected a JSON list of papers, found {type(papers).__name__}"
    bad = next((p for p in papers if not isinstance(p, dict)), None)
    if bad is not None:
        return f"the list contains a {type(bad).__name__} instead of a paper object"
    return None


def _corpus_error(path, problem: str) -> str:
    return (
        f"Existing corpus {path} cannot be used ({problem}). Search aborted before "
        "querying any provider so the accumulated corpus is not overwritten; restore "
        "corpus.json from a project backup or move it aside to start a new corpus."
    )


async def _load_existing_corpus(path) -> list[dict]:
    """Return the saved corpus, or [] when there is none yet.

    Raises ``CorpusLoadError`` before any provider is queried when the file
    cannot be read or is not a list of paper objects: replacing it with this
    run's results would silently lose every earlier search.
    """
    if not path.exists():
        return []
    try:
        papers = await asyncio.to_thread(_read_corpus_file, path)
    except (OSError, ValueError) as exc:  # ValueError covers JSON and decode errors
        raise CorpusLoadError(_corpus_error(path, f"{type(exc).__name__}: {exc}")) from exc
    problem = _corpus_shape_problem(papers)
    if problem:
        raise CorpusLoadError(_corpus_error(path, problem))
    logger.info("Loaded existing corpus with %d papers from %s", len(papers), path)
    return papers


# ── Live provider table ────────────────────────────────────


@dataclass(frozen=True)
class _SearchQueries:
    elicit: str
    semantic: str
    boolean_raw: str
    boolean_terms: list[str]

    @property
    def has_boolean(self) -> bool:
        return bool(self.boolean_raw or self.boolean_terms)


def _single_boolean_query(boolean_terms: list[str] | None) -> str:
    """The one Boolean query; more than one is rejected instead of dropped."""
    queries = [str(t).strip() for t in (boolean_terms or []) if str(t).strip()]
    if len(queries) > 1:
        raise ValueError(
            f"Stage 1 takes one Boolean query, got {len(queries)}. Combine them "
            "into a single query (for example with OR) so none is silently ignored."
        )
    return queries[0] if queries else ""


def _search_queries(question: str, semantic_terms, boolean_terms) -> _SearchQueries:
    """One query per provider family (no per-term fan-out)."""
    terms = [str(t).strip() for t in (semantic_terms or []) if str(t).strip()]
    semantic = " ".join((question or "").split()) or " ".join(terms)
    return _SearchQueries(
        elicit=build_elicit_query(question, semantic_terms),
        semantic=semantic,
        boolean_raw=_single_boolean_query(boolean_terms),
        boolean_terms=terms,
    )


@dataclass(frozen=True)
class _LiveProvider:
    choice: ProviderChoice
    raw_name: str
    query: str
    skip_reason: str | None  # None: the provider can run
    make_client: Callable[[], Any]
    normalize: Callable[[dict], dict]


def _first_reason(*checks: tuple[bool, str]) -> str | None:
    return next((reason for failed, reason in checks if failed), None)


def _pubmed_query(queries: _SearchQueries, fulltext_only: bool, normalize):
    query = _field_query(_PUBMED_FIELD, queries.boolean_raw, queries.boolean_terms)
    if not fulltext_only:
        return query, normalize
    return _pubmed_free_full_text_query(query), _normalize_pm_free_full_text(normalize)


def _live_providers(config, queries: _SearchQueries, fulltext_only: bool) -> list[_LiveProvider]:
    """Every live-path provider in search order.

    Client classes are looked up when a client is made, so tests can stub them.
    """
    from acquire import (
        elicit,
        openalex,
        pubmed,
        scopus,
        semantic_scholar,
        web_of_science,
    )
    from acquire import normalizers as norm

    no_semantic = (not queries.semantic, "no semantic query provided")
    no_terms = (not queries.has_boolean, "no query terms provided")
    pm_query, pm_normalize = _pubmed_query(queries, fulltext_only, norm.normalize_pm_paper)
    boolean = (queries.boolean_raw, queries.boolean_terms)
    return [
        _LiveProvider(
            ProviderChoice.ELICIT, "gui_elicit_raw.json", queries.elicit,
            _first_reason((not config.elicit_api_key, "ELICIT_API_KEY not configured"),
                          (not queries.elicit, "no natural-language question provided")),
            lambda: elicit.ElicitClient(api_key=config.elicit_api_key),
            norm.normalize_elicit_paper),
        _LiveProvider(
            ProviderChoice.S2, "gui_s2_raw.json", queries.semantic, _first_reason(no_semantic),
            lambda: semantic_scholar.SemanticScholarClient(api_key=config.s2_api_key),
            norm.normalize_s2_paper),
        _LiveProvider(
            ProviderChoice.OPENALEX, "gui_openalex_raw.json", queries.semantic,
            _first_reason(no_semantic),
            lambda: openalex.OpenAlexClient(api_key=config.openalex_api_key),
            norm.normalize_oa_paper),
        _LiveProvider(
            ProviderChoice.PUBMED, "gui_pubmed_raw.json", pm_query, _first_reason(no_terms),
            lambda: pubmed.PubMedClient(api_key=config.pubmed_api_key,
                                        email=config.pubmed_email, tool=config.pubmed_tool),
            pm_normalize),
        _LiveProvider(
            ProviderChoice.WOS, "gui_wos_raw.json", _field_query(_WOS_FIELD, *boolean),
            _first_reason((not config.wos_api_key, "WOS_API_KEY not configured"), no_terms),
            lambda: web_of_science.WebOfScienceClient(api_key=config.wos_api_key),
            norm.normalize_wos_paper),
        _LiveProvider(
            ProviderChoice.SCOPUS, "gui_scopus_raw.json", _field_query(_SCOPUS_FIELD, *boolean),
            _first_reason((not config.scopus_api_key, "SCOPUS_API_KEY not configured"), no_terms),
            lambda: scopus.ScopusClient(api_key=config.scopus_api_key,
                                        insttoken=config.scopus_insttoken),
            norm.normalize_scopus_paper),
    ]


def _report_skip(label: str, reason: str, progress: Callable[[str], None]) -> None:
    message = f"Skipping {label} — {reason}"
    logger.info(message)
    progress(message)


def _plan_providers(
    providers: list[_LiveProvider],
    selected: set[ProviderChoice],
    max_results: int,
    progress: Callable[[str], None],
) -> list[tuple[_LiveProvider, int]]:
    """Pair each provider that will run with its share of the paper limit.

    Providers that cannot run (no key, no query) are reported and left out
    BEFORE the limit is split, so their share goes to the providers that run.
    Raises ``ValueError`` when no selected provider can run.
    """
    chosen = [p for p in providers if p.choice in selected]
    for provider in chosen:
        if provider.skip_reason:
            _report_skip(provider.choice.value, provider.skip_reason, progress)
    runnable = [p for p in chosen if not p.skip_reason]
    if not runnable:
        reasons = "; ".join(f"{p.choice.value}: {p.skip_reason}" for p in chosen)
        raise ValueError(f"No selected search provider can run ({reasons}).")
    limits = provider_result_limits(max_results, {p.choice for p in runnable})
    for provider in runnable:
        if limits[provider.choice] <= 0:
            _report_skip(provider.choice.value, "no share of paper limit", progress)
    return [(p, limits[p.choice]) for p in runnable if limits[p.choice] > 0]


def _search_call(query: str, limit: int) -> Callable:
    return lambda client: client.search(query, max_results=limit)


async def _run_providers(
    planned: list[tuple[_LiveProvider, int]],
    config,
    progress: Callable[[str], None],
) -> tuple[list[dict], list[dict], dict[str, str | None]]:
    """Search every planned provider concurrently; failures are isolated per provider.

    Each provider has its own httpx client, so there is no shared state to
    serialize on — searching them one at a time only added up every
    provider's latency instead of paying roughly the slowest one (Stage 1
    review round 1, item 4). Cancellation is checked once before dispatch;
    ``PipelineSearchWorker`` forcing the asyncio future (gui.workers) still
    makes mid-flight cancellation prompt on the live path.

    Returns the fetched papers, the manifest records and each provider's error
    (or None).
    """
    from acquire.orchestrator import save_raw

    # Cancellation was checked by ``run_gui_search`` immediately before dispatch
    # (the gathered searches are cancelled mid-flight by the worker's future).
    for provider, _limit in planned:
        label = provider.choice.value
        logger.info("%s query: %s...", label, provider.query[:80])
        # The literal query sent is shown so a cross-provider field-tag
        # mismatch (review round 1, item 2) is visible while the search runs,
        # not only in the search manifest afterwards.
        progress(f"Searching {label}: {provider.query[:80]}")

    results = await asyncio.gather(*(
        _search_one_provider(
            provider.choice.value, provider.make_client(),
            _search_call(provider.query, limit), provider.normalize, limit,
            config.raw_dir / provider.raw_name, save_raw, query=provider.query,
        )
        for provider, limit in planned
    ))

    papers: list[dict] = []
    records: list[dict] = []
    outcomes: dict[str, str | None] = {}
    for (provider, _limit), (found, record) in zip(planned, results, strict=True):
        label = provider.choice.value
        outcomes[label] = record["error"]
        records.append(record)
        if record["error"]:
            progress(f"WARNING: {label} search failed: {record['error']}")
        else:
            note = " (truncated — see manifest)" if record["truncated"] else ""
            progress(f"{label}: {record['retrieved']} papers{note}")
        papers.extend(found)
    return papers, records, outcomes


# ── Citation snowball, merge, minimum-results check ────────


async def _snowball_new_papers(
    config, papers: list[dict], progress: Callable[[str], None],
) -> tuple[list[dict], dict | None]:
    """Snowball around this run's papers when ``citation_snowball_enabled``.

    Expands around the top-cited Semantic Scholar papers as seeds (via
    ``acquire.orchestrator``'s shared snowball helpers). Returns the records,
    tagged ``citation_snowball``, and their manifest entry (None when the
    snowball did not run).
    """
    if not getattr(config, "citation_snowball_enabled", False):
        return [], None
    from acquire.orchestrator import (
        SNOWBALL_SOURCE,
        _manifest_provider_entry,
        _snowball_expand,
        save_raw,
    )

    records = await _snowball_expand(config, papers)
    if records is None:
        progress("Citation snowball skipped: no Semantic Scholar seed papers in this search")
        return [], None
    if records:
        save_raw(config.raw_dir / "gui_s2_snowball_raw.json", records)
    progress(f"Citation snowball: {len(records)} citing/cited records "
             "(citation_snowball_enabled)")
    return records, _manifest_provider_entry(SNOWBALL_SOURCE, len(records))


def _merge_into_corpus(existing: list[dict], new: list[dict]) -> tuple[list[dict], dict]:
    """Deduplicate existing + new, drop abstract-less records; return the counts."""
    from acquire.deduplicator import deduplicate_papers

    combined = existing + new
    deduplicated = deduplicate_papers(combined)
    corpus = [p for p in deduplicated if p.get("abstract")]
    logger.info("Merging: %d existing + %d new = %d before dedup; %d after dedup + "
                "abstract filter", len(existing), len(new), len(combined), len(corpus))
    return corpus, {
        "records_before_dedup": len(combined),
        "duplicates_removed": len(combined) - len(deduplicated),
        "no_abstract_removed": len(deduplicated) - len(corpus),
    }


def _warn_below_minimum(count: int, min_results: int, progress: Callable[[str], None]) -> None:
    if count >= min_results:
        return
    message = (
        f"WARNING: corpus has {count} papers, below the minimum of {min_results}. "
        "Consider broadening search terms or increasing the paper limit."
    )
    logger.warning(message)
    progress(message)


# ── Main search coroutine ──────────────────────────────────


async def run_gui_search(
    natural_language_question: str,
    semantic_terms: list[str],
    boolean_terms: list[str],
    provider: "ProviderChoice | list[ProviderChoice]",
    config,
    max_results: int = 500,
    min_results: int = 1,
    progress_callback: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
    fulltext_only: bool = False,
) -> "SearchResult":
    """Search the selected providers and merge the results into corpus.json.

    One query per provider family: *natural_language_question* (Elicit, else
    S2/OpenAlex), *semantic_terms* (S2/OpenAlex fallback, OR-ed Boolean
    fallback) and at most one Boolean query in *boolean_terms* (PubMed, WoS,
    Scopus; ``ValueError`` for more). Providers that cannot run are reported
    and the paper limit is split across the rest (``ValueError`` if none can).
    The existing corpus is loaded first (``CorpusLoadError`` before any
    provider is queried if it is unusable), then extended additively; with
    ``config.citation_snowball_enabled`` the run's papers are snowballed (see
    ``_snowball_new_papers``). A failed provider is reported and keeps its old raw
    file; if all fail, ``RuntimeError`` and nothing is saved. *min_year*/
    *max_year*/*fulltext_only* filter only this run's papers (undated kept).
    """
    from acquire.orchestrator import save_raw
    from core.models import SearchRunRecord

    validate_year_range(min_year, max_year)
    queries = _search_queries(natural_language_question, semantic_terms, boolean_terms)

    def _progress(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    def _raise_if_cancelled() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise asyncio.CancelledError("Search cancelled by user")

    config.ensure_dirs()
    _raise_if_cancelled()
    planned = _plan_providers(
        _live_providers(config, queries, fulltext_only),
        normalize_provider_choices(provider), max_results, _progress,
    )
    output_path = config.corpus_dir / "corpus.json"
    existing_papers = await _load_existing_corpus(output_path)
    total_limit = max(1, int(max_results or 1))
    logger.info("Paper result cap: total=%d, per-provider=%s", total_limit,
                {p.choice.value: limit for p, limit in planned})
    _progress(f"Paper limit: up to {total_limit} total across {len(planned)} providers")

    _raise_if_cancelled()
    fetched, provider_records, outcomes = await _run_providers(planned, config, _progress)
    provider_errors = _check_provider_outcomes(outcomes)
    if provider_errors:
        _progress(f"WARNING: {len(provider_errors)} of {len(outcomes)} search providers "
                  f"failed ({', '.join(provider_errors)}); results are incomplete")

    _raise_if_cancelled()
    snowball, snowball_record = await _snowball_new_papers(config, fetched, _progress)
    provider_records = provider_records + ([snowball_record] if snowball_record else [])
    fetched = [*fetched, *snowball]
    logger.info("New papers fetched this run: %d", len(fetched))
    new_papers, filter_removals = _filter_new_papers(
        fetched, min_year, max_year, fulltext_only, _progress
    )
    _raise_if_cancelled()
    _progress(f"Deduplicating {len(new_papers)} new papers...")
    corpus, merge_counts = _merge_into_corpus(existing_papers, new_papers)
    _warn_below_minimum(len(corpus), min_results, _progress)

    # Save results (additive — overwrites with the merged superset)
    _raise_if_cancelled()
    save_raw(output_path, corpus)
    _record_search_run(config, SearchRunRecord(
        entry_point="gui_search",
        providers=provider_records,
        corpus_before=len(existing_papers),
        records_retrieved=len(fetched),
        records_removed_by_filters=filter_removals,
        corpus_after=len(corpus),
        filters={"min_year": min_year, "max_year": max_year, "fulltext_only": fulltext_only},
        **merge_counts,
    ))

    return SearchResult(corpus, provider_errors)
