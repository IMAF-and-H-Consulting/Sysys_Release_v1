"""Async Scopus API client."""
import asyncio
import logging

import httpx

from acquire.http_retry import RetryPolicy, get_with_retry, raise_for_status_redacted

logger = logging.getLogger(__name__)

SCOPUS_BASE = "https://api.elsevier.com/content"
# COMPLETE view carries ``dc:description`` (the abstract); STANDARD does not.
SCOPUS_COMPLETE_VIEW = "COMPLETE"
# Status codes Scopus returns when the key is not entitled to the COMPLETE view.
_VIEW_NOT_ENTITLED_STATUSES = frozenset({401, 403})
_RETRY_POLICY = RetryPolicy()


class ScopusClient:
    """Async client for the Scopus Search API.

    HTTP and transport errors propagate to the caller so a failed search is
    never mistaken for a zero-hit search. The client asks for the COMPLETE view
    (which includes abstracts) and falls back to the default STANDARD view once
    if the key is not entitled to it.
    """

    def __init__(self, api_key: str = "", insttoken: str = "", rate_limit_delay: float = 0.35):
        headers: dict[str, str] = {"Accept": "application/json"}
        if api_key:
            headers["X-ELS-APIKey"] = api_key
        if insttoken:
            headers["X-ELS-Insttoken"] = insttoken
        self._client = httpx.AsyncClient(
            base_url=SCOPUS_BASE,
            headers=headers,
            timeout=30.0,
        )
        self._delay = rate_limit_delay
        self._use_complete_view = True
        self.last_total_hits: int | None = None

    async def _get_page(self, params: dict[str, str]) -> dict:
        """Fetch one result page, downgrading to STANDARD view if not entitled.

        Transient 429/5xx responses and transport errors are retried per the
        shared policy (``acquire.http_retry``); the COMPLETE/STANDARD view
        downgrade only triggers on the non-retryable 401/403 "not entitled"
        statuses.
        """
        if self._use_complete_view:
            resp = await get_with_retry(
                self._client, "/search/scopus",
                params={**params, "view": SCOPUS_COMPLETE_VIEW},
                policy=_RETRY_POLICY, label="Scopus",
            )
            if resp.status_code not in _VIEW_NOT_ENTITLED_STATUSES:
                return raise_for_status_redacted(resp).json()
            logger.warning(
                "Scopus COMPLETE view refused (HTTP %d); retrying with the STANDARD "
                "view, which has no abstracts",
                resp.status_code,
            )
            self._use_complete_view = False
        resp = await get_with_retry(
            self._client, "/search/scopus", params=params,
            policy=_RETRY_POLICY, label="Scopus",
        )
        return raise_for_status_redacted(resp).json()

    async def search(self, query: str, max_results: int = 200) -> list[dict]:
        """Search Scopus and return raw result entries.

        Raises ``httpx.HTTPError`` on any failed page, including after earlier
        pages succeeded, rather than returning a silently partial list.
        """
        if not query:
            return []

        papers: list[dict] = []
        start = 0
        page_size = min(25, max_results)
        total = 0

        while start < max_results:
            params = {
                "query": query,
                "count": str(page_size),
                "start": str(start),
            }
            logger.info("Scopus search: start=%d query='%s'", start, query[:80])
            await asyncio.sleep(self._delay)

            data = await self._get_page(params)
            results = data.get("search-results", {})
            entries = results.get("entry", [])
            total = int(results.get("opensearch:totalResults", 0))

            if not entries:
                break

            papers.extend(entries)
            start += len(entries)
            if start >= total or start >= max_results:
                break

        self.last_total_hits = total
        if total > len(papers[:max_results]):
            logger.warning(
                "Scopus: retrieved %d of %d hits (capped by max_results=%d)",
                len(papers[:max_results]), total, max_results,
            )
        logger.info("Scopus search complete: %d papers", len(papers))
        return papers[:max_results]

    async def close(self):
        await self._client.aclose()
