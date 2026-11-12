"""Async OpenAlex API client."""
import asyncio
import logging
from dataclasses import replace

import httpx

from acquire.http_retry import RetryPolicy, get_with_retry, raise_for_status_redacted

logger = logging.getLogger(__name__)

OA_BASE = "https://api.openalex.org"
OPENALEX_SELECT_FIELDS = (
    "id,ids,doi,title,publication_year,cited_by_count,"
    "abstract_inverted_index,authorships,type,"
    "open_access,best_oa_location,primary_location,has_content,content_url"
)
OPENALEX_FALLBACK_SELECT_FIELDS = (
    "id,ids,doi,title,publication_year,cited_by_count,"
    "abstract_inverted_index,authorships,type,"
    "open_access,best_oa_location,primary_location"
)


def invert_abstract(inverted_index: dict | None) -> str | None:
    """Reconstruct abstract text from OpenAlex inverted-index format."""
    if not inverted_index:
        return None
    try:
        word_positions: list[tuple[int, str]] = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        word_positions.sort(key=lambda x: x[0])
        return " ".join(w for _, w in word_positions)
    except Exception:
        return None


_PER_PAGE = 100
# HTTP statuses OpenAlex returns when it rejects the requested select fields.
_SELECT_REJECTED_STATUSES = frozenset({400, 500})
_RETRY_POLICY = RetryPolicy()
# With the full select set, a 500 is first treated as "field set rejected", so
# it is not retried with the same fields.
_PRIMARY_SELECT_POLICY = replace(
    _RETRY_POLICY, retry_statuses=_RETRY_POLICY.retry_statuses - {500},
)


class OpenAlexClient:
    """Async client for the OpenAlex API."""

    def __init__(self, api_key: str = "", mailto: str = "", rate_limit_delay: float = 0.1):
        headers: dict[str, str] = {}
        contact = mailto or (api_key if api_key and "@" in api_key else "")
        if contact:
            # OpenAlex uses mailto for the polite pool
            headers["User-Agent"] = f"mailto:{contact}"
        # A genuine OpenAlex premium key (not an email) goes on the query string.
        self._premium_key = "" if "@" in api_key else api_key
        self._client = httpx.AsyncClient(
            base_url=OA_BASE,
            headers=headers,
            timeout=30.0,
        )
        self._delay = rate_limit_delay
        # Remembered across pages and queries once OpenAlex rejects the full set.
        self._select = OPENALEX_SELECT_FIELDS

    def _request_params(self, params: dict, select: str) -> dict:
        """A new params dict with the select fields and premium key added."""
        extra = {"api_key": self._premium_key} if self._premium_key else {}
        return {**params, "select": select, **extra}

    async def _send(self, path: str, params: dict, select: str) -> httpx.Response:
        policy = (_PRIMARY_SELECT_POLICY if select != OPENALEX_FALLBACK_SELECT_FIELDS
                  else _RETRY_POLICY)
        await asyncio.sleep(self._delay)
        return await get_with_retry(
            self._client, path, params=self._request_params(params, select),
            policy=policy, label="OpenAlex",
        )

    async def _get_with_select_fallback(
        self, path: str, params: dict
    ) -> httpx.Response:
        """GET with select-field fallback and the shared retry policy.

        Errors are raised with ``api_key``/``mailto`` redacted from the message.
        """
        select = params.get("select") or self._select
        resp = await self._send(path, params, select)
        if (resp.status_code in _SELECT_REJECTED_STATUSES
                and select != OPENALEX_FALLBACK_SELECT_FIELDS):
            logger.warning("OpenAlex: field set rejected (%d), retrying with fallback",
                           resp.status_code)
            self._select = OPENALEX_FALLBACK_SELECT_FIELDS
            resp = await self._send(path, params, OPENALEX_FALLBACK_SELECT_FIELDS)
        return raise_for_status_redacted(resp)

    async def search(
        self,
        search_text: str,
        extra_filter: str = "has_abstract:true",
        max_results: int = 1000,
    ) -> list[dict]:
        """Search OpenAlex for works matching *search_text*."""
        if not search_text:
            return []

        papers: list[dict] = []
        page = 1

        while len(papers) < max_results:
            params: dict[str, str] = {
                "search": search_text,
                "per_page": str(_PER_PAGE),
                "page": str(page),
            }
            if extra_filter:
                params["filter"] = extra_filter

            logger.info("OpenAlex search: page=%d query='%s'", page, search_text[:80])
            resp = await self._get_with_select_fallback("/works", params)
            data = resp.json()

            results = data.get("results", [])
            total_count = data.get("meta", {}).get("count", 0)
            logger.info("OpenAlex: page %d → %d results (total=%d)", page, len(results), total_count)

            if not results:
                break

            papers.extend(results)
            page += 1
            # Stop once every hit is in hand; a short page is the last page.
            if len(papers) >= total_count or len(results) < _PER_PAGE:
                break

        logger.info("OpenAlex search complete: %d papers", len(papers))
        return papers[:max_results]

    async def close(self):
        await self._client.aclose()
