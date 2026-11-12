"""Async Semantic Scholar Graph API client."""
import asyncio
import logging

import httpx

from acquire.http_retry import RetryPolicy, get_with_retry, parse_retry_after

logger = logging.getLogger(__name__)

S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = (
    "paperId,title,abstract,year,citationCount,externalIds,"
    "authors,fieldsOfStudy,publicationTypes,tldr,openAccessPdf"
)
S2_FALLBACK_FIELDS = (
    "paperId,title,abstract,year,citationCount,externalIds,"
    "authors,fieldsOfStudy,publicationTypes"
)


class SemanticScholarClient:
    """Async client for the Semantic Scholar Academic Graph API."""

    def __init__(
        self,
        api_key: str = "",
        rate_limit_delay: float = 1.1,
        max_retries: int = 5,
        initial_backoff: float = 5.0,
        max_backoff: float = 5.0,
    ):
        headers: dict[str, str] = {}
        if api_key:
            headers["x-api-key"] = api_key
        self._client = httpx.AsyncClient(
            base_url=S2_BASE,
            headers=headers,
            timeout=30.0,
        )
        self._delay = rate_limit_delay
        # The 5 s cap applies to Retry-After as well as to the back-off.
        self._policy = RetryPolicy(
            max_attempts=max_retries + 1,
            base_backoff=initial_backoff,
            max_backoff=max_backoff,
            max_retry_after=max_backoff,
        )

    @staticmethod
    def _parse_retry_after(value: str | None) -> float | None:
        """Parse a Retry-After header value (delta-seconds form) into seconds."""
        return parse_retry_after(value)

    async def _request_with_backoff(self, path: str, params: dict) -> httpx.Response:
        """GET through the shared retry policy (429/5xx and transport errors).

        ``Retry-After`` is honoured up to ``max_backoff``. After exhausting
        ``max_retries`` the final (still-failing) response is returned so the
        caller can decide how to handle it.
        """
        await asyncio.sleep(self._delay)
        return await get_with_retry(
            self._client, path, params=params, policy=self._policy, label="S2",
        )

    async def _get_with_field_fallback(
        self,
        path: str,
        params: dict,
        *,
        fields: str,
        fallback_fields: str | None = None,
    ) -> httpx.Response:
        """GET with automatic field-set fallback on 400 and shared retry policy."""
        resp = await self._request_with_backoff(path, {**params, "fields": fields})
        if resp.status_code == 400 and fallback_fields:
            logger.warning("S2: field set rejected, retrying with fallback fields")
            resp = await self._request_with_backoff(
                path, {**params, "fields": fallback_fields},
            )
        return resp

    async def search(self, query: str, max_results: int = 1000) -> list[dict]:
        """Search for papers. Returns raw S2 paper dicts."""
        if not query:
            return []

        papers: list[dict] = []
        offset = 0
        page_size = 100

        while offset < max_results:
            limit = min(page_size, max_results - offset)
            params = {"query": query, "limit": str(limit), "offset": str(offset)}

            logger.info("S2 search: offset=%d limit=%d query='%s'", offset, limit, query[:80])
            resp = await self._get_with_field_fallback(
                "/paper/search", params, fields=S2_FIELDS, fallback_fields=S2_FALLBACK_FIELDS,
            )

            resp.raise_for_status()
            data = resp.json()

            total_available = data.get("total", 0)
            batch = data.get("data", [])
            logger.info("S2: batch=%d total_available=%d", len(batch), total_available)

            if not batch:
                break

            papers.extend(batch)
            offset += len(batch)
            if offset >= total_available:
                break

        logger.info("S2 search complete: %d papers", len(papers))
        return papers[:max_results]

    async def get_paper(self, paper_id: str) -> dict | None:
        """Fetch a single paper by S2 ID, DOI, or other external ID."""
        try:
            resp = await self._get_with_field_fallback(
                f"/paper/{paper_id}", {}, fields=S2_FIELDS, fallback_fields=S2_FALLBACK_FIELDS,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.warning("S2 get_paper(%s) failed: %s", paper_id, exc)
            return None

    async def get_citations(self, paper_id: str, limit: int = 100) -> list[dict]:
        """Get papers that cite the given paper."""
        try:
            params = {"limit": str(limit)}
            resp = await self._get_with_field_fallback(
                f"/paper/{paper_id}/citations", params,
                fields=S2_FIELDS, fallback_fields=S2_FALLBACK_FIELDS,
            )
            resp.raise_for_status()
            return [c.get("citingPaper", {}) for c in resp.json().get("data", [])]
        except Exception as exc:
            logger.warning("S2 get_citations(%s) failed: %s", paper_id, exc)
            return []

    async def get_references(self, paper_id: str, limit: int = 100) -> list[dict]:
        """Get papers referenced by the given paper."""
        try:
            params = {"limit": str(limit)}
            resp = await self._get_with_field_fallback(
                f"/paper/{paper_id}/references", params,
                fields=S2_FIELDS, fallback_fields=S2_FALLBACK_FIELDS,
            )
            resp.raise_for_status()
            return [r.get("citedPaper", {}) for r in resp.json().get("data", [])]
        except Exception as exc:
            logger.warning("S2 get_references(%s) failed: %s", paper_id, exc)
            return []

    async def close(self):
        await self._client.aclose()
