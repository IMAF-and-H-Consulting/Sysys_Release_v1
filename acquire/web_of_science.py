"""Async Web of Science Starter API client."""
import asyncio
import logging

import httpx

from acquire.http_retry import RetryPolicy, get_with_retry, raise_for_status_redacted

logger = logging.getLogger(__name__)

WOS_BASE = "https://api.clarivate.com/apis/wos-starter/v1"
_RETRY_POLICY = RetryPolicy()


class WebOfScienceClient:
    """Async client for the Web of Science Starter API."""

    def __init__(self, api_key: str = "", rate_limit_delay: float = 0.35):
        headers: dict[str, str] = {}
        if api_key:
            headers["X-ApiKey"] = api_key
        self._client = httpx.AsyncClient(
            base_url=WOS_BASE,
            headers=headers,
            timeout=30.0,
        )
        self._delay = rate_limit_delay
        self.last_total_hits: int | None = None

    async def search(self, query: str, max_results: int = 200) -> list[dict]:
        """Search WoS and return raw hit dicts."""
        if not query:
            return []

        papers: list[dict] = []
        total = 0
        page = 1
        per_page = min(50, max_results)

        while len(papers) < max_results:
            params = {
                "q": query,
                "limit": str(per_page),
                "page": str(page),
            }
            logger.info("WoS search: page=%d query='%s'", page, query[:80])
            await asyncio.sleep(self._delay)

            # Transient failures are retried per the shared policy
            # (acquire.http_retry); a page that still fails after retries must
            # not look like the end of the result set.
            resp = await get_with_retry(
                self._client, "/documents", params=params,
                policy=_RETRY_POLICY, label="WoS",
            )
            data = raise_for_status_redacted(resp).json()

            hits = data.get("hits", [])
            total = int(data.get("metadata", {}).get("total", 0) or 0)

            if not hits:
                break

            papers.extend(hits)
            page += 1
            if len(papers) >= total or len(papers) >= max_results:
                break

        self.last_total_hits = total
        if total > len(papers[:max_results]):
            logger.warning(
                "WoS: retrieved %d of %d hits (capped by max_results=%d)",
                len(papers[:max_results]), total, max_results,
            )
        logger.info("WoS search complete: %d papers", len(papers))
        return papers[:max_results]

    async def close(self):
        await self._client.aclose()
