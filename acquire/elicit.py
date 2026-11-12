"""Async Elicit API client."""
import logging

import httpx

from acquire.http_retry import RetryPolicy, post_with_retry, raise_for_status_redacted

logger = logging.getLogger(__name__)

ELICIT_BASE = "https://elicit.com/api/v2"
ELICIT_MAX_RESULTS = 100
_RETRY_POLICY = RetryPolicy()


class ElicitClient:
    """Thin async wrapper around the Elicit search API.

    HTTP and transport errors propagate so a failed search is never mistaken
    for a zero-hit search.
    """

    def __init__(self, api_key: str = ""):
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=ELICIT_BASE,
            headers=headers,
            timeout=60.0,
        )
        # Set when a search asked for more than ELICIT_MAX_RESULTS: the API
        # has no total-hits field, so this is the only truncation signal
        # available (Stage 1 review round 1, item 1).
        self.hit_hard_cap: bool = False

    async def search(self, query: str, max_results: int = 100) -> list[dict]:
        """Search Elicit and return raw paper dicts."""
        if not query:
            return []

        limit = min(max_results, ELICIT_MAX_RESULTS)
        self.hit_hard_cap = max_results > ELICIT_MAX_RESULTS
        if self.hit_hard_cap:
            logger.warning(
                "Elicit: requested %d results but the API returns at most %d",
                max_results, ELICIT_MAX_RESULTS,
            )
        logger.info("Elicit search: query='%s' max_results=%d", query[:80], limit)

        resp = await post_with_retry(
            self._client, "/search/papers",
            json={"query": query, "maxResults": limit},
            policy=_RETRY_POLICY, label="Elicit",
        )
        data = raise_for_status_redacted(resp).json()
        papers = data.get("papers", data.get("results", []))
        self.hit_hard_cap = max_results > ELICIT_MAX_RESULTS and len(papers) >= ELICIT_MAX_RESULTS
        return papers[:max_results]

    async def close(self):
        await self._client.aclose()
