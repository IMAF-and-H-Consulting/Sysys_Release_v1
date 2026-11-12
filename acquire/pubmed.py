"""Async PubMed E-Utilities client."""
import asyncio
import logging
import re
from xml.etree import ElementTree

import httpx

from acquire.http_retry import RetryPolicy, get_with_retry, raise_for_status_redacted

logger = logging.getLogger(__name__)

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_SOURCE = "pubmed"
EFETCH_BATCH_SIZE = 200
_RETRY_POLICY = RetryPolicy(max_attempts=3)
_LEADING_YEAR_RE = re.compile(r"^\s*(\d{4})")


def _element_text(element) -> str:
    """Return all text inside *element*, including text in inline markup."""
    if element is None:
        return ""
    return "".join(element.itertext()).strip()


def _article_title(art) -> str:
    """ArticleTitle with inline markup kept, falling back to VernacularTitle."""
    return _element_text(art.find("ArticleTitle")) or _element_text(art.find("VernacularTitle"))


def _leading_year(text: str | None) -> int | None:
    """The leading four-digit year of *text* ("1998 Dec-1999 Jan" -> 1998)."""
    match = _LEADING_YEAR_RE.match(text or "")
    return int(match.group(1)) if match else None


def _publication_year(art) -> int | None:
    """PubDate Year, then PubDate MedlineDate, then ArticleDate Year; else None."""
    pub_date = art.find("Journal/JournalIssue/PubDate")
    if pub_date is not None:
        year = (_leading_year(pub_date.findtext("Year"))
                or _leading_year(pub_date.findtext("MedlineDate")))
        if year:
            return year
    for article_date in art.findall("ArticleDate"):
        year = _leading_year(article_date.findtext("Year"))
        if year:
            return year
    return None


def _esearch_error(payload: dict) -> str:
    """Return an error description for a failed esearch payload, or ''."""
    if payload.get("error"):
        return str(payload["error"])
    result = payload.get("esearchresult")
    if not isinstance(result, dict):
        return "response has no esearchresult"
    if result.get("ERROR"):
        return str(result["ERROR"])
    errors = {k: v for k, v in (result.get("errorlist") or {}).items() if v}
    if errors and int(result.get("count", 0) or 0) == 0:
        return f"query errors with zero hits: {errors}"
    if errors:
        logger.warning("PubMed esearch reported query issues: %s", errors)
    return ""


def parse_pubmed_xml(xml_text: str) -> list[dict]:
    """Parse PubMed efetch XML into paper dicts."""
    papers = []
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError as exc:
        logger.error("PubMed XML parse error: %s", exc)
        return []

    for article in root.iter("PubmedArticle"):
        try:
            medline = article.find("MedlineCitation")
            if medline is None:
                continue

            pmid_el = medline.find("PMID")
            pmid = pmid_el.text if pmid_el is not None else ""

            art = medline.find("Article")
            if art is None:
                continue

            title = _article_title(art)

            abstract_el = art.find("Abstract")
            abstract = ""
            if abstract_el is not None:
                parts = []
                for at in abstract_el.findall("AbstractText"):
                    label = at.get("Label", "")
                    text = "".join(at.itertext())
                    if label:
                        parts.append(f"{label}: {text}")
                    else:
                        parts.append(text)
                abstract = " ".join(parts)

            authors = []
            author_list = art.find("AuthorList")
            if author_list is not None:
                for au in author_list.findall("Author"):
                    last = au.findtext("LastName", "")
                    fore = au.findtext("ForeName", "")
                    if last:
                        authors.append(f"{last}, {fore}".strip(", "))

            year = _publication_year(art)

            doi = ""
            article_ids = article.find("PubmedData")
            if article_ids is not None:
                for aid in article_ids.iter("ArticleId"):
                    if aid.get("IdType") == "doi" and aid.text:
                        doi = aid.text
                        break

            papers.append({
                "title": title or "",
                "abstract": abstract or "",
                "authors": authors,
                "year": year,
                "doi": doi,
                "pmid": pmid,
                "source": PUBMED_SOURCE,
            })
        except Exception as exc:
            logger.warning("PubMed: error parsing article: %s", exc)
            continue

    return papers


class PubMedClient:
    """Async client for PubMed E-Utilities (esearch + efetch)."""

    def __init__(self, api_key: str = "", email: str = "", tool: str = "sysys"):
        self._params: dict[str, str] = {}
        if api_key:
            self._params["api_key"] = api_key
        if email:
            self._params["email"] = email
            self._params["tool"] = tool
        self._client = httpx.AsyncClient(timeout=30.0)
        self._delay = 0.1 if api_key else 0.34
        self.last_total_hits: int | None = None

    async def _request_with_retry(self, url: str, params: dict) -> httpx.Response:
        """GET through the shared retry policy; errors never carry api_key/email."""
        resp = await get_with_retry(
            self._client, url, params=params, policy=_RETRY_POLICY, label="PubMed",
        )
        return raise_for_status_redacted(resp)

    async def _esearch(self, query: str, retmax: int = 10000) -> dict:
        """Run esearch and return the esearchresult dict.

        Results are relevance-ranked (E-utilities defaults to newest-first).
        Raises ``RuntimeError`` when E-utilities reports a query error, so a
        malformed query never reads as a zero-hit search.
        """
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": str(retmax),
            "sort": "relevance",
            "usehistory": "y",
            "retmode": "json",
            **self._params,
        }
        logger.info("PubMed esearch: query='%s'", query[:120])
        resp = await self._request_with_retry(ESEARCH_URL, params)
        payload = resp.json()
        error = _esearch_error(payload)
        if error:
            raise RuntimeError(f"PubMed esearch failed for query '{query[:120]}': {error}")
        return payload["esearchresult"]

    async def _efetch_batch(
        self, webenv: str, query_key: str, retstart: int, retmax: int = 200
    ) -> list[dict]:
        """Fetch one batch of PubMed articles via efetch XML."""
        params = {
            "db": "pubmed",
            "retmode": "xml",
            "rettype": "abstract",
            "WebEnv": webenv,
            "query_key": query_key,
            "retstart": str(retstart),
            "retmax": str(retmax),
            **self._params,
        }
        await asyncio.sleep(self._delay)
        resp = await self._request_with_retry(EFETCH_URL, params)
        return parse_pubmed_xml(resp.text)

    async def search(self, query: str, max_results: int = 5000) -> list[dict]:
        """Search PubMed and return parsed paper dicts."""
        if not query:
            return []

        esearch = await self._esearch(query, retmax=max_results)
        count = int(esearch.get("count", 0))
        self.last_total_hits = count
        webenv = esearch.get("webenv", "")
        query_key = esearch.get("querykey", "1")
        id_list = esearch.get("idlist", [])

        logger.info("PubMed esearch: %d hits, %d IDs, WebEnv=%s",
                     count, len(id_list), (webenv or "N/A")[:20])

        if not id_list and not webenv:
            return []

        papers: list[dict] = []
        total = min(max_results, count)
        retstart = 0

        while retstart < total:
            retmax = min(EFETCH_BATCH_SIZE, total - retstart)
            batch = await self._efetch_batch(webenv, query_key, retstart, retmax)
            # Advance by what was requested, not what parsed: book articles and
            # parse failures must not make the next batch overlap this one.
            retstart += retmax
            papers.extend(batch)
            logger.info("PubMed: fetched %d/%d", len(papers), total)
        papers = papers[:max_results]

        if count > len(papers):
            logger.warning(
                "PubMed: retrieved %d of %d hits (capped by max_results=%d)",
                len(papers), count, max_results,
            )
        logger.info("PubMed search complete: %d papers", len(papers))
        return papers

    async def close(self):
        await self._client.aclose()
