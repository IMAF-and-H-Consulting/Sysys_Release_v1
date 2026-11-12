"""Helpers for downloading and normalizing full-text sources."""

from __future__ import annotations

import asyncio
import html
import json
import logging
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse

from core.io import (
    atomic_write_bytes as _write_bytes_atomic,
)
from core.io import (
    atomic_write_json,
)
from core.io import (
    atomic_write_text as _write_text_atomic,
)

_WARNED_NO_PDF_EXTRACTOR = False
OA_HINT_FIELDS = (
    "pmc",
    "s2_open_access_pdf_url",
    "openalex_is_oa",
    "openalex_oa_status",
    "openalex_oa_url",
    "openalex_any_repository_has_fulltext",
    "openalex_best_oa_url",
    "openalex_best_oa_pdf_url",
    "openalex_best_oa_host_type",
    "openalex_primary_url",
    "openalex_primary_pdf_url",
    "openalex_primary_host_type",
    "openalex_has_pdf_content",
    "openalex_has_grobid_xml",
    "openalex_content_url",
)
_HOST_LIMITS = {
    "api.unpaywall.org": {"concurrency": 1, "min_interval": 0.34},
    "pmc.ncbi.nlm.nih.gov": {"concurrency": 1, "min_interval": 0.34},
    "www.ncbi.nlm.nih.gov": {"concurrency": 1, "min_interval": 0.34},
    "eutils.ncbi.nlm.nih.gov": {"concurrency": 1, "min_interval": 0.34},
}
_DEFAULT_HOST_LIMIT = {"concurrency": 4, "min_interval": 0.0}

# Public alias kept for callers that import write_json_atomic from this module.
def write_json_atomic(path: Path, data) -> None:
    atomic_write_json(path, data, ensure_ascii=True)


def load_download_cache(log_path: Path) -> dict[str, dict]:
    if not log_path.exists():
        return {}
    try:
        with open(log_path, encoding="utf-8") as f:
            rows = json.load(f)
    except Exception:
        return {}

    cache: dict[str, dict] = {}
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            key = download_cache_key(row)
            if key:
                cache[key] = row
    return cache


def download_cache_key(row: dict) -> str:
    return str(
        row.get("doi")
        or row.get("key")
        or row.get("pmc")
        or row.get("title", "")
    ).strip()


def reusable_download_record(row: dict) -> bool:
    text_path = Path(str(row.get("path") or ""))
    if text_path.suffix.lower() != ".txt" or not text_path.exists():
        return False
    source_path = str(row.get("source_path") or "").strip()
    if source_path and not Path(source_path).exists():
        archive_path = str(row.get("source_archive_path") or "").strip()
        if not archive_path or not Path(archive_path).exists():
            return False
    return True


def download_record_status(row: dict) -> str:
    status = str(row.get("status") or "").strip().lower()
    if status:
        return status
    if reusable_download_record(row):
        return "downloaded"
    return ""


def recover_download_artifact(output_dir: Path, safe_name: str) -> dict | None:
    text_path = output_dir / f"{safe_name}.txt"
    if not text_path.exists():
        return None

    source_path = output_dir / f"{safe_name}.pdf"
    source_format = "pdf"
    if not source_path.exists():
        source_path = output_dir / f"{safe_name}.html"
        source_format = "html"
    if not source_path.exists():
        source_path = Path("")
        source_format = "text"

    try:
        chars = len(text_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    row = {
        "path": str(text_path),
        "source_format": source_format,
        "source_url": "",
        "chars": chars,
    }
    if str(source_path):
        row["source_path"] = str(source_path)
    return row


def copy_oa_hint_fields(row: dict) -> dict:
    return {
        key: row.get(key)
        for key in OA_HINT_FIELDS
        if key in row and row.get(key) not in (None, "", False)
    }


class AsyncHostThrottle:
    """Per-host concurrency/interval limits plus an optional global cap.

    Per-host semaphores alone leave total concurrency unbounded across many
    hosts; ``max_total`` bounds it (and should not exceed the HTTP client's
    connection pool, or requests queue for the pool and hit pool timeouts).
    """

    def __init__(self, default_concurrency: int = 4, max_total: int | None = None):
        self._default_concurrency = max(1, int(default_concurrency))
        self._global = asyncio.Semaphore(max(1, int(max_total))) if max_total else None
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._next_allowed: dict[str, float] = {}

    def _config_for(self, host: str) -> dict[str, float | int]:
        config = dict(_DEFAULT_HOST_LIMIT)
        config["concurrency"] = self._default_concurrency
        config.update(_HOST_LIMITS.get(host, {}))
        return config

    async def _wait_for_slot(self, host: str, min_interval: float) -> None:
        if min_interval <= 0:
            return
        lock = self._locks.setdefault(host, asyncio.Lock())
        async with lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            wait_for = max(0.0, self._next_allowed.get(host, 0.0) - now)
            if wait_for:
                await asyncio.sleep(wait_for)
            self._next_allowed[host] = loop.time() + min_interval

    async def get(self, client, url: str, **kwargs):
        host = urlparse(url).netloc.lower()
        config = self._config_for(host)
        semaphore = self._semaphores.setdefault(
            host,
            asyncio.Semaphore(max(1, int(config["concurrency"]))),
        )
        async with semaphore:
            await self._wait_for_slot(host, float(config["min_interval"]))
            if self._global is None:
                return await client.get(url, **kwargs)
            async with self._global:
                return await client.get(url, **kwargs)


def _retry_delay_seconds(response, attempt: int) -> float:
    retry_after = (response.headers.get("retry-after") or "").strip()
    if retry_after.isdigit():
        return max(1.0, float(retry_after))
    return min(8.0, float(2 ** max(0, attempt - 1)))


async def fetch_response_async(
    client,
    throttle: AsyncHostThrottle,
    url: str,
    *,
    max_attempts: int = 3,
    **request_kwargs,
):
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = await throttle.get(client, url, **request_kwargs)
        except Exception as e:
            last_exc = e
            if attempt == max_attempts:
                raise
            await asyncio.sleep(min(8.0, float(2 ** max(0, attempt - 1))))
            continue

        if response.status_code in {429, 500, 502, 503, 504} and attempt < max_attempts:
            await asyncio.sleep(_retry_delay_seconds(response, attempt))
            continue
        return response

    if last_exc:
        raise last_exc
    return None


def clean_html_to_text(raw_html: str) -> str:
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(raw_html, "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "header", "footer"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
    except ImportError:
        text = re.sub(r"(?is)<script.*?>.*?</script>", " ", raw_html)
        text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
        text = re.sub(r"(?is)<[^>]+>", " ", text)
        text = html.unescape(text)
    return normalize_plaintext(text)


def normalize_plaintext(text: str) -> str:
    text = text.replace("\x00", " ").replace("\ufeff", " ").replace("\ufffd", " ")
    text = html.unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def _extract_attr(tag: str, name: str) -> str:
    match = re.search(
        rf"""\b{name}\s*=\s*(['"])(.*?)\1""",
        tag,
        re.IGNORECASE | re.DOTALL,
    )
    return html.unescape(match.group(2)).strip() if match else ""


def _looks_like_pdf_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.path.lower().endswith(".pdf")


def content_is_pdf(content: bytes) -> bool:
    """Return whether bytes contain a PDF header near the start of the file."""
    return b"%PDF-" in content[:1024]


def response_is_pdf(response) -> bool:
    """Identify PDFs by body signature, not server metadata or URL suffixes."""
    return content_is_pdf(response.content)


def find_pdf_url_in_html(raw_html: str, base_url: str) -> str:
    for match in re.finditer(r"<meta[^>]+>", raw_html, re.IGNORECASE):
        tag = match.group(0)
        if "citation_pdf_url" in tag.lower():
            value = _extract_attr(tag, "content")
            if value:
                return urljoin(base_url, value)

    href_patterns = [
        r"""href\s*=\s*(['"])([^'"]+\.pdf(?:\?[^'"]*)?)\1""",
        r"""data-readcube-pdf-url\s*=\s*(['"])([^'"]+\.pdf(?:\?[^'"]*)?)\1""",
    ]
    for pattern in href_patterns:
        match = re.search(pattern, raw_html, re.IGNORECASE)
        if match:
            return urljoin(base_url, html.unescape(match.group(2)))
    return ""


def _resolve_pdf_extractor():
    extractors = []

    try:
        import pdftotext  # type: ignore

        def _pdftotext_extract(path: Path) -> str:
            with open(path, "rb") as f:
                pdf = pdftotext.PDF(f)
            return "\n\n".join(pdf)

        extractors.append(("pdftotext", _pdftotext_extract))
    except ImportError:
        pass

    try:
        from pypdf import PdfReader  # type: ignore

        def _pypdf_extract(path: Path) -> str:
            reader = PdfReader(str(path))
            return "\n\n".join(page.extract_text() or "" for page in reader.pages)

        extractors.append(("pypdf", _pypdf_extract))
    except ImportError:
        pass

    try:
        from PyPDF2 import PdfReader as LegacyPdfReader  # type: ignore

        def _pypdf2_extract(path: Path) -> str:
            reader = LegacyPdfReader(str(path))
            return "\n\n".join(page.extract_text() or "" for page in reader.pages)

        extractors.append(("PyPDF2", _pypdf2_extract))
    except ImportError:
        pass

    return extractors


def extract_pdf_text(path: Path, logger: logging.Logger | None = None) -> tuple[str, str]:
    global _WARNED_NO_PDF_EXTRACTOR

    extractors = _resolve_pdf_extractor()
    if not extractors:
        if logger and not _WARNED_NO_PDF_EXTRACTOR:
            logger.warning(
                "PDF extraction unavailable. Install one of: pdftotext, pypdf, or PyPDF2."
            )
            _WARNED_NO_PDF_EXTRACTOR = True
        return "", ""

    for name, extractor in extractors:
        try:
            text = normalize_plaintext(extractor(path))
            if text:
                return text, name
        except Exception as e:
            if logger:
                logger.warning("PDF extraction failed with %s for %s: %s", name, path.name, e)

    return "", ""


def _save_html_result(
    raw_html: str,
    url: str,
    output_dir: Path,
    safe_name: str,
    min_text_chars: int,
) -> dict | None:
    clean_text = clean_html_to_text(raw_html)
    if len(clean_text) < min_text_chars:
        return None

    source_path = output_dir / f"{safe_name}.html"
    text_path = output_dir / f"{safe_name}.txt"
    _write_text_atomic(source_path, raw_html)
    _write_text_atomic(text_path, clean_text)
    return {
        "path": str(text_path),
        "source_path": str(source_path),
        "source_format": "html",
        "source_url": url,
        "chars": len(clean_text),
    }


def _save_pdf_result(
    pdf_bytes: bytes,
    url: str,
    output_dir: Path,
    safe_name: str,
    logger: logging.Logger,
    min_text_chars: int,
) -> dict | None:
    if not content_is_pdf(pdf_bytes):
        logger.info("Ignoring non-PDF response from %s", url)
        return None

    source_path = output_dir / f"{safe_name}.pdf"
    _write_bytes_atomic(source_path, pdf_bytes)
    clean_text, extractor = extract_pdf_text(source_path, logger=logger)
    if len(clean_text) < min_text_chars:
        return None

    text_path = output_dir / f"{safe_name}.txt"
    _write_text_atomic(text_path, clean_text)
    return {
        "path": str(text_path),
        "source_path": str(source_path),
        "source_format": "pdf",
        "source_url": url,
        "pdf_extractor": extractor,
        "chars": len(clean_text),
    }


def _fallback_html_url(pdf_url: str) -> str:
    if _looks_like_pdf_url(pdf_url):
        parsed = urlparse(pdf_url)
        return urlunparse(parsed._replace(path=parsed.path[:-4]))
    return ""


def download_preferred_fulltext(
    client,
    url: str,
    output_dir: Path,
    safe_name: str,
    logger: logging.Logger,
    min_text_chars: int = 1200,
) -> dict | None:
    """Fetch a full-text source, preferring provider PDF links when usable."""
    response = client.get(url)
    if response.status_code != 200:
        return None

    resolved_url = str(response.url)
    if response_is_pdf(response) or _looks_like_pdf_url(resolved_url):
        result = _save_pdf_result(
            response.content,
            resolved_url,
            output_dir,
            safe_name,
            logger,
            min_text_chars=min_text_chars,
        )
        if result:
            return result

        fallback_url = _fallback_html_url(resolved_url)
        if fallback_url and fallback_url != resolved_url:
            logger.info("PDF extraction unavailable or empty for %s; trying landing page", safe_name)
            html_response = client.get(fallback_url)
            if html_response.status_code == 200 and not response_is_pdf(html_response):
                result = _save_html_result(
                    html_response.text,
                    str(html_response.url),
                    output_dir,
                    safe_name,
                    min_text_chars=min_text_chars,
                )
                if result:
                    return result
        return None

    raw_html = response.text
    pdf_url = find_pdf_url_in_html(raw_html, resolved_url)
    if pdf_url:
        pdf_response = client.get(pdf_url)
        if pdf_response.status_code == 200 and response_is_pdf(pdf_response):
            result = _save_pdf_result(
                pdf_response.content,
                str(pdf_response.url),
                output_dir,
                safe_name,
                logger,
                min_text_chars=min_text_chars,
            )
            if result:
                logger.info("Preferred provider PDF for %s", safe_name)
                return result
            logger.warning("PDF download for %s did not yield usable text; falling back to HTML", safe_name)

    return _save_html_result(
        raw_html,
        resolved_url,
        output_dir,
        safe_name,
        min_text_chars=min_text_chars,
    )


async def download_preferred_fulltext_async(
    client,
    throttle: AsyncHostThrottle,
    url: str,
    output_dir: Path,
    safe_name: str,
    logger: logging.Logger,
    min_text_chars: int = 1200,
) -> dict | None:
    """Async variant of full-text fetch with host-aware throttling and retries."""
    response = await fetch_response_async(client, throttle, url)
    if response is None or response.status_code != 200:
        return None

    resolved_url = str(response.url)
    if response_is_pdf(response) or _looks_like_pdf_url(resolved_url):
        result = await asyncio.to_thread(
            _save_pdf_result,
            response.content,
            resolved_url,
            output_dir,
            safe_name,
            logger,
            min_text_chars=min_text_chars,
        )
        if result:
            return result

        fallback_url = _fallback_html_url(resolved_url)
        if fallback_url and fallback_url != resolved_url:
            logger.info("PDF extraction unavailable or empty for %s; trying landing page", safe_name)
            html_response = await fetch_response_async(client, throttle, fallback_url)
            if (
                html_response is not None
                and html_response.status_code == 200
                and not response_is_pdf(html_response)
            ):
                result = await asyncio.to_thread(
                    _save_html_result,
                    html_response.text,
                    str(html_response.url),
                    output_dir,
                    safe_name,
                    min_text_chars=min_text_chars,
                )
                if result:
                    return result
        return None

    raw_html = response.text
    pdf_url = find_pdf_url_in_html(raw_html, resolved_url)
    if pdf_url:
        pdf_response = await fetch_response_async(client, throttle, pdf_url)
        if (
            pdf_response is not None
            and pdf_response.status_code == 200
            and response_is_pdf(pdf_response)
        ):
            result = await asyncio.to_thread(
                _save_pdf_result,
                pdf_response.content,
                str(pdf_response.url),
                output_dir,
                safe_name,
                logger,
                min_text_chars=min_text_chars,
            )
            if result:
                logger.info("Preferred provider PDF for %s", safe_name)
                return result
            logger.warning("PDF download for %s did not yield usable text; falling back to HTML", safe_name)

    return await asyncio.to_thread(
        _save_html_result,
        raw_html,
        resolved_url,
        output_dir,
        safe_name,
        min_text_chars=min_text_chars,
    )
