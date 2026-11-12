"""Stage 7 text extraction from downloaded PDF and HTML full text.

Split out of ``stages.s7_fulltext``, which re-exports every name here.
"""
import logging
import re
from pathlib import Path

# Shares the Stage 7 logger so log records keep their original name.
logger = logging.getLogger("stages.s7_fulltext")

_MIN_BINARY_TEXT_CHARS = 100


def _content_looks_like_html(content: bytes) -> bool:
    prefix = content.lstrip()[:64].lower()
    return prefix.startswith((b"<!doctype html", b"<html", b"<?xml"))


def _text_from_unknown_content(content: bytes) -> str:
    """Text from a response that is neither PDF nor HTML, or ``""``.

    Only content that decodes as UTF-8 is treated as markup; arbitrary binary
    data would otherwise be decoded as Latin-1 and passed on as garbage text.
    """
    try:
        content.decode("utf-8")
    except UnicodeDecodeError:
        return ""
    text = _extract_text_from_html(content)
    return text if len(text) > _MIN_BINARY_TEXT_CHARS else ""


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------
def _extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from PDF. Try fitz (PyMuPDF) -> pypdf -> skip."""
    # Try PyMuPDF (fitz)
    try:
        import fitz  # type: ignore[import-untyped]
        doc = fitz.open(str(pdf_path))
        pages = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()
        text = "\n".join(pages).strip()
        if text:
            logger.debug("  PDF extracted with fitz: %d chars", len(text))
            return text
    except ImportError:
        logger.debug("  fitz (PyMuPDF) not available")
    except Exception as exc:
        logger.debug("  fitz extraction failed: %s", exc)

    # Try pypdf
    try:
        from pypdf import PdfReader  # type: ignore[import-untyped]
        reader = PdfReader(str(pdf_path))
        pages = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                pages.append(t)
        text = "\n".join(pages).strip()
        if text:
            logger.debug("  PDF extracted with pypdf: %d chars", len(text))
            return text
    except ImportError:
        logger.debug("  pypdf not available")
    except Exception as exc:
        logger.debug("  pypdf extraction failed: %s", exc)

    logger.warning("  Could not extract text from PDF: %s", pdf_path.name)
    return ""


# ---------------------------------------------------------------------------
# HTML text extraction
# ---------------------------------------------------------------------------
def _extract_text_from_html(content: bytes) -> str:
    """Extract text from HTML. Try BeautifulSoup -> regex fallback."""
    html_str = ""
    for enc in ("utf-8", "latin-1"):
        try:
            html_str = content.decode(enc)
            break
        except (UnicodeDecodeError, AttributeError):
            continue
    if not html_str and isinstance(content, str):
        html_str = content

    if not html_str:
        return ""

    # Try BeautifulSoup
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_str, "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "header", "footer"]):
            tag.decompose()
        text = soup.get_text(separator=" ").strip()
        if text and len(text) > 100:
            logger.debug("  HTML extracted with BeautifulSoup: %d chars", len(text))
            return text
    except ImportError:
        logger.debug("  BeautifulSoup not available")
    except Exception as exc:
        logger.debug("  BeautifulSoup extraction failed: %s", exc)

    # Fallback: regex strip tags
    text = re.sub(r"<script[^>]*>.*?</script>", "", html_str, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text:
        logger.debug("  HTML extracted with regex fallback: %d chars", len(text))
    return text
