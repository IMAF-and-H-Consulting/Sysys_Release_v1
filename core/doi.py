"""Single source of truth for DOI handling across every pipeline stage.

Two forms exist, and both come from :func:`strip_doi_prefix`:

* **storage form** (:func:`strip_doi_prefix`) — resolver/``doi:`` prefixes
  removed, original case kept. Use when writing a DOI into a record, because
  Stage 7 derives fulltext cache filenames from the stored value.
* **comparison form** (:func:`normalize_doi`) — storage form lowercased.
  Use for any key, dedup, or equality check. DOIs are case-insensitive.
"""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import unquote

DOI_RESOLVER = "https://doi.org/"

# Stripped case-insensitively, repeatedly, in any combination
# ("https://doi.org/doi:10.…", "doi: https://dx.doi.org/10.…").
_RESOLVER_PREFIXES = tuple(
    scheme + host
    for scheme in ("https://", "http://", "")
    for host in ("www.doi.org/", "dx.doi.org/", "doi.org/")
)
_DOI_LABEL = re.compile(r"^doi\s*[:\s]\s*", re.IGNORECASE)  # "doi:", "DOI: ", "DOI 10."
_PERCENT_ESCAPE = re.compile(r"%[0-9A-Fa-f]{2}")
# Sentence punctuation that exports leave after a DOI; never part of a real one.
_TRAILING_PUNCTUATION = ".,;"


def _strip_one_prefix(doi: str) -> tuple[str, bool]:
    """Remove one resolver URL prefix or ``doi`` label; flag a resolver URL."""
    lowered = doi.lower()
    for prefix in _RESOLVER_PREFIXES:
        if lowered.startswith(prefix):
            return doi[len(prefix):].lstrip(), True
    match = _DOI_LABEL.match(doi)
    if match:
        return doi[match.end():], False
    return doi, False


def strip_doi_prefix(value: Any) -> str:
    """Return the DOI without resolver URL or ``doi:`` prefix, case preserved.

    Also URL-decodes ``%2F``-style escapes, drops a resolver URL's query or
    fragment, and trims trailing sentence punctuation.
    """
    doi = str(value or "").strip()
    was_url = False
    while True:
        stripped, from_resolver = _strip_one_prefix(doi)
        if stripped == doi:
            break
        doi, was_url = stripped, was_url or from_resolver
    if was_url:
        doi = re.split(r"[?#]", doi, maxsplit=1)[0]
    if _PERCENT_ESCAPE.search(doi):
        doi = unquote(doi)
    return doi.strip().rstrip(_TRAILING_PUNCTUATION).strip()


def normalize_doi(value: Any) -> str:
    """Return the comparison form of a DOI, or ``""`` when there is none."""
    return strip_doi_prefix(value).lower()


def doi_url(value: Any) -> str:
    """Return a resolver link for any DOI spelling, or ``""`` when there is none."""
    doi = strip_doi_prefix(value)
    return f"{DOI_RESOLVER}{doi}" if doi else ""
