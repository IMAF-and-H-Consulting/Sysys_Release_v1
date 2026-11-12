"""Provider record normalizers — one public function per search provider.

Each ``normalize_*_paper`` maps one raw provider record to the common corpus
schema and returns a new dict. ``gui.search_runner.run_gui_search`` (the live
Stage 1 path) uses these directly.
"""

from __future__ import annotations

from acquire.openalex import invert_abstract
from acquire.pubmed import PUBMED_SOURCE
from core.doi import strip_doi_prefix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalize_external_id(value: str | None, prefixes: tuple[str, ...] = ()) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    for prefix in prefixes:
        text = text.removeprefix(prefix)
    # Canonical PubMed/PMC URLs end in "/": keep it out of the identifier.
    text = text.rstrip("/")
    if text.startswith("https://") or text.startswith("http://"):
        text = text.rstrip("/").split("/")[-1]
    return text or None


def _normalize_pmcid(value: str | None) -> str | None:
    pmcid = _normalize_external_id(
        value,
        prefixes=(
            "https://www.ncbi.nlm.nih.gov/pmc/articles/",
            "http://www.ncbi.nlm.nih.gov/pmc/articles/",
            "https://pmc.ncbi.nlm.nih.gov/articles/",
            "http://pmc.ncbi.nlm.nih.gov/articles/",
        ),
    )
    if not pmcid:
        return None
    if pmcid.isdigit():
        pmcid = f"PMC{pmcid}"
    return pmcid.upper()


def _normalize_pmid(value: str | None) -> str | None:
    return _normalize_external_id(
        value,
        prefixes=(
            "https://pubmed.ncbi.nlm.nih.gov/",
            "http://pubmed.ncbi.nlm.nih.gov/",
        ),
    )


def _location_url(location: dict | None) -> str:
    if not isinstance(location, dict):
        return ""
    return (
        location.get("pdf_url")
        or location.get("landing_page_url")
        or ""
    )


def _location_pdf_url(location: dict | None) -> str:
    if not isinstance(location, dict):
        return ""
    return location.get("pdf_url") or ""


def _location_host_type(location: dict | None) -> str:
    if not isinstance(location, dict):
        return ""
    return ((location.get("source") or {}).get("type") or "").strip()


def _first_nonempty(*values):
    """Return the first non-empty / non-None value (rejects [], {}, '')."""
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


def _parse_year(value) -> int | None:
    """Coerce *value* to an integer year, handling date strings like '2024-01-15'."""
    if value in (None, ""):
        return None
    text = str(value).strip()
    if len(text) >= 4 and text[:4].isdigit():
        return int(text[:4])
    return int(text) if text.isdigit() else None


def _as_int(value) -> int | None:
    """Coerce a citation-count-like value to int, or None if it is not numeric."""
    if value is None or value == "":
        return None
    text = str(value).strip()
    return int(text) if text.isdigit() else None


def _iter_identifier_candidates(value) -> list[dict]:
    """Coerce varied identifier structures into a list of {type, value} dicts."""
    if isinstance(value, dict):
        if {"type", "value"} <= set(value):
            return [value]
        out = []
        for key, item in value.items():
            if isinstance(item, dict):
                candidate = dict(item)
                candidate.setdefault("type", key)
                out.append(candidate)
            else:
                out.append({"type": key, "value": item})
        return out
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _identifier_value(value, *names: str) -> str | None:
    """Extract an identifier by type name from varied identifier structures."""
    wanted = {name.lower() for name in names}
    for candidate in _iter_identifier_candidates(value):
        ctype = str(
            candidate.get("type")
            or candidate.get("name")
            or candidate.get("@type")
            or ""
        ).lower()
        if ctype in wanted:
            return _first_nonempty(
                candidate.get("value"),
                candidate.get("id"),
                candidate.get("identifier"),
                candidate.get("content"),
            )
    if isinstance(value, dict):
        for name in names:
            direct = value.get(name) or value.get(name.lower()) or value.get(name.upper())
            if direct:
                return direct
    return None


def _extract_links(value) -> list[str]:
    """Extract URLs from a list of link objects or plain strings."""
    out = []
    items = value if isinstance(value, list) else [value]
    for item in items:
        if not isinstance(item, dict):
            continue
        href = item.get("url") or item.get("href") or item.get("@href")
        if href:
            out.append(href)
    return out


def _extract_wos_authors(value) -> list[str]:
    """Extract author names from various Web of Science response structures."""
    if isinstance(value, dict):
        value = value.get("authors") or value.get("author") or value.get("names") or []
    if not isinstance(value, list):
        return []
    authors = []
    for item in value:
        if not isinstance(item, dict):
            continue
        name = _first_nonempty(
            item.get("displayName"),
            item.get("fullName"),
            item.get("wosStandard"),
            item.get("name"),
        )
        if name:
            authors.append(str(name))
    return authors


def _extract_scopus_authors(value, creator: str | None = None) -> list[str]:
    """Extract author names from Scopus response structures."""
    if isinstance(value, dict):
        value = value.get("author") or []
    authors = []
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, dict):
                continue
            name = _first_nonempty(
                item.get("authname"),
                item.get("ce:indexed-name"),
                item.get("preferred-name", {}).get("ce:indexed-name")
                if isinstance(item.get("preferred-name"), dict) else None,
            )
            if name:
                authors.append(str(name))
    if not authors and creator:
        authors.append(creator)
    return authors


# ---------------------------------------------------------------------------
# Normalizers — one per provider
# ---------------------------------------------------------------------------
def normalize_elicit_paper(paper: dict) -> dict:
    return {
        "title": paper.get("title", ""),
        "abstract": paper.get("abstract"),
        "year": paper.get("year"),
        "doi": paper.get("doi"),
        "pmid": paper.get("pmid"),
        "elicit_id": paper.get("elicitId"),
        "citation_count": paper.get("citedByCount"),
        "tldr": None,
        "authors": paper.get("authors") or [],
        "fields_of_study": None,
        "venue": paper.get("venue"),
        "urls": paper.get("urls") or [],
        "source": "elicit",
    }


def normalize_s2_paper(paper: dict) -> dict:
    ext_ids = paper.get("externalIds") or {}
    open_access_pdf = paper.get("openAccessPdf") or {}
    return {
        "title": paper.get("title", ""),
        "abstract": paper.get("abstract"),
        "year": paper.get("year"),
        "doi": ext_ids.get("DOI") or paper.get("doi"),
        "pmid": ext_ids.get("PubMed"),
        "s2_id": paper.get("paperId"),
        "citation_count": paper.get("citationCount"),
        "tldr": (paper.get("tldr") or {}).get("text"),
        "authors": [a.get("name", "") for a in (paper.get("authors") or [])],
        "fields_of_study": paper.get("fieldsOfStudy"),
        "s2_open_access_pdf_url": open_access_pdf.get("url") or "",
        "source": "s2",
    }


def normalize_oa_paper(paper: dict) -> dict:
    doi = strip_doi_prefix(paper.get("doi")) or None
    authors = []
    for authorship in (paper.get("authorships") or []):
        name = authorship.get("author", {}).get("display_name", "")
        if name:
            authors.append(name)
    ids = paper.get("ids") or {}
    open_access = paper.get("open_access") or {}
    best_oa_location = paper.get("best_oa_location") or {}
    primary_location = paper.get("primary_location") or {}
    has_content = paper.get("has_content") or {}
    abstract = paper.get("abstract")
    if not abstract:
        abstract = invert_abstract(paper.get("abstract_inverted_index"))
    return {
        "title": paper.get("title", ""),
        "abstract": abstract,
        "year": paper.get("publication_year"),
        "doi": doi,
        "pmid": _normalize_pmid(ids.get("pmid")),
        "pmc": _normalize_pmcid(ids.get("pmcid")),
        "oa_id": paper.get("id"),
        "citation_count": paper.get("cited_by_count"),
        "tldr": None,
        "authors": authors,
        "fields_of_study": None,
        "openalex_is_oa": bool(open_access.get("is_oa")),
        "openalex_oa_status": open_access.get("oa_status") or "",
        "openalex_oa_url": open_access.get("oa_url") or "",
        "openalex_any_repository_has_fulltext": bool(
            open_access.get("any_repository_has_fulltext")
        ),
        "openalex_best_oa_url": _location_url(best_oa_location),
        "openalex_best_oa_pdf_url": _location_pdf_url(best_oa_location),
        "openalex_best_oa_host_type": _location_host_type(best_oa_location),
        "openalex_primary_url": _location_url(primary_location),
        "openalex_primary_pdf_url": _location_pdf_url(primary_location),
        "openalex_primary_host_type": _location_host_type(primary_location),
        "openalex_has_pdf_content": bool(has_content.get("pdf")),
        "openalex_has_grobid_xml": bool(has_content.get("grobid_xml")),
        "openalex_content_url": paper.get("content_url") or "",
        "source": "openalex",
    }


def normalize_pm_paper(paper: dict) -> dict:
    """PubMed papers are already mostly normalized by PubMedClient.

    Returns a new dict (the input is not changed) with the provider-neutral
    fields filled and the lowercase ``pubmed`` source label, also for legacy
    records parsed as ``"PubMed"``.
    """
    defaults = {
        "citation_count": None,
        "tldr": None,
        "fields_of_study": None,
        "s2_id": None,
        "oa_id": None,
    }
    return {**defaults, **paper, "source": PUBMED_SOURCE}


def normalize_wos_paper(paper: dict) -> dict:
    source_raw = paper.get("source")
    source = source_raw if isinstance(source_raw, dict) else {}
    identifiers = paper.get("identifiers") or {}
    links = paper.get("links") or []
    citations_raw = paper.get("citations")
    citation_count = _first_nonempty(
        paper.get("timesCited"),
        paper.get("citationsCount"),
        citations_raw.get("count") if isinstance(citations_raw, dict) else None,
    )
    names_raw = paper.get("names")
    names = names_raw if isinstance(names_raw, dict) else {}
    return {
        "title": _first_nonempty(
            paper.get("title"),
            paper.get("documentTitle"),
            names.get("title"),
        ) or "",
        "abstract": _first_nonempty(
            paper.get("abstract"),
            paper.get("abstractText"),
            paper.get("summary"),
        ),
        "year": _parse_year(
            _first_nonempty(
                source.get("publishYear"),
                source.get("publishedBiblioYear"),
                paper.get("publicationYear"),
                paper.get("year"),
            )
        ),
        "doi": _identifier_value(identifiers, "doi", "DOI"),
        "pmid": _normalize_pmid(_identifier_value(identifiers, "pmid", "pubmed", "PMID")),
        "wos_id": _first_nonempty(paper.get("uid"), paper.get("id")),
        "citation_count": _as_int(citation_count),
        "tldr": None,
        "authors": _extract_wos_authors(paper.get("names")),
        "fields_of_study": None,
        "venue": _first_nonempty(source.get("sourceTitle"), source.get("title")),
        "urls": _extract_links(links),
        "source": "wos",
    }


def normalize_scopus_paper(paper: dict) -> dict:
    cover_date = _first_nonempty(
        paper.get("prism:coverDate"),
        paper.get("prism:coverDisplayDate"),
        paper.get("coverDate"),
    )
    citation_count = paper.get("citedby-count")
    creator = _first_nonempty(paper.get("dc:creator"), paper.get("creator"))
    return {
        "title": _first_nonempty(paper.get("dc:title"), paper.get("title")) or "",
        # subtypeDescription is the document type ("Article", "Review"), never
        # the abstract. STANDARD-view records have no abstract at all.
        "abstract": _first_nonempty(
            paper.get("dc:description"),
            paper.get("description"),
        ),
        "document_type": _first_nonempty(
            paper.get("subtypeDescription"),
            paper.get("subtype"),
        ),
        "year": _parse_year(cover_date),
        "doi": _first_nonempty(paper.get("prism:doi"), paper.get("doi")),
        "pmid": _normalize_pmid(_first_nonempty(paper.get("pubmed-id"), paper.get("pubmedId"))),
        "scopus_id": _first_nonempty(paper.get("eid"), paper.get("identifier")),
        "citation_count": _as_int(citation_count),
        "tldr": None,
        "authors": _extract_scopus_authors(paper.get("author"), creator=str(creator or "")),
        "fields_of_study": None,
        "venue": _first_nonempty(
            paper.get("prism:publicationName"),
            paper.get("publicationName"),
        ),
        "urls": _extract_links(paper.get("link")),
        "scopus_openaccess": str(paper.get("openaccess") or ""),
        "source": "scopus",
    }
