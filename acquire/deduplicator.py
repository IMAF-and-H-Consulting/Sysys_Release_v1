"""Deduplicate paper records while keeping the richest available copy."""
import html
import logging
import re
import unicodedata

from core.doi import normalize_doi

logger = logging.getLogger(__name__)

TITLE_SIMILARITY_THRESHOLD = 90
# Fuzzy title matches never merge records published further apart than this.
MAX_FUZZY_YEAR_GAP = 1
# Identifier fields whose secondary values are kept under ``id_aliases``.
ALIAS_FIELDS = ("doi", "pmid")
# List fields where the preferred record's non-empty list wins outright:
# author names come in provider-specific formats ("Smith, John", "John Smith",
# "Smith J."), so a union would double every author.
PREFERRED_ONLY_LIST_FIELDS = frozenset({"authors"})
_NON_WORD_RE = re.compile(r"[^\w\s]+")

IDENTIFIER_FIELDS = (
    "doi",
    "pmid",
    "pmc",
    "pmcid",
    "s2_id",
    "paperId",
    "oa_id",
    "openalex_id",
    "wos_id",
    "scopus_id",
    "elicit_id",
)

RICH_METADATA_FIELDS = (
    "doi",
    "pmid",
    "pmc",
    "pmcid",
    "s2_id",
    "oa_id",
    "wos_id",
    "scopus_id",
    "elicit_id",
    "title",
    "abstract",
    "year",
    "authors",
    "venue",
    "journal",
    "citation_count",
    "tldr",
    "fields_of_study",
    "urls",
    "url",
)

FULL_TEXT_FIELDS = (
    "full_text",
    "fulltext",
    "full_text_path",
    "full_text_source_path",
    "full_text_source_archive_path",
)

FULL_TEXT_AVAILABILITY_FIELDS = (
    "full_text_url",
    "full_text_source_url",
    "s2_open_access_pdf_url",
    "openalex_oa_url",
    "openalex_best_oa_url",
    "openalex_best_oa_pdf_url",
    "openalex_primary_url",
    "openalex_primary_pdf_url",
    "openalex_content_url",
    "oa_url",
)

FULL_TEXT_FLAGS = (
    "has_full_text",
    "open_access",
    "openalex_is_oa",
    "openalex_any_repository_has_fulltext",
    "openalex_has_pdf_content",
    "openalex_has_grobid_xml",
)


def _clean_identifier(value) -> str | None:
    """Normalize generic identifier-like values for exact duplicate keys."""
    if value in (None, "", [], {}):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.startswith("https://") or text.startswith("http://"):
        text = text.rstrip("/").split("/")[-1]
    return text.lower()


def _aliases(paper: dict, field: str) -> list:
    aliases = paper.get("id_aliases")
    values = aliases.get(field) if isinstance(aliases, dict) else None
    return list(values) if isinstance(values, list) else []


def _doi_set(paper: dict) -> set[str]:
    """Normalized primary and alias DOIs of *paper*."""
    candidates = [paper.get("doi"), *_aliases(paper, "doi")]
    return {doi for doi in (normalize_doi(value) for value in candidates) if doi}


def _pmid_set(paper: dict) -> set[str]:
    candidates = [paper.get("pmid"), *_aliases(paper, "pmid")]
    return {pmid for pmid in (_clean_identifier(value) for value in candidates) if pmid}


def _dedupe_keys(paper: dict) -> list[str]:
    """Return exact-match keys that should collapse records into one paper."""
    keys: list[str] = [f"doi:{doi}" for doi in sorted(_doi_set(paper))]
    keys.extend(
        f"pmid:{pmid}" for pmid in sorted(_pmid_set(paper))
        if _clean_identifier(paper.get("pmid")) != pmid
    )

    canonical_fields = {
        "pmcid": "pmc",
        "paperId": "s2_id",
        "oa_id": "openalex",
        "openalex_id": "openalex",
    }
    for field in IDENTIFIER_FIELDS:
        if field == "doi":
            continue
        ident = _clean_identifier(paper.get(field))
        if ident:
            keys.append(f"{canonical_fields.get(field, field)}:{ident}")
    return keys


def _has_value(value) -> bool:
    if value in (None, "", [], {}):
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _source_count(value) -> int:
    if not value:
        return 0
    if isinstance(value, list):
        return len([item for item in value if _has_value(item)])
    return len([part for part in str(value).split("+") if part.strip()])


def _as_count(value) -> int:
    """Coerce a count-like value to int; malformed legacy values count as 0."""
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def _full_text_rank(paper: dict) -> int:
    """Rank full-text strength: embedded text/file first, then OA availability."""
    if paper.get("text_source") == "full_text":
        return 3
    if any(_has_value(paper.get(field)) for field in FULL_TEXT_FIELDS):
        return 3
    if _as_count(paper.get("full_text_chars")) > 0:
        return 3
    if any(_has_value(paper.get(field)) for field in FULL_TEXT_AVAILABILITY_FIELDS):
        return 2
    if any(bool(paper.get(field)) for field in FULL_TEXT_FLAGS):
        return 2
    return 0


def _metadata_value_count(paper: dict) -> int:
    return sum(1 for field in RICH_METADATA_FIELDS if _has_value(paper.get(field)))


def _identifier_count(paper: dict) -> int:
    return len(_dedupe_keys(paper))


def _paper_quality_key(paper: dict) -> tuple:
    """Sort key for choosing the canonical copy of a duplicate group.

    Priority is:
    1. actual/available full text
    2. connected identifiers such as DOI, PMID, PMCID, and provider IDs
    3. richer bibliographic metadata
    4. longer abstract/full text payloads
    5. citation count and number of contributing sources
    """
    abstract_len = len(str(paper.get("abstract") or ""))
    full_text_len = len(str(paper.get("full_text") or paper.get("fulltext") or ""))
    citations = _as_count(paper.get("citation_count"))
    return (
        _full_text_rank(paper),
        _identifier_count(paper),
        _metadata_value_count(paper),
        full_text_len,
        abstract_len,
        citations,
        _source_count(paper.get("source")),
    )


def _merge_unique_lists(existing: list, new: list) -> list:
    merged = list(existing)
    seen = {str(item).strip().lower() for item in merged if _has_value(item)}
    for item in new:
        marker = str(item).strip().lower()
        if marker and marker not in seen:
            merged.append(item)
            seen.add(marker)
    return merged


def _merge_sources(existing, new) -> str:
    """Join ``+``-separated source labels, ignoring case when de-duplicating."""
    parts = [part for part in str(existing or "").split("+") if part]
    seen = {part.lower() for part in parts}
    for part in str(new or "").split("+"):
        if part and part.lower() not in seen:
            parts.append(part)
            seen.add(part.lower())
    return "+".join(parts)


def _merged_aliases(merged: dict, preferred: dict, secondary: dict) -> dict[str, list]:
    """``id_aliases`` for a merge: every known DOI/PMID except the kept primary."""
    clean = {"doi": normalize_doi, "pmid": _clean_identifier}
    out: dict[str, list] = {}
    for field in ALIAS_FIELDS:
        primary = clean[field](merged.get(field))
        seen = {primary} if primary else set()
        kept: list = []
        for value in (*_aliases(preferred, field), secondary.get(field),
                      *_aliases(secondary, field)):
            key = clean[field](value)
            if key and key not in seen:
                kept.append(value)
                seen.add(key)
        if kept:
            out[field] = kept
    return out


def _merge_value(field: str, current, incoming):
    """The merged value of one field (*current* comes from the preferred copy)."""
    if field == "source":
        return _merge_sources(current, incoming) if incoming else current
    if isinstance(current, list) and isinstance(incoming, list):
        if field in PREFERRED_ONLY_LIST_FIELDS:
            return current if _has_value(current) else list(incoming)
        return _merge_unique_lists(current, incoming)
    if isinstance(current, dict) and isinstance(incoming, dict):
        return {**incoming, **current}
    if not _has_value(current) and _has_value(incoming):
        return incoming
    return current


def _merge_papers(existing: dict, new: dict) -> dict:
    """Merge duplicate records, keeping the highest-quality copy as canonical.

    Returns a new dict; secondary DOIs/PMIDs that differ from the kept ones are
    preserved under ``id_aliases``.
    """
    if _paper_quality_key(new) > _paper_quality_key(existing):
        preferred, secondary = new, existing
    else:
        preferred, secondary = existing, new

    merged = {
        **preferred,
        **{key: _merge_value(key, preferred.get(key), val)
           for key, val in secondary.items() if key != "id_aliases"},
    }
    aliases = _merged_aliases(merged, preferred, secondary)
    without_aliases = {key: val for key, val in merged.items() if key != "id_aliases"}
    return {**without_aliases, "id_aliases": aliases} if aliases else without_aliases


def normalize_title(title) -> str:
    """Lowercase title with HTML entities, width forms and punctuation folded."""
    text = unicodedata.normalize("NFKC", html.unescape(str(title or "")))
    return " ".join(_NON_WORD_RE.sub(" ", text.lower()).split())


def _title_similarity(a: str, b: str) -> float:
    """Jaccard word-set similarity x 100 of the normalized titles."""
    words_a = set(normalize_title(a).split())
    words_b = set(normalize_title(b).split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return (len(intersection) / len(union)) * 100


def _year_of(paper: dict) -> int | None:
    text = str(paper.get("year") or "").strip()
    return int(text[:4]) if text[:4].isdigit() else None


def _identity_conflict(a: dict, b: dict) -> bool:
    """True when two records are provably different studies.

    Different non-empty DOIs or PMIDs, or publication years more than
    ``MAX_FUZZY_YEAR_GAP`` apart (companion papers, protocol vs results).
    """
    dois_a, dois_b = _doi_set(a), _doi_set(b)
    if dois_a and dois_b and not dois_a & dois_b:
        return True
    pmids_a, pmids_b = _pmid_set(a), _pmid_set(b)
    if pmids_a and pmids_b and not pmids_a & pmids_b:
        return True
    year_a, year_b = _year_of(a), _year_of(b)
    return year_a is not None and year_b is not None and abs(year_a - year_b) > MAX_FUZZY_YEAR_GAP


_TITLE_LEADING_ARTICLES = frozenset({"the", "a", "an"})


def _block_tokens(paper: dict) -> tuple[str, str, str]:
    """Year, first-author token, and title prefix used to block fuzzy compares."""
    year = str(paper.get("year") or "").strip()
    authors = paper.get("authors") or []
    if isinstance(authors, str):
        authors = [authors]
    author = ""
    if authors:
        first = str(authors[0] or "").strip().lower()
        if "," in first:
            token = first.split(",", 1)[0]
        else:
            parts = first.split()
            if len(parts) >= 2 and len(parts[-1]) <= 2:
                token = parts[0]
            else:
                token = parts[-1] if parts else ""
        author = "".join(ch for ch in token if ch.isalnum())[:12]
    title = normalize_title(paper.get("title"))
    words = [w for w in title.split() if len(w) > 2 and w not in _TITLE_LEADING_ARTICLES]
    prefix = words[0][:6] if words else ""
    return year, author, prefix


def _block_keys(year: str, author: str, prefix: str) -> list[tuple[str, str, str]]:
    keys: list[tuple[str, str, str]] = []
    if year and author:
        keys.append(("ya", year, author))
    elif year:
        keys.append(("y", year, ""))
    elif author:
        keys.append(("a", "", author))
    if prefix:
        keys.append(("p", prefix, ""))
    if not keys:
        keys.append(("u", "", ""))
    return keys


def deduplicate_papers(papers: list[dict]) -> list[dict]:
    """Deduplicate by exact identifiers, then by fuzzy title match."""
    before = len(papers)

    # Phase 1: exact identifier clustering. This catches DOI, PMID, PMCID, and
    # provider IDs, including bridge records that connect two partial records.
    parent = list(range(before))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    seen_keys: dict[str, int] = {}
    for index, paper in enumerate(papers):
        for key in _dedupe_keys(paper):
            if key in seen_keys:
                union(index, seen_keys[key])
            else:
                seen_keys[key] = index

    grouped: dict[int, list[dict]] = {}
    group_order: list[int] = []
    for index, paper in enumerate(papers):
        root = find(index)
        if root not in grouped:
            grouped[root] = []
            group_order.append(root)
        grouped[root].append(paper)

    unique: list[dict] = []
    for root in group_order:
        group = grouped[root]
        merged = group[0]
        for paper in group[1:]:
            merged = _merge_papers(merged, paper)
        unique.append(merged)

    after_exact = len(unique)

    # Phase 2: Fuzzy title dedup, blocked by year, first author, and title prefix
    # so each paper is compared only with plausible neighbors.
    final: list[dict] = []
    seen_titles: list[tuple[str, int]] = []
    blocks: dict[tuple[str, str, str], list[int]] = {}

    for p in unique:
        title = (p.get("title") or "").strip().lower()
        if not title:
            final.append(p)
            seen_titles.append(("", len(final) - 1))
            continue

        year, author, prefix = _block_tokens(p)
        candidate_positions: list[int] = []
        seen_positions: set[int] = set()
        for key in _block_keys(year, author, prefix):
            for pos in blocks.get(key, []):
                if pos not in seen_positions:
                    seen_positions.add(pos)
                    candidate_positions.append(pos)

        match_position = None
        for pos in candidate_positions:
            seen, seen_idx = seen_titles[pos]
            if (seen and not _identity_conflict(final[seen_idx], p)
                    and _title_similarity(title, seen) >= TITLE_SIMILARITY_THRESHOLD):
                match_position = pos
                break
        if match_position is None:
            pos = len(final)
            seen_titles.append((title, pos))
            final.append(p)
            for key in _block_keys(year, author, prefix):
                blocks.setdefault(key, []).append(pos)
        else:
            final_idx = seen_titles[match_position][1]
            final[final_idx] = _merge_papers(final[final_idx], p)
            updated_title = (final[final_idx].get("title") or "").strip().lower()
            if updated_title:
                seen_titles[match_position] = (updated_title, final_idx)

    removed = before - len(final)
    logger.info(
        "Deduplication: %d -> %d papers (%d removed: %d by exact ID, %d by title)",
        before, len(final), removed,
        before - after_exact, after_exact - len(final),
    )
    return final
