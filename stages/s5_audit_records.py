"""Stage 5 input records: validation, field coercion and paper identity."""
import hashlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

# 64-bit title digest: collision-free in practice for any realistic corpus.
_TITLE_DIGEST_BYTES = 8


def _text(value: Any) -> str:
    """A string field, or "" for null/non-string values (``dict.get`` keeps nulls)."""
    return value if isinstance(value, str) else ""


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _validate_records(data: Any, label: str) -> list[dict]:
    """The dict records of a JSON list; malformed entries are logged and skipped.

    A top-level object is rejected: iterating it would audit its keys.
    """
    if not isinstance(data, list):
        raise TypeError(f"{label} must contain a JSON list of records, "
                         f"got {type(data).__name__}")
    records = [r for r in data if isinstance(r, dict)]
    if len(records) < len(data):
        logger.warning("  Skipped %d malformed %s record(s) that were not JSON objects",
                       len(data) - len(records), label)
    return records


# ── Paper identity ────────────────────────────────────────────────────────

def _bibliographic_id(paper: dict) -> str:
    """doi/pmid/pmcid or title-digest id, or "" when nothing identifies the paper."""
    for key in ("doi", "pmid", "pmcid"):
        value = str(paper.get(key) or "").strip()
        if value:
            return f"{key}:{value.lower()}"
    title = str(paper.get("title") or "").strip().lower()
    if title:
        year = str(paper.get("year") or paper.get("publication_year") or "")
        digest = hashlib.blake2b(f"{title}|{year}".encode(),
                                 digest_size=_TITLE_DIGEST_BYTES).hexdigest()
        return f"title:{digest}"
    return ""


def _derive_paper_id(paper: dict, index: int) -> str:
    """Build a stable identifier for a paper that lacks a usable one.

    Prefers real bibliographic identifiers so the same paper keeps the same
    id across runs, and only falls back to positional numbering when nothing
    identifying is present.
    """
    return _bibliographic_id(paper) or f"index:{index}"


def _own_id(record: dict) -> str:
    return str(record.get("paper_id") or record.get("id") or "").strip()


def _unique_id(pid: str, seen: set[str]) -> str:
    suffix = 2
    while f"{pid}#{suffix}" in seen:
        suffix += 1
    return f"{pid}#{suffix}"


def _normalize_paper_ids(claims: list[dict], profiles: list[dict]) -> int:
    """Ensure every paper has a unique, non-empty ``paper_id``, in place.

    Stage 5 keys audit rows, per-paper aggregation, and the Phase 4 grounding
    flags on ``paper_id``. Blank or duplicated ids therefore cause one paper's
    audit rows to overwrite another's, so Phase 4 can flag or remove the
    wrong records. Normalizing once here means Phase 1 and Phase 4 cannot
    disagree about identity. Duplicates become ``pid#2``, ``pid#3`` …

    Profiles take their own ``paper_id or id``, else a bibliographic id; they
    never get a positional id, which would attach them to an unrelated claim.

    Returns the number of claim ids that were derived or disambiguated.
    """
    seen: set[str] = set()
    derived = duplicates = 0
    for index, paper in enumerate(claims):
        original = _own_id(paper)
        pid = original or _derive_paper_id(paper, index)
        derived += not original
        if pid in seen:
            pid = _unique_id(pid, seen)
            duplicates += 1
        seen.add(pid)
        paper["paper_id"] = pid

    for profile in profiles:
        profile_id = _own_id(profile) or _bibliographic_id(profile)
        if profile_id:
            profile["paper_id"] = profile_id

    if derived:
        logger.info("  Derived paper_id for %d record(s) from doi/pmid/title", derived)
    if duplicates:
        logger.warning(
            "  Disambiguated %d duplicate paper_id value(s) with a '#n' suffix; "
            "audit aggregation and the Phase 4 grounding check would otherwise have "
            "merged distinct papers", duplicates,
        )
    return derived + duplicates


def _profile_lookup_ids(pid: str) -> tuple[str, ...]:
    """Ids under which a claim's profile may be filed (``pid#2`` → ``pid``)."""
    base, sep, suffix = pid.rpartition("#")
    return (pid, base) if sep and suffix.isdigit() and base else (pid,)
