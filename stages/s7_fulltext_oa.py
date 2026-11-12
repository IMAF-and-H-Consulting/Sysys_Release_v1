"""Stage 7 Phase 2 helpers: Unpaywall open-access lookups and their cache.

Split out of ``stages.s7_fulltext``, which re-exports every name here and
keeps the ``check_oa_status`` driver (tests patch its module globals).
"""
import json
import logging
import time
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import quote

from core.io import atomic_write_json

# Shares the Stage 7 logger so log records keep their original name.
logger = logging.getLogger("stages.s7_fulltext")

OA_LOOKUP_SKIPPED_REASON = (
    "Unpaywall lookup skipped: no contact email is configured (set the PubMed "
    "email in Settings). Unpaywall requires a real contact address, so no "
    "placeholder is sent."
)

_UNPAYWALL_API = "https://api.unpaywall.org/v2"
_UNPAYWALL_TIMEOUT_SECONDS = 15
_UNPAYWALL_DELAY_SECONDS = 0.1  # ~10 requests per second
# Persist the cache every N fresh lookups so a crash keeps earlier results.
_OA_CACHE_FLUSH_EVERY = 25
# Closed-access answers expire (embargoes end); open-access answers do not.
_NEGATIVE_OA_CACHE_TTL_DAYS = 30
OA_STATUS_OPEN = "open"
OA_STATUS_CLOSED = "closed"
OA_STATUS_ERROR = "error"
OA_STATUS_SKIPPED = "skipped"


def _unpaywall_url(doi: str) -> str:
    """Unpaywall URL with the DOI path-encoded (a SICI ``#`` is not a fragment)."""
    return f"{_UNPAYWALL_API}/{quote(doi, safe='/')}"


def _load_oa_cache(cache_path: Path | None) -> dict[str, dict]:
    if not cache_path or not cache_path.exists():
        return {}
    try:
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
    except (OSError, ValueError) as exc:
        logger.warning("OA cache %s is unreadable (%s); starting a fresh cache", cache_path, exc)
        return {}
    if not isinstance(cache, dict):
        logger.warning("OA cache %s is not a JSON object; starting a fresh cache", cache_path)
        return {}
    logger.info("  OA cache: loaded %d entries", len(cache))
    return cache


def _save_oa_cache(cache_path: Path | None, cache: dict[str, dict]) -> None:
    if not cache_path:
        return
    try:
        atomic_write_json(cache_path, cache)
    except OSError as exc:
        logger.warning("Could not save OA cache: %s", exc)


def _cache_entry_is_fresh(entry: dict, today: date) -> bool:
    """Open-access entries never expire; closed ones expire after the TTL.

    Legacy entries without a parseable ``queried`` date are kept.
    """
    if entry.get("is_oa"):
        return True
    try:
        queried = date.fromisoformat(str(entry.get("queried") or ""))
    except ValueError:
        return True
    return (today - queried).days <= _NEGATIVE_OA_CACHE_TTL_DAYS


def _apply_oa_entry(target: dict, entry: dict) -> None:
    target["is_oa"] = bool(entry.get("is_oa", False))
    target["oa_url"] = entry.get("oa_url", "")
    target["host_type"] = entry.get("host_type", "")
    target["oa_status"] = OA_STATUS_OPEN if target["is_oa"] else OA_STATUS_CLOSED


def _mark_oa_unknown(target: dict, status: str) -> None:
    """A failed or skipped lookup: not known to be open access, but not "closed"."""
    target["is_oa"] = False
    target["oa_url"] = ""
    target["host_type"] = ""
    target["oa_status"] = status


def _entry_from_unpaywall(data: dict) -> dict:
    is_oa = bool(data.get("is_oa", False))
    best_loc = data.get("best_oa_location") or {}
    if not (is_oa and best_loc):
        return {"is_oa": is_oa, "oa_url": "", "host_type": ""}
    oa_url = (
        best_loc.get("url_for_pdf")
        or best_loc.get("url_for_landing_page")
        or best_loc.get("url", "")
    )
    return {"is_oa": is_oa, "oa_url": oa_url, "host_type": best_loc.get("host_type", "")}


def _query_unpaywall(session: Any, doi: str, email: str) -> dict | None:
    """Return a cache entry for ``doi``, or ``None`` when the API call failed."""
    resp = session.get(
        _unpaywall_url(doi), params={"email": email}, timeout=_UNPAYWALL_TIMEOUT_SECONDS,
    )
    if resp.status_code == 200:
        entry = _entry_from_unpaywall(resp.json())
    elif resp.status_code == 404:
        entry = {"is_oa": False, "oa_url": "", "host_type": ""}
    else:
        logger.warning("  Unpaywall HTTP %d for DOI %s", resp.status_code, doi)
        return None
    return {**entry, "queried": time.strftime("%Y-%m-%d")}


def _lookup_oa(session: Any, doi: str, email: str) -> dict | None:
    try:
        return _query_unpaywall(session, doi, email)
    except Exception as exc:  # network errors, bad JSON: report as an error
        logger.warning("  Unpaywall error for DOI %s: %s", doi, exc)
        return None
