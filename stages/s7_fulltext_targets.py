"""Stage 7 Phase 1: identify high-value targets (SRs, MAs, RCTs).

Split out of ``stages.s7_fulltext``, which re-exports every name here.
"""
import json
import logging
import re
from pathlib import Path
from typing import Any

from core.claims import claim_value

# Shares the Stage 7 logger so log records keep their original name.
logger = logging.getLogger("stages.s7_fulltext")

TARGET_TYPES = ["systematic_review_meta_analysis", "randomized_controlled_trial"]


def _profile_study_type(profile: Any) -> str:
    if not isinstance(profile, dict):
        return ""
    method_profile = profile.get("method_profile")
    if not isinstance(method_profile, dict):
        return ""
    return str(method_profile.get("study_method_type") or "")


def _target_record(paper: dict, method_type: str) -> dict:
    return {
        "doi": str(paper.get("doi") or ""),
        "pmid": str(paper.get("pmid") or ""),
        "title": str(paper.get("title") or ""),
        "year": paper.get("year", ""),
        "method_type": method_type,
    }


def _add_target(targets: list[dict], seen_ids: set[str], record: dict) -> bool:
    uid = record["doi"] or record["pmid"]
    if not uid or uid in seen_ids:
        return False
    seen_ids.add(uid)
    targets.append(record)
    return True


def _load_json_list(path: Path, label: str) -> list:
    """Load a JSON array; a missing, unreadable or non-list file yields ``[]``."""
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        logger.warning("Could not load %s: %s", label, exc)
        return []
    if not isinstance(data, list):
        logger.warning("Ignoring %s: expected a JSON list, got %s", label, type(data).__name__)
        return []
    return data


def identify_targets(
    config, target_types: list[str] | None = None,
    claims_path: str | None = None,
) -> list[dict]:
    """Find SRs, MAs, RCTs from method_profiles_validated.json + claims.

    Records are validated one at a time, so a malformed profile or claim is
    skipped instead of discarding the rest of its file.
    """
    target_types = target_types or TARGET_TYPES
    targets: list[dict] = []
    seen_ids: set[str] = set()

    # Source 1: method_profiles_validated.json
    mp_path = config.data_dir / "method_review" / "method_profiles_validated.json"
    profiles = _load_json_list(mp_path, "method profiles")
    for p in profiles:
        study_type = _profile_study_type(p)
        if _matches_target_type(study_type, target_types):
            _add_target(targets, seen_ids, _target_record(p, study_type))
    if profiles:
        logger.info("  Method profiles: %d targets from %d papers",
                    len(targets), len(profiles))
    del profiles

    # Source 2: claims
    if claims_path:
        claims_file = Path(claims_path)
    else:
        # resolve_claims_path falls back to claims.json when the filtered
        # file from Stage 5 exists but is empty
        from process.paths import resolve_claims_path
        claims_file = resolve_claims_path(config)

    claims = _load_json_list(claims_file, "claims")
    claims_added = 0
    for p in claims:
        if not isinstance(p, dict):
            continue
        # Stage 3 nests methodology under "claims".
        method_field = str(claim_value(p, "methodology", "") or "").lower()
        if _matches_target_type(method_field, target_types):
            claims_added += _add_target(
                targets, seen_ids, _target_record(p, method_field[:60]))
    if claims:
        logger.info("  Claims: %d additional targets from %d papers",
                    claims_added, len(claims))
    del claims

    # Breakdown by type
    type_counts: dict[str, int] = {}
    for t in targets:
        mt = t["method_type"]
        type_counts[mt] = type_counts.get(mt, 0) + 1
    if type_counts:
        breakdown = ", ".join(f"{k}: {v}" for k, v in type_counts.items())
        logger.info("  Target breakdown: %s", breakdown)

    # Filter out papers without DOIs (can't query Unpaywall)
    with_doi = [t for t in targets if t.get("doi")]
    without_doi = len(targets) - len(with_doi)
    if without_doi:
        logger.warning("  %d targets lack DOIs — skipping OA lookup for those", without_doi)

    return with_doi


def _type_keywords(target_types: list[str]) -> list[str]:
    """Convert target type identifiers to search keywords."""
    keywords = []
    for tt in target_types:
        keywords.append(tt.lower().replace("_", " "))
        # Also add component keywords
        if "systematic" in tt.lower():
            keywords.extend(["systematic review", "meta-analysis", "meta analysis"])
        if "randomized" in tt.lower():
            keywords.extend(["randomized controlled trial", "randomised controlled trial", "rct"])
    return list(set(keywords))


def _normalize_design_text(text: str) -> str:
    """Lowercase with ``_``/``-`` as spaces, so enums and prose compare alike."""
    return " ".join(re.sub(r"[_\-]+", " ", str(text).lower()).split())


def _matches_target_type(text: str, target_types: list[str]) -> bool:
    """Whether a study-design enum or methodology text names a target type.

    Keywords match whole words, so "rct" does not match "infarct", and a
    "non-" or "quasi-" prefix (non-randomized controlled trial) is no match.
    """
    normalized = _normalize_design_text(text)
    if not normalized:
        return False
    return any(
        re.search(
            rf"(?<!\bnon )(?<!\bquasi )\b{re.escape(_normalize_design_text(kw))}\b",
            normalized,
        )
        for kw in _type_keywords(target_types)
    )
