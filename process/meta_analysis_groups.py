"""The researcher's effect decisions and suggested pooling groups.

Every calculable or converted effect gets a row the researcher reviews: include
it or not, which group it pools in, and (for statistics that carry no sign)
its direction. Decisions live in ``claims/effect_decisions.json`` keyed by
``effect_id`` so they survive re-running Stage 2c and the meta-analysis.

Outcome-label suggestions are only suggested: effects whose outcome labels are
similar and that share a measure family are proposed together, and pool only
once the researcher confirms them. Model groups take their place in the table
only after the two semantic gates in ``meta_analysis_model_groups`` pass.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from core.io import atomic_write_json
from process.meta_analysis_effects import STATUS_CALCULABLE, STATUS_CONVERTED

logger = logging.getLogger(__name__)

DECISIONS_FILE = "effect_decisions.json"
REVIEW_FILE = "effects_for_review.json"
DECISIONS_VERSION = 1
OUTCOME_SIMILARITY = 0.5      # token Jaccard for outcome labels to share a suggestion
POPULATION_SIMILARITY = 0.3   # populations must not clearly differ
MIN_SUGGESTED_PAPERS = 2      # a suggestion only makes sense if it could pool
_UNSIGNED_ROUTES = frozenset({"f_to_d", "chi2_to_phi", "p_to_r"})
_STOPWORDS = frozenset(
    "a an and are as at by for from in of on or the to vs versus with without between "
    "among group groups score scores level levels".split())


def decisions_path(config) -> Path:
    return Path(config.claims_dir) / DECISIONS_FILE


def load_decisions(config) -> dict[str, dict]:
    """``{effect_id: {"include", "group", "direction"}}``; empty if none saved."""
    path = decisions_path(config)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Could not read effect decisions %s: %s", path, exc)
        return {}
    decisions = data.get("decisions") if isinstance(data, dict) else None
    return {str(k): dict(v) for k, v in (decisions or {}).items() if isinstance(v, dict)}


def save_decisions(config, decisions: dict[str, dict]) -> Path:
    path = decisions_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, {"version": DECISIONS_VERSION, "decisions": decisions})
    return path


_SUFFIXES = ("ions", "ion", "ives", "ive", "ness", "ity", "ies", "es", "s")
MIN_STEM = 4


def _stem(word: str) -> str:
    """Light suffix stripping so 'depression' and 'depressive' compare equal."""
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= MIN_STEM:
            return word[:-len(suffix)]
    return word


def _tokens(text: str) -> frozenset[str]:
    words = re.findall(r"[a-z0-9]+", str(text or "").casefold())
    return frozenset(_stem(w) for w in words if w not in _STOPWORDS and len(w) > 1)


def _jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def _family(row: dict) -> str:
    scale = str(row.get("analysis_scale") or "")
    route = str(row.get("route") or "")
    used = str(row.get("effect_used") or "")
    if "Log" in scale or used.split("=")[0] in {"or", "aor", "rr", "hr"}:
        return "ratio"
    if scale == "Fisher's z" or route in {"chi2_to_phi", "z_to_r", "p_to_r"} or used.startswith("r="):
        return "correlation"
    return "smd"


def suggest_groups(rows: list[dict]) -> dict[str, str]:
    """``{effect_id: suggested group label}`` for effects that look alike."""
    clusters: list[dict] = []
    for row in rows:
        if row.get("status") not in (STATUS_CALCULABLE, STATUS_CONVERTED):
            continue
        outcome = _tokens(row.get("outcome"))
        if not outcome:
            continue
        population, family = _tokens(row.get("population")), _family(row)
        for cluster in clusters:
            if (cluster["family"] == family
                    and _jaccard(cluster["outcome"], outcome) >= OUTCOME_SIMILARITY
                    and (not cluster["population"] or not population
                         or _jaccard(cluster["population"], population) >= POPULATION_SIMILARITY)):
                cluster["rows"].append(row)
                break
        else:
            clusters.append({"family": family, "outcome": outcome, "population": population,
                             "rows": [row]})
    suggestions: dict[str, str] = {}
    for cluster in clusters:
        papers = {r["article_id"] for r in cluster["rows"]}
        if len(papers) < MIN_SUGGESTED_PAPERS:
            continue
        first = cluster["rows"][0]
        label = str(first.get("outcome") or "").strip()
        if first.get("population"):
            label = f"{label} — {str(first['population']).strip()}"
        for row in cluster["rows"]:
            suggestions[row["effect_id"]] = label
    return suggestions


def review_rows(rows: list[dict], decisions: dict[str, dict],
                proposals: dict[str, dict] | None = None) -> list[dict]:
    """Rows for the effects table: every calculable or converted effect.

    *proposals* (``{effect_id: {"group", ...}}``) are the model's pooling
    groups; without one, an effect gets the outcome-label suggestion.
    """
    suggestions = suggest_groups(rows)
    proposals = proposals or {}
    table = []
    for row in rows:
        if row.get("status") not in (STATUS_CALCULABLE, STATUS_CONVERTED) and \
                row.get("reason_code") != "excluded_by_researcher":
            continue
        decision = decisions.get(row["effect_id"], {})
        default_include = (row.get("status") == STATUS_CALCULABLE
                           or (row.get("status") == STATUS_CONVERTED and row.get("sign_known")))
        table.append({
            "effect_id": row["effect_id"], "article_id": row["article_id"], "title": row["title"],
            "test_text": row["test_text"], "effect_used": row["effect_used"],
            "route": row.get("route") or "", "analysis_scale": row.get("analysis_scale", ""),
            "analysis_estimate": row.get("analysis_estimate", ""),
            "analysis_se": row.get("analysis_se", ""),
            "uncertainty_source": row.get("uncertainty_source", ""),
            "assumptions": row.get("assumptions", ""), "status": row["status"],
            "reason": row.get("reason", ""), "outcome": row.get("outcome", ""),
            "comparison": row.get("comparison", ""), "population": row.get("population", ""),
            "sign_known": row.get("route") not in _UNSIGNED_ROUTES,
            "suggested_group": proposals.get(row["effect_id"], {}).get("group")
            or suggestions.get(row["effect_id"], ""),
            "suggested_reversed": bool(proposals.get(row["effect_id"], {}).get("reversed")),
            "suggested_group_audit_id": str(
                proposals.get(row["effect_id"], {}).get("audit_id") or ""
            ),
            "suggested_group_validation": str(
                proposals.get(row["effect_id"], {}).get("validation") or ""
            ),
            "reversed": bool(decision.get("reversed")),
            "group": str(decision.get("group") or ""),
            "include": bool(decision.get("include", default_include)),
            "direction": str(decision.get("direction") or ""),
        })
    return table


def write_review(path: Path, rows: list[dict], decisions: dict[str, dict],
                 proposals: dict[str, dict] | None = None) -> Path:
    atomic_write_json(Path(path), {"version": DECISIONS_VERSION,
                                   "effects": review_rows(rows, decisions, proposals)})
    return Path(path)
