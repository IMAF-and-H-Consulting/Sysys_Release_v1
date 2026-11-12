"""
NeuroQuery API bridge for cross-referencing hypothesis components
with predicted brain activation patterns.

Queries the NeuroQuery API, computes convergence scores between predicted
and observed brain regions, and generates component-level reports.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from core.claims import claim_list
from process.paths import load_paper_list, resolve_claims_corpus

logger = logging.getLogger(__name__)

# ── Region normalisation helpers ──────────────────────────────────────

_REGION_ALIASES: dict[str, str] = {
    "dlpfc": "dorsolateral prefrontal cortex",
    "vmpfc": "ventromedial prefrontal cortex",
    "ofc": "orbitofrontal cortex",
    "acc": "anterior cingulate cortex",
    "pcc": "posterior cingulate cortex",
    "sma": "supplementary motor area",
    "stg": "superior temporal gyrus",
    "ifg": "inferior frontal gyrus",
    "mfg": "middle frontal gyrus",
    "sfg": "superior frontal gyrus",
    "ipl": "inferior parietal lobule",
    "spl": "superior parietal lobule",
    "mtg": "middle temporal gyrus",
    "itg": "inferior temporal gyrus",
    "tpj": "temporoparietal junction",
    "pfc": "prefrontal cortex",
    "vlpfc": "ventrolateral prefrontal cortex",
}


def _normalise_region(name: str) -> str:
    """Lowercase, strip whitespace, expand common abbreviations."""
    name = name.strip().lower()
    name = re.sub(r'\s+', ' ', name)
    return _REGION_ALIASES.get(name, name)


# A shorter name only matches inside a longer one when it carries an
# anatomical word beyond these generic/positional terms ("cortex" alone
# would otherwise match most predicted regions).
_GENERIC_REGION_TOKENS = {
    "area", "areas", "region", "regions", "cortex", "cortical", "gyrus", "gyri",
    "lobe", "lobule", "nucleus", "left", "right", "bilateral", "l", "r",
    "anterior", "posterior", "superior", "inferior", "middle", "medial",
    "lateral", "dorsal", "ventral", "the", "of",
}


def _region_tokens(name: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _normalise_region(name))


def _contains_tokens(longer: list[str], shorter: list[str]) -> bool:
    width = len(shorter)
    return any(longer[i:i + width] == shorter for i in range(len(longer) - width + 1))


def _regions_match(a: str, b: str) -> bool:
    """Match region names on whole tokens after normalisation.

    Equal names match; otherwise the shorter name's tokens must appear as a
    contiguous run in the longer one and include a non-generic word, so
    "left amygdala" ~ "amygdala" but "cortex" !~ "anterior cingulate cortex".
    """
    a_tokens, b_tokens = _region_tokens(a), _region_tokens(b)
    if not a_tokens or not b_tokens:
        return False
    if a_tokens == b_tokens:
        return True
    shorter, longer = sorted((a_tokens, b_tokens), key=len)
    if not set(shorter) - _GENERIC_REGION_TOKENS:
        return False
    return _contains_tokens(longer, shorter)


# ── NeuroQuery API ────────────────────────────────────────────────────

_NEUROQUERY_URL = "https://neuroquery.org/api/query"
_NEUROQUERY_TIMEOUT_SECONDS = 5


def query_neuroquery(terms: str) -> dict:
    """
    Query the NeuroQuery API.

    POST to https://neuroquery.org/api/query with {"text": terms}.
    Returns the parsed JSON object, or an empty dict on any failure or a
    non-object response. The response schema is checked by the caller
    (``_predicted_regions``).
    """
    try:
        import requests
    except ImportError:
        logger.warning("requests library not available; cannot query NeuroQuery")
        return {}

    try:
        resp = requests.post(
            _NEUROQUERY_URL,
            json={"text": terms},
            timeout=_NEUROQUERY_TIMEOUT_SECONDS,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        logger.warning("NeuroQuery API timed out for query: %s", terms[:80])
        return {}
    except requests.exceptions.ConnectionError:
        logger.warning("NeuroQuery API unreachable for query: %s", terms[:80])
        return {}
    except (requests.exceptions.RequestException, ValueError) as exc:
        logger.warning("NeuroQuery API error for query '%s': %s", terms[:80], exc)
        return {}
    if not isinstance(data, dict):
        logger.warning(
            "NeuroQuery API returned %s, not a JSON object, for query: %s",
            type(data).__name__, terms[:80],
        )
        return {}
    return data


def _region_name(entry: Any) -> str:
    if not isinstance(entry, dict):
        return ""
    return str(entry.get("region") or "").strip()


def _predicted_regions(nq_result: Any, query: str = "") -> list[dict] | None:
    """Named ``{"region", "weight"}`` entries from a response, or None.

    None means no usable prediction: a failed call, a non-object response,
    or one without a non-empty ``brain_regions`` list of named regions.
    """
    if not isinstance(nq_result, dict) or not nq_result:
        return None
    regions = nq_result.get("brain_regions")
    named = [r for r in regions if _region_name(r)] if isinstance(regions, list) else []
    if not named:
        logger.warning(
            "NeuroQuery response for '%s' has no usable 'brain_regions' list "
            "(keys: %s); convergence not scored",
            query[:80], sorted(nq_result)[:10],
        )
        return None
    return named


# ── Convergence scoring ───────────────────────────────────────────────

def compute_convergence_score(
    predicted_regions: list[dict] | None,
    observed_regions: list[str],
) -> dict:
    """
    Compare predicted activation regions (from NeuroQuery) with observed
    mentions (from extracted claims).

    Parameters
    ----------
    predicted_regions : list[dict] | None
        From NeuroQuery: [{"region": "amygdala", "weight": 0.8}, ...]
    observed_regions : list[str]
        From claims: ["amygdala", "prefrontal cortex", ...]

    Returns
    -------
    dict with convergence_score, matched/unmatched region lists, and counts.
    ``convergence_score`` is None when there is no named predicted region
    (no prediction is not the same as measured non-convergence). Only named
    regions count toward ``total_predicted``.
    """
    names = [_region_name(p) for p in predicted_regions or []]
    names = [n for n in names if n]
    observed = list(observed_regions)
    matched = [n for n in names if any(_regions_match(n, o) for o in observed)]
    score = round(len(matched) / len(names), 3) if names else None
    return {
        "convergence_score": score,
        "observed_not_predicted": [
            o for o in observed if not any(_regions_match(o, n) for n in names)
        ],
        "predicted_not_observed": [n for n in names if n not in matched],
        "matched_regions": matched,
        "total_predicted": len(names),
        "total_observed": len(observed),
    }


# ── Linking claims to components ──────────────────────────────────────

def load_convergence_claims(config: Any) -> list[dict]:
    """Claims for convergence: the resolved corpus file (Stage 5 filtered
    claims when present), not every JSON file in the claims directory."""
    path = resolve_claims_corpus(config)
    return load_paper_list(path) if path is not None else []


def _component_keywords(comp: dict) -> list[str]:
    keywords = comp.get("keywords") or []
    if isinstance(keywords, str):
        keywords = [keywords]
    terms = [str(k).strip() for k in keywords if str(k).strip()]
    return terms or [str(comp.get("label", "")).strip()]


def _claim_text(claim: dict) -> str:
    parts = [str(claim.get("title") or ""), str(claim.get("abstract") or "")]
    parts.extend(claim_list(claim, "findings"))
    return " ".join(parts)


def _keyword_patterns(keywords: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(rf"\b{re.escape(k)}\b", re.IGNORECASE) for k in keywords if k]


def _claim_links_component(
    claim: dict, comp_id: Any, patterns: list[re.Pattern[str]],
) -> bool:
    """Linked by an explicit ``component_id``, else by a component keyword."""
    claim_comp = claim.get("component_id")
    if claim_comp:
        return str(claim_comp) == str(comp_id)
    text = _claim_text(claim)
    return any(p.search(text) for p in patterns)


def _linked_observations(
    claims: list[dict], comp_id: Any, keywords: list[str],
) -> tuple[list[str], int]:
    """(de-duplicated regions, number of linked claims) for one component."""
    patterns = _keyword_patterns(keywords)
    linked = [c for c in claims if isinstance(c, dict) and _claim_links_component(c, comp_id, patterns)]
    regions = [r for c in linked for r in claim_list(c, "key_entities")]
    return list(dict.fromkeys(regions)), len(linked)


# ── Component-level convergence report ────────────────────────────────

def _component_entry(comp: dict, claims: list[dict]) -> tuple[str, dict]:
    comp_id = comp.get("id", comp.get("label", "unknown"))
    label = comp.get("label", comp_id)
    keywords = _component_keywords(comp)
    query_str = " ".join(keywords) or str(label)
    predicted = _predicted_regions(query_neuroquery(query_str), query_str)
    observed, n_linked = _linked_observations(claims, comp_id, keywords)
    entry = {
        **compute_convergence_score(predicted, observed),
        "component_id": comp_id,
        "component_label": label,
        "query": query_str,
        "neuroquery_available": predicted is not None,
        "linked_claims": n_linked,
    }
    return str(comp_id), entry


def _save_report(report: dict, config: Any) -> None:
    output_dir = getattr(config, "output_dir", None)
    if not output_dir:
        return
    out_path = Path(output_dir) / "neuroquery_convergence.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Saved convergence report to %s", out_path)


def generate_convergence_report(
    components: list[dict],
    claims: list[dict],
    config: Any = None,
) -> dict:
    """
    For each hypothesis component:
    1. Build a query string from component keywords (label if none)
    2. Query NeuroQuery and validate the response schema
    3. Collect brain regions (``key_entities``) from claims linked to this
       component: by ``component_id`` when a claim has one, otherwise when
       its title/abstract/findings mention a component keyword
    4. Compute the convergence score (None when no usable prediction)

    Parameters
    ----------
    components : list[dict]
        Hypothesis components with 'id', 'label', 'keywords' fields.
    claims : list[dict]
        Extracted claims, ideally from ``load_convergence_claims(config)``.
    config : optional
        If provided and has output_dir, saves results to
        config.output_dir / "neuroquery_convergence.json".

    Returns
    -------
    dict mapping component_id -> convergence results, including
    ``neuroquery_available`` and ``linked_claims``.
    """
    report = dict(_component_entry(comp, claims) for comp in components)
    if config is not None:
        _save_report(report, config)
    return report
