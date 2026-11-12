"""
GRADE evidence certainty assessment for the Sysys pipeline.

Implements the Grading of Recommendations Assessment, Development and
Evaluation (GRADE) framework, rating certainty of evidence across five
domains: risk of bias, inconsistency, indirectness, imprecision, and
publication bias.  Each hypothesis component receives an overall certainty
rating (High / Moderate / Low / Very Low) derived from the domain ratings.

The module also generates a Summary of Findings (SoF) table suitable for
display in the GUI and export to Markdown / JSON.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from core.claims import claim_list, claim_value
from core.io import load_json as _load_json
from process.grade_models import (
    ComponentGrade,
    DomainRating,
    GradeCertainty,
    GradeConcern,
)
from process.grade_report import (  # noqa: F401  (re-exported)
    generate_sof_table,
    save_grade_outputs,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HIGH_QUALITY_DESIGNS = {
    "randomized_controlled_trial",
    "systematic_review_meta_analysis",
    "rct",
    "meta-analysis",
    "systematic review",
}

_CERTAINTY_LADDER = [
    GradeCertainty.HIGH,
    GradeCertainty.MODERATE,
    GradeCertainty.LOW,
    GradeCertainty.VERY_LOW,
]


def _evidence_tier(study_method_type: str) -> str:
    """Map a study design label to an evidence tier.

    Covers the method-type vocabulary emitted by
    ``process.method_review.ALLOWED_METHOD_TYPES`` (plus legacy aliases);
    anything unrecognized falls through to "unknown".
    """
    t = study_method_type.lower()
    if t in ("rct", "meta-analysis", "systematic review",
             "randomized_controlled_trial", "systematic_review_meta_analysis"):
        return "high"
    if t in ("cohort", "case-control",
             "cohort_observational", "longitudinal", "case_control"):
        return "moderate"
    if t in ("case report", "expert opinion",
             "case_report_case_series", "cross_sectional", "narrative_review",
             "preclinical_animal", "in_vitro_cellular", "imaging_ml_classifier",
             "genetic_association", "mixed_methods", "methods_protocol"):
        return "low"
    return "unknown"


# ---------------------------------------------------------------------------
# Domain assessors
# ---------------------------------------------------------------------------

def assess_risk_of_bias(
    method_profiles: list[dict],
    component_studies: list[str],
) -> DomainRating:
    """Assess risk of bias based on study-design quality.

    Logic:
    - Count studies by evidence_tier for this component's articles.
    - >50 % "low" tier  -> very serious concern (downgrade 2)
    - >50 % "moderate"  -> serious concern (downgrade 1)
    - >50 % "high"      -> no concern
    - Also factor in the Stage 4 ``needs_manual_review`` flag (legacy
      ``flagged_for_review``) and the has_control_group rate.
    """
    study_set = set(component_studies)
    relevant = [mp for mp in method_profiles if mp.get("article_id") in study_set]
    n = len(relevant)
    if n == 0:
        return DomainRating(
            domain="risk_of_bias",
            concern=GradeConcern.VERY_SERIOUS,
            rationale="No method-profile data available for component studies.",
            downgrade_levels=2,
            data_points={"n_profiles": 0},
        )

    tier_counts: dict[str, int] = {"high": 0, "moderate": 0, "low": 0, "unknown": 0}
    flagged = 0
    control_count = 0
    for mp in relevant:
        tier = _evidence_tier(mp.get("study_method_type", "unknown"))
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        if mp.get("needs_manual_review") or mp.get("flagged_for_review"):
            flagged += 1
        if mp.get("has_control_group"):
            control_count += 1

    low_pct = (tier_counts["low"] + tier_counts["unknown"]) / n
    mod_pct = tier_counts["moderate"] / n
    control_rate = control_count / n
    flagged_rate = flagged / n

    # Determine concern level
    if low_pct > 0.5 or flagged_rate > 0.4:
        concern = GradeConcern.VERY_SERIOUS
        downgrade = 2
    elif mod_pct > 0.5 or low_pct > 0.3:
        concern = GradeConcern.SERIOUS
        downgrade = 1
    else:
        concern = GradeConcern.NO_CONCERN
        downgrade = 0

    # Extra penalty if very few studies have controls
    if control_rate < 0.3 and downgrade < 2:
        concern = GradeConcern.SERIOUS if concern == GradeConcern.NO_CONCERN else GradeConcern.VERY_SERIOUS
        downgrade = min(downgrade + 1, 2)

    parts = [
        f"{n} studies evaluated.",
        f"Tier breakdown: {tier_counts['high']} high, {tier_counts['moderate']} moderate, "
        f"{tier_counts['low']} low, {tier_counts['unknown']} unknown.",
    ]
    if flagged:
        parts.append(f"{flagged} flagged for review.")
    if control_rate < 0.5:
        parts.append(f"Only {control_rate:.0%} have a control group.")

    return DomainRating(
        domain="risk_of_bias",
        concern=concern,
        rationale=" ".join(parts),
        downgrade_levels=downgrade,
        data_points={
            "n_profiles": n,
            "tier_counts": tier_counts,
            "flagged": flagged,
            "control_rate": round(control_rate, 3),
        },
    )


def assess_inconsistency(
    interrater_records: list[dict],
    robustness_data: dict,
    component_studies: list[str],
) -> DomainRating:
    """Assess inconsistency (heterogeneity across studies) for one component.

    Accepts optional historical per-component agreement records. Each record
    gives an original and second-rater score; we derive a per-paper agreement
    score and standard deviation, then average them across the component's
    articles.  Robustness stability is folded in if a global stability score
    is available.
    """
    signals: list[str] = []
    worst_concern = GradeConcern.NO_CONCERN
    worst_downgrade = 0

    study_set = set(component_studies)
    component_records = [
        r for r in interrater_records
        if isinstance(r, dict) and r.get("article_id") in study_set
    ]

    agreement = None
    score_std = None
    if component_records:
        scores = []
        stds = []
        for r in component_records:
            orig = r.get("original_score")
            second = r.get("second_score")
            try:
                orig_i = int(orig) if orig is not None else None
                second_i = int(second) if second is not None else None
            except (ValueError, TypeError):
                orig_i = None
                second_i = None
            if orig_i is not None and second_i is not None:
                diff = abs(orig_i - second_i)
                scores.append(1.0 - diff / 5.0)
                stds.append(diff / 5.0)
        if scores:
            agreement = sum(scores) / len(scores)
        if stds:
            score_std = sum(stds) / len(stds)

    if agreement is not None:
        if agreement < 0.4:
            worst_concern = GradeConcern.VERY_SERIOUS
            worst_downgrade = 2
            signals.append(f"Mean agreement {agreement:.2f} (<0.4 — very low).")
        elif agreement < 0.7:
            worst_concern = GradeConcern.SERIOUS
            worst_downgrade = 1
            signals.append(f"Mean agreement {agreement:.2f} (moderate).")
        else:
            signals.append(f"Mean agreement {agreement:.2f} (good).")

    if score_std is not None and score_std > 0.3:
        signals.append(f"High score variability (std={score_std:.2f}).")
        if worst_downgrade < 1:
            worst_concern = GradeConcern.SERIOUS
            worst_downgrade = 1

    # --- robustness stability (global, available only if present) ---
    stability = None
    if isinstance(robustness_data, dict):
        stability = robustness_data.get("stability_score")

    if stability is not None:
        if stability < 0.5:
            signals.append(f"Robustness stability {stability:.2f} (low).")
            if worst_downgrade < 2:
                worst_concern = GradeConcern.VERY_SERIOUS if worst_downgrade >= 1 else GradeConcern.SERIOUS
                worst_downgrade = min(worst_downgrade + 1, 2)
        elif stability < 0.7:
            signals.append(f"Robustness stability {stability:.2f} (moderate).")
        else:
            signals.append(f"Robustness stability {stability:.2f} (good).")

    if agreement is None and stability is None:
        signals.append("Insufficient data to assess inconsistency; manual assessment needed.")
        worst_concern = GradeConcern.SERIOUS
        worst_downgrade = 1

    return DomainRating(
        domain="inconsistency",
        concern=worst_concern,
        rationale=" ".join(signals) if signals else "Insufficient data.",
        downgrade_levels=worst_downgrade,
        data_points={
            "n_component_records": len(component_records),
            "mean_agreement": round(agreement, 3) if agreement is not None else None,
            "mean_score_std": round(score_std, 3) if score_std is not None else None,
            "stability_score": stability,
        },
    )


def assess_indirectness(
    claims: list[dict],
    component: dict,
) -> DomainRating:
    """Assess indirectness — how directly evidence addresses the hypothesis.

    Checks what fraction of claims mention the component's keywords and
    whether conditions/key entities in claims match the hypothesis focus.
    """
    keywords = [kw.lower() for kw in component.get("keywords", [])]
    comp_label = component.get("label", "").lower()

    if not claims:
        return DomainRating(
            domain="indirectness",
            concern=GradeConcern.VERY_SERIOUS,
            rationale="No claims available to assess directness.",
            downgrade_levels=2,
            data_points={"n_claims": 0, "keyword_match_rate": 0},
        )

    keyword_hits = 0
    for claim in claims:
        # Combine all text fields for matching
        text = " ".join([
            " ".join(claim_list(claim, "findings")),
            " ".join(claim_list(claim, "mechanisms")),
            " ".join(claim_list(claim, "conditions")),
            " ".join(claim_list(claim, "key_entities")),
            str(claim_value(claim, "methodology", "") or ""),
        ]).lower()

        if comp_label and comp_label in text or any(kw in text for kw in keywords):
            keyword_hits += 1

    match_rate = keyword_hits / len(claims) if claims else 0

    if match_rate < 0.2:
        concern = GradeConcern.VERY_SERIOUS
        downgrade = 2
        rationale = (
            f"Only {match_rate:.0%} of {len(claims)} claims directly reference "
            f"the component keywords. Most evidence is about related but not "
            f"identical constructs."
        )
    elif match_rate < 0.5:
        concern = GradeConcern.SERIOUS
        downgrade = 1
        rationale = (
            f"{match_rate:.0%} of {len(claims)} claims reference the component "
            f"keywords. Evidence is partially indirect."
        )
    else:
        concern = GradeConcern.NO_CONCERN
        downgrade = 0
        rationale = (
            f"{match_rate:.0%} of {len(claims)} claims directly reference "
            f"the component keywords."
        )

    return DomainRating(
        domain="indirectness",
        concern=concern,
        rationale=rationale,
        downgrade_levels=downgrade,
        data_points={
            "n_claims": len(claims),
            "keyword_match_rate": round(match_rate, 3),
        },
    )


def assess_imprecision(
    robustness_data: dict,
    n_studies: int,
    component_id: str = "",
) -> DomainRating:
    """Assess imprecision (wide confidence intervals, small samples).

    - Bootstrap CI width: >0.5 very serious, 0.3-0.5 serious, <0.3 no concern
    - n_studies < 3  -> very serious
    - n_studies < 5  -> serious

    The bootstrap CI is currently computed across the whole corpus (Stage 9),
    so it is reported as a global reference while the sample-size concern is
    component-specific.
    """
    signals: list[str] = []
    worst_concern = GradeConcern.NO_CONCERN
    worst_downgrade = 0

    # --- CI width ---
    ci_width = None
    if isinstance(robustness_data, dict):
        bootstrap = robustness_data.get("bootstrap") or {}
        if isinstance(bootstrap, dict):
            ci_lower = bootstrap.get("ci_lower")
            ci_upper = bootstrap.get("ci_upper")
            if ci_lower is not None and ci_upper is not None:
                ci_width = ci_upper - ci_lower

    if ci_width is not None:
        if ci_width > 0.5:
            worst_concern = GradeConcern.VERY_SERIOUS
            worst_downgrade = 2
            signals.append(f"Bootstrap CI width {ci_width:.2f} (very wide).")
        elif ci_width > 0.3:
            worst_concern = GradeConcern.SERIOUS
            worst_downgrade = 1
            signals.append(f"Bootstrap CI width {ci_width:.2f} (wide).")
        else:
            signals.append(f"Bootstrap CI width {ci_width:.2f} (acceptable).")
    else:
        signals.append("No bootstrap CI data available.")

    # --- sample size ---
    if n_studies < 3:
        signals.append(f"Only {n_studies} studies (very small evidence base).")
        if worst_downgrade < 2:
            worst_concern = GradeConcern.VERY_SERIOUS
            worst_downgrade = 2
    elif n_studies < 5:
        signals.append(f"Only {n_studies} studies (small evidence base).")
        if worst_downgrade < 1:
            worst_concern = GradeConcern.SERIOUS
            worst_downgrade = 1
    else:
        signals.append(f"{n_studies} studies.")

    if component_id:
        signals.append(f"Component: {component_id}.")
    if ci_width is not None:
        signals.append("Bootstrap CI is corpus-global; component-specific CI not available.")

    return DomainRating(
        domain="imprecision",
        concern=worst_concern,
        rationale=" ".join(signals),
        downgrade_levels=worst_downgrade,
        data_points={
            "ci_width": round(ci_width, 3) if ci_width is not None else None,
            "n_studies": n_studies,
            "component_id": component_id,
        },
    )


# Formal funnel-asymmetry tests are only interpreted from this many studies
# (Cochrane Handbook 13.3.5.4); below it the count heuristic applies.
MIN_STUDIES_FOR_FORMAL_BIAS_TEST = 10
EGGER_SIGNIFICANT_P = 0.05
EGGER_BORDERLINE_P = 0.10
TRIMFILL_SHIFT_CONCERN = 0.2


def _formal_bias_inputs(n_studies: int, results: Mapping | None) -> dict | None:
    """Usable formal test results, or None when the tests did not really run.

    Not usable: skipped, fewer than ``MIN_STUDIES_FOR_FORMAL_BIAS_TEST``
    studies, or the "not run" sentinels (Egger SE of 0, p outside (0, 1]).
    """
    if not isinstance(results, Mapping) or results.get("skipped"):
        return None
    p_value = results.get("eggers_p", results.get("egger_p_value"))
    k = results.get("n_studies", n_studies)
    if not isinstance(p_value, (int, float)) or not 0.0 < p_value <= 1.0:
        return None
    if k < MIN_STUDIES_FOR_FORMAL_BIAS_TEST or results.get("eggers_se") == 0:
        return None
    return {
        "p_value": float(p_value),
        "trimfill_n": results.get("trimfill_n_imputed", 0) or 0,
        "trimfill_adj": results.get("trimfill_adjusted_effect"),
        "trimfill_orig": results.get("trimfill_original_effect"),
        "failsafe": results.get("failsafe_n", 0) or 0,
    }


def _formal_bias_rating(n_studies: int, inputs: dict) -> DomainRating:
    """Publication bias from formal tests; at most one level (GRADE: "strongly suspected")."""
    p_value = inputs["p_value"]
    if p_value < EGGER_SIGNIFICANT_P:
        parts = [f"Egger's test significant (p={p_value:.3f}), suggesting publication bias."]
    elif p_value < EGGER_BORDERLINE_P:
        parts = [f"Egger's test borderline (p={p_value:.3f})."]
    else:
        parts = [f"Egger's test non-significant (p={p_value:.3f})."]
    suspected = p_value < EGGER_BORDERLINE_P

    trimfill_n = inputs["trimfill_n"]
    if trimfill_n > 0 and inputs["trimfill_adj"] is not None and inputs["trimfill_orig"] is not None:
        shift = abs(inputs["trimfill_adj"] - inputs["trimfill_orig"])
        parts.append(f"Trim-and-fill imputed {trimfill_n} studies (adjusted effect shift: {shift:.3f}).")
        suspected = suspected or shift > TRIMFILL_SHIFT_CONCERN

    failsafe = inputs["failsafe"]
    if failsafe > 0:
        threshold = 5 * n_studies + 10
        relation = "below" if failsafe < threshold else "above"
        parts.append(f"Fail-safe N = {failsafe} ({relation} {threshold} threshold).")
        suspected = suspected or failsafe < threshold

    return DomainRating(
        domain="publication_bias",
        concern=GradeConcern.SERIOUS if suspected else GradeConcern.NO_CONCERN,
        rationale=" ".join(parts),
        downgrade_levels=1 if suspected else 0,
        data_points={"egger_p": p_value, "n_studies": n_studies, "tested": True,
                     "trimfill_n_imputed": trimfill_n, "failsafe_n": failsafe},
    )


def assess_publication_bias(
    n_studies: int,
    bias_test_results: dict | None = None,
) -> DomainRating:
    """Assess publication bias using Egger's test, trim-and-fill, and fail-safe N.

    Formal results (from ``process.meta_analysis.run_bias_tests``; keys
    ``eggers_p`` or legacy ``egger_p_value``) are used only when the tests
    actually ran on at least ``MIN_STUDIES_FOR_FORMAL_BIAS_TEST`` studies, and
    downgrade by at most one level (audit 15-M4).  Otherwise a study-count
    heuristic assumes a serious concern.
    """
    inputs = _formal_bias_inputs(n_studies, bias_test_results)
    if inputs is not None:
        return _formal_bias_rating(n_studies, inputs)

    if n_studies < MIN_STUDIES_FOR_FORMAL_BIAS_TEST:
        rationale = (f"Only {n_studies} studies — too few to assess publication bias. "
                     f"Serious concern assumed by default.")
    else:
        rationale = (f"{n_studies} studies available but no formal bias test performed. "
                     f"Publication bias untested; serious concern assumed.")
    return DomainRating(
        domain="publication_bias",
        concern=GradeConcern.SERIOUS,
        rationale=rationale,
        downgrade_levels=1,
        data_points={"n_studies": n_studies, "tested": False},
    )


# ---------------------------------------------------------------------------
# Overall certainty
# ---------------------------------------------------------------------------

def compute_overall_certainty(
    starting: str,
    domains: list[DomainRating],
) -> GradeCertainty:
    """Compute overall GRADE certainty.

    Start at HIGH (for RCTs/SRs) or LOW (for observational).
    Downgrade by the sum of ``downgrade_levels`` across all 5 domains.
    """
    start_idx = 0 if starting == "High" else 2  # HIGH=0, LOW=2
    total_downgrade = sum(d.downgrade_levels for d in domains)
    final_idx = min(start_idx + total_downgrade, len(_CERTAINTY_LADDER) - 1)
    return _CERTAINTY_LADDER[final_idx]


# ---------------------------------------------------------------------------
# Component-level GRADE
# ---------------------------------------------------------------------------

def grade_component(
    component: dict,
    claims: list[dict],
    method_profiles: list[dict],
    interrater_data: list[dict],
    robustness_data: dict,
    component_studies: list[str],
    bias_test_results: dict | None = None,
) -> ComponentGrade:
    """Run full GRADE assessment for one hypothesis component."""
    n_studies = len(component_studies)

    # Determine starting certainty from study designs
    study_set = set(component_studies)
    high_design_count = 0
    for mp in method_profiles:
        if mp.get("article_id") in study_set:
            smt = mp.get("study_method_type", "unknown").lower()
            if smt in _HIGH_QUALITY_DESIGNS:
                high_design_count += 1

    starting = "High" if n_studies > 0 and high_design_count / max(n_studies, 1) > 0.5 else "Low"

    # Assess all 5 domains
    rob = assess_risk_of_bias(method_profiles, component_studies)
    incon = assess_inconsistency(interrater_data, robustness_data, component_studies)
    indir = assess_indirectness(claims, component)
    imprec = assess_imprecision(robustness_data, n_studies, component_id=component.get("id", ""))
    pub_bias = assess_publication_bias(n_studies, bias_test_results)

    domains = [rob, incon, indir, imprec, pub_bias]
    overall = compute_overall_certainty(starting, domains)

    # Summary text
    concern_parts = []
    for d in domains:
        if d.concern != GradeConcern.NO_CONCERN:
            level = "serious" if d.concern == GradeConcern.SERIOUS else "very serious"
            concern_parts.append(f"{d.domain.replace('_', ' ')} ({level})")

    if concern_parts:
        summary = (
            f"{overall.value} certainty ({starting} start). "
            f"Downgraded for: {', '.join(concern_parts)}."
        )
    else:
        summary = f"{overall.value} certainty ({starting} start). No concerns identified."

    return ComponentGrade(
        component_id=component.get("id", ""),
        component_label=component.get("label", ""),
        n_studies=n_studies,
        starting_certainty=starting,
        domains=domains,
        overall_certainty=overall,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Pipeline-level GRADE
# ---------------------------------------------------------------------------

def _claim_article_id(claim: Mapping[str, Any]) -> str:
    """Article id of a claims record; Stage 3 records carry DOI/PMID, not article_id."""
    for key in ("article_id", "doi", "pmid"):
        value = claim.get(key)
        if value:
            return str(value)
    return ""


def _claim_match_text(claim: Mapping[str, Any]) -> str:
    """Findings and conditions of a claims record (nested under ``claims`` by Stage 3)."""
    return " ".join(claim_list(claim, "findings") + claim_list(claim, "conditions")).lower()


def _infer_component_articles(components: list[dict],
                              claims: list[Mapping[str, Any]]) -> dict[str, set[str]]:
    """Map component id -> article ids whose claims mention its label or keywords."""
    texts = [(_claim_article_id(c), _claim_match_text(c)) for c in claims]
    mapping: dict[str, set[str]] = {}
    for comp in components:
        keywords = [str(kw).lower() for kw in comp.get("keywords", [])]
        label = str(comp.get("label", "")).lower()
        matched = {
            aid for aid, text in texts
            if aid and ((label and label in text) or any(kw in text for kw in keywords))
        }
        if matched:
            mapping[comp.get("id", "")] = matched
    return mapping


def _stage_output(config, *parts: str) -> Any:
    """Load ``config.data_dir/<parts>``; None when the config has no data dir."""
    data_dir = getattr(config, "data_dir", None)
    if data_dir is None:
        return None
    return _load_json(Path(data_dir).joinpath(*parts))


def _load_components(config, session: Mapping[str, Any]) -> list[dict]:
    """Session components (legacy), else the components built from the config."""
    from core.research_components import build_hypothesis_components_from_config

    raw = session.get("hypothesis_components")
    if isinstance(raw, list) and raw:
        return [dict(c) for c in raw if isinstance(c, Mapping)]
    return [asdict(c) for c in build_hypothesis_components_from_config(config)]


def _flatten_method_profile(record: Mapping[str, Any]) -> dict:
    """A Stage 4 validated record as the flat profile the GRADE domains read."""
    profile = record.get("method_profile")
    profile = profile if isinstance(profile, Mapping) else {}
    return {**profile, "article_id": _claim_article_id(record)}


def _load_method_profiles(config, session: Mapping[str, Any]) -> list[dict]:
    """Session method profiles (legacy), else Stage 4's validated profiles."""
    mp_dict = session.get("method_profiles")
    if isinstance(mp_dict, Mapping) and mp_dict:
        return [{**mp, "article_id": mp.get("article_id", aid)}
                for aid, mp in mp_dict.items() if isinstance(mp, Mapping)]
    records = _stage_output(config, "method_review", "method_profiles_validated.json")
    if not isinstance(records, list):
        return []
    return [_flatten_method_profile(r) for r in records if isinstance(r, Mapping)]


def _load_robustness(config, session: Mapping[str, Any]) -> dict:
    """Session robustness result (legacy), else Stage 9's robustness_results.json."""
    result = session.get("robustness_result")
    if not isinstance(result, dict):
        result = _stage_output(config, "validation", "robustness_results.json")
    return result if isinstance(result, dict) else {}


def _load_claims(config, session: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Stage 5 filtered claims, else the session's claim data (legacy)."""
    claims_raw = _load_json(config.claims_dir / "claims_filtered.json")
    if not isinstance(claims_raw, list):
        cd = session.get("claim_data", {})
        claims_raw = list(cd.values()) if isinstance(cd, dict) else []
    return [c for c in claims_raw if isinstance(c, Mapping)]


def grade_all_components(config, *, save: bool = True) -> list[ComponentGrade]:
    """Load all pipeline data and run GRADE for every hypothesis component.

    Reads the files the stages write: components from the research config,
    ``claims/claims_filtered.json`` (Stage 5),
    ``method_review/method_profiles_validated.json`` (Stage 4) and
    ``validation/robustness_results.json`` (Stage 9). A legacy
    ``analysis_session.json`` in the output dir, when present, takes
    precedence for the fields it holds. Returns [] when there are no claims.

    With ``save`` (the default) non-empty results are written to
    ``grade_results.json`` and ``grade_sof_table.md`` in the output dir.
    """
    grades: list[ComponentGrade] = []

    # --- load pipeline outputs ------------------------------------------------
    session = _load_json(config.output_dir / "analysis_session.json")
    if not isinstance(session, dict):
        session = {}

    claims_raw = _load_claims(config, session)
    if not claims_raw:
        logger.warning("GRADE: no claims found (run Stage 5 first); nothing to grade.")
        return grades

    components_raw = _load_components(config, session)
    if not components_raw:
        logger.warning("No hypothesis components found in session data or config.")
        return grades

    method_profiles_raw = _load_method_profiles(config, session)
    robustness_raw = _load_robustness(config, session)

    # Evidence links — used to map articles to components
    evidence_links = session.get("evidence_links", [])

    # --- build component→article mapping ---
    comp_article_map: dict[str, set[str]] = {}
    for link in evidence_links:
        if isinstance(link, dict):
            cid = link.get("component_id", "")
            aid = link.get("article_id", "")
            if cid and aid:
                comp_article_map.setdefault(cid, set()).add(aid)

    # If no evidence links, try to infer from claims + keyword matching
    if not comp_article_map and claims_raw:
        comp_article_map = _infer_component_articles(components_raw, claims_raw)

    # --- grade each component ------------------------------------------------
    for comp in components_raw:
        cid = comp.get("id", "")
        study_set = comp_article_map.get(cid, set())
        component_studies = list(study_set)

        # Filter claims to those from component's articles
        comp_claims = [c for c in claims_raw if _claim_article_id(c) in study_set]

        grade = grade_component(
            component=comp,
            claims=comp_claims,
            method_profiles=method_profiles_raw,
            interrater_data=[],  # retired stage outputs are not required
            robustness_data=robustness_raw,
            component_studies=component_studies,
        )
        grades.append(grade)

    logger.info("GRADE assessment completed for %d components.", len(grades))
    if save and grades:
        results_path, sof_path = save_grade_outputs(config.output_dir, grades)
        logger.info("GRADE results saved: %s, %s", results_path, sof_path)
    return grades
