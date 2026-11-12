"""Corpus-wide meta-analysis with explicit compatible synthesis groups."""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import CancelledError
from dataclasses import asdict, replace
import csv
import io
import json
import logging
import math

from core.io import atomic_write_json, atomic_write_text
from process.llm_budget import resolve_model_capacity
from process.meta_analysis_bias import run_bias_tests
from process.meta_analysis_conversions import harmonize_to_smd
from process.meta_analysis_groups import REVIEW_FILE, load_decisions, write_review
from process.meta_analysis_inputs import (
    evaluate_metrics, prepare_corpus_metrics, statistics_funnel,
)
from process.meta_analysis_statistics_audit import (
    STATISTICS_AUDIT_FILE, funnel_lines, write_statistics_audit,
)
from process.meta_analysis_model_groups import (
    GROUPING_REVIEW_FILE, PROPOSALS_FILE, propose_groups, save_grouping_review,
    save_proposals, with_proposals,
)
from process.meta_analysis_models import PooledEstimate
from process.meta_analysis_plots import create_forest_plot_figure, create_funnel_plot_figure
from process.meta_analysis_pooling import (
    benjamini_hochberg, pool_random_effects, resolve_meta_analysis_input,
)
from process.term_normalization import inspect_statistical_normalization

logger = logging.getLogger("process.meta_analysis")

# Pooled log ratios and Fisher's z are also reported on their natural scale.
BACK_TRANSFORMS = {
    **{kind: (kind.upper(), math.exp) for kind in ("or", "aor", "rr", "hr")},
    "r": ("r", math.tanh),
}
RESEARCHER_GROUP = "researcher_group"
SMD_SPLIT_REASON = (
    "Cohen's d and Hedges' g results for the same outcome are pooled separately "
    "(d lacks a usable sample size for the small-sample correction)")
COMPOSITE_NOTE = "mean of {k} estimates from this paper in this group (correlation assumed 1)"
COMPOSITE_REASON = "Several estimates in one group were combined into one study estimate"


def _group_key(study):
    # An assigned group is the researcher's or the model's harmonized
    # outcome/comparison decision. Otherwise use the extracted labels
    # conservatively; never guess that different outcomes estimate the same
    # quantity just because both are ORs.
    if study.group_source in ("researcher", "model_validated"):
        # Mixed effect types in an assigned group are put on one scale.
        return RESEARCHER_GROUP, study.analysis_group
    if study.analysis_group:
        return study.effect_type, study.analysis_group
    if all((study.outcome.strip(), study.comparison.strip(), study.population.strip())):
        # Stage 2c owns canonicalization and records the raw mapping.  Exact
        # labels are used here so a stale input cannot be silently merged during
        # scientific pooling.
        return (study.effect_type, study.outcome,
                study.comparison, study.population)
    return None


def _harmonized(effects, paper_audit):
    """A researcher group with several effect types, on the SMD scale."""
    if len({s.analysis_scale for s in effects}) == 1:
        return effects
    harmonized = []
    for study in effects:
        converted, why = harmonize_to_smd(study)
        if converted is None:
            paper_audit[study.article_id]["analysis_reasons"].add(
                f"Left out of group '{study.analysis_group}': {why}")
        else:
            harmonized.append(converted)
    return harmonized


def _composite(estimates):
    """One study estimate from several in a group: their mean.

    Their correlation is unknown, so it is taken as 1 (Borenstein et al. 2009,
    ch. 24): the SE is the mean SE, never smaller than one estimate's.
    """
    k = len(estimates)
    effect = sum(s.effect_size for s in estimates) / k
    se = sum(s.standard_error for s in estimates) / k
    first = estimates[0]
    return replace(first, effect_size=effect, standard_error=se, variance=se * se,
                   ci_lower=effect - 1.96 * se, ci_upper=effect + 1.96 * se,
                   sample_size=max(s.sample_size for s in estimates),
                   assumptions=(*first.assumptions, COMPOSITE_NOTE.format(k=k)))


def _independent_estimates(effects, paper_audit):
    """One estimate per article; several distinct ones are combined (see ``_composite``)."""
    by_article = defaultdict(list)
    for study in effects:
        # Repeated extraction of the same estimate is one result.
        fingerprint = (study.effect_size, study.variance, study.sample_size)
        if all((s.effect_size, s.variance, s.sample_size) != fingerprint
               for s in by_article[study.article_id]):
            by_article[study.article_id].append(study)
    independent = []
    for aid, estimates in by_article.items():
        if len(estimates) > 1:
            paper_audit[aid]["analysis_reasons"].add(COMPOSITE_REASON)
            independent.append(_composite(estimates))
        else:
            independent.append(estimates[0])
    return independent


def back_transform(effect_type, pooled):
    """Pooled ratio / correlation on its natural scale (exp or tanh); None otherwise."""
    if pooled is None or effect_type not in BACK_TRANSFORMS:
        return None
    measure, transform = BACK_TRANSFORMS[effect_type]
    return {"measure": measure,
            **{key: round(transform(getattr(pooled, attr)), 6) for key, attr in (
                ("effect", "pooled_effect"), ("ci_lower", "ci_lower"), ("ci_upper", "ci_upper"))}}


def _flag_smd_splits(candidates, paper_audit):
    """Cohen's d (no usable n) and Hedges' g of one outcome land in separate groups."""
    for key, effects in candidates.items():
        if key[0] == "d" and ("g", *key[1:]) in candidates:
            for study in effects + candidates[("g", *key[1:])]:
                paper_audit[study.article_id]["analysis_reasons"].add(SMD_SPLIT_REASON)


def _group_studies(studies, paper_audit):
    """Pool each compatible group.

    Returns (groups, pooled article ids, {id(input study): weighted copy}).
    Pooling never modifies the input studies; each group's ``studies`` are
    the pooled copies carrying random-effects weights (audit 15-M1).
    """
    candidates = defaultdict(list)
    for study in studies:
        key = _group_key(study)
        if key is None:
            paper_audit[study.article_id]["analysis_reasons"].add(
                "Outcome, comparison and population need assignment")
        else:
            candidates[key].append(study)
    _flag_smd_splits(candidates, paper_audit)
    groups, pooled_ids, weighted = [], set(), {}
    for key, effects in candidates.items():
        if key[0] == RESEARCHER_GROUP:
            effects = _harmonized(effects, paper_audit)
            if not effects:
                continue
        independent = _independent_estimates(effects, paper_audit)
        example = effects[0]
        label = example.analysis_group or (
            f"{example.outcome}; {example.comparison}; {example.population}")
        group = {"label": label, "effect_type": example.effect_type,
                 "source": example.group_source or "labels",
                 "analysis_scale": example.analysis_scale,
                 "n_studies": len(independent), "pooled": None,
                 "studies": independent, "reason": "", "bias": None,
                 "back_transformed": None}
        if len(independent) >= 2:
            pooled = pool_random_effects(independent)
            weighted.update(zip(map(id, independent), pooled.studies))
            group.update(pooled=pooled, studies=pooled.studies,
                         bias=run_bias_tests(pooled.studies),
                         back_transformed=back_transform(example.effect_type, pooled))
            pooled_ids.update(s.article_id for s in independent)
        else:
            group["reason"] = "Fewer than two independent studies with compatible results"
            for study in independent:
                paper_audit[study.article_id]["analysis_reasons"].add(group["reason"])
        groups.append(group)
    return groups, pooled_ids, weighted


def coverage_lines(results, *, include_reasons=True):
    coverage = results.get("coverage", {})
    lines = [
        f"Papers assessed: {coverage.get('n_corpus_papers', 0)}",
        f"Papers with statistical candidates: {coverage.get('n_papers_with_statistics', 0)}",
        f"Statistical candidates: {coverage.get('n_statistical_results', 0)}",
        f"Papers with calculable linked effects: {coverage.get('n_papers_with_calculable_effects', 0)}",
        f"Papers contributing to pooled estimates: {results.get('n_studies', 0)}",
        f"Compatible groups pooled: {sum(g['pooled'] is not None for g in results.get('groups', []))}",
    ]
    if "n_fulltext_papers" in coverage:
        lines.insert(1, f"Text coverage: {coverage['n_fulltext_papers']} full text; "
                     f"{coverage['n_abstract_only_papers']} abstract only")
    if "n_rechecked" in coverage:
        lines.append(f"Statistics re-checked by the model: {coverage['n_rechecked']} "
                     f"({coverage['n_recheck_failed']} still without a verdict)")
        bundles = (coverage.get("model_recheck") or {}).get("bundles") or {}
        planning = bundles.get("planning") or {}
        if bundles:
            lines.append(
                "Model re-check token actions: "
                f"{bundles.get('requests', 0)} requests; "
                f"{bundles.get('splits', 0)} automatic splits; "
                f"{bundles.get('recovered_statistics', 0)} statistics recovered; "
                f"{bundles.get('terminal_failures', 0)} terminal failures; "
                f"mode {planning.get('mode', 'not recorded')} "
                f"({planning.get('reason', 'no reason recorded')})"
            )
    if coverage.get("n_unverified_statistics"):
        lines.append(f"Statistics without a context verdict (no outcome label, cannot be grouped): "
                     f"{coverage['n_unverified_statistics']}")
    term_status = coverage.get("term_normalization") or {}
    if term_status:
        status = term_status.get("status")
        if status == "no_structured_labels":
            lines.append(
                "Structured statistical labels: none were available; no label "
                "canonicalization or label-based grouping was possible"
            )
        elif status == "canonical":
            lines.append(
                "Structured statistical labels: canonical Stage 2c input "
                f"({term_status.get('marked_statistics', 0)}/"
                f"{term_status.get('labeled_statistics', 0)} labeled statistics carry "
                f"{term_status.get('policy_id', 'the recorded policy')})"
            )
        else:
            if status == "configuration_mismatch":
                detail = (
                    f"{term_status.get('config_mismatches', 0)} labeled statistic marker(s) "
                    "do not match the active project vocabulary"
                )
            else:
                detail = (
                    f"{term_status.get('unmarked_statistics', 0)} labeled statistics "
                    "lack the current marker"
                )
            lines.append(
                "Structured statistical labels: "
                f"{status or 'unknown'} input; {detail}. "
                "They are grouped exactly as saved; meta-analysis performs no downstream "
                "normalization. Rerun Stage 2c for canonical labels and raw-value provenance "
                "under the active project vocabulary."
            )
    grouping = results.get("model_grouping")
    if grouping:
        lines.append(
            f"Model semantic grouping: {grouping.get('candidates_returned', 0)} candidates reviewed; "
            f"{len(grouping['groups'])} passed all gates "
            f"({grouping['requests']} requests, status {grouping['status']})"
        )
        lines.append(
            f"Grouping coverage: {grouping.get('items_considered', 0)} estimate items; "
            f"{grouping.get('items_without_any_candidate', 0)} never placed in a candidate; "
            f"{grouping.get('items_explicitly_unmatched_in_all_views', 0)} explicitly unmatched "
            "after complete cross-paper opportunity coverage; "
            f"{grouping.get('items_unresolved_due_to_incomplete_comparison_coverage', 0)} "
            "unresolved because split or failed views left comparison opportunities unseen"
        )
        lines += [f"Model grouping error: {error}" for error in grouping["errors"]]
    lines += funnel_lines(coverage.get("statistics_funnel"))
    reasons = coverage.get("analysis_reasons", {}) if include_reasons else {}
    for reason, count in reasons.items():
        lines.append(f"{reason}: {count} papers")
    return lines


def groups_for_report(groups):
    """Stable report order: completed pooled analyses before non-pooled groups.

    This is deliberately presentation-only.  The analysis and serialized group
    collection retain their original order so saved identities, review choices,
    FDR keys, and plot selection are not changed merely to improve readability.
    Eligibility policy stays in the analysis: the renderer uses only the
    resulting ``pooled`` state and never reimplements a study-count threshold.
    """
    groups = list(groups or [])
    return [group for group in groups if group.get("pooled") is not None] + [
        group for group in groups if group.get("pooled") is None
    ]


def _save(results, config, studies):
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    from process.meta_analysis_narrative import save_evidence_report
    narrative = save_evidence_report(results, config)
    grouping_review_path = None
    if results.get("model_grouping") is not None:
        grouping_review_path = save_grouping_review(
            output / "meta_analysis_evidence" / GROUPING_REVIEW_FILE,
            results["model_grouping"],
        )
        results["model_grouping_review_path"] = str(grouping_review_path)
    path = output / "meta_analysis_results.json"
    serial = {key: results[key] for key in (
        "input_source", "input_path", "n_studies", "coverage", "skipped", "reason")}
    serial["model_grouping"] = results.get("model_grouping")
    serial["model_grouping_review_path"] = results.get("model_grouping_review_path")
    serial["narrative_sources_path"] = results["narrative_sources_path"]
    serial["suggested_groups"] = results["evidence_synthesis"]["suggested_groups"]
    serial["groups"] = [
        {**{k: v for k, v in group.items() if k not in {"pooled", "studies", "bias"}},
         "pooled": asdict(group["pooled"]) if group["pooled"] else None,
         "bias": asdict(group["bias"]) if group["bias"] else None,
         "studies": [asdict(s) for s in group["studies"]]}
        for group in results["groups"]]
    serial["studies"] = [asdict(s) for s in studies]
    serial["fdr_by_group"] = results["fdr_by_group"]
    # Preserve the established single-group schema; never manufacture a grand
    # pooled effect across distinct outcome or measure groups.
    if results["pooled"]:
        serial.update({k: v for k, v in asdict(results["pooled"]).items()
                       if k not in {"studies", "n_studies"}})
    atomic_write_json(path, serial)
    results["save_path"] = str(path)
    csv_path = output / "meta_analysis_paper_audit.csv"
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["article_id", "title", "n_statistical_results",
                                              "n_calculable_effects", "pooled", "reasons"])
    writer.writeheader()
    writer.writerows(results["paper_audit"])
    atomic_write_text(csv_path, buffer.getvalue())
    results["audit_path"] = str(csv_path)
    stats_path = output / STATISTICS_AUDIT_FILE
    write_statistics_audit(stats_path, results.get("statistic_rows", []))
    results["statistics_audit_path"] = str(stats_path)
    review_path = write_review(output / "meta_analysis_evidence" / REVIEW_FILE,
                               results.get("statistic_rows", []), results.get("decisions", {}),
                               results.get("model_groups"))
    results["effects_review_path"] = str(review_path)
    report_path = output / "meta_analysis_report.md"
    lines = ["# Meta-analysis", "", *[f"- {line}" for line in coverage_lines(results, include_reasons=False)], ""]
    if results["reason"]:
        lines += [results["reason"], ""]
    if results["groups"]:
        lines += [
            "Groups that produced a pooled estimate are reported first; groups not "
            "pooled are listed afterward with the analysis reason.",
            "",
        ]
    for group in groups_for_report(results["groups"]):
        source = (
            " Model group passed the title/research-question gate and every recorded "
            "scientific compatibility gate."
            if group.get("source") == "model_validated" else ""
        )
        lines += [f"## {group['label']}", "", f"Measure: {group['analysis_scale']}; "
                  f"independent studies: {group['n_studies']}.{source}", ""]
        pooled = group["pooled"]
        if pooled:
            lines += [f"Pooled estimate: {pooled.pooled_effect:.6g}; "
                      f"95% CI [{pooled.ci_lower:.6g}, {pooled.ci_upper:.6g}]; "
                      f"I² = {pooled.i_squared:.1f}%.", ""]
            back = group.get("back_transformed")
            if back:
                lines += [f"As {back['measure']}: {back['effect']:.6g}; "
                          f"95% CI [{back['ci_lower']:.6g}, {back['ci_upper']:.6g}].", ""]
        else:
            lines += [group["reason"] + ".", ""]
    if grouping_review_path is not None:
        relative_grouping_review = f"meta_analysis_evidence/{grouping_review_path.name}"
        lines += [
            "Model semantic grouping audit (paper titles, extracted statistics, every matching "
            f"characteristic, and every rejection): [{grouping_review_path.name}]"
            f"({relative_grouping_review}).",
            "",
        ]
    lines += [f"Paper-by-paper accounting: [{csv_path.name}]({csv_path.name}).", "",
              f"Statistic-by-statistic accounting (what was found, used or passed up, and why): "
              f"[{stats_path.name}]({stats_path.name}).", "",
              "Candidate extraction uses every retained paper and its saved full text when available. "
              "A paper count is not a count of compatible effect estimates. "
              "Missing uncertainty or interpretation stays unresolved.", ""]
    lines.append(narrative)
    atomic_write_text(report_path, "\n".join(lines))
    results["report_path"] = str(report_path)


def _read_explicit_input(path):
    """Load an explicitly selected input; a missing or unreadable file is an error.

    An empty analysis would otherwise overwrite earlier results (audit 15-L5).
    """
    if not path.exists():
        raise FileNotFoundError(f"Meta-analysis input not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Meta-analysis input {path} is not valid UTF-8 JSON: {exc}") from exc


def add_result_figures(results):
    """Draw the forest and funnel figures of the first pooled group in place.

    Shows a placeholder when no group pooled. The figures use pyplot, so a GUI
    that runs the analysis in a thread calls this on its own thread afterwards.
    """
    completed = [g for g in results["groups"] if g["pooled"] is not None]
    displayed = completed[0] if completed else None
    results["forest_fig"] = create_forest_plot_figure(
        displayed["pooled"] if displayed else PooledEstimate(),
        title=displayed["label"] if displayed else "No compatible group ready to pool")
    results["funnel_fig"] = create_funnel_plot_figure(
        displayed["studies"] if displayed else [],
        displayed["pooled"].pooled_effect if displayed else 0,
        displayed["bias"] if displayed else None)


def run_full_analysis(config, *, input_source="stage2c", metrics_path=None,
                      with_figures=True, use_model=False, context_tokens=0,
                      progress=None, cancel_event=None):
    """Assess the whole corpus and pool separately within compatible groups.

    ``with_figures=False`` skips the forest/funnel figures (see ``add_result_figures``).
    ``use_model`` asks the configured model about statistics without a context
    verdict and for pooling groups across papers. A positive ``context_tokens``
    value is an explicit per-run window; ``0``/``None`` resolves the configured
    or live loaded-model capacity (see ``meta_analysis_model_groups``).
    """
    path, source = resolve_meta_analysis_input(
        config, input_source=input_source, metrics_path=metrics_path)
    capacity = None
    effective_context_tokens = context_tokens
    if use_model:
        capacity = resolve_model_capacity(
            config,
            requested_context_tokens=(
                context_tokens if context_tokens and context_tokens > 0 else None
            ),
            model=config.llm_model,
        )
        effective_context_tokens = capacity.context_window_tokens
    if source == "stage2c":
        raw, coverage = prepare_corpus_metrics(
            config, path, recheck=use_model, context_tokens=effective_context_tokens,
            progress=progress, cancel_event=cancel_event)
    else:
        raw, coverage = _read_explicit_input(path), {}
    term_status = inspect_statistical_normalization(raw, config)
    coverage["term_normalization"] = term_status
    if term_status["status"] not in {"canonical", "no_structured_labels"}:
        message = (
            f"Statistical-label input is {term_status['status']}: "
            f"{term_status['unmarked_statistics']} labeled statistic(s) lack a matching "
            "Stage 2c normalization marker for the active project vocabulary. Meta-analysis "
            "will use every label exactly as saved and will not normalize during pooling; "
            "rerun Stage 2c for canonical labels."
        )
        logger.warning(message)
        if progress:
            progress(f"WARNING: {message}")
    decisions = load_decisions(config) if getattr(config, "claims_dir", None) else {}
    allow_paper_n = bool(getattr(config, "meta_analysis_use_paper_sample_size", False))
    studies, parsed_audit, statistic_rows = evaluate_metrics(
        raw, allow_paper_n=allow_paper_n, decisions=decisions)
    proposals, grouping = {}, None
    if use_model:
        proposals, grouping = propose_groups(
            config, statistic_rows, context_tokens=effective_context_tokens,
            progress=progress, cancel_event=cancel_event)
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledError("Meta-analysis cancelled; earlier results were kept")
    if use_model:
        save_proposals(config.output_dir / "meta_analysis_evidence" / PROPOSALS_FILE,
                       proposals, grouping)
        if proposals:
            studies, parsed_audit, statistic_rows = evaluate_metrics(
                raw, allow_paper_n=allow_paper_n, decisions=with_proposals(decisions, proposals))
    papers = coverage.pop("papers", None)
    if papers is None:
        papers = [{"article_id": r["article_id"], "title": r["title"], "n_statistical_results": 0}
                  for r in parsed_audit]
    audit = {row["article_id"]: {**row, "n_calculable_effects": 0,
                               "analysis_reasons": set()} for row in papers}
    for row in parsed_audit:
        current = audit[row["article_id"]]
        current["n_calculable_effects"] = row["n_calculable_effects"]
        current["analysis_reasons"].update(row["exclusion_reasons"])
    for row in audit.values():
        if not row["n_calculable_effects"] and not row["analysis_reasons"]:
            row["analysis_reasons"].add("No exported calculable effect; see extraction review")
    groups, pooled_ids, weighted = _group_studies(studies, audit)
    completed = [g for g in groups if g["pooled"] is not None]
    coverage.setdefault("n_corpus_papers", len(audit))
    if capacity is not None:
        coverage["model_capacity"] = capacity.as_dict()
    coverage.setdefault("n_papers_with_statistics", len(raw))
    coverage.setdefault("n_statistical_results", sum(
        len(entry.get("statistical_tests", [])) or len(entry.get("effect_sizes", [])) for entry in raw))
    coverage["n_papers_with_calculable_effects"] = len({s.article_id for s in studies})
    coverage["n_calculable_effects"] = len(studies)
    coverage["statistics_funnel"] = statistics_funnel(statistic_rows)
    coverage["analysis_reasons"] = dict(Counter(
        reason for row in audit.values() for reason in row["analysis_reasons"]))
    results = {
        "input_source": source, "input_path": coverage.get("prepared_input_path", str(path)),
        "n_studies": len(pooled_ids), "groups": groups, "coverage": coverage,
        "pooled": completed[0]["pooled"] if len(completed) == 1 else None,
        "bias": completed[0]["bias"] if len(completed) == 1 else None,
        "skipped": not completed,
        "reason": "" if completed else "No compatible group currently has two independent, calculable study estimates.",
        "paper_audit": [{"article_id": aid, "title": row["title"],
                         "n_statistical_results": row["n_statistical_results"],
                         "n_calculable_effects": row["n_calculable_effects"],
                         "pooled": aid in pooled_ids,
                         "reasons": "; ".join(sorted(row["analysis_reasons"]))}
                        for aid, row in audit.items()],
    }
    results["fdr_by_group"] = benjamini_hochberg(
        {str(i): group["pooled"].p_value for i, group in enumerate(groups) if group["pooled"]})
    if with_figures:
        add_result_figures(results)
    results["statistic_rows"] = statistic_rows
    results["decisions"] = decisions
    results["model_groups"] = proposals
    results["model_grouping"] = grouping
    _save(results, config, [weighted.get(id(s), s) for s in studies])
    return results
