"""Stage 8 robustness: main entry point.

Split out of stages/s9_robustness.py, which exposes ``run`` lazily to avoid
a circular import (this module imports ``split_half`` from the facade).
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from core.io import atomic_write_json
from process.paths import load_paper_list, resolve_claims_corpus, resolve_validation_dir
from stages._common import announce_no_llm, is_cancelled, make_progress
from stages.s9_robustness import _not_estimable, split_half
from stages.s9_robustness_analyses import (
    condition_sensitivity_analysis,
    condition_sensitivity_summary,
    permutation_test,
    sensitivity_analysis,
)
from stages.s9_robustness_config import (
    RobustnessOutcome,
    _load_condition_config,
    _load_outcome,
    condition_names,
)
from stages.s9_robustness_report import (
    LOW_OUTCOME_COVERAGE,
    _write_network_report,
    _write_report,
)
from stages.s9_robustness_stats import RobustnessCancelled, bootstrap_ci

# Same logger name as the original single module (stages.s9_robustness).
logger = logging.getLogger("stages.s9_robustness")

CANONICAL_RESULTS_NAME = "robustness_results.json"
# Label shuffles behind the network edge-significance threshold.
NETWORK_PERMUTATIONS = 500
# Edges kept in the summary JSON (the full list goes to condition_networks.json).
SUMMARY_TOP_EDGES = 20


# ── Main entry point ──────────────────────────────────────────────────────

def _mark_significant_edges(edges: list[dict], threshold: float) -> list[dict]:
    """Copies of ``edges`` flagged significant when weight exceeds ``threshold``.

    ``threshold`` is the 95th percentile of the permutation null maximum, so
    only weights strictly above it are significant (as the
    ``compute_edge_threshold`` docstring states).  The null maxima are small
    tied integers, and ``>=`` inflated the family-wise error to 0.08-0.11
    (audit 12-M4).
    """
    return [{**edge, "significant": edge["weight"] > threshold} for edge in edges]


@dataclass(frozen=True)
class _RunContext:
    """Everything the individual analyses need from ``run``'s arguments."""

    progress: Callable[[str], None]
    check_cancel: Callable[[], None]
    config: Any
    seed: int
    bootstrap_iterations: int
    split_iterations: int
    permutation_iterations: int
    condition_patterns: dict
    condition_ranking: list[str]
    outcome: RobustnessOutcome
    validation_dir: Path


def _validate_iterations(**counts: Any) -> None:
    """Fail fast on iteration counts that would crash deep in an analysis (12-L4)."""
    for name, value in counts.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _warn_unknown_kwargs(progress: Callable[[str], None], kw: dict,
                         ignored_llm: list[str]) -> None:
    """Name keyword arguments Stage 8 does not know (often a misspelling)."""
    unknown = sorted(set(kw) - set(ignored_llm))
    if unknown:
        progress(
            f"WARNING: Stage 8 ignored unknown argument(s): {', '.join(unknown)}. "
            "Check the spelling (e.g. bootstrap_iterations, split_iterations, "
            "permutation_iterations)."
        )


def _resolve_results_path(config, output_dir, output_path) -> tuple[Path, Path]:
    """``(results JSON, directory for reports and network artifacts)``.

    The GUI selects a JSON file, while legacy script callers select a folder.
    Resolve the parent before mkdir so an existing results file can be rerun.
    """
    if output_path is None and output_dir:
        legacy_path = Path(output_dir).expanduser()
        if legacy_path.suffix.lower() == ".json" and not legacy_path.is_dir():
            output_path = legacy_path
    results_path = Path(output_path).expanduser() if output_path is not None else None
    if results_path is not None and results_path.is_dir():
        raise ValueError(f"Results JSON must be a file, not a directory: {results_path}")
    validation_dir = resolve_validation_dir(
        config, results_path.parent if results_path is not None else output_dir
    )
    return results_path or validation_dir / CANONICAL_RESULTS_NAME, validation_dir


def _canonical_results_path(config, results_path: Path) -> Path | None:
    """The run's canonical results file when ``results_path`` is elsewhere.

    Stage 10, GRADE, the committee brief and the Pipeline Board read only
    ``<data_dir>/validation/robustness_results.json`` (audit 12-M8). Without a
    config there is no run to keep in sync, so nothing is mirrored.
    """
    if config is None:
        return None
    canonical = resolve_validation_dir(config) / CANONICAL_RESULTS_NAME
    if canonical.resolve() == results_path.resolve():
        return None
    return canonical


def _summary_payload(results: dict) -> dict:
    """Results without the full edge list, which is too large for the summary
    JSON (it is saved in condition_networks.json)."""
    network = results.get("network_analysis")
    if not isinstance(network, dict) or "edges" not in network:
        return results
    return {**results, "network_analysis":
            {k: v for k, v in network.items() if k != "edges"}}


def _save_results(results: dict, results_path: Path, config,
                  progress: Callable[[str], None]) -> dict[str, str]:
    """Write the results JSON (and its canonical copy); return the paths written."""
    payload = _summary_payload(results)
    atomic_write_json(results_path, payload)
    progress(f"Results saved: {results_path}")
    saved = {"results_path": str(results_path)}
    canonical = _canonical_results_path(config, results_path)
    if canonical is not None:
        atomic_write_json(canonical, payload)
        progress(f"Canonical copy for Stage 9 / GRADE / Pipeline Board: {canonical}")
        saved["canonical_results_path"] = str(canonical)
    return saved


# ── Outcome coverage (audit 12-M6) ────────────────────────────────────────

def _outcome_coverage(papers: list[dict], outcome: RobustnessOutcome,
                      progress: Callable[[str], None]) -> tuple[int, str]:
    """``(papers with an outcome value, skip reason or "")``, warning as needed."""
    n_papers = len(papers)
    n_with = sum(1 for paper in papers if outcome.has_value(paper))
    progress(f"Papers with a usable '{outcome.field}' value: {n_with} of {n_papers}")
    if n_with == 0:
        reason = (
            f"no paper has a value for outcome field '{outcome.field}' (misnamed "
            "field, or Stage 3 extraction not run); support rates would all be 0"
        )
        progress(f"WARNING: {reason}. Outcome analyses are SKIPPED.")
        return 0, reason
    if n_with / n_papers < LOW_OUTCOME_COVERAGE:
        progress(
            f"WARNING: only {n_with} of {n_papers} papers have a '{outcome.field}' "
            "value; the rest count as non-supporting and deflate every support rate."
        )
    return n_with, ""


def _outcome_skips(reason: str) -> dict[str, dict]:
    """Skipped results for every analysis that counts outcome support."""
    return {
        "split_half": _not_estimable(reason),
        "sensitivity": {"thresholds": {}, "stable": None,
                        "skipped": True, "reason": reason},
        "permutation": {"observed_correlation": None, "p_value": None,
                        "n_permutations": 0, "significant": False,
                        "condition_gradient": {}, "skipped": True, "reason": reason},
        "condition_sensitivity": {"skipped": True, "reason": reason},
    }


# ── Analyses ──────────────────────────────────────────────────────────────

def _run_bootstrap(papers: list[dict], ctx: _RunContext) -> dict:
    ctx.progress("Analysis 1/6: Bootstrap confidence intervals...")
    bs = bootstrap_ci(papers, n_iter=ctx.bootstrap_iterations, seed=ctx.seed,
                      check_cancel=ctx.check_cancel)
    if bs.get("skipped"):
        ctx.progress(f"  WARNING: Bootstrap SKIPPED: {bs.get('reason', '')}")
        return bs
    ctx.progress(
        f"  Bootstrap: mean={bs['mean']:.3f}, "
        f"95% CI=[{bs['ci_lower']:.3f}, {bs['ci_upper']:.3f}] "
        f"({bs['n_scored']}/{bs['n_papers']} papers scored)"
    )
    if bs["n_unscored"] or bs["n_unparseable"]:
        ctx.progress(
            f"  WARNING: {bs['n_unscored']} paper(s) without a relevance_score and "
            f"{bs['n_unparseable']} with an unparseable one were EXCLUDED from the "
            "bootstrap (not counted as 0)."
        )
    return bs


def _format_stat(value) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "n/a"


def _run_split_half(papers: list[dict], ctx: _RunContext) -> dict:
    ctx.progress("Analysis 2/6: Split-half reliability...")
    sh = split_half(
        papers, n_splits=ctx.split_iterations, seed=ctx.seed,
        condition_patterns=ctx.condition_patterns,
        condition_ranking=condition_names(ctx.condition_patterns),
        outcome=ctx.outcome, check_cancel=ctx.check_cancel,
    )
    if sh.get("estimable", False):
        ctx.progress(
            f"  Split-half: Spearman-Brown r={sh['mean_correlation']:.3f}, "
            f"τ={_format_stat(sh.get('kendall_tau'))}, "
            f"ICC={_format_stat(sh.get('icc'))} ({sh['interpretation']})"
        )
    else:
        ctx.progress(f"  WARNING: Split-half NOT ESTIMABLE: {sh.get('note', '')}")
    return sh


def _run_sensitivity(papers: list[dict], ctx: _RunContext) -> dict:
    ctx.progress("Analysis 3/6: Sensitivity analysis (thresholds 1-5)...")
    sa = sensitivity_analysis(papers, outcome=ctx.outcome)
    ctx.progress(f"  Sensitivity: stable={sa['stable']}")
    return sa


def _run_permutation(papers: list[dict], ctx: _RunContext) -> dict:
    ctx.progress("Analysis 4/6: Permutation test (condition gradient)...")
    pt = permutation_test(
        papers, ctx.condition_patterns, ctx.condition_ranking,
        n_perm=ctx.permutation_iterations, seed=ctx.seed,
        outcome=ctx.outcome, check_cancel=ctx.check_cancel,
    )
    if pt.get("skipped"):
        ctx.progress(f"  WARNING: Permutation test SKIPPED: {pt.get('reason', '')}")
    else:
        ctx.progress(
            f"  Permutation: r={pt['observed_correlation']:.4f}, "
            f"p={pt['p_value']:.4f}, significant={pt['significant']}"
        )
    return pt


def _run_condition_sensitivity(papers: list[dict], ctx: _RunContext) -> dict:
    ctx.progress("Analysis 6/6: Co-occurring condition sensitivity...")
    sens = condition_sensitivity_analysis(
        papers, ctx.condition_patterns, condition_names(ctx.condition_patterns),
        outcome=ctx.outcome,
    )
    ctx.progress("  " + condition_sensitivity_summary(sens))
    return sens


def _network_skip_reason(patterns: dict) -> str:
    has_targets = any(values for category, values in patterns.items()
                      if category != "conditions")
    if patterns["conditions"] and has_targets:
        return ""
    return ("Configure graph_analysis conditions and target entities to enable "
            "network analysis.")


def _communities_progress(communities: dict) -> str:
    if communities.get("unavailable"):
        return f"  WARNING: Communities unavailable: {communities['unavailable']}"
    return (f"  Communities: {communities['n_communities']} "
            f"(modularity={communities['modularity']:.3f})")


def _build_network(papers: list[dict], ctx: _RunContext) -> tuple[dict, list[dict]]:
    """Network summary and the full edge list (raises on failure)."""
    from process import graph_analysis

    patterns = graph_analysis.load_patterns(ctx.config)
    entities = graph_analysis.extract_corpus_entities(papers, patterns)
    edges = graph_analysis.build_cooccurrence_edges(papers, patterns, entities=entities)
    threshold = graph_analysis.compute_edge_threshold(
        papers, patterns, n_permutations=NETWORK_PERMUTATIONS, seed=ctx.seed,
        entities=entities, check_cancel=ctx.check_cancel,
    )
    edges = _mark_significant_edges(edges, threshold)
    n_significant = sum(1 for e in edges if e["significant"])
    ctx.progress(f"  Edge threshold: {threshold} ({n_significant}/{len(edges)} significant)")

    communities = graph_analysis.detect_communities(
        [e for e in edges if e["significant"]], seed=ctx.seed)
    ctx.progress(_communities_progress(communities))

    network = {
        "n_edges": len(edges),
        "n_significant_edges": n_significant,
        "edge_threshold": threshold,
        "n_conditions": len({e["source"] for e in edges}),
        "n_regions": len({e["target"] for e in edges}),
        "top_edges": edges[:SUMMARY_TOP_EDGES],
        "edges": edges,
        "communities": communities,
    }
    reason = _network_skip_reason(patterns)
    if reason:
        network = {**network, "skipped": True, "reason": reason}
        ctx.progress(f"  WARNING: Network analysis SKIPPED: {reason}")
    return network, edges


def _save_network(network: dict, edges: list[dict], n_papers: int,
                  ctx: _RunContext) -> dict[str, str]:
    """Write condition_networks.json and its report; return their paths."""
    network_path = ctx.validation_dir / "condition_networks.json"
    atomic_write_json(network_path, {
        "n_papers": n_papers,
        "n_edges": network["n_edges"],
        "n_significant_edges": network["n_significant_edges"],
        "edge_threshold": network["edge_threshold"],
        "conditions": sorted({e["source"] for e in edges}),
        "regions": sorted({e["target"] for e in edges}),
        "edges": edges,
        "communities": network["communities"],
        "skipped": network.get("skipped", False),
        "reason": network.get("reason", ""),
    })
    ctx.progress(f"  Networks: {network['n_edges']} edges, {network['n_conditions']} "
                 f"conditions, {network['n_regions']} target entities")
    ctx.progress(f"  Saved: {network_path}")
    report_path = ctx.validation_dir / "condition_networks_report.md"
    _write_network_report(edges, report_path, n_papers, network.get("reason", ""))
    return {"network_path": str(network_path), "network_report_path": str(report_path)}


def _run_network(papers: list[dict], ctx: _RunContext) -> tuple[dict, dict[str, str]]:
    """``(network_analysis result, artifact paths)``; failures become ``error``."""
    ctx.progress("Analysis 5/6: Condition–entity co-occurrence networks...")
    try:
        network, edges = _build_network(papers, ctx)
        return network, _save_network(network, edges, len(papers), ctx)
    except RobustnessCancelled:
        raise
    except Exception as exc:
        logger.warning("Network analysis failed: %s", exc, exc_info=True)
        ctx.progress(f"  ERROR: Network analysis FAILED ({type(exc).__name__}: {exc}); "
                     f"the other analyses continue, but there is no network result.")
        return {"error": str(exc)}, {}


def _run_analyses(papers: list[dict], ctx: _RunContext, outcome_reason: str) -> dict:
    """All six analyses in order; raises RobustnessCancelled on cancel."""
    skips = _outcome_skips(outcome_reason) if outcome_reason else {}
    results: dict[str, Any] = {}
    ctx.check_cancel()
    results["bootstrap"] = _run_bootstrap(papers, ctx)
    ctx.check_cancel()
    results["split_half"] = skips.get("split_half") or _run_split_half(papers, ctx)
    ctx.check_cancel()
    results["sensitivity"] = skips.get("sensitivity") or _run_sensitivity(papers, ctx)
    ctx.check_cancel()
    results["permutation"] = skips.get("permutation") or _run_permutation(papers, ctx)
    ctx.check_cancel()
    results["network_analysis"], paths = _run_network(papers, ctx)
    results.update(paths)
    ctx.check_cancel()
    results["condition_sensitivity"] = (skips.get("condition_sensitivity")
                                        or _run_condition_sensitivity(papers, ctx))
    return results


def _announce_conditions(ctx: _RunContext) -> None:
    names = condition_names(ctx.condition_patterns)
    if names:
        ctx.progress(f"Conditions: {names}")
    else:
        ctx.progress("No conditions configured; condition analyses will be skipped.")
    if ctx.condition_ranking:
        ctx.progress(f"Condition ranking: {ctx.condition_ranking}")
    else:
        ctx.progress("No ordered co_occurring_conditions list; the gradient test "
                     "will be skipped.")


def _empty_corpus_result(corpus_path: Path, results_path: Path, config,
                         progress: Callable[[str], None]) -> dict:
    progress(
        f"WARNING: {corpus_path} contains 0 papers. NOTHING WAS ANALYSED; "
        f"all six robustness analyses are empty. Check that the input file "
        f"is the scored/extracted corpus, not an empty or wrong file."
    )
    empty_result = {
        "n_papers": 0,
        "bootstrap": {},
        "split_half": {},
        "sensitivity": {},
        "permutation": {},
        "network_analysis": {},
    }
    return {**empty_result, **_save_results(empty_result, results_path, config, progress)}


def _report_skips(results: dict, progress: Callable[[str], None]) -> None:
    skipped = skipped_analyses(results)
    progress("═" * 60)
    if skipped:
        progress(
            f"STAGE 8 FINISHED WITH {len(skipped)} OF 6 ANALYSES SKIPPED OR "
            f"FAILED — read before citing these results:"
        )
        for name, why in skipped:
            progress(f"  - {name}: {why}")
    else:
        progress("STAGE 8 COMPLETE: all 6 analyses produced results")
    progress("═" * 60)


def run(
    claims_path: str | None = None,
    output_dir: str | None = None,
    config=None,
    seed: int = 42,
    bootstrap_iterations: int = 10000,
    split_iterations: int = 1000,
    permutation_iterations: int = 10000,
    cancel_event=None,
    progress_callback=None,
    output_path: str | Path | None = None,
    **kw,
) -> dict:
    """Run all 6 robustness analyses.

    Parameters
    ----------
    claims_path : str, optional
        Explicit path to claims JSON.  If not provided, standard locations
        under ``config.data_dir`` are searched.
    output_dir : str, optional
        Override for output directory. JSON file paths from older GUI callers
        are accepted as an alias for ``output_path``.
    output_path : str or Path, optional
        Exact results JSON file; takes precedence over ``output_dir``.
        Reports and network artifacts are saved beside this file.  When it is
        not the run's canonical ``validation/robustness_results.json``, that
        file is refreshed too so downstream stages see this run.
    config : Config, optional
        Pipeline configuration object.
    seed : int
        Random seed for reproducibility.
    bootstrap_iterations, split_iterations, permutation_iterations : int
        Iteration counts for each analysis (each at least 1).
    cancel_event : threading.Event or similar, optional
        Set to cancel the run; honoured inside the long loops too.
    progress_callback : callable, optional
        Called with progress message strings.
    """
    _validate_iterations(bootstrap_iterations=bootstrap_iterations,
                         split_iterations=split_iterations,
                         permutation_iterations=permutation_iterations)
    _cancelled = partial(is_cancelled, cancel_event)
    _progress = make_progress(logger, progress_callback, log_with_callback=False)

    def _check_cancel() -> None:
        if _cancelled():
            raise RobustnessCancelled()

    _progress("═" * 60)
    _progress("STAGE 8: ROBUSTNESS ANALYSES")
    _progress("═" * 60)
    _warn_unknown_kwargs(_progress, kw, announce_no_llm(_progress, "Stage 8 (Robustness)", kw))

    results_path, validation_dir = _resolve_results_path(config, output_dir, output_path)
    corpus_path = resolve_claims_corpus(config, claims_path)
    if corpus_path is None:
        _progress("ERROR: No claims corpus found. Run earlier stages first.")
        return {"error": "No claims corpus found"}

    papers = load_paper_list(corpus_path)
    n_papers = len(papers)
    _progress(f"Loaded {n_papers} papers from {corpus_path}")

    condition_patterns, condition_ranking = _load_condition_config(config)
    outcome = _load_outcome(config)
    ctx = _RunContext(
        progress=_progress, check_cancel=_check_cancel, config=config, seed=seed,
        bootstrap_iterations=bootstrap_iterations, split_iterations=split_iterations,
        permutation_iterations=permutation_iterations,
        condition_patterns=condition_patterns, condition_ranking=condition_ranking,
        outcome=outcome, validation_dir=validation_dir,
    )
    _announce_conditions(ctx)
    _progress(f"Outcome measured: {outcome.field}")

    if n_papers == 0:
        return _empty_corpus_result(corpus_path, results_path, config, _progress)

    n_with_outcome, outcome_reason = _outcome_coverage(papers, outcome, _progress)
    try:
        analyses = _run_analyses(papers, ctx, outcome_reason)
    except RobustnessCancelled:
        _progress("Cancelled.")
        return {"cancelled": True}

    results: dict[str, Any] = {"n_papers": n_papers, "outcome_field": outcome.field,
                               "n_with_outcome": n_with_outcome, **analyses}
    results.update(_save_results(results, results_path, config, _progress))

    report_path = validation_dir / "robustness_report.md"
    _write_report(results, report_path, n_papers, condition_ranking)
    results["report_path"] = str(report_path)
    _report_skips(results, _progress)
    return results


def skipped_analyses(results: dict) -> list[tuple[str, str]]:
    """List ``(analysis, reason)`` for every analysis that produced no result."""
    out: list[tuple[str, str]] = []
    bs = results.get("bootstrap") or {}
    if not bs or bs.get("skipped"):
        out.append(("Bootstrap CI", bs.get("reason", "no result")))
    sh = results.get("split_half") or {}
    if not sh.get("estimable", False):
        out.append(("Split-half reliability", sh.get("note", "no result")))
    sa = results.get("sensitivity") or {}
    if not sa or sa.get("skipped"):
        out.append(("Sensitivity analysis", sa.get("reason", "no result")))
    pt = results.get("permutation") or {}
    if not pt or pt.get("skipped"):
        out.append(("Permutation test", pt.get("reason", "no result")))
    net = results.get("network_analysis") or {}
    if not net or net.get("skipped") or net.get("error"):
        out.append(("Network analysis",
                    net.get("error") or net.get("reason", "no result")))
    cs = results.get("condition_sensitivity") or {}
    if not cs or cs.get("skipped"):
        out.append(("Condition sensitivity", cs.get("reason", "no result")))
    return out
