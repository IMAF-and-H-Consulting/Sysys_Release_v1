"""Plain-language summaries of Stage 8 / Stage 9 results for the GUI.

Pure functions (no Qt) so the done-handlers in ``gui/main_window.py`` stay
short and the wording is unit-testable. Each summary is a list of
``(level, message)`` pairs for ``MainWindow._log`` plus a one-line status.
The point is to never report "complete" when a stage errored, was empty, or
skipped the analyses a reader would rely on.
"""
from __future__ import annotations

from dataclasses import dataclass, field

NO_LLM_LOG_LINE = (
    "This stage makes NO LLM calls: it is local statistics on files already "
    "on disk. No provider (Ollama, OpenAI, Anthropic, custom endpoint) is "
    "contacted."
)


@dataclass(frozen=True)
class StageSummary:
    """What to log, what to show in the status label, and the board state."""

    lines: list[tuple[str, str]] = field(default_factory=list)
    status: str = ""
    board_state: str = "done"
    board_summary: str = ""


def summarize_robustness(result: object) -> StageSummary:
    """Summarise a Stage 8 result dict (or anything the worker returned)."""
    if not isinstance(result, dict):
        return StageSummary(
            lines=[("ERROR", f"Stage 8 returned an unexpected result: {result!r}")],
            status="Stage 8 FAILED: unexpected result",
            board_state="failed",
            board_summary="Unexpected result",
        )
    if result.get("error"):
        return StageSummary(
            lines=[("ERROR", f"Stage 8 FAILED: {result['error']}. Nothing was analysed.")],
            status=f"Stage 8 FAILED: {result['error']}",
            board_state="failed",
            board_summary=str(result["error"]),
        )
    if result.get("cancelled"):
        return StageSummary(
            lines=[("WARNING", "Stage 8 was cancelled; results are incomplete.")],
            status="Stage 8 cancelled",
            board_state="pending",
            board_summary="Cancelled",
        )

    from stages.s9_robustness_run import skipped_analyses

    n_papers = result.get("n_papers", 0)
    lines: list[tuple[str, str]] = []
    if not n_papers:
        lines.append(("WARNING", "Stage 8 analysed 0 papers. NO robustness results exist."))
        return StageSummary(
            lines=lines,
            status="Stage 8 ran on an EMPTY corpus; nothing analysed",
            board_state="done",
            board_summary="0 papers: nothing analysed",
        )

    lines += _robustness_metric_lines(result)
    skipped = skipped_analyses(result)
    for name, why in skipped:
        lines.append(("WARNING", f"{name} SKIPPED: {why}"))
    computed = 6 - len(skipped)
    if skipped:
        status = (
            f"Stage 8 done: {n_papers} papers, only {computed}/6 analyses "
            f"produced results ({len(skipped)} skipped; see log)"
        )
    else:
        status = f"Stage 8 done: {n_papers} papers, all 6 analyses produced results"
    report = result.get("report_path", "")
    if report:
        lines.append(("INFO", f"  Report: {report}"))
    return StageSummary(
        lines=lines,
        status=status,
        board_state="done",
        board_summary=f"{n_papers} papers, {computed}/6 analyses",
    )


def _robustness_metric_lines(result: dict) -> list[tuple[str, str]]:
    lines: list[tuple[str, str]] = []
    bs = result.get("bootstrap") or {}
    if bs and not bs.get("skipped"):
        lines.append(("INFO",
                      f"Bootstrap CI: mean={bs['mean']:.3f}, "
                      f"95% CI=[{bs['ci_lower']:.3f}, {bs['ci_upper']:.3f}]"))
    sh = result.get("split_half") or {}
    if sh.get("estimable"):
        lines.append(("INFO",
                      f"Split-half: r={sh.get('mean_correlation', 0):.3f}, "
                      f"ICC={_optional_stat(sh.get('icc'))} "
                      f"({sh.get('interpretation', '?')})"))
    sensitivity = _sensitivity_line(result.get("sensitivity") or {})
    if sensitivity:
        lines.append(("INFO", sensitivity))
    pt = result.get("permutation") or {}
    if pt and not pt.get("skipped"):
        lines.append(("INFO",
                      f"Permutation: r={pt['observed_correlation']:.4f}, "
                      f"p={pt['p_value']:.4f}, "
                      f"significant={pt.get('significant', False)}"))
    sens = result.get("condition_sensitivity") or {}
    if sens and not sens.get("skipped"):
        from stages.s9_robustness import condition_sensitivity_summary
        lines.append(("INFO", condition_sensitivity_summary(sens)))
    return lines


def _optional_stat(value) -> str:
    """A statistic to 3 decimals, or "n/a" when it could not be computed."""
    return "n/a" if value is None else f"{value:.3f}"


def _sensitivity_line(sa: dict) -> str:
    """Sensitivity verdict; "" when skipped (the skip warning says why)."""
    if not sa or sa.get("skipped"):
        return ""
    if sa.get("stable") is None:
        return "Sensitivity: not evaluable (no relevance threshold has papers)"
    return f"Sensitivity: stable={sa['stable']}"


def summarize_verify(result: object) -> StageSummary:
    """Summarise a Stage 9 result dict (or anything the worker returned)."""
    if not isinstance(result, dict):
        return StageSummary(
            lines=[("ERROR", f"Stage 9 returned an unexpected result: {result!r}")],
            status="Stage 9 FAILED: unexpected result",
            board_state="failed",
            board_summary="Unexpected result",
        )
    if result.get("error"):
        return StageSummary(
            lines=[("ERROR", f"Stage 9 FAILED: {result['error']}. Nothing was verified.")],
            status=f"Stage 9 FAILED: {result['error']}",
            board_state="failed",
            board_summary=str(result["error"]),
        )
    if result.get("cancelled"):
        return StageSummary(
            lines=[("WARNING", "Stage 9 was cancelled; nothing was saved.")],
            status="Stage 9 cancelled",
            board_state="failed",
            board_summary="Cancelled",
        )

    if result.get("review_report"):
        n = result["n_papers"]
        warnings = result.get("warnings", [])
        status = (f"Stage 9: final review of {n} papers, {result['n_findings']} saved findings, "
                  f"{result['n_themes']} themes; internal checks passed")
        return StageSummary(lines=[("INFO", status)] + [("WARNING", w) for w in warnings],
                            status=status, board_state="done",
                            board_summary=f"Final review: {n} papers")

    summary = result.get("summary", {})
    total = summary.get("total_checks", 0)
    passed = summary.get("passed", 0)
    failed = summary.get("failed", 0)
    warnings = summary.get("warnings", 0)
    lines: list[tuple[str, str]] = []

    if result.get("warning"):
        lines.append(("WARNING", f"Stage 9: {result['warning']}. NOTHING WAS VERIFIED."))
        return StageSummary(
            lines=lines,
            status=f"Stage 9 verified NOTHING: {result['warning']}",
            board_state="done",
            board_summary="Nothing verified",
        )

    lines.append(("INFO",
                  f"Verification: {total} checks — {passed} passed, "
                  f"{failed} failed, {warnings} unverified"))
    for check in result.get("checks", []):
        if check.get("status") == "FAIL":
            lines.append(("WARNING",
                          f"Abstract number WRONG: '{check.get('claim')}' — data "
                          f"gives {check.get('actual')} ({check.get('note', '')})"))

    if total == 0:
        lines.append(("WARNING",
                      "The abstract has no numbers Stage 9 recognises. "
                      "NOTHING WAS VERIFIED."))
        status = "Stage 9 verified NOTHING: no recognisable numbers in the abstract"
    elif passed == 0 and failed == 0:
        lines.append(("WARNING",
                      f"None of the {total} numbers could be matched to data. "
                      f"NOTHING WAS CONFIRMED."))
        status = f"Stage 9: {total} numbers found, NONE could be checked"
    elif failed:
        status = (f"Stage 9: {failed} of {total} abstract numbers are WRONG "
                  f"({passed} ok, {warnings} unverified)")
    else:
        status = f"Stage 9: {passed} of {total} numbers ok, {warnings} unverified"

    report = result.get("report_path", "")
    if report:
        lines.append(("INFO", f"  Report: {report}"))
    return StageSummary(
        lines=lines,
        status=status,
        board_state="done",
        board_summary=f"{total} checks, {passed} passed, {failed} failed",
    )
