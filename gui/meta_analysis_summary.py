"""Readable coverage and group results for the meta-analysis panel."""
from process.meta_analysis_run import coverage_lines, groups_for_report


def summary_text(results):
    lines = ["Meta-analysis", "", *coverage_lines(results, include_reasons=False), ""]
    if results.get("reason"):
        lines += [results["reason"], ""]
    groups = results.get("groups", [])
    if groups:
        lines += [
            "Groups that produced a pooled estimate are shown first; groups not pooled "
            "follow with the analysis reason.",
            "",
        ]
    for group in groups_for_report(groups):
        lines += [group["label"], f"  {group['n_studies']} independent studies; {group['analysis_scale']}"]
        pooled = group["pooled"]
        if pooled is not None:
            lines += [f"  Pooled effect: {pooled.pooled_effect:.4f}; "
                      f"95% CI [{pooled.ci_lower:.4f}, {pooled.ci_upper:.4f}]",
                      f"  Z = {pooled.z_value:.3f}, p = {pooled.p_value:.6g}",
                      f"  I² = {pooled.i_squared:.1f}%, τ² = {pooled.tau_squared:.6f}"]
            back = group.get("back_transformed")
            if back:
                lines.append(f"  As {back['measure']}: {back['effect']:.4f}; "
                             f"95% CI [{back['ci_lower']:.4f}, {back['ci_upper']:.4f}]")
            if group["n_studies"] >= 10:
                bias = group.get("bias")
                if bias:
                    lines.append(f"  Publication bias: {bias.overall_interpretation}")
            else:
                lines.append("  Fewer than 10 studies; publication bias tests have limited power.")
        else:
            lines.append("  " + group["reason"])
        lines.append("")
    if results.get("evidence_synthesis"):
        from process.meta_analysis_narrative import render_evidence_report
        lines += [render_evidence_report(results["evidence_synthesis"])]
    enhancement = results.get("enhancement")
    if enhancement:
        lines += ["Model enhancement: " + enhancement["status"]]
        if enhancement.get("report_path"):
            from pathlib import Path
            lines += ["Enhanced report: " + enhancement["report_path"],
                      Path(enhancement["report_path"]).read_text(encoding="utf-8")]
        if enhancement.get("error"):
            lines += ["Enhancement issue: " + enhancement["error"]]
    if results.get("report_path"):
        lines += ["Report: " + results["report_path"], "Paper audit: " + results["audit_path"]]
    return "\n".join(lines)
