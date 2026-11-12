"""Plain-language summary of a Stage 6 (Synthesize) result for the GUI.

A pure function (no Qt), like ``gui.robustness_verify_summary``. Stage 6 reports
"complete" only when it produced an abstract: clusters and narratives without
one leave Stages 8-9 nothing to check, so that is a failure the user must see.
"""
from __future__ import annotations

from gui.robustness_verify_summary import StageSummary


def summarize_synthesis(result: dict) -> StageSummary:
    """Board state and wording for a Stage 6 result dict."""
    n_clusters = int(result.get("n_clusters") or 0)
    n_narratives = int(result.get("narratives_count") or 0)
    parts = [f"{n_clusters} cluster{'' if n_clusters == 1 else 's'}"] if n_clusters else []
    if result.get("status") == "partial":
        unresolved = result.get("unresolved_model_steps") or []
        progress_path = result.get("synthesis_progress_path") or "the Stage 6 progress artifact"
        first_action = ""
        if unresolved:
            first = unresolved[0]
            message = first.get("message") if isinstance(first, dict) else first
            if str(message or "").strip():
                first_action = f" Unresolved action: {str(message).strip()}"
        return StageSummary(
            lines=[(
                "WARNING",
                f"Stage 6 is PARTIAL: {n_narratives}/{n_clusters} cluster narrative(s) "
                f"were saved and {len(unresolved)} model step(s) remain unresolved. "
                f"Compatible reruns resume from {progress_path} instead of discarding "
                f"completed findings.{first_action}",
            )],
            status=(
                f"Synthesis partial — {n_narratives}/{n_clusters} narratives saved; "
                f"{len(unresolved)} model step(s) unresolved.{first_action}"
            ),
            board_state="failed",
            board_summary="; ".join([*parts, f"{n_narratives} narratives saved", "partial"]),
        )
    if result.get("abstract_path"):
        return StageSummary(
            status="Synthesis complete.",
            board_state="done",
            board_summary="; ".join([*parts, "abstract saved"]),
        )
    why = "; ".join(w for w in result.get("warnings") or [] if "abstract" in w.lower())
    return StageSummary(
        lines=[("ERROR", "Stage 6 produced NO abstract, so there is nothing for Stage 9 to "
                         "verify." + (f" {why}" if why else ""))],
        status="Synthesis produced NO abstract; see the log.",
        board_state="failed",
        board_summary="; ".join([*parts, "no abstract"]),
    )
