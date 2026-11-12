"""On-disk status of each pipeline stage, for the Pipeline Board.

The single place that decides whether a stage's output exists and how to
summarize it. Stdlib-only on purpose: the board refreshes on project open, and
importing the ``stages`` package would load the ML stack (several seconds)
on the GUI thread. ``tests/test_stage_outputs.py`` guards that.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.io import load_json

if TYPE_CHECKING:
    from core.config import Config

StageStatus = dict[str, str]

PENDING_SUMMARY = "No output yet"

# Current and retired validation artifacts must never count as a run comparison.
_COMPARE_RESULT_SKIP = {
    "robustness_results.json",
    "verification_results.json",
    "interrater_results.json",
    "interrater_summary.json",
}


def stage_output_files(cfg: Config) -> dict[str, Path]:
    """The primary file each stage writes, by name.

    One table for the Pipeline Board and run snapshots
    (``core.snapshots.save_snapshot``), so neither looks somewhere no stage
    writes.
    """
    validation = cfg.data_dir / "validation"
    return {
        "corpus": cfg.corpus_dir / "corpus.json",
        "relevant": cfg.claims_dir / "relevant.json",
        "scored_all": cfg.claims_dir / "scored_all.json",
        "claims": cfg.claims_dir / "claims.json",
        "method_profiles": cfg.data_dir / "method_review" / "method_profiles_validated.json",
        "claims_filtered": cfg.claims_dir / "claims_filtered.json",
        "synthesis": cfg.clusters_dir / "clusters.json",
        "abstract_draft": cfg.clusters_dir / "abstract_draft.md",
        "fulltext_metrics": cfg.data_dir / "fulltext" / "auto_extracted_metrics.json",
        "robustness": validation / "robustness_results.json",
        "verification": validation / "verification_results.json",
        "run_compare": validation / "run_compare_results.json",
    }


def _list_len(data: Any) -> int | None:
    if isinstance(data, list):
        return len(data)
    return None


def _has_text(path: Path) -> bool:
    """True when ``path`` holds something other than whitespace (a 0-byte draft is not output)."""
    try:
        return bool(path.read_text(encoding="utf-8").strip())
    except (OSError, UnicodeDecodeError):
        return False


def _done(summary: str) -> StageStatus:
    return {"status": "done", "summary": summary}


def _pending() -> StageStatus:
    return {"status": "pending", "summary": PENDING_SUMMARY}


def _summarize_search(cfg: Config) -> StageStatus:
    data = load_json(stage_output_files(cfg)["corpus"])
    n = _list_len(data)
    if n is None:
        return _pending()
    label = "paper" if n == 1 else "papers"
    return _done(f"{n} {label}")


def _summarize_score(cfg: Config) -> StageStatus:
    files = stage_output_files(cfg)
    relevant = load_json(files["relevant"])
    n = _list_len(relevant)
    if n is None:
        return _pending()
    scored = load_json(files["scored_all"])
    total = _list_len(scored)
    if total is not None:
        return _done(f"{n} relevant of {total} scored")
    label = "paper" if n == 1 else "papers"
    return _done(f"{n} relevant {label}")


def _summarize_extract(cfg: Config) -> StageStatus:
    data = load_json(stage_output_files(cfg)["claims"])
    n = _list_len(data)
    if n is None:
        return _pending()
    label = "paper" if n == 1 else "papers"
    return _done(f"{n} {label} extracted")


def _summarize_validate(cfg: Config) -> StageStatus:
    data = load_json(stage_output_files(cfg)["method_profiles"])
    n = _list_len(data)
    if n is None:
        return _pending()
    flagged = 0
    for record in data:
        if not isinstance(record, dict):
            continue
        profile = record.get("method_profile") or {}
        if isinstance(profile, dict) and profile.get("needs_manual_review"):
            flagged += 1
    label = "profile" if n == 1 else "profiles"
    if flagged:
        return _done(f"{n} {label}, {flagged} flagged")
    return _done(f"{n} {label}")


def _summarize_audit(cfg: Config) -> StageStatus:
    data = load_json(stage_output_files(cfg)["claims_filtered"])
    n = _list_len(data)
    if n is None:
        return _pending()
    return _done(f"{n} kept")


def _summarize_synthesize(cfg: Config) -> StageStatus:
    files = stage_output_files(cfg)
    clusters_path, abstract_path = files["synthesis"], files["abstract_draft"]
    clusters_data = load_json(clusters_path)
    n_clusters: int | None = None
    if isinstance(clusters_data, list):
        n_clusters = len(clusters_data)
    elif isinstance(clusters_data, dict):
        nested = clusters_data.get("clusters")
        if isinstance(nested, list):
            n_clusters = len(nested)
    abstract_exists = _has_text(abstract_path)
    if n_clusters is None and not abstract_exists:
        return _pending()
    parts: list[str] = []
    if n_clusters is not None:
        label = "cluster" if n_clusters == 1 else "clusters"
        parts.append(f"{n_clusters} {label}")
    if abstract_exists:
        parts.append("abstract saved")
    else:
        # Clusters without a usable abstract: the stage did not finish its job.
        parts.append("no abstract")
        return {"status": "pending", "summary": "; ".join(parts)}
    return _done("; ".join(parts) if parts else "synthesis output present")


def _summarize_fulltext(cfg: Config) -> StageStatus:
    path = stage_output_files(cfg)["fulltext_metrics"]
    if not path.exists():
        return _pending()
    data = load_json(path)
    n = _list_len(data)
    if n is not None:
        with_metrics = 0
        for record in data:
            if isinstance(record, dict) and record.get("metrics"):
                with_metrics += 1
        if with_metrics:
            return _done(f"{n} papers, {with_metrics} with metrics")
        return _done(f"{n} papers in metrics file")
    return _done("metrics file present")


def _summarize_robustness(cfg: Config) -> StageStatus:
    data = load_json(stage_output_files(cfg)["robustness"])
    if not isinstance(data, dict):
        return _pending()
    n = data.get("n_papers")
    if n:
        label = "paper" if n == 1 else "papers"
        return _done(f"{n} {label} analysed")
    return _done("robustness results saved")


def _summarize_verify(cfg: Config) -> StageStatus:
    data = load_json(stage_output_files(cfg)["verification"])
    if not isinstance(data, dict):
        return _pending()
    if data.get("review_report"):
        report = Path(data.get("report_path", ""))
        if not report.is_file():
            return _pending()
        return _done(f"Final review: {data.get('n_papers', 0)} papers")
    summary = data.get("summary") or {}
    total = summary.get("total_checks", 0)
    passed = summary.get("passed", 0)
    if not (total or passed):
        return _done("nothing verified" if "total_checks" in summary else "verification report saved")
    label = f"{total} checks, {passed} passed"
    failed, unverified = summary.get("failed", 0), summary.get("warnings", 0)
    if failed or unverified:
        label += f", {failed} failed, {unverified} unverified"
    return _done(label)


def _summarize_compare(cfg: Config) -> StageStatus:
    default_path = stage_output_files(cfg)["run_compare"]
    validation = default_path.parent
    data = load_json(default_path)
    if isinstance(data, dict):
        n_runs = data.get("n_runs", 0)
        interp = (data.get("overall_summary") or {}).get("interpretation")
        if n_runs:
            summary = f"{n_runs} runs compared"
            if interp:
                summary = f"{summary} ({interp})"
            return _done(summary)
        return _done("comparison saved")
    if not validation.exists():
        return _pending()
    try:
        candidates = sorted(validation.glob("*_results.json"))
    except OSError:
        return _pending()
    for path in candidates:
        if path.name in _COMPARE_RESULT_SKIP:
            continue
        payload = load_json(path)
        if not isinstance(payload, dict):
            continue
        n_runs = payload.get("n_runs", 0)
        if n_runs:
            return _done(f"{n_runs} runs compared")
        return _done("comparison saved")
    return _pending()


_STAGE_CHECKS: dict[int, Callable[[Config], StageStatus]] = {
    1: _summarize_search,
    2: _summarize_score,
    3: _summarize_extract,
    4: _summarize_validate,
    5: _summarize_audit,
    6: _summarize_synthesize,
    7: _summarize_fulltext,
    8: _summarize_robustness,
    9: _summarize_verify,
    10: _summarize_compare,
}

STAGE_NUMBERS = tuple(_STAGE_CHECKS)


def check_output(stage_num: int, cfg: Config) -> StageStatus:
    """Return ``{"status": "done"|"pending", "summary": str}`` for one stage.

    Missing or unreadable files mean pending. Raises KeyError for an unknown
    stage number.
    """
    return _STAGE_CHECKS[stage_num](cfg)
