"""Run snapshot system for living reviews.

Provides save/load/list/diff functionality for pipeline run snapshots,
enabling longitudinal comparison of systematic review updates over time.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from core.doi import normalize_doi
from core.io import atomic_write_json, atomic_write_text
from core.stage_outputs import stage_output_files

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────


@dataclass
class Snapshot:
    """A frozen record of pipeline state at a point in time."""
    run_id: str
    timestamp: str                  # ISO format
    config_hash: str                # hash of research_config.yaml for comparison
    stage_data: dict                # {stage_name: {file_path: str, record_count: int, hash: str}}
    summary: dict                   # {n_corpus: int, n_relevant: int, n_claims: int, ...}
    # Per-article state frozen at snapshot time ({key, title, doi, year,
    # relevance_score}); None for legacy snapshots saved before it existed.
    articles: list[dict] | None = None


@dataclass
class SnapshotDiff:
    """Comparison between two snapshots."""
    old_run_id: str
    new_run_id: str
    new_articles: list[dict] = field(default_factory=list)
    removed_articles: list[dict] = field(default_factory=list)
    score_changes: list[dict] = field(default_factory=list)
    claim_changes: list[dict] = field(default_factory=list)
    grade_changes: dict = field(default_factory=dict)
    summary_text: str = ""


# ── File hashing ──────────────────────────────────────────────────────────


def _file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except OSError:
        return ""


_RECORD_LIST_KEYS = ("articles", "records", "items", "claims", "results", "clusters")


def _count_records(path: Path) -> int:
    """Count records in a JSON file.

    A list counts its items, a dict its record list (``articles``,
    ``clusters`` …) or, when every value is a record, its entries. A summary
    dict (e.g. robustness results) has no records and counts 0.
    """
    data = _load_json_safe(path)
    if isinstance(data, list):
        return len(data)
    if not isinstance(data, dict):
        return 0
    for key in _RECORD_LIST_KEYS:
        if isinstance(data.get(key), list):
            return len(data[key])
    if data and all(isinstance(v, dict) for v in data.values()):
        return len(data)
    return 0


def _load_json_safe(path: Path) -> list | dict | None:
    """Load a JSON file, returning None on failure."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):  # ValueError covers JSON and UTF-8 decode errors
        return None


# ── Config hashing ────────────────────────────────────────────────────────


def _config_hash(config) -> str:
    """Hash the research config to detect configuration changes between runs."""
    config_str = json.dumps({
        "hypothesis": config.hypothesis_text,
        "provider": config.llm_provider,
        "model": config.llm_model,
        "threshold": config.relevance_threshold,
    }, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


# ── Snapshot save/load/list ───────────────────────────────────────────────


def save_snapshot(config, run_id: str = "") -> Path:
    """Save current pipeline state as a timestamped snapshot.

    1. Generate run_id from timestamp if not provided
    2. Walk all pipeline output files, compute record counts and file hashes
    3. Collect summary statistics
    4. Save to config.output_dir / "snapshots" / f"{run_id}.json"
    """
    now = datetime.now()
    if not run_id:
        run_id = now.strftime("run_%Y%m%d_%H%M%S")

    snapshot_dir = config.output_dir / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    stage_data = _stage_data(config)

    # Summary statistics
    summary = {
        "n_corpus": stage_data.get("corpus", {}).get("record_count", 0),
        "n_relevant": stage_data.get("relevant", {}).get("record_count", 0),
        "n_claims": stage_data.get("claims", {}).get("record_count", 0),
        "n_claims_filtered": stage_data.get("claims_filtered", {}).get("record_count", 0),
        "stages_completed": len(stage_data),
        "total_files": len(stage_data),
    }

    snapshot = Snapshot(
        run_id=run_id,
        timestamp=now.isoformat(),
        config_hash=_config_hash(config),
        stage_data=stage_data,
        summary=summary,
        articles=_frozen_articles(config),
    )

    out_path = snapshot_dir / f"{run_id}.json"
    atomic_write_json(out_path, asdict(snapshot), ensure_ascii=True)

    logger.info("Snapshot saved: %s (%d stages)", run_id, len(stage_data))
    return out_path


def _snapshot_files(config) -> dict[str, Path]:
    """Files a snapshot records: each stage's output plus the analysis results."""
    return {
        **stage_output_files(config),
        "meta_analysis": config.output_dir / "meta_analysis_results.json",
        "final_review": config.output_dir / "final_review_report.md",
        "final_review_sources": config.output_dir / "final_review_sources.json",
        # Where Stage 6 and the committee brief read GRADE results from.
        "grade": config.output_dir / "grade_results.json",
    }


def _stage_data(config) -> dict:
    """{stage: {file_path, record_count, hash}} for every output that exists."""
    return {
        name: {
            "file_path": str(path),
            "record_count": _count_records(path) if path.suffix == ".json" else 0,
            "hash": _file_hash(path),
        }
        for name, path in _snapshot_files(config).items()
        if path.exists()
    }


def load_snapshot(path: Path) -> Snapshot:
    """Load a snapshot from JSON."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return Snapshot(**data)


def _timestamp_key(snapshot: Snapshot) -> datetime:
    try:
        return datetime.fromisoformat(snapshot.timestamp)
    except (TypeError, ValueError):
        logger.warning("Snapshot %s has an unreadable timestamp %r; listing it last.",
                       snapshot.run_id, snapshot.timestamp)
        return datetime.min


def list_snapshots(config) -> list[Snapshot]:
    """List all snapshots, sorted by their saved timestamp (newest first)."""
    snapshot_dir = config.output_dir / "snapshots"
    if not snapshot_dir.exists():
        return []

    snapshots = []
    for path in sorted(snapshot_dir.glob("*.json")):
        try:
            snapshots.append(load_snapshot(path))
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("Failed to load snapshot %s: %s", path, exc)
    return sorted(snapshots, key=_timestamp_key, reverse=True)


# ── Snapshot diff ─────────────────────────────────────────────────────────


def _article_key(article: dict) -> str:
    """Stable key for an article: DOI if available, else lowercase title."""
    doi = normalize_doi(article.get("doi"))
    if doi:
        return f"doi:{doi}"
    title = article.get("title", "")
    return f"title:{title.lower().strip()}"


def _article_list(data) -> list[dict]:
    """Records from a stage file that is a list or a {"articles": [...]} dict."""
    if isinstance(data, dict):
        data = data.get("articles", [])
    if not isinstance(data, list):
        return []
    return [a for a in data if isinstance(a, dict)]


def _score_of(article: dict):
    return article.get("relevance_score", article.get("score"))


def _frozen_articles(config) -> list[dict]:
    """Freeze per-article identity and relevance score at snapshot time.

    Scores come from the corpus record when present, otherwise from
    Stage 2's ``claims_dir/scored_all.json``.
    """
    corpus = _article_list(_load_json_safe(config.corpus_dir / "corpus.json"))
    claims_dir = getattr(config, "claims_dir", None)
    scored = _article_list(_load_json_safe(claims_dir / "scored_all.json")) if claims_dir else []
    scores = {_article_key(a): _score_of(a) for a in scored}
    frozen = []
    for article in corpus:
        key = _article_key(article)
        score = _score_of(article)
        frozen.append({
            "key": key,
            "title": article.get("title", ""),
            "doi": article.get("doi", ""),
            "year": article.get("year", ""),
            "relevance_score": score if score is not None else scores.get(key),
        })
    return frozen


def _snapshot_articles(snapshot: Snapshot) -> list[dict] | None:
    """Articles as they were when ``snapshot`` was taken, or None if unknown.

    Legacy snapshots stored only a path and hash. Their corpus file can be
    used only while it is byte-identical to what was snapshotted.
    """
    if snapshot.articles is not None:
        return snapshot.articles
    corpus = snapshot.stage_data.get("corpus", {})
    path, stored_hash = corpus.get("file_path", ""), corpus.get("hash", "")
    if not path or not stored_hash or _file_hash(Path(path)) != stored_hash:
        return None
    return [{**a, "key": _article_key(a), "relevance_score": _score_of(a)}
            for a in _article_list(_load_json_safe(Path(path)))]


def _article_summary(article: dict) -> dict:
    return {
        "title": article.get("title", ""),
        "doi": article.get("doi", ""),
        "year": article.get("year", ""),
    }


def _score_change(old_art: dict, new_art: dict) -> dict | None:
    old_score = old_art.get("relevance_score")
    new_score = new_art.get("relevance_score")
    if old_score is None or new_score is None:
        return None
    try:
        if abs(float(old_score) - float(new_score)) <= 0.01:
            return None
    except (ValueError, TypeError):
        if str(old_score) == str(new_score):  # non-numeric: compare as labels
            return None
    return {"title": new_art.get("title", ""), "old_score": old_score,
            "new_score": new_score}


def _diff_articles(diff: SnapshotDiff, old_articles: list[dict],
                   new_articles: list[dict]) -> None:
    """Fill new/removed articles and score changes from frozen article lists."""
    old_by_key = {a["key"]: a for a in old_articles}
    new_by_key = {a["key"]: a for a in new_articles}
    old_keys, new_keys = set(old_by_key), set(new_by_key)
    diff.new_articles.extend(_article_summary(new_by_key[k]) for k in sorted(new_keys - old_keys))
    diff.removed_articles.extend(
        _article_summary(old_by_key[k]) for k in sorted(old_keys - new_keys))
    for key in sorted(old_keys & new_keys):
        change = _score_change(old_by_key[key], new_by_key[key])
        if change:
            diff.score_changes.append(change)


def diff_snapshots(old: Snapshot, new: Snapshot, config=None) -> SnapshotDiff:
    """Compare two snapshots and identify changes.

    1. Use each snapshot's frozen article list (never the live corpus file)
    2. Diff by DOI/title to find new/removed articles
    3. Compare relevance scores for shared articles
    4. Compare claim data for shared articles
    5. Compare GRADE results if available
    6. Generate summary text
    """
    diff = SnapshotDiff(old_run_id=old.run_id, new_run_id=new.run_id)

    old_articles = _snapshot_articles(old)
    new_articles = _snapshot_articles(new)
    unavailable = [snap.run_id for snap, arts in ((old, old_articles), (new, new_articles))
                   if arts is None]
    if not unavailable:
        _diff_articles(diff, old_articles, new_articles)

    # Claim changes — compare by hash of stage data
    old_claims_hash = old.stage_data.get("claims", {}).get("hash", "")
    new_claims_hash = new.stage_data.get("claims", {}).get("hash", "")
    if old_claims_hash and new_claims_hash and old_claims_hash != new_claims_hash:
        old_n = old.stage_data.get("claims", {}).get("record_count", 0)
        new_n = new.stage_data.get("claims", {}).get("record_count", 0)
        diff.claim_changes.append({
            "type": "count_change",
            "old_count": old_n,
            "new_count": new_n,
        })

    # GRADE changes
    old_grade_hash = old.stage_data.get("grade", {}).get("hash", "")
    new_grade_hash = new.stage_data.get("grade", {}).get("hash", "")
    if old_grade_hash and new_grade_hash and old_grade_hash != new_grade_hash:
        diff.grade_changes["changed"] = True

    # Config change detection
    config_changed = old.config_hash != new.config_hash

    # Generate summary
    parts = [f"Comparison: {old.run_id} -> {new.run_id}"]
    if config_changed:
        parts.append("WARNING: Pipeline configuration changed between runs.")
    if unavailable:
        parts.append(
            "Article-level comparison unavailable: snapshot(s) "
            f"{', '.join(unavailable)} predate frozen article data and their "
            "corpus file has changed since.")
    else:
        parts.append(f"New articles: {len(diff.new_articles)}")
        parts.append(f"Removed articles: {len(diff.removed_articles)}")
        parts.append(f"Score changes: {len(diff.score_changes)}")
    if diff.claim_changes:
        parts.append(f"Claim data changed (old: {diff.claim_changes[0].get('old_count', '?')}, new: {diff.claim_changes[0].get('new_count', '?')})")
    if diff.grade_changes:
        parts.append("GRADE assessment results changed.")

    # Stage completion comparison
    old_stages = set(old.stage_data.keys())
    new_stages = set(new.stage_data.keys())
    added_stages = new_stages - old_stages
    removed_stages = old_stages - new_stages
    if added_stages:
        parts.append(f"New stages completed: {', '.join(sorted(added_stages))}")
    if removed_stages:
        parts.append(f"Stages no longer present: {', '.join(sorted(removed_stages))}")

    diff.summary_text = "\n".join(parts)
    return diff


# ── Diff report ───────────────────────────────────────────────────────────


def _cell(value) -> str:
    """A markdown table cell: pipes escaped, line breaks flattened."""
    return " ".join(str(value if value is not None else "").split()).replace("|", "\\|")


def _table_row(*values) -> str:
    return "| " + " | ".join(_cell(v) for v in values) + " |"


def generate_diff_report(diff: SnapshotDiff, output_path: Path) -> Path:
    """Save diff as a formatted markdown report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Snapshot Comparison Report",
        "",
        f"**Old run:** {diff.old_run_id}",
        f"**New run:** {diff.new_run_id}",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        diff.summary_text,
        "",
    ]

    if diff.new_articles:
        lines.extend([
            "## New Articles",
            "",
            "| Title | DOI | Year |",
            "|-------|-----|------|",
        ])
        for art in diff.new_articles:
            lines.append(_table_row(art.get("title", ""), art.get("doi", ""), art.get("year", "")))
        lines.append("")

    if diff.removed_articles:
        lines.extend([
            "## Removed Articles",
            "",
            "| Title | DOI | Year |",
            "|-------|-----|------|",
        ])
        for art in diff.removed_articles:
            lines.append(_table_row(art.get("title", ""), art.get("doi", ""), art.get("year", "")))
        lines.append("")

    if diff.score_changes:
        lines.extend([
            "## Score Changes",
            "",
            "| Title | Old Score | New Score |",
            "|-------|-----------|-----------|",
        ])
        for sc in diff.score_changes:
            lines.append(_table_row(sc.get("title", ""), sc.get("old_score", ""),
                                    sc.get("new_score", "")))
        lines.append("")

    if diff.claim_changes:
        lines.extend([
            "## Claim Changes",
            "",
        ])
        for cc in diff.claim_changes:
            lines.append(f"- {cc.get('type', '')}: old={cc.get('old_count', '?')}, new={cc.get('new_count', '?')}")
        lines.append("")

    if diff.grade_changes:
        lines.extend([
            "## GRADE Changes",
            "",
            "GRADE assessment results differ between the two runs.",
            "",
        ])

    lines.extend([
        "---",
        "*Generated by Sysys Living Review Snapshot System*",
    ])

    atomic_write_text(output_path, "\n".join(lines) + "\n")
    logger.info("Diff report saved to %s", output_path)
    return output_path
