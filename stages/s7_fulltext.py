"""Stage 7: Full-text retrieval & deep metric extraction.

Identifies high-value papers (SRs, MAs, RCTs) from method profiles,
checks open-access status via Unpaywall, downloads full text (PDF/HTML),
and extracts quantitative metrics (effect sizes, CIs, sample sizes, p-values).

No LLM calls — only HTTP downloads + regex extraction.
"""
import hashlib
import json
import logging
import re
import time
from collections.abc import Callable, Iterator
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote  # noqa: F401  (re-export; used in s7_fulltext_oa)

if TYPE_CHECKING:
    from core.config import Config

from functools import partial

from acquire.fulltext import content_is_pdf
from core.claims import claim_value  # noqa: F401  (re-export; used in s7_fulltext_targets)
from core.io import (
    atomic_append_jsonl,
    atomic_write_json,
    iter_jsonl_records,
    jsonl_to_json_array,
)
from process.statistical_extractor_numbers import (  # noqa: F401  (re-export; used in s7_fulltext_metrics)
    normalize_number,
    parse_count,
)
from stages._common import is_cancelled, make_progress

# Split-out phases, re-exported so ``stages.s7_fulltext.<name>`` keeps working
# (callers and tests patch names such as ``identify_targets`` on this module).
from stages.s7_fulltext_metrics import (  # noqa: F401
    _CI_LEVELS,
    _CI_RE,
    _D_G_RE,
    _DECIMAL,
    _ETA_SQUARED_RE,
    _MAX_ABS_STANDARDISED_EFFECT,
    _METRIC_TEXT_CHARS,
    _N_EQUALS_RE,
    _P_VALUE_RE,
    _RATIO_RANGE,
    _RATIO_RE,
    _SAMPLE_SIZE_RANGE,
    _SIGNED_NUM,
    _TOTAL_N_RE,
    _append_unique,
    _extract_confidence_intervals,
    _extract_effect_sizes,
    _extract_p_values,
    _extract_sample_sizes,
    _get_context,
    extract_metrics_from_text,
)
from stages.s7_fulltext_oa import (  # noqa: F401
    _NEGATIVE_OA_CACHE_TTL_DAYS,
    _OA_CACHE_FLUSH_EVERY,
    _UNPAYWALL_API,
    _UNPAYWALL_DELAY_SECONDS,
    _UNPAYWALL_TIMEOUT_SECONDS,
    OA_LOOKUP_SKIPPED_REASON,
    OA_STATUS_CLOSED,
    OA_STATUS_ERROR,
    OA_STATUS_OPEN,
    OA_STATUS_SKIPPED,
    _apply_oa_entry,
    _cache_entry_is_fresh,
    _entry_from_unpaywall,
    _load_oa_cache,
    _lookup_oa,
    _mark_oa_unknown,
    _query_unpaywall,
    _save_oa_cache,
    _unpaywall_url,
)
from stages.s7_fulltext_targets import (  # noqa: F401
    TARGET_TYPES,
    _add_target,
    _load_json_list,
    _matches_target_type,
    _normalize_design_text,
    _profile_study_type,
    _target_record,
    _type_keywords,
    identify_targets,
)
from stages.s7_fulltext_text import (  # noqa: F401
    _MIN_BINARY_TEXT_CHARS,
    _content_looks_like_html,
    _extract_text_from_html,
    _extract_text_from_pdf,
    _text_from_unknown_content,
)

logger = logging.getLogger(__name__)


def _user_agent(email: str = "") -> str:
    contact = f"; mailto:{email}" if email else ""
    return f"SystS-Pipeline/1.0 (Systematic Review Tool{contact}) python-requests"


_USER_AGENT = _user_agent()


def _import_requests():
    """Import requests with a clear install hint for missing environments."""
    try:
        import requests
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'requests' for Stage 7 full-text retrieval. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return requests


def _resume_key(record: dict) -> str:
    """Return a stable identity shared by targets and persisted JSONL rows."""
    for field in ("doi", "pmid", "title"):
        value = str(record.get(field) or "").strip().casefold()
        if value:
            return f"{field}:{value}"
    return str(record.get("resume_key") or "").strip()


def _latest_jsonl_by_key(path: Path, allowed_keys: set[str]) -> dict[str, dict]:
    """Load only the latest small metadata/metrics row for each active paper."""
    latest: dict[str, dict] = {}
    for record in iter_jsonl_records(path):
        if not isinstance(record, dict):
            continue
        key = _resume_key(record)
        if key and key in allowed_keys:
            latest[key] = record
    return latest


def _has_extracted_metrics(record: dict) -> bool:
    metrics = record.get("metrics") or {}
    return isinstance(metrics, dict) and any(
        isinstance(values, list) and bool(values)
        for values in metrics.values()
    )


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def run(
    config: "Config | None" = None,
    email: str | None = None,
    claims_path: str | None = None,
    workers: int | None = None,
    resume: bool = True,
    cancel_event: Any = None,
    progress_callback: Callable[[str], None] | None = None,
    **kw,
) -> dict:
    """Identify targets -> check OA -> download -> extract metrics."""
    assert config is not None, "s7_fulltext.run() requires a config"

    _progress = make_progress(logger, progress_callback)

    _cancelled = partial(is_cancelled, cancel_event)

    email = str(email or getattr(config, "pubmed_email", "") or "").strip()
    workers = max(1, int(
        workers or getattr(config, "fulltext_download_workers", 8) or 8
    ))
    streaming = bool(getattr(config, "fulltext_streaming", True))
    download_batch_size = max(1, int(
        getattr(config, "fulltext_download_batch_size", 50) or 50
    ))

    _progress("Stage 7: Full-Text Retrieval & Metric Extraction")
    _progress(f"  Email for Unpaywall: {email}" if email else f"WARNING: {OA_LOOKUP_SKIPPED_REASON}")
    _progress(f"  Workers: {workers}")
    if streaming:
        _progress("  Streaming mode: one paper at a time with resumable JSONL")
    else:
        _progress(f"  Buffered mode: batches of {download_batch_size} papers")

    # Ensure output directory
    ft_dir = config.data_dir / "fulltext"
    ft_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = ft_dir / "auto_extracted_metrics.json"
    metrics_jsonl_path = ft_dir / "auto_extracted_metrics.jsonl"
    download_log_path = ft_dir / "download_log.json"
    download_log_jsonl_path = ft_dir / "download_log.jsonl"

    # ── 1. Identify targets ───────────────────────────────────────────
    _progress("Phase 1: Identifying high-value targets...")
    targets = identify_targets(config, claims_path=claims_path)
    _progress(f"  Identified {len(targets)} high-value targets")

    if _cancelled():
        return {"cancelled": True}

    if not targets:
        _progress("No targets found — nothing to retrieve. Stage 7 complete.")
        result = {
            "targets": 0, "oa_papers": 0, "downloaded": 0,
            "metrics_extracted": 0, "output_path": str(ft_dir),
        }
        warning = _clear_outputs_for_empty_run([metrics_path, download_log_path])
        if warning:
            _progress(f"WARNING: {warning}")
            result["warning"] = warning
        return result

    if _cancelled():
        return {"cancelled": True}

    # ── 2. Check OA status ────────────────────────────────────────────
    _progress("Phase 2: Checking open-access status via Unpaywall...")
    cache_path = ft_dir / "oa_status.json"
    oa_targets = check_oa_status(
        targets, email=email, progress_callback=progress_callback,
        cache_path=cache_path, cancel_event=cancel_event,
    )
    oa_papers = [t for t in oa_targets if t.get("is_oa")]
    _progress(f"  {len(oa_papers)}/{len(targets)} papers are open access")
    oa_summary = _oa_lookup_summary(oa_targets, email, _progress)

    if _cancelled():
        return {"cancelled": True}

    # ── 3–5. Retrieve full text, extract metrics, and persist incrementally
    _progress("Phase 3: Downloading full text for OA papers...")
    keyed_oa_papers: list[dict] = []
    current_keys: set[str] = set()
    for index, paper in enumerate(oa_papers):
        keyed = dict(paper)
        keyed["resume_key"] = _resume_key(keyed) or f"position:{index}"
        if keyed["resume_key"] not in current_keys:
            keyed_oa_papers.append(keyed)
            current_keys.add(keyed["resume_key"])

    if not resume:
        for path in (metrics_jsonl_path, download_log_jsonl_path):
            path.unlink(missing_ok=True)
    download_rows = _latest_jsonl_by_key(download_log_jsonl_path, current_keys)
    metric_rows = _latest_jsonl_by_key(metrics_jsonl_path, current_keys)
    completed_keys = {
        key
        for key, row in download_rows.items()
        if row.get("status") == "ok" and isinstance(metric_rows.get(key, {}).get("metrics"), dict)
    }
    resumed_count = len(completed_keys)
    pending_papers = [
        paper for paper in keyed_oa_papers
        if paper["resume_key"] not in completed_keys
    ]
    if resumed_count:
        _progress(
            f"  Resuming from JSONL: {resumed_count} completed, "
            f"{len(pending_papers)} remaining"
        )

    def _buffered_results() -> Iterator[dict]:
        for start in range(0, len(pending_papers), download_batch_size):
            if _cancelled():
                break
            batch = pending_papers[start:start + download_batch_size]
            yield from retrieve_fulltext(
                batch,
                output_dir=ft_dir,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
                workers=workers,
            )

    if streaming:
        result_iterator: Iterator[dict] = retrieve_fulltext_iter(
            pending_papers,
            output_dir=ft_dir,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
            workers=workers,
            batch_size=download_batch_size,
        )
    else:
        result_iterator = _buffered_results()

    for paper in result_iterator:
        if _cancelled():
            break
        key = _resume_key(paper)
        download_record = {
            "resume_key": key,
            "doi": paper.get("doi", ""),
            "pmid": paper.get("pmid", ""),
            "title": paper.get("title", ""),
            "status": "ok" if paper.get("text") else "failed",
            "format": paper.get("format", "unknown"),
            "file": paper.get("saved_file", ""),
            "size_bytes": paper.get("size_bytes", 0),
        }
        text = paper.get("text", "")
        if text:
            paper_id = paper.get("doi") or paper.get("pmid") or key
            metrics = extract_metrics_from_text(text, paper_id=paper_id)
            metric_record = {
                "resume_key": key,
                "doi": paper.get("doi", ""),
                "pmid": paper.get("pmid", ""),
                "title": paper.get("title", ""),
                "method_type": paper.get("method_type", ""),
                "metrics": metrics,
            }
            atomic_append_jsonl(metrics_jsonl_path, metric_record)
            if key:
                metric_rows[key] = metric_record
            # Allow the text to be garbage-collected before the next iteration.
            paper["text"] = ""
        # The download row commits the completed metrics write.
        atomic_append_jsonl(download_log_jsonl_path, download_record)
        if key:
            download_rows[key] = download_record
        if _cancelled():
            break

    completed_keys = {
        key
        for key, row in download_rows.items()
        if row.get("status") == "ok" and isinstance(metric_rows.get(key, {}).get("metrics"), dict)
    }
    success_count = len(completed_keys)
    papers_with_metrics = sum(
        1 for key in completed_keys if _has_extracted_metrics(metric_rows[key])
    )

    if _cancelled():
        return {
            "cancelled": True, "downloaded": success_count,
            "metrics_extracted": papers_with_metrics, "resumed": resumed_count,
            "remaining": len(current_keys - completed_keys),
        }

    _progress(
        f"  Downloaded and extracted text from {success_count}/{len(keyed_oa_papers)} papers"
    )
    _progress(
        f"  Extracted metrics from {papers_with_metrics}/{len(keyed_oa_papers)} papers"
    )

    # Materialize deduplicated JSON views while retaining the append-only JSONL
    # checkpoints for resume. Crash-truncated lines are ignored by the reader.
    if metrics_jsonl_path.exists():
        jsonl_to_json_array(
            metrics_jsonl_path,
            metrics_path,
            record_key=_resume_key,
            include_keys=completed_keys,
        )
    else:
        atomic_write_json(metrics_path, [])
    if download_log_jsonl_path.exists():
        jsonl_to_json_array(
            download_log_jsonl_path,
            download_log_path,
            record_key=_resume_key,
            include_keys=current_keys,
        )
    else:
        atomic_write_json(download_log_path, [])

    _progress(f"  Saved: {metrics_path}")
    cancelled = _cancelled()
    if cancelled:
        _progress(
            "Stage 7 cancelled; JSONL checkpoints preserved for the next run"
        )
    else:
        _progress("Stage 7 complete")

    return {
        "targets": len(targets),
        "oa_papers": len(oa_papers),
        "downloaded": success_count,
        "metrics_extracted": papers_with_metrics,
        "resumed": resumed_count,
        "remaining": len(current_keys - completed_keys),
        "cancelled": cancelled,
        "output_path": str(metrics_path),
        **oa_summary,
    }


def _oa_lookup_summary(targets: list[dict], email: str, progress: Callable[[str], None]) -> dict:
    """Result fields that keep failed or skipped OA lookups apart from closed access."""
    errors = sum(1 for t in targets if t.get("oa_status") == OA_STATUS_ERROR)
    summary: dict[str, Any] = {"oa_errors": errors}
    if errors:
        progress(f"  WARNING: {errors} Unpaywall lookups failed; they are retried on the next run")
    if not email:
        summary["oa_lookup_skipped"] = OA_LOOKUP_SKIPPED_REASON
    return summary


def _has_records(path: Path) -> bool:
    try:
        with open(path, encoding="utf-8") as f:
            return bool(json.load(f))
    except (OSError, ValueError):
        return False


def _clear_outputs_for_empty_run(paths: list[Path]) -> str | None:
    """Write ``[]`` to each output; return a warning naming any non-empty one cleared.

    A run with 0 targets must not leave a previous run's results looking
    current, but replacing real results with nothing must be visible.
    """
    cleared = [p.name for p in paths if p.exists() and _has_records(p)]
    for path in paths:
        atomic_write_json(path, [])
    if not cleared:
        return None
    warning = (f"Stage 7 found 0 targets and cleared previous results in "
               f"{', '.join(cleared)}; check the method profiles and claims inputs.")
    logger.warning(warning)
    return warning


# ---------------------------------------------------------------------------
# Phase 2: Check OA status via Unpaywall (helpers in stages.s7_fulltext_oa)
# ---------------------------------------------------------------------------
def check_oa_status(
    targets: list[dict],
    email: str = "",
    progress_callback: Callable[[Any], None] | None = None,
    progress_label: str | None = None,
    cache_path: Path | None = None,
    cancel_event: Any = None,
) -> list[dict]:
    """Query Unpaywall API for each target's OA status.

    Sets ``is_oa``, ``oa_url``, ``host_type`` and ``oa_status`` (``open``,
    ``closed``, ``error`` or ``skipped``) on each target. Failed lookups are
    never cached. Without a contact ``email`` only cached answers are used.
    """
    requests = _import_requests()
    email = str(email or "").strip()
    if not email:
        logger.warning(OA_LOOKUP_SKIPPED_REASON)

    _cancelled = partial(is_cancelled, cancel_event)
    cache = _load_oa_cache(cache_path)
    today = date.today()

    session = requests.Session()
    session.headers.update({"User-Agent": _user_agent(email)})

    total = len(targets)
    cached_hits = 0
    api_hits = 0
    api_errors = 0

    def _emit_oa_progress(current: int) -> None:
        msg = f"  OA check [{current}/{total}] — {api_hits} queried, {cached_hits} cached"
        logger.info(msg)
        if not progress_callback:
            return
        if progress_label:
            progress_callback({
                "message": f"{progress_label}: {current}/{total} checked",
                "current": current,
                "total": total,
            })
        else:
            progress_callback(msg)

    for i, target in enumerate(targets):
        if _cancelled():
            break

        doi = target.get("doi", "")
        if not doi:
            continue

        cached = cache.get(doi)
        if isinstance(cached, dict) and _cache_entry_is_fresh(cached, today):
            _apply_oa_entry(target, cached)
            cached_hits += 1
        elif not email:
            _mark_oa_unknown(target, OA_STATUS_SKIPPED)
        else:
            entry = _lookup_oa(session, doi, email)
            if entry is None:
                _mark_oa_unknown(target, OA_STATUS_ERROR)
                api_errors += 1
            else:
                _apply_oa_entry(target, entry)
                cache[doi] = entry
                api_hits += 1
                if api_hits % _OA_CACHE_FLUSH_EVERY == 0:
                    _save_oa_cache(cache_path, cache)
            time.sleep(_UNPAYWALL_DELAY_SECONDS)

        if (i + 1) % 10 == 0 or i + 1 == total:
            _emit_oa_progress(i + 1)

    _save_oa_cache(cache_path, cache)

    logger.info("  OA status: %d cached, %d API hits, %d errors",
                cached_hits, api_hits, api_errors)

    return targets


# ---------------------------------------------------------------------------
# Phase 3: Retrieve full text
# ---------------------------------------------------------------------------
def retrieve_fulltext(
    oa_papers: list[dict],
    output_dir: Path,
    progress_callback: Callable[[str], None] | None = None,
    cancel_event: Any = None,
    workers: int = 8,
    **kwargs: Any,
) -> list[dict]:
    """Download full text and extract plain text from PDF/HTML."""
    return list(retrieve_fulltext_iter(
        oa_papers,
        output_dir,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
        workers=workers,
        **kwargs,
    ))


# ---------------------------------------------------------------------------
# Phase 3b: Streaming full-text retrieval
# ---------------------------------------------------------------------------
def retrieve_fulltext_iter(
    oa_papers: list[dict],
    output_dir: Path,
    progress_callback: Callable[[str], None] | None = None,
    cancel_event: Any = None,
    workers: int = 8,
    batch_size: int = 50,
) -> Iterator[dict]:
    """Yield full-text result entries one at a time to bound memory usage.

    Downloads are performed sequentially (one active HTTP request at a time),
    which keeps only the current response content and extracted text resident.
    The ``batch_size`` parameter controls how often progress is emitted and is
    reserved for future batched concurrency; it does not affect peak memory in
    the current implementation.
    """
    requests = _import_requests()

    _cancelled = partial(is_cancelled, cancel_event)

    session = requests.Session()
    session.headers.update({"User-Agent": _USER_AGENT})

    total = len(oa_papers)
    for i, paper in enumerate(oa_papers):
        if _cancelled():
            break

        doi = paper.get("doi", "")
        title = str(paper.get("title") or "")[:60]
        oa_url = paper.get("oa_url", "")

        if not oa_url:
            logger.warning("  [%d/%d] No OA URL for %s — skipping", i + 1, total, doi)
            yield {**paper, "text": "", "format": "none", "saved_file": ""}
            continue

        result_entry: dict[str, Any] = {
            **paper, "text": "", "format": "unknown",
            "saved_file": "", "size_bytes": 0, "success": False,
        }

        try:
            resp = session.get(oa_url, timeout=60, allow_redirects=True, stream=True)
            resp.raise_for_status()
            _reject_oversized(resp)
            _save_and_extract(resp, output_dir, _safe_filename(doi), result_entry)

            status = "ok" if result_entry["text"] else "no-text"
            result_entry["success"] = status == "ok"
            size_kb = result_entry["size_bytes"] // 1024
            msg = (f"  [{i + 1}/{total}] Downloaded: {title} "
                   f"({result_entry['format']}, {size_kb}KB) [{status}]")
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        except Exception as exc:
            logger.warning("  [%d/%d] Download failed for %s: %s",
                           i + 1, total, doi, exc)
            result_entry["format"] = "error"
            if progress_callback:
                progress_callback(f"  [{i + 1}/{total}] Failed: {title} — {exc}")

        yield result_entry


# Responses are streamed; a body declared larger than this is never read.
_MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024


def _reject_oversized(resp: Any) -> None:
    declared = str(resp.headers.get("content-length") or "").strip()
    if declared.isdigit() and int(declared) > _MAX_DOWNLOAD_BYTES:
        resp.close()
        raise ValueError(
            f"response too large ({declared} bytes > {_MAX_DOWNLOAD_BYTES} byte limit)"
        )


def _save_and_extract(resp: Any, output_dir: Path, safe_name: str, entry: dict) -> None:
    """Save the response body and fill ``entry`` with its format and text."""
    content_type = resp.headers.get("content-type", "").lower()
    content = resp.content
    if len(content) > _MAX_DOWNLOAD_BYTES:
        raise ValueError(f"response too large ({len(content)} bytes)")
    entry["size_bytes"] = len(content)

    if content_is_pdf(content):
        fmt, suffix = "pdf", ".pdf"
    elif "html" in content_type or "xml" in content_type or _content_looks_like_html(content):
        fmt, suffix = "html", ".html"
    else:
        fmt, suffix = "unknown", ".bin"
    path = output_dir / f"{safe_name}{suffix}"
    path.write_bytes(content)
    entry["saved_file"] = str(path)

    if fmt == "pdf":
        entry["format"], entry["text"] = "pdf", _extract_text_from_pdf(path)
    elif fmt == "html":
        entry["format"], entry["text"] = "html", _extract_text_from_html(content)
    else:
        text = _text_from_unknown_content(content)
        if text:
            entry["format"], entry["text"] = "html", text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_NAME_HASH_CHARS = 10
# Stem + "_" + hash stays within the former 120-character name limit.
_SAFE_STEM_CHARS = 120 - 1 - _NAME_HASH_CHARS


def _safe_filename(doi: str) -> str:
    """Convert a DOI to a safe filename that stays unique per DOI.

    The stem is sanitised and truncated (filesystem limits); a short hash of
    the full DOI keeps punctuation variants and long shared prefixes apart.
    """
    stem = re.sub(r"[^\w\-.]", "_", doi)[:_SAFE_STEM_CHARS]
    digest = hashlib.sha256(doi.encode("utf-8")).hexdigest()[:_NAME_HASH_CHARS]
    return f"{stem}_{digest}"
