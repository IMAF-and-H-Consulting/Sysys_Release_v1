"""Stage 2b: Optional full-text corpus enrichment before relevance scoring.

Downloads legally accessible open-access full text for papers in the corpus,
adds cleaned plaintext to the paper records, and writes an enriched corpus file
that Stage 2 can score against.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

from acquire.fulltext import (
    AsyncHostThrottle,
    copy_oa_hint_fields,
    download_cache_key,
    download_preferred_fulltext_async,
    download_record_status,
    load_download_cache,
    recover_download_artifact,
    reusable_download_record,
    write_json_atomic,
)
from core.io import atomic_write_json_array
from stages._common import deliver_progress, is_cancelled

logger = logging.getLogger(__name__)


def _emit_progress(progress_callback, message: str, current: int | None = None, total: int | None = None) -> None:
    """Best-effort progress reporting hook for GUI callers."""
    if not progress_callback:
        return
    payload: dict[str, Any] = {"message": message}
    if current is not None:
        payload["current"] = current
    if total is not None:
        payload["total"] = total
    deliver_progress(progress_callback, payload)


_is_cancelled = is_cancelled


def _paper_key(paper: dict) -> str:
    return (
        paper.get("doi")
        or paper.get("pmc")
        or paper.get("pmid")
        or paper.get("s2_id")
        or paper.get("oa_id")
        or paper.get("title", "")
    )


_NAME_HASH_CHARS = 10


def _safe_name(identifier: str) -> str:
    """Filesystem-safe stem with a short hash of the full identifier.

    Sanitising alone maps ``a(b)``, ``a:b`` and ``a/b`` to the same stem, so
    disk recovery could adopt another paper's text; the hash keeps them apart.
    """
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", identifier).strip("_") or "paper"
    digest = hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:_NAME_HASH_CHARS]
    return f"{stem}_{digest}"


# HTTP statuses that may succeed later (rate limits, server trouble, timeouts).
_TRANSIENT_HTTP_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
_HTTP_OK = 200
_FAILED_STATUSES = frozenset({"error", "http_error"})
# Total in-flight downloads across all hosts (per-host limits come from
# ``workers``), and the HTTP timeouts for each request and pool wait.
_MAX_CONCURRENT_DOWNLOADS = 32
_HTTP_TIMEOUT_SECONDS = 30.0
_HTTP_POOL_TIMEOUT_SECONDS = 120.0


class _StatusRecordingThrottle(AsyncHostThrottle):
    """Host throttle that remembers the last HTTP status seen for each URL."""

    def __init__(self, default_concurrency: int = 4, max_total: int | None = None):
        super().__init__(default_concurrency=default_concurrency, max_total=max_total)
        self.statuses: dict[str, int] = {}

    async def get(self, client, url: str, **kwargs):
        response = await super().get(client, url, **kwargs)
        status = getattr(response, "status_code", None)
        if isinstance(status, int):
            self.statuses[url] = status
        return response


def _is_transient_http_status(status) -> bool:
    return not isinstance(status, int) or status in _TRANSIENT_HTTP_STATUSES or status >= 500


def _is_terminal_download_row(row: dict) -> bool:
    """True when a cached outcome will not change on retry.

    Only a 200 response without usable text, or a permanent HTTP error such as
    404, is final. Exceptions, rate limits, 5xx responses, and legacy
    ``no_usable_text`` rows that never recorded an HTTP status are retried.
    """
    status = download_record_status(row)
    if status == "no_usable_text":
        return row.get("http_status") == _HTTP_OK
    if status == "http_error":
        return not _is_transient_http_status(row.get("http_status"))
    return False


def _unusable_download_row(target: dict, url: str, http_status) -> dict:
    """Log row for a fetch that returned no usable text."""
    row = {**target, "source_url": url}
    if isinstance(http_status, int):
        row["http_status"] = http_status
    if isinstance(http_status, int) and http_status != _HTTP_OK:
        row["status"] = "http_error"
        row["error"] = f"HTTP {http_status}"
    else:
        row["status"] = "no_usable_text"
    return row


def _download_outcome_counts(rows: dict[str, dict], targets: list[dict]) -> dict[str, int]:
    """Count failed (retryable) and unavailable (terminal) eligible targets."""
    failed = unavailable = 0
    for target in targets:
        if not (target.get("is_oa") and target.get("oa_url")):
            continue
        row = rows.get(download_cache_key(target))
        status = download_record_status(row or {})
        if status in _FAILED_STATUSES:
            failed += 1
        elif status == "no_usable_text":
            unavailable += 1
    return {"download_failed": failed, "download_unavailable": unavailable}


def _build_targets(corpus: list[dict]) -> list[dict]:
    targets: list[dict] = []
    seen: set[str] = set()

    for paper in corpus:
        key = _paper_key(paper)
        if not key or key in seen:
            continue
        seen.add(key)
        targets.append({
            "key": key,
            "doi": paper.get("doi"),
            "pmc": paper.get("pmc"),
            "title": paper.get("title", ""),
            "year": paper.get("year"),
            **copy_oa_hint_fields(paper),
        })

    return targets


# Corpus OA hints in fallback order after Unpaywall and PMC: direct PDFs first,
# then landing pages. Each entry is (field, host-type field, source label).
_OA_HINT_PRIORITY = (
    ("openalex_best_oa_pdf_url", "openalex_best_oa_host_type", "openalex"),
    ("s2_open_access_pdf_url", "", "semantic_scholar"),
    ("openalex_oa_url", "", "openalex"),
    ("openalex_best_oa_url", "openalex_best_oa_host_type", "openalex"),
)


def _pmc_url(pmc: str) -> str:
    return f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc}/"


def _oa_location(target: dict, oa: dict | None) -> tuple[str, str, str]:
    """Return ``(url, host_type, source)`` for the best available OA copy.

    Order: Unpaywall's URL, PMC, OpenAlex/Semantic Scholar PDF hints, then
    landing-page hints. Any Unpaywall miss (error, closed, no URL) falls back.
    """
    if oa and oa.get("is_oa") and oa.get("oa_url"):
        return oa["oa_url"], oa.get("host_type", ""), "unpaywall"
    if target.get("pmc"):
        return _pmc_url(target["pmc"]), "repository", "pmc"
    for field, host_field, source in _OA_HINT_PRIORITY:
        url = str(target.get(field) or "").strip()
        if url:
            return url, str(target.get(host_field) or "") if host_field else "", source
    return "", "", ""


def _merge_oa_locations(targets: list[dict], oa_rows: list[dict]) -> list[dict]:
    by_doi = {row.get("doi"): row for row in oa_rows if row.get("doi")}
    merged: list[dict] = []

    for target in targets:
        row = dict(target)
        oa = by_doi.get(target.get("doi"))
        if oa and oa.get("resolver"):
            row["resolver"] = oa.get("resolver", "")
        if oa and oa.get("oa_status"):
            row["unpaywall_status"] = oa["oa_status"]
        url, host_type, source = _oa_location(target, oa)
        row["is_oa"] = bool(url)
        row["oa_url"] = url
        row["host_type"] = host_type
        if source:
            row["oa_url_source"] = source
        merged.append(row)

    return merged


def _download_fulltext(
    targets: list[dict],
    output_dir: Path,
    progress_callback=None,
    cancel_event=None,
    log_path: Path | None = None,
    workers: int = 8,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_path or output_dir / "download_log.json"
    persisted_rows = load_download_cache(log_path)
    # The download log is rewritten wholesale on each flush (atomic JSON);
    # batch the in-loop flushes so a large corpus doesn't pay O(n^2) bytes.
    # The unconditional write after the loop still covers completion/cancel;
    # per-paper files on disk let recover_download_artifact() redo any
    # unflushed work after a hard crash.
    _flush_every = 25
    _writes_pending = 0

    def _flush_log(force: bool = False) -> None:
        nonlocal _writes_pending
        _writes_pending += 1
        if force or _writes_pending >= _flush_every:
            write_json_atomic(log_path, list(persisted_rows.values()))
            _writes_pending = 0
    eligible_keys = {
        download_cache_key(target)
        for target in targets
        if target.get("is_oa") and target.get("oa_url")
    }
    downloaded: list[dict] = [
        row for key, row in persisted_rows.items()
        if key in eligible_keys and reusable_download_record(row)
    ]
    eligible = [target for target in targets if target.get("is_oa") and target.get("oa_url")]
    total = len(eligible)
    processed = 0
    logger.info("Stage 2b: %d papers are eligible for full-text download", total)
    _emit_progress(progress_callback, "Preparing corpus full-text downloads...", 0, max(1, total))

    if total == 0:
        logger.warning("Stage 2b: no open-access full-text targets were found in the selected corpus")
        return downloaded

    if downloaded:
        logger.info(
            "Stage 2b: reusing %d completed corpus full-text downloads from %s",
            len(downloaded),
            log_path,
        )
    pending: list[dict] = []

    for target in targets:
        if _is_cancelled(cancel_event):
            logger.info("Stage 2b: stop requested after %d/%d full-text items", processed, total)
            break
        url = target.get("oa_url", "")
        if not target.get("is_oa") or not url:
            continue

        processed += 1
        ident = target.get("doi") or target.get("pmc") or target["key"]
        safe_name = _safe_name(ident)
        cache_key = download_cache_key(target)
        cached = persisted_rows.get(cache_key)
        if cached and reusable_download_record(cached):
            if download_record_status(cached) != "downloaded":
                cached["status"] = "downloaded"
                persisted_rows[cache_key] = cached
                _flush_log()
            logger.info("  Reused cached corpus full text for %s", ident)
            _emit_progress(
                progress_callback,
                f"Reused cached corpus full text {processed}/{total}: {ident}",
                processed,
                max(1, total),
            )
            continue
        recovered = recover_download_artifact(output_dir, safe_name)
        if recovered:
            row = {
                **target,
                **recovered,
                "status": "downloaded",
            }
            downloaded.append(row)
            persisted_rows[cache_key] = row
            _flush_log()
            logger.info("  Recovered existing corpus full text from disk for %s", ident)
            _emit_progress(
                progress_callback,
                f"Recovered existing corpus full text {processed}/{total}: {ident}",
                processed,
                max(1, total),
            )
            continue
        cached_status = download_record_status(cached or {})
        if cached and _is_terminal_download_row(cached):
            logger.info("  Skipping previously checked corpus full text for %s (%s)", ident, cached_status)
            _emit_progress(
                progress_callback,
                f"Skipping previously checked corpus full text {processed}/{total}: {ident} ({cached_status})",
                processed,
                max(1, total),
            )
            continue
        pending.append({
            "target": target,
            "safe_name": safe_name,
            "cache_key": cache_key,
            "ident": ident,
        })

    async def _download_pending():
        import httpx

        per_host = max(1, int(workers))
        max_total = max(per_host, _MAX_CONCURRENT_DOWNLOADS)
        throttle = _StatusRecordingThrottle(default_concurrency=per_host, max_total=max_total)
        # The global cap matches the connection pool, so requests never queue
        # for a pooled connection long enough to hit the pool timeout.
        limits = httpx.Limits(max_connections=max_total, max_keepalive_connections=max_total)
        timeout = httpx.Timeout(_HTTP_TIMEOUT_SECONDS, pool=_HTTP_POOL_TIMEOUT_SECONDS)
        async with httpx.AsyncClient(
            timeout=timeout, limits=limits, follow_redirects=True,
        ) as client:
            async def _download_one(item: dict) -> dict:
                target = item["target"]
                url = target.get("oa_url", "")
                try:
                    result = await download_preferred_fulltext_async(
                        client,
                        throttle,
                        url,
                        output_dir,
                        item["safe_name"],
                        logger,
                    )
                    if result:
                        return {
                            "cache_key": item["cache_key"],
                            "row": {
                                **target,
                                **result,
                                "status": "downloaded",
                            },
                        }
                    return {
                        "cache_key": item["cache_key"],
                        "row": _unusable_download_row(
                            target, url, throttle.statuses.get(url)
                        ),
                    }
                except Exception as e:
                    return {
                        "cache_key": item["cache_key"],
                        "row": {
                            **target,
                            "status": "error",
                            "source_url": url,
                            "error": str(e),
                        },
                    }

            tasks = [asyncio.create_task(_download_one(item)) for item in pending]
            completed = 0
            try:
                for task in asyncio.as_completed(tasks):
                    if _is_cancelled(cancel_event):
                        logger.info(
                            "Stage 2b: stop requested after %d/%d async full-text items",
                            completed,
                            len(pending),
                        )
                        for pending_task in tasks:
                            if not pending_task.done():
                                pending_task.cancel()
                        break

                    payload = await task
                    completed += 1
                    row = payload["row"]
                    cache_key = payload["cache_key"]
                    persisted_rows[cache_key] = row
                    if row.get("status") == "downloaded":
                        downloaded.append(row)
                        logger.info(
                            "  Full text %d/%d: %s (%s)",
                            processed - len(pending) + completed,
                            total,
                            row.get("doi") or row.get("pmc") or row.get("key"),
                            row.get("source_format", "unknown"),
                        )
                    elif row.get("status") == "no_usable_text":
                        logger.warning(
                            "  Full text skipped for %s: no usable text extracted",
                            row.get("doi") or row.get("pmc") or row.get("key"),
                        )
                    else:
                        logger.warning(
                            "  Full text failed for %s: %s",
                            row.get("doi") or row.get("pmc") or row.get("key"),
                            row.get("error", ""),
                        )
                    _flush_log()
                    _emit_progress(
                        progress_callback,
                        f"Processed corpus full text {processed - len(pending) + completed}/{total}",
                        processed - len(pending) + completed,
                        max(1, total),
                    )
            finally:
                await asyncio.gather(*tasks, return_exceptions=True)

    if pending:
        asyncio.run(_download_pending())

    write_json_atomic(log_path, list(persisted_rows.values()))
    return downloaded


def _merge_into_download_log(log_path: Path, rows: list[dict]) -> dict[str, dict]:
    """Fold ``rows`` into the persisted download log and return the merged rows.

    ``update_fulltext_store`` rewrites archive paths on the downloaded rows, so
    they are merged rather than replacing the log. A wholesale overwrite would
    drop the "no_usable_text" and "error" entries that let the next run skip
    papers it has already checked.
    """
    persisted_rows = load_download_cache(log_path)
    for row in rows:
        persisted_rows[download_cache_key(row)] = row
    write_json_atomic(log_path, list(persisted_rows.values()))
    return persisted_rows


def run(
    config,
    corpus_path: Path | None = None,
    output_path: Path | None = None,
    email: str | None = None,
    progress_callback=None,
    cancel_event=None,
) -> dict:
    """Build a corpus enriched with full-text plaintext where available."""
    from stages.s7_fulltext import OA_LOOKUP_SKIPPED_REASON, check_oa_status

    corpus_path = corpus_path or config.corpus_dir / "corpus.json"
    output_path = output_path or config.corpus_dir / "corpus_fulltext.json"
    email = str(email or getattr(config, "pubmed_email", "") or "").strip()
    oa_skip: dict[str, str] = {}
    if not email:
        # Never send a placeholder contact; PMC and corpus OA hints still apply.
        oa_skip = {"oa_lookup_skipped": OA_LOOKUP_SKIPPED_REASON}
        logger.warning("Stage 2b: %s", OA_LOOKUP_SKIPPED_REASON)
        _emit_progress(progress_callback, f"WARNING: {OA_LOOKUP_SKIPPED_REASON}")

    logger.info("Stage 2b: loading corpus from %s", corpus_path)
    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)

    targets = _build_targets(corpus)
    _emit_progress(progress_callback, f"Identified {len(targets)} corpus targets", 0, max(1, len(targets)))
    doi_targets = [t for t in targets if t.get("doi")]
    logger.info(
        "Stage 2b: identified %d unique targets, %d with DOI for OA lookup",
        len(targets),
        len(doi_targets),
    )
    _emit_progress(progress_callback, "Checking corpus open-access availability...", 0, max(1, len(targets)))
    oa_cache_path = config.data_dir / "fulltext_corpus" / "oa_status.json"
    oa_rows = check_oa_status(
        doi_targets,
        email=email,
        progress_callback=progress_callback,
        progress_label="Checking corpus open-access availability",
        cache_path=oa_cache_path,
        cancel_event=cancel_event,
    ) if doi_targets else []
    if _is_cancelled(cancel_event):
        _emit_progress(progress_callback, "Stop requested during corpus open-access lookup", 0, max(1, len(targets)))
        return {
            "paper_count": len(corpus),
            "fulltext_count": 0,
            "corpus_path": str(output_path),
            "download_dir": str(config.data_dir / "fulltext_corpus"),
            **oa_skip,
            "cancelled": True,
        }
    targets = _merge_oa_locations(targets, oa_rows)
    oa_count = sum(1 for target in targets if target.get("is_oa") and target.get("oa_url"))
    logger.info("Stage 2b: %d targets have an open-access full-text URL", oa_count)

    download_dir = config.data_dir / "fulltext_corpus"
    download_log_path = download_dir / "download_log.json"
    downloaded = _download_fulltext(
        targets,
        download_dir,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
        log_path=download_log_path,
        workers=max(1, int(config.fulltext_download_workers)),
    )
    store_metadata = {
        "archive_dir": str(download_dir / "source_archive"),
        "archived_originals": 0,
        "paperqa": {"available": False},
    }
    if downloaded:
        from process.fulltext_store import update_fulltext_store

        _emit_progress(
            progress_callback,
            "Indexing extracted full text and archiving originals...",
            0,
            max(1, len(downloaded)),
        )
        downloaded, store_metadata = update_fulltext_store(
            downloaded,
            download_dir,
            workers=max(1, int(config.fulltext_download_workers)),
            paperqa_enabled=bool(getattr(config, "paperqa_index_enabled", True)),
            paperqa_batch_size=int(getattr(config, "paperqa_index_batch_size", 100)),
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )
        paperqa_meta = store_metadata.get("paperqa", {})
        if paperqa_meta.get("available"):
            logger.info(
                "Stage 2b: PaperQA indexed %d new texts (%d total) -> %s",
                paperqa_meta.get("added", 0),
                paperqa_meta.get("indexed_total", 0),
                paperqa_meta.get("docs_path", ""),
            )
        else:
            logger.info(
                "Stage 2b: PaperQA indexing skipped (%s)",
                paperqa_meta.get("reason")
                or paperqa_meta.get("error", "paper-qa not installed"),
            )
        logger.info(
            "Stage 2b: archived %d original full-text sources -> %s",
            store_metadata.get("archived_originals", 0),
            store_metadata.get("archive_dir", ""),
        )
    # Record archive paths before any cancel return: the originals are gone
    # once archived, so a log still pointing at them loses source provenance.
    persisted_rows = _merge_into_download_log(download_log_path, downloaded)
    if _is_cancelled(cancel_event):
        _emit_progress(progress_callback, "Stop requested during corpus full-text download", 0, max(1, len(downloaded)))
        return {
            "paper_count": len(corpus),
            "fulltext_count": len(downloaded),
            "corpus_path": str(output_path),
            "download_dir": str(download_dir),
            "archive_dir": store_metadata.get("archive_dir", ""),
            "paperqa": store_metadata.get("paperqa", {}),
            **oa_skip,
            "cancelled": True,
        }

    # Index the downloads by paper key, holding only the file location — not
    # the text.  Reading every document up front would put the whole corpus
    # worth of full text in memory at once, on top of the corpus itself and
    # the enriched copy of it.
    downloaded_map: dict[str, dict] = {}
    for row in downloaded:
        downloaded_map[row["key"]] = {
            "full_text_path": row["path"],
            "full_text_source_path": row.get("source_path") or row["path"],
            "full_text_source_archive_path": row.get("source_archive_path", ""),
            "full_text_source_format": row.get("source_format", "html"),
            "full_text_source_url": row.get("source_url", ""),
        }

    outcome_counts = _download_outcome_counts(persisted_rows, targets)
    if outcome_counts["download_failed"]:
        logger.warning(
            "Stage 2b: %d full-text downloads failed (errors, rate limits or server "
            "errors); they will be retried on the next run",
            outcome_counts["download_failed"],
        )

    total_papers = len(corpus)
    stats = {"enriched": 0, "unreadable": 0, "cancelled": False}

    def _enriched_records():
        """Yield enriched records one at a time.

        Each full text is read immediately before its record is serialized and
        released straight after, so peak memory is one document rather than
        all of them.
        """
        for idx, paper in enumerate(corpus, start=1):
            record = dict(paper)
            meta = downloaded_map.get(_paper_key(paper))

            if meta and not stats["cancelled"]:
                if _is_cancelled(cancel_event):
                    # Keep emitting the remaining papers so the output still
                    # covers the whole corpus; they simply fall back to their
                    # abstracts, which is what a partial run produced before.
                    stats["cancelled"] = True
                else:
                    try:
                        clean_text = Path(meta["full_text_path"]).read_text(
                            encoding="utf-8")
                    except OSError as exc:
                        stats["unreadable"] += 1
                        logger.warning(
                            "  Could not read downloaded full text for %s (%s); "
                            "falling back to abstract",
                            _paper_key(paper) or "unknown", exc,
                        )
                    else:
                        record.update(meta)
                        record["full_text"] = clean_text
                        record["full_text_chars"] = len(clean_text)
                        record["text_source"] = "full_text"
                        stats["enriched"] += 1

            record.setdefault("text_source", "abstract")

            if idx % 50 == 0 or idx == total_papers:
                _emit_progress(
                    progress_callback,
                    f"Writing enriched corpus {idx}/{total_papers}",
                    idx,
                    max(1, total_papers),
                )
            yield record

    paper_count = atomic_write_json_array(output_path, _enriched_records())

    enriched_count = stats["enriched"]
    if stats["unreadable"]:
        logger.warning(
            "Stage 2b: %d downloaded full texts could not be read from disk",
            stats["unreadable"],
        )

    if stats["cancelled"]:
        _emit_progress(
            progress_callback,
            "Stop requested while embedding downloaded full text",
            enriched_count,
            max(1, total_papers),
        )
        return {
            "paper_count": paper_count,
            "fulltext_count": enriched_count,
            "corpus_path": str(output_path),
            "download_dir": str(download_dir),
            "archive_dir": store_metadata.get("archive_dir", ""),
            "paperqa": store_metadata.get("paperqa", {}),
            **outcome_counts,
            **oa_skip,
            "cancelled": True,
        }

    logger.info(
        "Stage 2b complete: %d/%d papers enriched with full text -> %s",
        enriched_count,
        paper_count,
        output_path,
    )
    _emit_progress(
        progress_callback,
        f"Corpus full-text enrichment complete: {enriched_count}/{paper_count} papers",
        enriched_count,
        max(1, paper_count),
    )
    return {
        "paper_count": paper_count,
        "fulltext_count": enriched_count,
        "corpus_path": str(output_path),
        "download_dir": str(download_dir),
        "archive_dir": store_metadata.get("archive_dir", ""),
        "paperqa": store_metadata.get("paperqa", {}),
        **outcome_counts,
        **oa_skip,
    }
