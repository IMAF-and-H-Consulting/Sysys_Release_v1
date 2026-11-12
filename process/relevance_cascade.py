"""Calibrated pw4 passage screening, preserving the package's exact protocol.

The 0/5 relevance_score is a binary compatibility projection, not a probability
or a calibrated ordinal rating. Filtering always uses screening.passed.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from functools import lru_cache
from pathlib import Path

from core.io import atomic_write_json

logger = logging.getLogger(__name__)

MODES = ("laya-bert", "bert-laya", "laya-only", "packaged-bert")
DEFAULT_PACKAGE = Path(__file__).resolve().parents[1] / "laya-screener-pw4"
MANIFEST_NAME = "SHA256SUMS.txt"
_HASH_CHUNK = 1 << 20
_MANIFEST_LINE = re.compile(r"^([0-9a-fA-F]{64}) [ *](.+)$")


class ModelIntegrityError(RuntimeError):
    """A pw4 model file is missing, altered, or not covered by SHA256SUMS.txt."""


def make_question(question: str) -> dict:
    return {
        "type": "choice",
        "instructions": f"Is this passage relevant to the research question: {question.strip()}",
        "criteria": {
            "A": "yes, it addresses the research question",
            "B": "no, it does not address the research question",
        },
    }


def probability_bin(p: float) -> int:
    return 1 + sum(p >= edge for edge in (0.2, 0.4, 0.6, 0.8))


def package_path(config=None) -> Path:
    value = getattr(config, "laya_screener_dir", "")
    return Path(value).expanduser().resolve() if value else DEFAULT_PACKAGE


def settings(config=None) -> dict:
    return {
        "package": str(package_path(config)),
        "device": getattr(config, "laya_device", "auto"),
        "batch_size": getattr(config, "laya_batch_size", 8),
        "cut_bin": getattr(config, "laya_cut_bin", 4),
        "brt_threshold": getattr(config, "laya_brt_threshold", 0.236),
    }


def model_signature(package: str | Path) -> list:
    """Invalidate resumes when a checkpoint, tokenizer or calibration changes."""
    root = Path(package)
    return [(str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted((root / "model").rglob("*")) if p.is_file()]


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


# Integrity (audit 21-M2). A local equivalent of laya_screener.verify_model_dir:
# importing the package's own module would run the code being verified.

def read_manifest(manifest: Path) -> dict[str, str]:
    """`shasum -a 256` output -> {relative path: lowercase hex digest}."""
    entries = {}
    for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        match = _MANIFEST_LINE.match(line)
        if not match:
            raise ModelIntegrityError(
                f"{manifest} line {number} is not a SHA-256 checksum line: {line[:80]!r}")
        entries[match.group(2)] = match.group(1).lower()
    return entries


@lru_cache(maxsize=64)
def _cached_sha256(path: str, size: int, mtime_ns: int) -> str:
    """Keyed by (size, mtime): a checkpoint is hashed once per process unless it changes."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entry_problem(root: Path, name: str, expected: str) -> str | None:
    path = root / name
    if not path.is_file():
        return f"{name}: missing"
    stat = path.stat()
    actual = _cached_sha256(str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    return None if actual == expected else f"{name}: sha256 {actual[:12]}... != expected {expected[:12]}..."


def verify_model_dir(package: Path, name: str) -> int:
    """Check every manifest entry under model/<name>/; returns how many files were verified."""
    manifest = Path(package) / MANIFEST_NAME
    if not manifest.is_file():
        raise ModelIntegrityError(
            f"Checksum manifest {manifest} is missing, so the {name} model cannot be verified. "
            "Re-extract the pw4 integration zip or set laya_screener_dir.")
    prefix = f"model/{name}/"
    entries = {n: d for n, d in read_manifest(manifest).items() if n.startswith(prefix)}
    if not entries:
        raise ModelIntegrityError(f"{manifest} lists no files under {prefix}; cannot verify the {name} model.")
    problems = [p for p in (_entry_problem(manifest.parent, n, d) for n, d in sorted(entries.items())) if p]
    if problems:
        raise ModelIntegrityError(
            f"The {name} model in {package} failed checksum verification ({len(problems)} of "
            f"{len(entries)} files): {'; '.join(problems)}. Re-extract the pw4 integration zip.")
    return len(entries)


def laya_truncation(answer: dict) -> dict:
    """Whether Laya's token window cut a passage (audit 21-M1); None when Laya does not say."""
    usage = answer.get("usage") or {}
    if "truncated" not in usage and "state_tokens_dropped" not in usage:
        return {"laya_truncated": None, "laya_state_tokens_dropped": None}
    dropped = int(usage.get("state_tokens_dropped") or 0)
    return {"laya_truncated": bool(usage.get("truncated", dropped > 0)),
            "laya_state_tokens_dropped": dropped}


_NOT_READ_BY_LAYA = {"laya_truncated": None, "laya_state_tokens_dropped": None}


def paper_passage(paper: dict, workload="abstract", abstract_chars=0,
                  fulltext_chars=0) -> tuple[str, str]:
    # Do not apply LLM prompt compression or question-to-claim rewriting:
    # normalization is already part of the calibrated Laya tokenizer.
    if workload == "fulltext" and paper.get("full_text"):
        text = str(paper["full_text"]).strip()
        limit = max(0, int(fulltext_chars or 0))
        return (text[:limit] if limit else text), "full_text"
    text = str(paper.get("abstract") or "").strip()
    limit = max(0, int(abstract_chars or 0))
    return (text[:limit] if limit else text), "abstract"


class CascadeScreener:
    def __init__(self, package=DEFAULT_PACKAGE, *, device="auto", batch_size=8,
                 cut_bin=4, brt_threshold=0.236, verify_checksums=True):
        if not isinstance(cut_bin, int) or not 1 <= cut_bin <= 5:
            raise ValueError("Laya cut_bin must be an integer from 1 to 5")
        if not math.isfinite(brt_threshold) or not 0 <= brt_threshold <= 1:
            raise ValueError("BRT threshold must be a probability from 0 to 1")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("Laya batch_size must be positive")
        self.package = Path(package)
        self.device = device
        self.batch_size = batch_size
        self.cut_bin = cut_bin
        self.brt_threshold = brt_threshold
        # Only an explicit False (for a model the manifest does not cover) skips 21-M2.
        self.verify_checksums = verify_checksums
        self.agent = self.brt = None
        self.last_timings = {}
        self.last_laya_truncation = None

    def prepare(self, mode):
        if mode not in MODES:
            raise ValueError(f"Unknown screening mode: {mode}")
        import torch
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        needed = ["brt-bert"] if mode == "packaged-bert" else ["laya"]
        if mode in ("laya-bert", "bert-laya"):
            needed.append("brt-bert")
        for name in needed:
            if not (self.package / "model" / name / "model.safetensors").is_file():
                raise FileNotFoundError(f"Missing {name} checkpoint in {self.package}. Extract the pw4 integration zip here or set laya_screener_dir.")
        if "laya" in needed and self.agent is None:
            self._verify("laya")
            import laya
            self.agent = laya.Agent(str(self.package / "model/laya"), device=self.device)
        if "brt-bert" in needed and self.brt is None:
            self._verify("brt-bert")
            from sentence_transformers import CrossEncoder
            self.brt = CrossEncoder(str(self.package / "model/brt-bert"), device=self.device)

    def _verify(self, name):
        if not self.verify_checksums:
            logger.warning("Loading the %s model from %s without checksum verification", name, self.package)
            return
        count = verify_model_dir(self.package, name)
        logger.info("Verified %d %s model files against %s", count, name, MANIFEST_NAME)

    @staticmethod
    def _probabilities(values, count):
        values = [float(v) for v in values]
        if len(values) != count or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError("Screening model returned invalid probabilities or prediction count")
        return values

    def laya_scores(self, question, passages):
        answers = self.agent.predict_batch(
            passages, {"relevant": make_question(question)}, batch_size=self.batch_size)
        values = self._probabilities(
            [a["answers"]["relevant"]["probabilities"]["A"] for a in answers], len(passages))
        # Side channel like last_timings, so laya_scores keeps returning plain probabilities.
        self.last_laya_truncation = [laya_truncation(a) for a in answers]
        return values

    def bert_scores(self, question, passages):
        values = self.brt.predict([(question, p) for p in passages],
                                  batch_size=self.batch_size, show_progress_bar=False)
        # The bundled 1-label CrossEncoder already applies sigmoid.
        return self._probabilities(values, len(passages))

    def screen(self, question, passages, mode="laya-bert"):
        question = question.strip()
        if not question:
            raise ValueError("Laya screening requires the original research question")
        if mode not in MODES:
            raise ValueError(f"Unknown screening mode: {mode}")
        self.last_timings = {"laya_seconds": 0.0, "bert_seconds": 0.0}
        if not passages:
            return []
        self.prepare(mode)
        laya, bert = [None] * len(passages), [None] * len(passages)
        truncation = [_NOT_READ_BY_LAYA] * len(passages)
        order = {"laya-bert": ("laya", "bert"), "bert-laya": ("bert", "laya"),
                 "laya-only": ("laya",), "packaged-bert": ("bert",)}[mode]
        active = [i for i, passage in enumerate(passages) if passage.strip()]
        for name in order:
            if not active:
                break
            start = time.perf_counter()
            self.last_laya_truncation = None
            values = getattr(self, name + "_scores")(question, [passages[i] for i in active])
            values = self._probabilities(values, len(active))
            self.last_timings[name + "_seconds"] = time.perf_counter() - start
            for i, value in zip(active, values):
                (laya if name == "laya" else bert)[i] = value
            if name == "laya" and self.last_laya_truncation is not None:
                for i, info in zip(active, self.last_laya_truncation):
                    truncation[i] = info
            active = [i for i in active if (
                probability_bin(laya[i]) >= self.cut_bin if name == "laya"
                else bert[i] >= self.brt_threshold)]
        kept = set(active)
        return [self._result_row(mode, laya[i], bert[i], i in kept, passages[i], truncation[i])
                for i in range(len(passages))]

    def _result_row(self, mode, laya_p, brt_p, passed, passage, truncation) -> dict:
        return {
            "mode": mode, "laya_p": laya_p,
            "laya_bin": probability_bin(laya_p) if laya_p is not None else None,
            "laya_passed": probability_bin(laya_p) >= self.cut_bin if laya_p is not None else None,
            "brt_p": brt_p, "brt_passed": brt_p >= self.brt_threshold if brt_p is not None else None,
            "passed": passed, "status": "scored" if passage.strip() else "missing_text",
            "cut_bin": self.cut_bin, "brt_threshold": self.brt_threshold, **truncation,
        }


def relevance_result(result, source="abstract") -> dict:
    return {"relevance_score": 5 if result["passed"] else 0,
            "relevance_quote": "", "relevance_quote_source": source,
            "relevance_quote_valid": False, "screening": result,
            "relevance_score_kind": "binary_screening_decision"}


def _resume_identity(papers, inputs, mode, opts, hypothesis) -> dict:
    """What a resumable result depends on. ``batch_size`` only groups the
    work, so changing it keeps the resume (audit 05-L6). Version 3 rows carry
    Laya truncation (audit 21-M1), so older checkpoints are not mixed in."""
    scoring = {key: value for key, value in opts.items() if key != "batch_size"}
    return {"version": 3, "mode": mode, "settings": scoring, "question": hypothesis,
            "papers": papers, "inputs": inputs, "models": model_signature(opts["package"])}


def _load_checkpoint(checkpoint: Path, total: int) -> list:
    """Resumed rows; an unreadable checkpoint is discarded, not fatal."""
    if not checkpoint.exists():
        return []
    try:
        results = json.loads(checkpoint.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Discarding unreadable cascade checkpoint %s: %s", checkpoint, exc)
        return []
    if not isinstance(results, list) or len(results) > total:
        raise ValueError(f"Invalid cascade checkpoint: {checkpoint}")
    return results


def _paper_ref(paper: dict, index: int) -> str:
    for key in ("id", "doi", "pmid", "title"):
        value = str(paper.get(key) or "").strip()
        if value:
            return value
    return f"index {index}"


def _missing_text_papers(results: list) -> list[str]:
    return [_paper_ref(p, i) for i, p in enumerate(results)
            if p["screening"]["status"] == "missing_text"]


def _report_missing_text(missing: list[str], total: int, progress_callback) -> None:
    """Papers with no text are not kept; ask for manual screening (audit 05-M8)."""
    if not missing:
        return
    message = (f"{len(missing)} paper(s) had no text to screen and were not kept; "
               "screen them manually (listed under missing_text_papers in "
               "screening_report.json).")
    logger.warning(message)
    if progress_callback:
        progress_callback({"message": message, "current": total, "total": total})


def _truncated_papers(results: list) -> list[str]:
    return [_paper_ref(p, i) for i, p in enumerate(results)
            if p["screening"].get("laya_truncated") is True]


def _report_truncated(truncated: list[str], total: int, progress_callback) -> None:
    """Laya judged these on a prefix of their text (audit 21-M1)."""
    if not truncated:
        return
    message = (f"{len(truncated)} paper(s) did not fit Laya's token window (512 for pw4) and were "
               "judged on a truncated passage (listed under laya_truncated_papers in "
               "screening_report.json).")
    logger.warning(message)
    if progress_callback:
        progress_callback({"message": message, "current": total, "total": total})


def filter_cascade(papers, config, *, hypothesis, workload="abstract",
                   cancel_event=None, progress_callback=None):
    """Batch, cancel and resume without mixing questions, inputs or experiments."""
    mode = config.relevance_screening_mode
    opts = settings(config)
    abstract_limit = max(0, int(config.score_abstract_chars or 0))
    fulltext_limit = max(0, int(config.score_fulltext_chars or 0))
    input_policy = {
        "action": (
            "explicit-user-character-limits"
            if abstract_limit or fulltext_limit else
            "complete-source-until-model-architecture-limit"
        ),
        "abstract_character_limit": abstract_limit or None,
        "fulltext_character_limit": fulltext_limit or None,
        "reason": (
            "Positive character limits were explicitly configured by the user."
            if abstract_limit or fulltext_limit else
            "No synthetic character cap was applied; Laya reports any truncation at its "
            "real 512-token architecture boundary."
        ),
    }
    logger.info("Laya/BRT input action: %s", input_policy["reason"])
    inputs = [paper_passage(p, workload, config.score_abstract_chars, config.score_fulltext_chars)
              for p in papers]
    key = fingerprint(_resume_identity(papers, inputs, mode, opts, hypothesis))
    checkpoint = config.claims_dir / f"cascade_checkpoint_{key}.json"
    results = _load_checkpoint(checkpoint, len(papers))
    screener = CascadeScreener(**opts)
    start = time.perf_counter()
    cancelled = False
    timings = {"laya_seconds": 0.0, "bert_seconds": 0.0}
    resumed = len(results)
    for offset in range(len(results), len(papers), screener.batch_size):
        if cancel_event is not None and cancel_event.is_set():
            cancelled = True
            break
        if progress_callback:
            progress_callback({"message": f"{mode}: scoring {offset}/{len(papers)}", "current": offset, "total": len(papers)})
        chunk = inputs[offset:offset + screener.batch_size]
        predictions = screener.screen(hypothesis, [text for text, _ in chunk], mode)
        results.extend({**papers[offset + i], **relevance_result(pred, chunk[i][1])}
                       for i, pred in enumerate(predictions))
        for name in timings:
            timings[name] += screener.last_timings[name]
        atomic_write_json(checkpoint, results)
    if cancel_event is not None and cancel_event.is_set():
        cancelled = True
    passed = [p for p in results if p["screening"]["passed"]]
    missing = _missing_text_papers(results)
    truncated = _truncated_papers(results)
    report = {"mode": mode, "settings": opts, "input_policy": input_policy,
              "question": hypothesis, "fingerprint": key,
              "input_count": len(papers), "scored_count": len(results), "kept_count": len(passed),
              "laya_evaluated": sum(p["screening"]["laya_p"] is not None for p in results),
              "bert_evaluated": sum(p["screening"]["brt_p"] is not None for p in results),
              "missing_text": len(missing), "missing_text_papers": missing,
              "laya_truncated": len(truncated), "laya_truncated_papers": truncated,
              "cancelled": cancelled, "resumed_count": resumed,
              "session_seconds": time.perf_counter() - start, "session_inference_seconds": timings,
              "score_kind": "binary_screening_decision", "quality_metrics": "Requires human labels"}
    atomic_write_json(config.claims_dir / "screening_report.json", report)
    # Keep a separate result for each experiment so baseline runs remain reviewable.
    atomic_write_json(config.claims_dir / "screening_experiments" / f"{key}.json",
                      {"report": report, "papers": results})
    if not cancelled:
        # Other experiments' checkpoints stay: each question/mode resumes on its own.
        checkpoint.unlink(missing_ok=True)
    _report_missing_text(missing, len(results), progress_callback)
    _report_truncated(truncated, len(results), progress_callback)
    if progress_callback:
        progress_callback({"message": f"{mode}: kept {len(passed)}/{len(results)}", "current": len(results), "total": len(papers)})
    return passed, cancelled, results
