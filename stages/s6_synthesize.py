"""Stage 6: Cluster claims, generate narratives, draft abstract."""
import hashlib
import json
import logging
from collections.abc import Callable
from concurrent.futures import CancelledError
from functools import partial
from pathlib import Path
from typing import Any

from core.config import DEFAULT_EMBEDDING_MODEL
from core.io import atomic_write_json, atomic_write_text
from process.term_normalization import inspect_claim_normalization
from stages._common import is_cancelled, make_progress

logger = logging.getLogger(__name__)

DEFAULT_CLUSTERS_FILENAME = "clusters.json"
SYNTHESIS_PROGRESS_FILENAME = "synthesis_progress.json"
# Every artifact a Stage 6 run writes next to the clusters file.
STAGE6_ARTIFACTS = (
    "narratives.json",
    "k_selection.json",
    "abstract_draft.md",
    "hypothesis_explanation.json",
    "hypothesis_explanation.md",
    "evidence_network.json",
    "synthesis_sources.json",
    "synthesis_sources.md",
    "synthesis_sources.bib",
    "clusters.partial.json",
    "narratives.partial.json",
    "k_selection.partial.json",
    "synthesis_partial.md",
    SYNTHESIS_PROGRESS_FILENAME,
)
_CANCELLED = {"cancelled": True}


def run(
    config=None,
    claims_path: str | None = None,
    n_clusters: int = 8,
    embedding_model: str | None = None,
    max_tokens: int | None = None,
    cancel_event: Any = None,
    progress_callback: Callable[[str], None] | None = None,
    output_path: str | None = None,
    k_max: int = 15,
    review_only: bool = False,
) -> dict:
    """Run Stage 6: cluster claims, generate per-cluster narratives, draft abstract.

    Steps
    -----
    1. Load claims (filtered or unfiltered)
    2. Cluster papers with embeddings
    3. Generate narrative per cluster via LLM
    4. Draft structured abstract via LLM
    5. Save clusters.json, narratives.json, abstract_draft.md and provenance

    *output_path* (the GUI "Synthesis Output" field) names the clusters
    file; every other artifact is written beside it. Default:
    ``config.clusters_dir / "clusters.json"``.

    Returns
    -------
    dict
        Summary with keys: total_papers, n_clusters, narratives_count,
        abstract_length, warnings, output paths.
    """
    from process.llm import set_config
    from process.paths import resolve_claims_path

    _progress = make_progress(logger, progress_callback)
    _cancelled = partial(is_cancelled, cancel_event)

    # ── 0. Configure LLM ────────────────────────────────────────────────
    if config is None:
        raise ValueError("config is required but was not provided")
    set_config(config)
    _progress(f"LLM: provider={config.llm_provider}, model={config.llm_model}")
    emb_model = embedding_model or DEFAULT_EMBEDDING_MODEL
    out_dir, clusters_name = _output_location(config, output_path)

    # ── 1. Load claims ──────────────────────────────────────────────────
    cp = Path(claims_path) if claims_path else resolve_claims_path(config)
    _progress(f"Loading claims from {cp}")
    papers = _load_papers(cp)
    _progress(f"Loaded {len(papers)} papers")
    if not papers:
        if review_only:
            raise ValueError("No articles available for cluster review")
        return _empty_run(out_dir, clusters_name, _progress)
    term_status = inspect_claim_normalization(papers, config)
    if term_status["status"] == "stale":
        fields = ", ".join(
            f"{name} ({detail['input_variants']} variants -> "
            f"{detail['canonical_terms']} canonical terms)"
            for name, detail in term_status["issues"].items()
        )
        raise ValueError(
            "The claims artifact contains uncanonical structured labels: "
            f"{fields}. Stage 6 will not normalize or hide this upstream data defect. "
            "Rerun Stage 3 (then the validation/audit stages that produce this claims "
            "input), or explicitly migrate the legacy claims artifact with "
            "scripts/normalize_structured_terms.py, so canonical labels and their "
            "raw-value audit are saved first."
        )
    input_warnings: list[str] = []
    if term_status["status"] not in {"canonical", "no_categorical_terms"}:
        if term_status["status"] == "configuration_mismatch":
            detail = (
                "Claims were canonicalized under a different normalization configuration "
                f"({term_status.get('config_mismatches', 0)} record marker(s) do not "
                "match the active project vocabulary)."
            )
        elif term_status["status"] == "mixed_markers":
            detail = "Only part of the claims artifact carries the current policy marker."
        else:
            detail = "Claims use a legacy artifact without the structured-term policy marker."
        warning = (
            f"{detail} No conflicting automatic variants were detected; Stage 6 will "
            "count labels exactly and will not normalize them. Rerun Stage 3 to save "
            "canonical labels and provenance under the active project vocabulary."
        )
        input_warnings.append(warning)
        _progress(f"WARNING: {warning}")
    if _cancelled():
        _progress("Cancelled before clustering")
        return dict(_CANCELLED)

    # ── 2. Cluster ──────────────────────────────────────────────────────
    clusters, k_info = _cluster(papers, 0 if review_only else n_clusters, emb_model, _progress, k_max=k_max)
    k_info["term_normalization"] = term_status
    warnings = [*input_warnings]
    if k_info.get("structure_warning"):
        warnings.append(k_info["structure_warning"])
    if _cancelled():
        _progress("Cancelled after clustering")
        return dict(_CANCELLED)

    if review_only:
        # Keep exploratory results separate from the completed synthesis.
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "k_review.json"
        atomic_write_json(path, k_info)
        return {"review_only": True, "total_papers": len(papers),
                "k_selection": k_info, "k_selection_path": str(path),
                "warnings": warnings}

    # ── 3-4. Narratives and abstract ────────────────────────────────────
    from process.llm_budget import resolve_model_capacity
    capacity = resolve_model_capacity(config, model=config.llm_model)
    response_override = int(max_tokens or 0)
    k_info["llm_capacity"] = capacity.as_dict()
    _progress(
        f"Token action: context={capacity.context_window_tokens} ({capacity.source}); "
        f"cluster response={response_override if response_override > 0 else 'auto'}; "
        "complete evidence is preferred and any selection/reduction is saved"
    )
    context = _load_context(config)
    warnings.extend(context["unreadable"].values())
    fingerprint = _synthesis_fingerprint(
        config, cp, clusters, capacity, response_override)
    k_info["synthesis_run_fingerprint"] = fingerprint
    config.ensure_dirs()
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / SYNTHESIS_PROGRESS_FILENAME
    progress_state = _load_synthesis_progress(progress_path, fingerprint)
    if progress_state is None:
        progress_state = _new_synthesis_progress(fingerprint, capacity)
    else:
        _progress(
            "Resuming Stage 6 from compatible progressive output: "
            f"{len(progress_state.get('narratives', []))} completed cluster narrative(s)"
        )

    def persist_progress() -> Path:
        return _persist_synthesis_progress(
            out_dir, clusters_name, clusters, k_info, progress_state)

    # Establish current-run artifacts before the first model request.  Every
    # subsequent successful request updates the same atomic files.
    persist_progress()
    try:
        drafted = _draft_texts(
            config, clusters, k_info, cp, context, response_override,
            capacity, cancel_event, _progress, progress_state, persist_progress)
    except CancelledError:
        progress_state["status"] = "cancelled"
        persist_progress()
        _progress("Cancelled during an LLM request")
        return dict(_CANCELLED)
    if drafted is None:
        progress_state["status"] = "cancelled"
        persist_progress()
        return dict(_CANCELLED)
    narratives = drafted["narratives"]
    abstract = drafted["abstract"]
    citations = drafted["citations"]
    for failure in drafted["errors"]:
        warnings.append(failure["message"])
    warnings.extend(_empty_reply_warnings(narratives, abstract, _progress))

    # ── 5. Promote usable outputs without destroying the last complete run ──
    all_clusters_finished = len(narratives) == len(clusters)
    complete_outputs = (
        drafted["status"] == "complete"
        and bool(abstract.strip())
        and all(str(item.get("narrative") or "").strip() for item in narratives)
    )
    partial_artifacts = progress_state.get("partial_artifacts") or {}
    if all_clusters_finished:
        # Cluster narratives are independently usable even when the abstract
        # is empty or unresolved, so retain the established core artifacts.
        paths = _save_core(
            out_dir, clusters_name, clusters, narratives, k_info, abstract, _progress)
        if not drafted["errors"]:
            paths.update(_save_provenance(
                out_dir, config, clusters, narratives, abstract, citations,
                context, k_info, _progress))
    else:
        # Do not replace the previous completed synthesis with an incomplete
        # cluster set. The partial paths are still returned and shown in-app.
        paths = {
            "clusters_path": partial_artifacts.get("clusters_path", ""),
            "narratives_path": partial_artifacts.get("narratives_path", ""),
        }

    progress_state["status"] = "complete" if complete_outputs else "partial"
    progress_state["abstract_status"] = "complete" if abstract.strip() else (
        progress_state.get("abstract_status") or "pending")
    progress_state["narratives"] = narratives
    persist_progress()
    paths["synthesis_progress_path"] = str(progress_path)
    paths["synthesis_partial_report_path"] = str(
        (progress_state.get("partial_artifacts") or {}).get("report_path", ""))

    if progress_state["status"] == "partial":
        _progress(
            f"Stage 6 partial: {len(narratives)}/{len(clusters)} cluster narrative(s) "
            f"saved; {len(drafted['errors'])} unresolved model step(s). "
            "Successful model work is retained for a compatible rerun."
        )
    elif warnings:
        _progress(f"Stage 6 complete with {len(warnings)} warning(s): " + "; ".join(warnings))
    else:
        _progress("Stage 6 complete")
    return {
        "total_papers": len(papers),
        "n_clusters": len(clusters),
        "narratives_count": len(narratives),
        "abstract_length": len(abstract.strip()),
        "k_selection": k_info,
        "embedding_backend": k_info.get("embedding_backend"),
        "status": progress_state["status"],
        "unresolved_model_steps": drafted["errors"],
        "resumed_narratives": drafted["resumed_narratives"],
        "warnings": warnings,
        **paths,
    }


# ---------------------------------------------------------------------------
# Input and output locations
# ---------------------------------------------------------------------------
def _output_location(config, output_path: str | None) -> tuple[Path, str]:
    """Directory and clusters filename; a suffix-less path is a directory."""
    if not output_path:
        return Path(config.clusters_dir), DEFAULT_CLUSTERS_FILENAME
    path = Path(output_path)
    if path.suffix:
        return path.parent, path.name
    return path, DEFAULT_CLUSTERS_FILENAME


def _load_papers(path: Path) -> list[dict]:
    """Paper records from a claims file: a list, or an object with an ``articles`` list."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get("articles"), list):
        data = data["articles"]
    if not isinstance(data, list) or not all(isinstance(p, dict) for p in data):
        raise ValueError(
            f"{path} must contain a list of paper records "
            "(or an object with an 'articles' list)")
    return data


def _empty_run(out_dir: Path, clusters_name: str, progress) -> dict:
    """Remove the previous run's artifacts so they are not presented as current."""
    removed = []
    partial_clusters_name = _partial_clusters_name(clusters_name)
    for name in (clusters_name, partial_clusters_name, *STAGE6_ARTIFACTS):
        path = out_dir / name
        if path.is_file():
            path.unlink()
            removed.append(name)
    warnings = ["No papers found — nothing to synthesize"]
    if removed:
        warnings.append("Removed stale Stage 6 outputs from a previous run: " + ", ".join(removed))
    for warning in warnings:
        progress(warning)
    return {"total_papers": 0, "n_clusters": 0, "narratives_count": 0,
            "abstract_length": 0, "warnings": warnings}


def _load_optional_json(path: Path) -> tuple[Any, str | None]:
    """``(data, None)``; ``(None, None)`` when absent; ``(None, error)`` when unreadable."""
    if not path.exists():
        return None, None
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, ValueError) as exc:
        logger.warning("Could not read optional synthesis context %s: %s", path, exc)
        return None, f"{path.name} could not be read: {exc}"


def _load_context(config) -> dict:
    """Optional Stage 4 method profiles and GRADE results, with read errors."""
    sources = {
        "method_profiles": config.data_dir / "method_review" / "method_profiles_validated.json",
        "grade_results": config.output_dir / "grade_results.json",
    }
    context: dict = {"unreadable": {}}
    for name, path in sources.items():
        context[name], error = _load_optional_json(path)
        if error:
            context["unreadable"][name] = error
    return context


def _claims_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cluster_member_ids(clusters: list[dict]) -> list[dict[str, Any]]:
    """Stable cluster membership used to bind checkpoints to this exact run."""
    return [
        {
            "cluster_id": cluster.get("cluster_id"),
            "papers": [
                str(
                    paper.get("paper_id")
                    or paper.get("doi")
                    or paper.get("pmid")
                    or paper.get("title")
                    or index
                )
                for index, paper in enumerate(cluster.get("papers", []) or [])
            ],
        }
        for cluster in clusters
    ]


def _synthesis_fingerprint(
    config,
    claims_path: Path,
    clusters: list[dict],
    capacity,
    response_override: int,
) -> str:
    """Fingerprint every input that can change a model-facing Stage 6 request."""
    payload = {
        "policy": "context-scaled-progressive-synthesis/v1",
        "claims_sha256": _claims_digest(claims_path),
        "clusters": _cluster_member_ids(clusters),
        "provider": str(getattr(config, "llm_provider", "") or ""),
        "model": str(getattr(config, "llm_model", "") or ""),
        "capacity": capacity.as_dict(),
        "cluster_output_override": int(response_override or 0),
        "abstract_output_override": int(
            getattr(config, "synthesis_abstract_max_tokens", 0) or 0),
        "research_config": getattr(config, "research_config", None),
    }
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, default=str,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _clusters_light(clusters: list[dict]) -> list[dict]:
    return [
        {**{key: value for key, value in cluster.items() if key != "papers"},
         "paper_titles": [paper.get("title", "")
                          for paper in cluster.get("papers", [])]}
        for cluster in clusters
    ]


def _partial_clusters_name(clusters_name: str) -> str:
    path = Path(clusters_name)
    suffix = path.suffix or ".json"
    stem = path.stem if path.suffix else path.name
    return f"{stem}.partial{suffix}"


def _partial_synthesis_markdown(state: dict[str, Any]) -> str:
    narratives = state.get("narratives") or []
    errors = state.get("errors") or []
    lines = [
        "# Partial Stage 6 synthesis",
        "",
        f"Status: **{state.get('status', 'in_progress')}**.",
        f"Completed cluster narratives: {len(narratives)}.",
        "",
        "This is current-run work, not a completed synthesis. Compatible reruns use the "
        "fingerprinted progress artifact and do not repeat saved model requests.",
        "",
    ]
    for narrative in narratives:
        lines.extend([
            f"## Cluster {narrative.get('cluster_id', '?')}",
            "",
            str(narrative.get("narrative") or "No narrative text was returned."),
            "",
        ])
    if errors:
        lines.extend(["## Unresolved model actions", ""])
        lines.extend(
            f"- {error.get('message') or error}" if isinstance(error, dict)
            else f"- {error}"
            for error in errors
        )
        lines.append("")
    return "\n".join(lines)


def _load_synthesis_progress(path: Path, fingerprint: str) -> dict[str, Any] | None:
    """Load only an incomplete checkpoint created for the exact same requests."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Ignoring unreadable Stage 6 progress checkpoint %s: %s", path, exc)
        return None
    if not isinstance(data, dict) or data.get("run_fingerprint") != fingerprint:
        logger.info("Stage 6 progress checkpoint does not match this run; starting fresh")
        return None
    if data.get("status") == "complete":
        return None
    return data


def _new_synthesis_progress(fingerprint: str, capacity) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_fingerprint": fingerprint,
        "status": "in_progress",
        "token_policy": "context-scaled-progressive-synthesis/v1",
        "model_capacity": capacity.as_dict(),
        "narratives": [],
        "completed_cluster_ids": [],
        "cluster_work": {},
        "errors": [],
        "abstract_status": "pending",
    }


def _persist_synthesis_progress(
    out_dir: Path,
    clusters_name: str,
    clusters: list[dict],
    k_info: dict,
    state: dict[str, Any],
) -> Path:
    """Durably expose successful model work after every Stage 6 request."""
    out_dir.mkdir(parents=True, exist_ok=True)
    partial_paths = {
        "clusters_path": out_dir / _partial_clusters_name(clusters_name),
        "narratives_path": out_dir / "narratives.partial.json",
        "k_selection_path": out_dir / "k_selection.partial.json",
        "report_path": out_dir / "synthesis_partial.md",
    }
    atomic_write_json(partial_paths["clusters_path"], _clusters_light(clusters))
    atomic_write_json(partial_paths["narratives_path"], state.get("narratives", []))
    atomic_write_json(partial_paths["k_selection_path"], k_info)
    atomic_write_text(partial_paths["report_path"], _partial_synthesis_markdown(state))
    state["partial_artifacts"] = {
        key: str(value) for key, value in partial_paths.items()
    }
    path = out_dir / SYNTHESIS_PROGRESS_FILENAME
    atomic_write_json(path, state)
    return path


# ---------------------------------------------------------------------------
# Clustering and LLM drafting
# ---------------------------------------------------------------------------
def _cluster(papers: list[dict], n_clusters: int, emb_model: str, progress, k_max: int = 15) -> tuple[list, dict]:
    from process import synthesizer as synth

    if n_clusters and n_clusters >= 1:
        progress(f"Clustering {len(papers)} papers into up to {n_clusters} clusters "
                 f"(embedding model: {emb_model})")
    else:
        progress(f"Clustering {len(papers)} papers with auto k "
                 f"(silhouette sweep, embedding model: {emb_model})")
    k_info: dict = {}
    clusters = synth.cluster_claims(
        papers, n_clusters=n_clusters, embedding_model=emb_model, k_info=k_info, k_max=k_max)
    if "article_count" in k_info:
        progress(f"Auto-k limits for {k_info['article_count']} articles: "
                 f"maximum K={k_info['search_max_k']}, small-cluster advisory threshold "
                 f"{k_info['min_cluster_size']} (2%); "
                 f"stopped: {k_info['stop_reason']}")
    if k_info.get("embedding_backend"):
        progress(f"Embedding backend used: {k_info['embedding_backend']}")
    if k_info.get("mode") == "auto":
        sweep = k_info.get("silhouette_scores", [])
        if sweep:
            progress("Silhouette sweep: " + ", ".join(
                f"k={s['k']}:{s['score']:.3f}" for s in sweep))
        progress(f"Auto-selected {k_info.get('chosen_k')} clusters by silhouette score")
    if k_info.get("structure_warning"):
        progress(f"WARNING: {k_info['structure_warning']}")
    progress(f"Formed {len(clusters)} clusters — sizes: {[c['paper_count'] for c in clusters]}")
    return clusters, k_info


def _draft_texts(config, clusters, k_info, claims_path, context, max_tokens,
                 model_capacity, cancel_event, progress, progress_state,
                 persist_progress):
    """Generate and checkpoint every independently usable model result."""
    from process import synthesizer as synth
    from process.llm_budget import BudgetPlanningError
    from process.llm_context import ContextWindowExceededError, OutputLimitExceededError

    research_config = getattr(config, "research_config", None)
    citations = synth.build_synthesis_source_citations(clusters)
    keys = {s["identity"]: s["citation_key"] for s in citations.get("sources", [])}
    cluster_ids = {cluster.get("cluster_id") for cluster in clusters}
    narratives: list[dict] = [
        dict(item) for item in progress_state.get("narratives", [])
        if isinstance(item, dict)
        and item.get("cluster_id") in cluster_ids
        and str(item.get("narrative") or "").strip()
    ]
    completed = {item.get("cluster_id") for item in narratives}
    resumed_narratives = len(narratives)
    progress_state.setdefault("error_history", []).extend(
        progress_state.get("errors", []))
    progress_state["errors"] = []
    progress_state["narratives"] = narratives
    progress_state["completed_cluster_ids"] = sorted(
        completed, key=lambda value: str(value))
    progress_state["status"] = "in_progress"
    persist_progress()

    recoverable = (
        OutputLimitExceededError,
        ContextWindowExceededError,
        BudgetPlanningError,
        RuntimeError,
        TimeoutError,
        ConnectionError,
        OSError,
    )

    for i, cluster in enumerate(clusters):
        if is_cancelled(cancel_event):
            progress(f"Cancelled during narrative generation (cluster {i})")
            return None
        cluster_id = cluster.get("cluster_id")
        if cluster_id in completed:
            progress(
                f"Reusing compatible saved narrative for cluster {cluster_id} "
                f"({cluster['paper_count']} papers)")
            continue
        progress(f"Generating narrative for cluster {i + 1}/{len(clusters)} "
                 f"({cluster['paper_count']} papers)")

        work_key = str(cluster_id)

        def checkpoint_cluster(payload, *, key=work_key):
            progress_state.setdefault("cluster_work", {})[key] = payload
            progress_state["status"] = "in_progress"
            persist_progress()

        try:
            narrative = synth.generate_cluster_narrative(
                model=config.llm_model, cluster=cluster, research_config=research_config,
                max_tokens=max_tokens, citation_keys=keys, model_capacity=model_capacity,
                cancel_event=cancel_event,
                checkpoint_callback=checkpoint_cluster,
                resume_state=progress_state.get("cluster_work", {}).get(work_key),
            )
        except CancelledError:
            raise
        except recoverable as exc:
            failure = {
                "phase": "cluster_narrative",
                "cluster_id": cluster_id,
                "error_type": type(exc).__name__,
                "message": (
                    f"Cluster {cluster_id} narrative is unresolved: {exc}. "
                    "Previously completed clusters and context batches were saved."
                ),
            }
            progress_state["errors"].append(failure)
            progress_state["status"] = "partial"
            persist_progress()
            progress(f"PARTIAL: {failure['message']}")
            continue
        narratives.append(narrative)
        narratives.sort(key=lambda item: str(item.get("cluster_id")))
        completed.add(cluster_id)
        progress_state["narratives"] = narratives
        progress_state["completed_cluster_ids"] = sorted(
            completed, key=lambda value: str(value))
        progress_state.setdefault("cluster_work", {}).pop(work_key, None)
        persist_progress()
        progress(f"Cluster {i + 1} narrative complete: "
                 f"{len(narrative.get('narrative', '') or '')} chars")
    progress(f"{len(narratives)}/{len(clusters)} cluster narratives generated and saved")
    if is_cancelled(cancel_event):
        progress("Cancelled before abstract generation")
        return None

    if len(completed) != len(clusters) or progress_state["errors"]:
        progress_state["status"] = "partial"
        progress_state["abstract_status"] = "blocked_by_incomplete_cluster_narratives"
        persist_progress()
        return {
            "narratives": narratives,
            "abstract": "",
            "citations": citations,
            "errors": list(progress_state["errors"]),
            "status": "partial",
            "resumed_narratives": resumed_narratives,
        }

    progress("Drafting structured abstract from cluster narratives")
    total_papers = sum(c.get("paper_count", 0) for c in clusters)
    facts = synth.build_pipeline_facts(
        total_papers=total_papers, clusters=clusters,
        k_info=k_info, claims_path=claims_path,
        grade_summary=synth._grade_summary(context.get("grade_results")))
    abstract_budget: dict = {}
    abstract_max_tokens = int(getattr(config, "synthesis_abstract_max_tokens", 0) or 0)
    try:
        abstract = synth.generate_abstract_draft(
            model=config.llm_model, cluster_narratives=narratives,
            total_papers=total_papers, research_config=research_config, pipeline_facts=facts,
            max_tokens=abstract_max_tokens, model_capacity=model_capacity,
            budget_metadata=abstract_budget, cancel_event=cancel_event)
    except CancelledError:
        raise
    except recoverable as exc:
        failure = {
            "phase": "abstract",
            "error_type": type(exc).__name__,
            "message": (
                f"Abstract generation is unresolved: {exc}. All completed cluster "
                "narratives remain saved and will be reused by a compatible rerun."
            ),
        }
        progress_state["errors"].append(failure)
        progress_state["status"] = "partial"
        progress_state["abstract_status"] = "unresolved"
        persist_progress()
        progress(f"PARTIAL: {failure['message']}")
        abstract = ""
    k_info["abstract_token_policy"] = abstract_budget
    if abstract and abstract.strip():
        progress_state["abstract_status"] = "complete"
        status = "complete"
    else:
        progress_state["abstract_status"] = (
            progress_state.get("abstract_status") or "empty_reply")
        status = "partial"
    progress_state["status"] = status
    progress_state["narratives"] = narratives
    persist_progress()
    progress(f"Abstract draft generated: {len(abstract or '')} chars")
    return {
        "narratives": narratives,
        "abstract": abstract or "",
        "citations": citations,
        "errors": list(progress_state["errors"]),
        "status": status,
        "resumed_narratives": resumed_narratives,
    }


def _empty_reply_warnings(narratives: list[dict], abstract: str, progress) -> list[str]:
    warnings = [
        f"LLM returned an empty narrative for cluster {n.get('cluster_id')}"
        for n in narratives if not str(n.get("narrative") or "").strip()
    ]
    if not abstract.strip():
        warnings.append(
            "LLM returned an empty abstract draft: the reply had no answer text. If LM Studio "
            "showed output, inspect the recorded token budget and provider finish reason. "
            "abstract_draft.md was NOT written or changed; increase an explicit override "
            "only if one is recorded, or choose a model with a larger output capability, "
            "then re-run Stage 6")
    for warning in warnings:
        progress(f"WARNING: {warning}")
    return warnings


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------
def _save_core(out_dir, clusters_name, clusters, narratives, k_info, abstract, progress) -> dict:
    paths = {
        "clusters_path": out_dir / clusters_name,
        "narratives_path": out_dir / "narratives.json",
    }
    atomic_write_json(paths["clusters_path"], _clusters_light(clusters))
    atomic_write_json(paths["narratives_path"], narratives)
    # A blank abstract is not an abstract: writing it would make a 0-byte file
    # look like finished output (and overwrite an earlier real draft).
    if abstract.strip():
        paths["abstract_path"] = out_dir / "abstract_draft.md"
        atomic_write_text(paths["abstract_path"], abstract)
    if k_info:
        paths["k_selection_path"] = out_dir / "k_selection.json"
        atomic_write_json(paths["k_selection_path"], k_info)
    for key, path in paths.items():
        progress(f"Saved {key.removesuffix('_path').replace('_', ' ')}: {path}")
    return {key: str(path) for key, path in paths.items() if key != "k_selection_path"}


def _save_provenance(out_dir, config, clusters, narratives, abstract, citations,
                     context, k_info, progress) -> dict:
    from process import synthesizer as synth
    from process.export import export_bibtex

    explanation = synth.build_hypothesis_explanation(
        clusters, narratives, abstract,
        research_config=getattr(config, "research_config", None),
        method_profiles=context.get("method_profiles"),
        grade_results=context.get("grade_results"),
        unreadable_inputs=context["unreadable"],
        extra_caveats=[k_info["structure_warning"]] if k_info.get("structure_warning") else [],
    )
    paths = {
        "hypothesis_explanation_path": out_dir / "hypothesis_explanation.json",
        "hypothesis_explanation_md_path": out_dir / "hypothesis_explanation.md",
        "evidence_network_path": out_dir / "evidence_network.json",
        "synthesis_sources_path": out_dir / "synthesis_sources.json",
        "synthesis_sources_md_path": out_dir / "synthesis_sources.md",
        "synthesis_sources_bib_path": out_dir / "synthesis_sources.bib",
    }
    atomic_write_json(paths["hypothesis_explanation_path"], explanation)
    atomic_write_text(paths["hypothesis_explanation_md_path"],
                      synth.render_hypothesis_explanation_markdown(explanation))
    atomic_write_json(paths["evidence_network_path"], synth.build_evidence_network(clusters))
    atomic_write_json(paths["synthesis_sources_path"], citations)
    atomic_write_text(paths["synthesis_sources_md_path"],
                      synth.render_synthesis_source_citations_markdown(citations))
    export_bibtex(citations.get("sources", []), paths["synthesis_sources_bib_path"])
    for key, path in paths.items():
        progress(f"Saved {key.removesuffix('_path').replace('_', ' ')}: {path}")
    return {key: str(path) for key, path in paths.items()}
