"""Path resolution utilities."""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_list_len(path: Path) -> int | None:
    """Return the length of a JSON list file, or None if unreadable/not a list."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # ValueError covers JSON and Unicode decode errors
        return None
    return len(data) if isinstance(data, list) else None


def resolve_claims_path(config, prefer_filtered: bool = True) -> Path:
    """Return claims_filtered.json if it exists, else claims.json.

    Parameters
    ----------
    config : Config
        Pipeline configuration (must have ``claims_dir`` attribute).
    prefer_filtered : bool
        When *True* (default), prefer the filtered variant produced by
        Stage 5.  Set to *False* to always return the unfiltered file.

    Returns
    -------
    Path
        Absolute path to the best available claims JSON file.
    """
    return _pick_claims_file(config.claims_dir, prefer_filtered)


def _warn_if_filtered_is_stale(filtered: Path, base: Path) -> None:
    """Stage 3 output rewritten after Stage 5 ran means the filter is out of date."""
    try:
        is_stale = base.stat().st_mtime > filtered.stat().st_mtime
    except OSError:
        return
    if is_stale:
        logger.warning(
            "%s is older than %s — Stage 3 claims changed after Stage 5 ran; "
            "using the filtered claims anyway (re-run Stage 5 to refresh them)",
            filtered, base,
        )


def _pick_claims_file(claims_dir: Path, prefer_filtered: bool = True) -> Path:
    filtered = claims_dir / "claims_filtered.json"
    base = claims_dir / "claims.json"
    if not prefer_filtered:
        logger.debug("Using base claims: %s", base)
        return base
    if not filtered.exists():
        if base.exists():
            logger.info(
                "%s not found (Stage 5 not run) — using unfiltered claims: %s",
                filtered.name, base,
            )
        return base
    # An empty filtered file means Stage 5 found nothing to keep — using
    # it would silently give downstream stages zero papers.
    n_filtered = _load_list_len(filtered)
    n_base = _load_list_len(base) if base.exists() else None
    if n_filtered == 0 and n_base:
        logger.warning(
            "%s is empty but %s has %d papers — falling back to the unfiltered claims",
            filtered, base, n_base,
        )
        return base
    if base.exists():
        _warn_if_filtered_is_stale(filtered, base)
    logger.debug("Using filtered claims: %s", filtered)
    return filtered


def _stage_data_dir(config) -> Path:
    """The run-scoped data dir, or ``./data`` when no config is given."""
    return config.data_dir if config is not None else Path.cwd() / "data"


def resolve_claims_corpus(config, claims_path: str | Path | None = None) -> Path | None:
    """Return the corpus Stages 8–10 analyse, or None when nothing exists.

    Order: explicit ``claims_path`` → Stage 5 filtered claims (falling back
    to Stage 3 claims when filtered is empty) → Stage 2 ``relevant.json``.
    Only the current run's data dir is searched, so a named run never
    picks up another run's claims.
    """
    if claims_path:
        path = Path(claims_path)
        return path if path.exists() else None
    data_dir = _stage_data_dir(config)
    claims_dir = config.claims_dir if config is not None else data_dir / "claims"
    candidates = [
        _pick_claims_file(claims_dir),
        claims_dir / "relevant.json",
        data_dir / "corpus" / "relevant.json",  # pre-2026 layout
    ]
    chosen = next((c for c in candidates if c.exists()), None)
    if chosen is not None and chosen.name == "relevant.json":
        logger.info("No Stage 3/5 claims found — using Stage 2 corpus: %s", chosen)
    return chosen


def resolve_validation_dir(config, output_dir: str | Path | None = None) -> Path:
    """Return (and create) the validation dir shared by Stages 8–10."""
    validation_dir = Path(output_dir) if output_dir else _stage_data_dir(config) / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    return validation_dir


def load_paper_list(path: Path) -> list[dict]:
    """Load a corpus file as a list of papers; keyed dicts become their values."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return list(data.values())
    return []
