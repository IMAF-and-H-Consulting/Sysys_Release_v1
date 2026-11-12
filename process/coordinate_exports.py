"""
Brain coordinate exports in NiMARE, GingerALE/Sleuth, and CSV formats.

Split out of ``process.coordinate_extractor``, which re-exports every name here.
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from process.coordinate_extractor import BrainCoordinate

# Shares the extractor's logger so log records keep their original name.
logger = logging.getLogger("process.coordinate_extractor")


# ── Export: NiMARE ─────────────────────────────────────────────────────

def _group_experiments(
    coordinates: list[BrainCoordinate],
) -> dict[str, dict[tuple[str, str], list[BrainCoordinate]]]:
    """study -> (contrast, space) -> peaks, preserving input order."""
    studies: dict[str, dict[tuple[str, str], list[BrainCoordinate]]] = {}
    for c in coordinates:
        experiments = studies.setdefault(c.article_id or "unknown", {})
        experiments.setdefault((c.contrast, c.space), []).append(c)
    return studies


def _warn_if_contrasts_pooled(coordinates: list[BrainCoordinate]) -> None:
    pooled = {c.article_id for c in coordinates if not c.contrast}
    if pooled:
        logger.warning(
            "%d studies have no contrast labels: all their peaks are pooled into one "
            "experiment per study (contrasts are not extracted from the text)",
            len(pooled),
        )


def _sample_size(peaks: list[BrainCoordinate]) -> int:
    return max((c.n_subjects for c in peaks), default=0)


def _nimare_contrast(peaks: list[BrainCoordinate]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    n = _sample_size(peaks)
    if n:
        metadata["sample_sizes"] = [n]
    transforms = sorted({c.space_transform for c in peaks if c.space_transform})
    if transforms:
        metadata["space_transform"] = "; ".join(transforms)
    return {
        "coords": {
            "space": peaks[0].space,
            "x": [c.x for c in peaks],
            "y": [c.y for c in peaks],
            "z": [c.z for c in peaks],
        },
        "metadata": metadata,
    }


def _nimare_contrast_id(contrast: str, space: str, n_spaces: int) -> str:
    base = contrast or "contrast_1"
    return f"{base}_{space}" if n_spaces > 1 else base


def export_nimare_dataset(
    coordinates: list[BrainCoordinate], output_path: Path
) -> Path:
    """
    Export as a NiMARE JSON dataset:
    ``{study: {"contrasts": {id: {"coords": {space, x, y, z},
    "metadata": {"sample_sizes": [n]}}}}}``.

    Peaks are grouped by article and contrast; a study whose peaks are in
    different spaces gets one contrast per space. Unlabelled contrasts are
    pooled (logged as a warning).
    """
    _warn_if_contrasts_pooled(coordinates)
    dataset: dict[str, Any] = {}
    for study_id, experiments in _group_experiments(coordinates).items():
        n_spaces = len({space for _, space in experiments})
        dataset[study_id] = {"contrasts": {
            _nimare_contrast_id(contrast, space, n_spaces): _nimare_contrast(peaks)
            for (contrast, space), peaks in experiments.items()
        }}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    logger.info("Exported NiMARE dataset to %s (%d studies)", output_path, len(dataset))
    return output_path


# ── Export: Sleuth / GingerALE ─────────────────────────────────────────

def _single_reference_space(
    coordinates: list[BrainCoordinate],
) -> tuple[list[BrainCoordinate], str]:
    """Sleuth files have one reference space; mixed input is converted to MNI."""
    spaces = {c.space for c in coordinates}
    if len(spaces) == 1:
        return coordinates, spaces.pop()
    if spaces:
        logger.info("Mixed coordinate spaces: converting Talairach peaks to MNI for Sleuth")
    # Looked up on the extractor module so patches of it still apply.
    from process import coordinate_extractor

    return coordinate_extractor.convert_tal_to_mni(coordinates), "MNI"


def _sleuth_block(study_id: str, contrast: str, peaks: list[BrainCoordinate]) -> list[str]:
    lines = [f"// {study_id}: {contrast}" if contrast else f"// {study_id}"]
    n = _sample_size(peaks)
    if n:
        lines.append(f"// Subjects={n}")
    else:
        logger.warning("Sleuth export: no sample size found for %s", study_id)
    lines.extend(f"{c.x}\t{c.y}\t{c.z}" for c in peaks)
    lines.append("")  # blank line between experiments
    return lines


def export_sleuth_format(
    coordinates: list[BrainCoordinate], output_path: Path
) -> Path:
    """
    Export as Sleuth/GingerALE text format:

    // Reference=MNI
    // article_id
    // Subjects=30
    x1  y1  z1
    ...
    """
    coords, ref_space = _single_reference_space(coordinates)
    _warn_if_contrasts_pooled(coords)
    lines: list[str] = [f"// Reference={ref_space}"]
    studies = _group_experiments(coords)
    for study_id, experiments in sorted(studies.items()):
        for (contrast, _space), peaks in experiments.items():
            lines.extend(_sleuth_block(study_id, contrast, peaks))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Exported Sleuth format to %s (%d studies)", output_path, len(studies))
    return output_path


# ── Export: CSV ────────────────────────────────────────────────────────

def export_csv(
    coordinates: list[BrainCoordinate], output_path: Path
) -> Path:
    """Export as CSV with all coordinate fields."""
    fieldnames = [
        "study_id", "x", "y", "z", "space", "stat_type", "stat_value",
        "cluster_size", "region", "contrast", "n_subjects",
        "space_ambiguous", "space_transform",
    ]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in coordinates:
            writer.writerow({
                "study_id": c.article_id,
                "x": c.x,
                "y": c.y,
                "z": c.z,
                "space": c.space,
                "stat_type": c.statistic_type,
                "stat_value": c.statistic_value,
                "cluster_size": c.cluster_size,
                "region": c.region_label,
                "contrast": c.contrast,
                "n_subjects": c.n_subjects,
                "space_ambiguous": c.space_ambiguous,
                "space_transform": c.space_transform,
            })

    logger.info("Exported CSV to %s (%d rows)", output_path, len(coordinates))
    return output_path
