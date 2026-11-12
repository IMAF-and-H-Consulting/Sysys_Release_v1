"""
Brain coordinate extraction from full-text articles.

Extracts MNI/Talairach coordinates using regex patterns, converts between
coordinate spaces, and exports in NiMARE, GingerALE/Sleuth, and CSV formats.
"""
from __future__ import annotations

import csv  # noqa: F401  (kept as a module attribute; used in coordinate_exports)
import json  # noqa: F401  (kept as a module attribute; used in coordinate_exports)
import logging
import re
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

# Exports, re-exported so ``process.coordinate_extractor.<name>`` keeps working.
from process.coordinate_exports import (  # noqa: F401
    _group_experiments,
    _nimare_contrast,
    _nimare_contrast_id,
    _sample_size,
    _single_reference_space,
    _sleuth_block,
    _warn_if_contrasts_pooled,
    export_csv,
    export_nimare_dataset,
    export_sleuth_format,
)

logger = logging.getLogger(__name__)


@dataclass
class BrainCoordinate:
    """A single peak activation coordinate extracted from a paper."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    space: str = "MNI"             # "MNI" or "Talairach"
    statistic_type: str = ""       # "Z", "T", "F", ""
    statistic_value: float = 0.0
    cluster_size: int = 0          # voxels, if reported
    region_label: str = ""         # anatomical label from paper
    contrast: str = ""             # what comparison produced this peak
    article_id: str = ""
    n_subjects: int = 0
    space_ambiguous: bool = False  # both spaces stated, none near this peak
    space_transform: str = ""      # set when converted from another space


# ── Text normalisation ────────────────────────────────────────────────

# Typographic minus signs that journals and PDF extraction use for negatives.
_MINUS_CHARS = str.maketrans({
    "−": "-",  # minus sign
    "–": "-",  # en dash
    "‒": "-",  # figure dash
    "‐": "-",  # hyphen
    "‑": "-",  # non-breaking hyphen
    "﹣": "-",  # small hyphen-minus
    "－": "-",  # fullwidth hyphen-minus
})

# Markup that carries CSS colour triplets rather than article text.
_MARKUP_NOISE = re.compile(
    r'<(style|script)\b[^>]*>.*?</\1\s*>'
    r'|\bstyle\s*=\s*(?:"[^"]*"|\'[^\']*\')'
    r'|\b(?:rgba?|hsla?)\s*\([^)]*\)',
    re.IGNORECASE | re.DOTALL,
)


def _normalize_text(text: str) -> str:
    """Map Unicode minus signs to '-' and drop CSS/script noise."""
    return _MARKUP_NOISE.sub(" ", text.translate(_MINUS_CHARS))


# ── Regex patterns ────────────────────────────────────────────────────

# Coordinate space detection. Explicit statements ("in MNI space", "Talairach
# coordinates") outrank bare mentions; Talairach tool/atlas citations
# ("Talairach Daemon", "Talairach and Tournoux") say nothing about the space.
_EXPLICIT_MNI = re.compile(
    r'\bMNI(?:[-\s]?152)?\s+(?:space|coordinates?|template|stereotaxic)\b'
    r'|\b(?:in|into|to)\s+(?:the\s+)?MNI\b'
    r'|\bMontreal\s+Neurological\s+Institute\b',
    re.IGNORECASE,
)
_EXPLICIT_TAL = re.compile(
    r'\bTalairach\s+(?:space|coordinates?|stereotaxic\s+space)\b'
    r'|\b(?:in|into|to)\s+(?:the\s+)?Talairach\s+(?:space|coordinates?)\b',
    re.IGNORECASE,
)
_MNI_PATTERN = re.compile(r'\bMNI\b', re.IGNORECASE)
_TAL_PATTERN = re.compile(
    r'\b(?:Talairach|TAL)\b'
    r'(?!\s*(?:Daemon|Client|labels?|and\s+Tournoux|&\s*Tournoux|,?\s*J\b))',
    re.IGNORECASE,
)
_SPACE_PATTERNS = (
    ("MNI", _EXPLICIT_MNI, _MNI_PATTERN),
    ("Talairach", _EXPLICIT_TAL, _TAL_PATTERN),
)
_REFERENCES_HEADING = re.compile(
    r'^[ \t]*(?:\d+\.?\s*)?(?:references|bibliography|literature\s+cited|works\s+cited)'
    r'[ \t]*:?[ \t]*$',
    re.IGNORECASE | re.MULTILINE,
)
_SPACE_BEFORE_TRIPLET = 300
_SPACE_BEFORE_TABLE = 400

_NUM = r'-?\d{1,3}(?:\.\d+)?'

# Parenthesised triplet: (x, y, z) or [x, y, z]
_PAREN_TRIPLET = re.compile(
    r'(?P<open>[\(\[])\s*'
    rf'({_NUM})\s*[,;/\s]\s*'
    rf'({_NUM})\s*[,;/\s]\s*'
    rf'({_NUM})\s*'
    r'[\)\]]'
)

# Words that mark a parenthesised triplet as a brain coordinate.
_COORD_CONTEXT = re.compile(
    r'\b(?:MNI|Talairach|coordinates?|peaks?|foc(?:us|i)|voxels?|maxim(?:um|a)'
    r'|x\s*,\s*y\s*,\s*z|BA\s*\d|Brodmann|activation|cluster'
    r'|gyr(?:us|i)|cort(?:ex|ical)|lob(?:e|ule)|sulc(?:us|i)|insula'
    r'|amygdala|hippocamp\w*|thalam\w*|putamen|caudate|striat\w*|cerebell\w*'
    r'|(?:pre)?cuneus|cingulate|operculum|pallidum|accumbens|fusiform'
    r'|frontal|temporal|parietal|occipital|hemisphere)\b'
    r'|\b[ZTF]\s*[=:]\s*\d',
    re.IGNORECASE,
)
_CONTEXT_BEFORE = 200
_CONTEXT_AFTER = 80

# Coordinate table header: consecutive x, y, z column names
_XYZ_HEADER = re.compile(
    r'(?<![A-Za-z])x\s*(?:\(mm\))?\s*[,;]?\s*'
    r'y\s*(?:\(mm\))?\s*[,;]?\s*'
    r'z(?![A-Za-z])\s*(?:\(mm\))?',
    re.IGNORECASE,
)
_HEADER_MAX_LEN = 160
_TABLE_BLOCK_CHARS = 3000
_ROW_MAX_LABEL_WORDS = 8

# Header column tokens: multi-word names first, then single words.
_HEADER_TOKEN = re.compile(
    r'no\.?\s*(?:of\s+)?voxels|cluster\s+(?:size|extent)|brodmann\s+areas?'
    r'|(?:peak\s+)?[ZTF][-\s](?:score|value|stat(?:istic)?)'
    r'|[^\s(),;|]+(?:\([^)]*\))?',
    re.IGNORECASE,
)
_HEADER_NOISE = re.compile(
    r'^(?:MNI|Talairach|Tal|coordinates?|coords?|\(?mm\)?|peak|[-–—/]+)$',
    re.IGNORECASE,
)
_STAT_COLUMN = re.compile(
    r'^(?:peak[-_\s]?)?(?P<type>[ZTF])(?:[-_\s]?(?:score|value|stat(?:istic)?))?$',
    re.IGNORECASE,
)
_CLUSTER_COLUMN = re.compile(
    r'^(?:k|ke|k_?e|voxels?|size|extent|cluster(?:\s+(?:size|extent))?'
    r'|no\.?\s*(?:of\s+)?voxels)$',
    re.IGNORECASE,
)
_OTHER_NUMERIC_COLUMN = re.compile(
    r'^(?:BA|brodmann(?:\s+areas?)?'
    r'|p(?:[-_]?(?:values?|unc|fwe|fdr|corr))?(?:\([^)]*\))?)$',
    re.IGNORECASE,
)

# Row cells: numbers, or placeholders that still occupy a column.
_NUMBER_CELL = re.compile(r'^[-+]?\d+(?:\.\d+)?$')
_FILLER_CELL = re.compile(r'^(?:\d+(?:[/-]\d+)+|-+|—|n/?a|ns)$', re.IGNORECASE)
_CELL_SUFFIX_MARKS = "*†‡§"

# Statistical value near coordinates
_STAT_PATTERN = re.compile(
    r'\b(?P<type>[ZTF])(?:[-\s]?(?:score|value))?'
    r'(?:\s*\(\s*\d+(?:\s*,\s*\d+)?\s*\))?'
    r'\s*[=:]\s*(?P<val>\d+(?:\.\d+)?)',
    re.IGNORECASE,
)

# Cluster size
_CLUSTER_PATTERN = re.compile(
    r'\bk(?:E)?\s*[=:]\s*(\d+)\b|\b(\d+)\s*voxels?\b',
    re.IGNORECASE,
)

# Sample size from context
_SAMPLE_PATTERN = re.compile(r'\b[Nn]\s*[=:]\s*(\d+)\b')
_PARTICIPANT_PATTERN = re.compile(
    r'\b(\d+)\s+(?:[a-z-]+\s+){0,2}'
    r'(?:participants|subjects|volunteers|patients|individuals|adults|children|adolescents)\b',
    re.IGNORECASE,
)

# A sentence ends at a newline or at . ! ? followed by whitespace.
_SENTENCE_BREAK = re.compile(r'\n|[.!?;](?=\s)')
_LOCAL_WINDOW = 80


# ── Coordinate range validation ───────────────────────────────────────

def _is_plausible_coord(x: float, y: float, z: float) -> bool:
    """Check that coordinates fall within plausible brain-space range."""
    return (-100 <= x <= 100) and (-130 <= y <= 100) and (-80 <= z <= 100)


# ── Extraction functions ──────────────────────────────────────────────

def _space_mentions(text: str, lo: int, hi: int) -> list[tuple[int, str, bool]]:
    """(position, space, is_explicit) for every space mention in text[lo:hi]."""
    return [
        (m.start(), space, is_explicit)
        for space, explicit_re, bare_re in _SPACE_PATTERNS
        for is_explicit, pattern in ((True, explicit_re), (False, bare_re))
        for m in pattern.finditer(text, max(0, lo), hi)
    ]


def _nearest_space(text: str, lo: int, hi: int, anchor: int) -> str | None:
    """Space of the mention closest to ``anchor``; explicit statements first."""
    mentions = _space_mentions(text, lo, hi)
    for want_explicit in (True, False):
        tier = [m for m in mentions if m[2] is want_explicit]
        if tier:
            return min(tier, key=lambda m: abs(m[0] - anchor))[1]
    return None


def _body_end(text: str) -> int:
    """Offset of the last reference-list heading (end of text if none)."""
    headings = list(_REFERENCES_HEADING.finditer(text))
    return headings[-1].start() if headings else len(text)


def _document_space(text: str) -> tuple[str, bool]:
    """Article-wide fallback space and whether both spaces are stated.

    The reference list is ignored and explicit statements outrank bare
    mentions; ties go to MNI.
    """
    mentions = _space_mentions(text, 0, _body_end(text))
    for want_explicit in (True, False):
        counts = Counter(space for _, space, explicit in mentions if explicit is want_explicit)
        if counts:
            space = "Talairach" if counts["Talairach"] > counts["MNI"] else "MNI"
            return space, len(counts) > 1
    return "MNI", False


@dataclass(frozen=True)
class _ArticleContext:
    """Normalised article text plus article-wide defaults."""
    text: str
    article_id: str
    n_subjects: int
    default_space: str
    default_ambiguous: bool

    def space_near(self, lo: int, hi: int, anchor: int) -> tuple[str, bool]:
        """(space, ambiguous): a nearby statement wins over the article default."""
        local = _nearest_space(self.text, lo, hi, anchor)
        if local:
            return local, False
        return self.default_space, self.default_ambiguous


def _sentence_end(text: str, end: int) -> int:
    hi = min(len(text), end + _LOCAL_WINDOW)
    after = _SENTENCE_BREAK.search(text, end, hi)
    return after.start() if after else hi


def _sentence_window(text: str, start: int, end: int) -> str:
    """Return the sentence containing text[start:end], capped at ±80 chars."""
    lo = max(0, start - _LOCAL_WINDOW)
    hi = min(len(text), end + _LOCAL_WINDOW)
    breaks_before = [m.end() for m in _SENTENCE_BREAK.finditer(text, lo, start)]
    if breaks_before:
        lo = breaks_before[-1]
    after = _SENTENCE_BREAK.search(text, end, hi)
    if after:
        hi = after.start()
    return text[lo:hi]


def _extract_stat_from_context(text: str, start: int, end: int) -> tuple[str, float]:
    """Look for a statistic in the same sentence as a coordinate."""
    m = _STAT_PATTERN.search(_sentence_window(text, start, end))
    if m:
        return m.group("type").upper(), float(m.group("val"))
    return "", 0.0


def _extract_cluster_from_context(text: str, start: int, end: int) -> int:
    """Look for a cluster size in the same sentence as a coordinate."""
    m = _CLUSTER_PATTERN.search(_sentence_window(text, start, end))
    if not m:
        return 0
    return int(m.group(1) or m.group(2))


def _extract_sample_from_context(text: str) -> int:
    """Extract sample size from 'N = 24' or '24 healthy participants'."""
    m = _SAMPLE_PATTERN.search(text) or _PARTICIPANT_PATTERN.search(text)
    return int(m.group(1)) if m else 0


def _is_citation_like(m: re.Match[str]) -> bool:
    """Square-bracketed non-negative integers read as a citation list."""
    if m.group("open") != "[":
        return False
    return all(re.fullmatch(r'\d+', m.group(i)) for i in (2, 3, 4))


def _has_coordinate_context(text: str, start: int, end: int) -> bool:
    """True when coordinate/anatomy/stat cues appear near a triplet."""
    window = text[max(0, start - _CONTEXT_BEFORE):end + _CONTEXT_AFTER]
    return bool(_COORD_CONTEXT.search(window))


def _extract_paren_triplets(ctx: _ArticleContext) -> list[BrainCoordinate]:
    """Strategy 1: bracketed (x, y, z) triplets with coordinate context."""
    text = ctx.text
    coords: list[BrainCoordinate] = []
    seen: set[tuple[float, float, float]] = set()
    for m in _PAREN_TRIPLET.finditer(text):
        x, y, z = float(m.group(2)), float(m.group(3)), float(m.group(4))
        if not _is_plausible_coord(x, y, z) or _is_citation_like(m):
            continue
        if not _has_coordinate_context(text, m.start(), m.end()) or (x, y, z) in seen:
            continue
        seen.add((x, y, z))
        space, ambiguous = ctx.space_near(
            m.start() - _SPACE_BEFORE_TRIPLET, _sentence_end(text, m.end()), m.start(),
        )
        stat_type, stat_val = _extract_stat_from_context(text, m.start(), m.end())
        coords.append(BrainCoordinate(
            x=x, y=y, z=z, space=space, space_ambiguous=ambiguous,
            statistic_type=stat_type, statistic_value=stat_val,
            cluster_size=_extract_cluster_from_context(text, m.start(), m.end()),
            article_id=ctx.article_id, n_subjects=ctx.n_subjects,
        ))
    return coords


# ── Table parsing ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class _TableLayout:
    """Column layout parsed from a coordinate table header."""
    n_before: int                    # numeric columns before x
    after: tuple[str, ...]           # kinds after z: "Z"/"T"/"F", "k" or ""


def _header_tokens(fragment: str) -> list[str]:
    tokens = [t.group(0) for t in _HEADER_TOKEN.finditer(fragment)]
    return [t for t in tokens if not _HEADER_NOISE.match(t)]


def _column_kind(token: str) -> str:
    """Classify a header token: stat type letter, 'k', 'num' or 'text'."""
    stat = _STAT_COLUMN.match(token)
    if stat:
        return stat.group("type").upper()
    if _CLUSTER_COLUMN.match(token):
        return "k"
    if _OTHER_NUMERIC_COLUMN.match(token):
        return "num"
    return "text"


def _parse_header(line: str) -> _TableLayout | None:
    """Locate x/y/z in a header line and classify surrounding columns."""
    if len(line) > _HEADER_MAX_LEN:
        return None
    m = _XYZ_HEADER.search(line)
    if not m:
        return None
    before = [_column_kind(t) for t in _header_tokens(line[:m.start()])]
    after = [_column_kind(t) for t in _header_tokens(line[m.end():])]
    return _TableLayout(
        n_before=sum(1 for kind in before if kind != "text"),
        after=tuple("" if kind in ("num", "text") else kind for kind in after),
    )


def _clean_cell(token: str) -> str:
    return token.rstrip(_CELL_SUFFIX_MARKS)


def _is_cell(token: str) -> bool:
    cell = _clean_cell(token)
    return bool(_NUMBER_CELL.match(cell) or _FILLER_CELL.match(cell))


def _split_row(line: str) -> tuple[str, list[str]] | None:
    """Split a row into its leading text label and following cell tokens."""
    tokens = line.split()
    first_cell = next((i for i, t in enumerate(tokens) if _is_cell(t)), None)
    if first_cell is None or first_cell > _ROW_MAX_LABEL_WORDS:
        return None
    cells: list[str] = []
    for token in tokens[first_cell:]:
        if not _is_cell(token):
            break
        cells.append(_clean_cell(token))
    return " ".join(tokens[:first_cell]), cells


def _cell_number(cells: list[str], index: int) -> float | None:
    if index >= len(cells) or not _NUMBER_CELL.match(cells[index]):
        return None
    return float(cells[index])


def _row_extras(
    layout: _TableLayout, cells: list[str], offset: int,
) -> tuple[str, float, int]:
    """Read stat type/value and cluster size from the row's own columns."""
    stat_type, stat_val, cluster = "", 0.0, 0
    for i, kind in enumerate(layout.after):
        value = _cell_number(cells, offset + i)
        if value is None:
            continue
        if kind == "k" and not cluster:
            cluster = int(value)
        elif kind in ("Z", "T", "F") and not stat_type:
            stat_type, stat_val = kind, value
    return stat_type, stat_val, cluster


def _parse_row(
    layout: _TableLayout, line: str,
) -> tuple[tuple[float, float, float], str, str, float, int] | None:
    """Map one table row to ((x, y, z), label, stat_type, stat_val, k)."""
    split = _split_row(line)
    if split is None:
        return None
    label, cells = split
    xyz = [_cell_number(cells, layout.n_before + i) for i in range(3)]
    if any(v is None for v in xyz):
        return None
    stat_type, stat_val, cluster = _row_extras(layout, cells, layout.n_before + 3)
    return (xyz[0], xyz[1], xyz[2]), label, stat_type, stat_val, cluster


def _extract_table_rows(ctx: _ArticleContext) -> list[BrainCoordinate]:
    """Strategy 2: rows under a header naming x, y and z columns.

    Each header starts a table; the space is read from its caption/header.
    """
    coords: list[BrainCoordinate] = []
    lines = ctx.text.split("\n")
    offset = 0
    for index, line in enumerate(lines):
        layout = _parse_header(line)
        if layout is not None:
            space = ctx.space_near(offset - _SPACE_BEFORE_TABLE, offset + len(line), offset)
            coords.extend(_rows_after_header(layout, lines[index + 1:], ctx, space))
        offset += len(line) + 1
    return coords


def _rows_after_header(
    layout: _TableLayout, lines: list[str], ctx: _ArticleContext,
    space: tuple[str, bool],
) -> list[BrainCoordinate]:
    """Rows of one table; duplicates are dropped within the table only."""
    coords: list[BrainCoordinate] = []
    seen: set[tuple[float, float, float]] = set()
    consumed = 0
    for line in lines:
        consumed += len(line) + 1
        if consumed > _TABLE_BLOCK_CHARS or _parse_header(line) is not None:
            break
        parsed = _parse_row(layout, line)
        if parsed is None:
            continue
        key, label, stat_type, stat_val, cluster = parsed
        if not _is_plausible_coord(*key) or key in seen:
            continue
        seen.add(key)
        if not stat_type:
            stat_type, stat_val = _extract_stat_from_context(line, 0, len(line))
        coords.append(BrainCoordinate(
            x=key[0], y=key[1], z=key[2], space=space[0], space_ambiguous=space[1],
            statistic_type=stat_type, statistic_value=stat_val,
            cluster_size=cluster or _extract_cluster_from_context(line, 0, len(line)),
            region_label=label, article_id=ctx.article_id, n_subjects=ctx.n_subjects,
        ))
    return coords


def _xyz(c: BrainCoordinate) -> tuple[float, float, float]:
    return c.x, c.y, c.z


def _enrich_row(row: BrainCoordinate, prose: BrainCoordinate | None) -> BrainCoordinate:
    """Fill a table row's missing stat/cluster from the same peak in the prose."""
    if prose is None:
        return row
    changes: dict[str, Any] = {}
    if not row.statistic_type and prose.statistic_type:
        changes.update(statistic_type=prose.statistic_type, statistic_value=prose.statistic_value)
    if not row.cluster_size and prose.cluster_size:
        changes["cluster_size"] = prose.cluster_size
    return replace(row, **changes)


def _merge_prose_into_rows(
    triplets: list[BrainCoordinate], rows: list[BrainCoordinate],
) -> list[BrainCoordinate]:
    """A table row replaces (and is enriched by) a prose triplet at the same peak."""
    prose = {_xyz(c): c for c in triplets}
    row_keys = {_xyz(c) for c in rows}
    kept = [c for c in triplets if _xyz(c) not in row_keys]
    return kept + [_enrich_row(row, prose.get(_xyz(row))) for row in rows]


def _warn_if_space_ambiguous(coords: list[BrainCoordinate], article_id: str) -> None:
    ambiguous = [c for c in coords if c.space_ambiguous]
    if ambiguous:
        logger.warning(
            "%s: both MNI and Talairach are stated; %d peak(s) without a nearby "
            "space statement were assigned %s (flagged space_ambiguous)",
            article_id or "article", len(ambiguous), ambiguous[0].space,
        )


def extract_coordinates_from_text(
    text: str, article_id: str = ""
) -> list[BrainCoordinate]:
    """
    Extract brain coordinates from full-text using regex patterns.

    Looks for:
    - Parenthesised triplets: (x, y, z), only near coordinate context
    - Table rows under an x/y/z header, mapped by column position
    - MNI vs Talairach from the nearest statement before each triplet or
      table (caption/header), else from the article body (references
      excluded); conflicting article-wide statements set ``space_ambiguous``
    - Statistical values (Z=, T=, F=) in the same sentence or row
    - Cluster sizes (k=, voxels) in the same sentence or row

    Duplicates are dropped within the prose and within each table; the
    same peak in two tables (contrasts) is kept for each. Unicode minus
    signs are normalised and CSS/script markup is ignored.
    """
    if not text or not text.strip():
        return []

    text = _normalize_text(text)
    space, ambiguous = _document_space(text)
    ctx = _ArticleContext(
        text=text, article_id=article_id,
        n_subjects=_extract_sample_from_context(text),
        default_space=space, default_ambiguous=ambiguous,
    )
    coords = _merge_prose_into_rows(_extract_paren_triplets(ctx), _extract_table_rows(ctx))
    _warn_if_space_ambiguous(coords, article_id)
    return coords


# ── Talairach → MNI conversion ────────────────────────────────────────

def _invert_4x4(matrix: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
    """Invert a 4x4 affine with Gauss-Jordan elimination."""
    size = 4
    aug = [list(matrix[row]) + [1.0 if col == row else 0.0 for col in range(size)] for row in range(size)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("Lancaster transform matrix is singular")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [value / scale for value in aug[col]]
        for row in range(size):
            if row == col:
                continue
            factor = aug[row][col]
            aug[row] = [value - factor * aug[col][index] for index, value in enumerate(aug[row])]
    return tuple(tuple(row[size:]) for row in aug)


def _apply_affine(
    matrix: tuple[tuple[float, ...], ...],
    x: float,
    y: float,
    z: float,
) -> tuple[float, float, float]:
    point = (x, y, z, 1.0)
    transformed = [sum(matrix[row][col] * point[col] for col in range(4)) for row in range(3)]
    return transformed[0], transformed[1], transformed[2]


# Lancaster et al. (2007) icbm_spm (MNI/SPM -> Talairach), inverted once.
_ICBM_SPM_TO_TAL = (
    (0.9254, 0.0024, -0.0118, -1.0207),
    (-0.0048, 0.9316, -0.0871, -1.7667),
    (0.0152, 0.0883, 0.8924, 4.0926),
    (0.0, 0.0, 0.0, 1.0),
)
_TAL_TO_ICBM_SPM = _invert_4x4(_ICBM_SPM_TO_TAL)


TAL_TO_MNI_TRANSFORM = "Lancaster 2007 tal2icbm_spm (inverse of icbm_spm2tal)"


def convert_tal_to_mni(coords: list[BrainCoordinate]) -> list[BrainCoordinate]:
    """
    Convert Talairach coordinates to MNI with Lancaster et al. (2007)
    tal2icbm_spm: the inverse of the published icbm_spm2tal affine.

    The SPM variant is applied to every study (FSL-normalised studies
    differ by a few mm); converted peaks record it in ``space_transform``.
    Returns a new list; Talairach coords are converted, MNI coords pass through.
    """
    result: list[BrainCoordinate] = []
    for c in coords:
        if c.space != "Talairach":
            result.append(c)
            continue
        mx, my, mz = _apply_affine(_TAL_TO_ICBM_SPM, c.x, c.y, c.z)
        result.append(replace(
            c, x=round(mx, 1), y=round(my, 1), z=round(mz, 1),
            space="MNI", space_transform=TAL_TO_MNI_TRANSFORM,
        ))
    return result


# ── Corpus-level extraction ───────────────────────────────────────────

# Extracted text first, then raw markup/PDF converted to text; one file per
# article stem across Stage 2b (fulltext_corpus) and Stage 7 (fulltext).
_SOURCE_PRIORITY = {".txt": 0, ".md": 0, ".html": 1, ".htm": 1, ".xml": 1, ".pdf": 2}
_MARKUP_SUFFIXES = (".html", ".htm", ".xml")


def _article_sources(dirs: list[Path]) -> dict[str, Path]:
    """Best readable file per article stem (extracted text wins over raw files)."""
    best: dict[str, tuple[tuple[int, int], Path]] = {}
    for dir_rank, ft_dir in enumerate(dirs):
        if not ft_dir.is_dir():
            logger.debug("Fulltext directory not found: %s", ft_dir)
            continue
        for fpath in sorted(ft_dir.iterdir()):
            priority = _SOURCE_PRIORITY.get(fpath.suffix.lower())
            if priority is None or not fpath.is_file():
                continue
            rank = (priority, dir_rank)
            current = best.get(fpath.stem)
            if current is None or rank < current[0]:
                best[fpath.stem] = (rank, fpath)
    return {stem: path for stem, (_, path) in sorted(best.items())}


def _read_article_text(path: Path) -> str:
    """Plain text of a fulltext file; HTML/XML and PDF are converted first."""
    from acquire import fulltext

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return fulltext.extract_pdf_text(path, logger=logger)[0]
    raw = path.read_text(encoding="utf-8", errors="replace")
    return fulltext.clean_html_to_text(raw) if suffix in _MARKUP_SUFFIXES else raw


def extract_coordinates_from_corpus(config: Any) -> list[BrainCoordinate]:
    """
    Extract coordinates from Stage 2b ``fulltext_corpus`` and Stage 7
    ``fulltext`` files, reading each article (file stem) once.

    Extracted ``.txt``/``.md`` text is preferred; raw HTML/XML and PDF
    downloads are converted to text, so markup is never scanned and
    PDF-only papers are included. Returns all coordinates.
    """
    dirs = [config.data_dir / "fulltext_corpus", config.data_dir / "fulltext"]
    sources = _article_sources(dirs)
    all_coords: list[BrainCoordinate] = []
    for article_id, fpath in sources.items():
        try:
            text = _read_article_text(fpath)
        except (OSError, ValueError) as exc:
            logger.warning("Could not read %s: %s", fpath, exc)
            continue
        if not text:
            logger.info("No text could be extracted from %s", fpath)
            continue
        all_coords.extend(extract_coordinates_from_text(text, article_id=article_id))

    logger.info(
        "Extracted %d coordinates from %d articles", len(all_coords), len(sources),
    )
    return all_coords
