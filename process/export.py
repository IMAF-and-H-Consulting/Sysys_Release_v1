"""Export utilities for Sysys pipeline outputs.

Provides BibTeX, RIS, and EndNote XML bibliography export, plus a one-click
report bundle that collects all pipeline artifacts into a single directory.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import unicodedata
from datetime import datetime
from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree, SubElement, indent

logger = logging.getLogger(__name__)


# ── Corpus loading ────────────────────────────────────────────────────────


def load_corpus_for_export(config) -> list[dict]:
    """Load the best available corpus (claims_filtered > relevant > corpus).

    Falls back through progressively less-filtered corpus files so that
    export works at any pipeline stage.
    """
    candidates = [
        config.claims_dir / "claims_filtered.json",
        config.claims_dir / "relevant.json",
        config.corpus_dir / "corpus.json",
        config.data_dir / "corpus.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:  # ValueError covers JSON and UTF-8 errors
            logger.warning("Failed to load corpus from %s: %s", path, exc)
            continue
        if isinstance(data, dict) and isinstance(data.get("articles"), list):
            data = data["articles"]
        if isinstance(data, list):
            return data
        logger.warning("Skipping %s: expected a list of papers or an 'articles' list", path)
    return []


# ── Citation key generation ───────────────────────────────────────────────


_INVALID_KEY_CHARS = re.compile(r"[^a-zA-Z0-9_:-]")
_SUFFIX_ALPHABET_SIZE = 26
_MAX_INITIALS = 3  # "Smith JA": a short all-caps trailing token is initials


def _preassigned_citation_key(paper: dict) -> str:
    """The paper's own ``citation_key`` reduced to valid BibTeX key characters."""
    existing = str(paper.get("citation_key") or "").strip()
    return _INVALID_KEY_CHARS.sub("", existing)


def _make_citation_key(paper: dict) -> str:
    """Generate a citation key from first author surname + year."""
    existing = _preassigned_citation_key(paper)
    if existing:
        return existing

    surname = _ascii_letters(_surname(_first_author(paper.get("authors")))) or "unknown"

    year = str(paper.get("year", ""))[:4]
    if not year.isdigit():
        year = "nd"

    return f"{surname}{year}"


def _first_author(authors) -> str:
    """First author name; string author lists are ``;``-separated."""
    if isinstance(authors, (list, tuple)):
        return str(authors[0]).strip() if authors else ""
    return str(authors or "").split(";")[0].strip()


def _looks_like_initials(token: str) -> bool:
    letters = token.replace(".", "").replace("-", "")
    return 0 < len(letters) <= _MAX_INITIALS and letters.isalpha() and letters.isupper()


def _surname(name: str) -> str:
    """Surname from "Last, First", "First Last" or PubMed-style "Last FM"."""
    if "," in name:
        tokens = name.split(",")[0].split()
        return tokens[-1] if tokens else ""
    tokens = name.split()
    if not tokens:
        return ""
    if len(tokens) > 1 and _looks_like_initials(tokens[-1]):
        return tokens[0]
    return tokens[-1]


def _ascii_letters(text: str) -> str:
    """Lower-case ASCII letters only, transliterating accents (Müller → muller)."""
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-zA-Z]", "", folded).lower()


def _key_suffix(index: int) -> str:
    """Letter suffix for the *index*-th duplicate: 0 → a, 25 → z, 26 → aa, …"""
    letters = ""
    index += 1
    while index > 0:
        index, remainder = divmod(index - 1, _SUFFIX_ALPHABET_SIZE)
        letters = chr(ord("a") + remainder) + letters
    return letters


def _reserve_preassigned_keys(corpus: list[dict]) -> dict[int, str]:
    """Map position → pre-assigned key for the first paper to claim each key."""
    reserved: dict[int, str] = {}
    claimed: set[str] = set()
    for index, paper in enumerate(corpus):
        key = _preassigned_citation_key(paper)
        if key and key not in claimed:
            reserved[index] = key
            claimed.add(key)
    return reserved


def _unique_key(base_key: str, used: set[str], next_suffix: dict[str, int]) -> str:
    """Return *base_key*, or its first free suffixed form, never one in *used*."""
    if base_key not in used:
        return base_key
    index = next_suffix.get(base_key, 0)
    while f"{base_key}{_key_suffix(index)}" in used:
        index += 1
    next_suffix[base_key] = index + 1
    return f"{base_key}{_key_suffix(index)}"


def assign_citation_keys(corpus: list[dict]) -> list[tuple[str, dict]]:
    """Assign stable, unique, valid BibTeX citation keys to paper records.

    A paper's own ``citation_key`` is kept when no earlier paper claims it.
    Every other key is ``surname+year`` with a letter suffix on collision
    (smith2023, smith2023a, …, smith2023z, smith2023aa), checked against all
    keys already used, including pre-assigned ones.
    """
    reserved = _reserve_preassigned_keys(corpus)
    used: set[str] = set(reserved.values())
    next_suffix: dict[str, int] = {}
    keyed: list[tuple[str, dict]] = []
    for index, paper in enumerate(corpus):
        key = reserved.get(index)
        if key is None:
            key = _unique_key(_make_citation_key(paper), used, next_suffix)
            used.add(key)
        keyed.append((key, paper))
    return keyed


# Braces become text commands with balanced ``{}`` so an unbalanced brace in
# a title can never end the BibTeX field early.
_BIBTEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "{": r"\textbraceleft{}",
    "}": r"\textbraceright{}",
    "&": r"\&",
    "%": r"\%",
    "#": r"\#",
    "_": r"\_",
    "$": r"\$",
    "^": r"\textasciicircum{}",
    "~": r"\textasciitilde{}",
}
_BIBTEX_SPECIALS = re.compile("[" + re.escape("".join(_BIBTEX_ESCAPES)) + "]")


def _escape_bibtex(text) -> str:
    """Escape every LaTeX/BibTeX special character in one pass."""
    if not text:
        return ""
    return _BIBTEX_SPECIALS.sub(lambda m: _BIBTEX_ESCAPES[m.group(0)], str(text))


def _author_list(authors) -> list[str]:
    """Author names from a list or a ``;``-separated string."""
    if isinstance(authors, (list, tuple)):
        names = [str(a) for a in authors]
    else:
        names = str(authors or "").split(";")
    return [name.strip() for name in names if name.strip()]


def _format_bibtex_authors(authors) -> str:
    """Format author list for BibTeX (Author, First and Author2, First2)."""
    return " and ".join(_escape_bibtex(name) for name in _author_list(authors))


def _warn_changed_keys(keyed: list[tuple[str, dict]]) -> None:
    """Log every given ``citation_key`` that the .bib cannot use verbatim (10-H3)."""
    for key, paper in keyed:
        given = str(paper.get("citation_key") or "").strip()
        if given and given != key:
            logger.warning(
                "Citation key %r was written as %r (invalid characters or a duplicate); "
                "documents citing %r will not resolve", given, key, given)


# ── BibTeX export ─────────────────────────────────────────────────────────


def export_bibtex(corpus: list[dict], output_path: Path) -> Path:
    """Export corpus as BibTeX (.bib).

    Generates one @article entry per paper with title, author, year,
    journal, doi, and abstract fields.  Missing fields are omitted.
    Duplicate citation keys get a letter suffix (smith2023, smith2023a, …).
    """
    output_path = Path(output_path)
    if not output_path.suffix:
        output_path = output_path.with_suffix(".bib")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    keyed = assign_citation_keys(corpus)
    _warn_changed_keys(keyed)

    for key, paper in keyed:
        entry_lines = [f"@article{{{key},"]

        title = paper.get("title", "")
        if title:
            entry_lines.append(f"    title = {{{_escape_bibtex(title)}}},")

        authors = paper.get("authors", "")
        if authors:
            entry_lines.append(f"    author = {{{_format_bibtex_authors(authors)}}},")

        year = paper.get("year", "")
        if year:
            entry_lines.append(f"    year = {{{year}}},")

        journal = paper.get("journal", "")
        if journal:
            entry_lines.append(f"    journal = {{{_escape_bibtex(journal)}}},")

        doi = paper.get("doi", "")
        if doi:
            entry_lines.append(f"    doi = {{{_escape_bibtex(doi)}}},")

        abstract = paper.get("abstract", "")
        if abstract:
            entry_lines.append(f"    abstract = {{{_escape_bibtex(abstract)}}},")

        entry_lines.append("}")
        lines.append("\n".join(entry_lines))

    output_path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Exported %d entries to BibTeX: %s", len(corpus), output_path)
    return output_path


# ── RIS export ────────────────────────────────────────────────────────────


def export_ris(corpus: list[dict], output_path: Path) -> Path:
    """Export corpus as RIS (.ris).

    RIS format uses tagged lines (TY, TI, AU, PY, JO, DO, AB, ER).
    Each author gets a separate AU tag.
    """
    output_path = Path(output_path)
    if not output_path.suffix:
        output_path = output_path.with_suffix(".ris")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    entries = ["\n".join(_ris_lines(paper)) for paper in corpus]

    output_path.write_text("\n\n".join(entries) + "\n", encoding="utf-8")
    logger.info("Exported %d entries to RIS: %s", len(corpus), output_path)
    return output_path


def _one_line(value) -> str:
    """Collapse all whitespace (including newlines) so a RIS value stays on its tag line."""
    return " ".join(str(value).split())


def _ris_lines(paper: dict) -> list[str]:
    lines = ["TY  - JOUR"]
    tagged = [("TI", paper.get("title", ""))]
    tagged += [("AU", author) for author in _author_list(paper.get("authors", ""))]
    tagged += [("PY", paper.get("year", "")), ("JO", paper.get("journal", "")),
               ("DO", paper.get("doi", "")), ("AB", paper.get("abstract", ""))]
    for tag, value in tagged:
        text = _one_line(value) if value else ""
        if text:
            lines.append(f"{tag}  - {text}")
    lines.append("ER  - ")
    return lines


# Characters XML 1.0 forbids (C0 controls other than tab/LF/CR, surrogates, U+FFFE/F).
_XML_ILLEGAL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")


def _xml_text(value) -> str:
    return _XML_ILLEGAL.sub("", str(value))


# ── EndNote XML export ────────────────────────────────────────────────────


def export_endnote_xml(corpus: list[dict], output_path: Path) -> Path:
    """Export corpus as EndNote XML.

    Generates an XML file following the EndNote import schema with
    record, title, authors, year, periodical, electronic-resource-num,
    and abstract elements.
    """
    output_path = Path(output_path)
    if not output_path.suffix:
        output_path = output_path.with_suffix(".xml")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    xml_root = Element("xml")
    records = SubElement(xml_root, "records")

    for paper in corpus:
        record = SubElement(records, "record")

        # Reference type
        ref_type = SubElement(record, "ref-type", name="Journal Article")
        ref_type.text = "17"

        # Contributors / authors
        contributors = SubElement(record, "contributors")
        authors_el = SubElement(contributors, "authors")
        for author in _author_list(paper.get("authors", "")):
            a_el = SubElement(authors_el, "author")
            a_el.text = _xml_text(author)

        # Title
        titles = SubElement(record, "titles")
        title_el = SubElement(titles, "title")
        title_el.text = _xml_text(paper.get("title", "") or "")

        # Journal
        journal = paper.get("journal", "")
        if journal:
            periodical = SubElement(titles, "secondary-title")
            periodical.text = _xml_text(journal)

        # Year
        year = paper.get("year", "")
        if year:
            dates = SubElement(record, "dates")
            year_el = SubElement(dates, "year")
            year_el.text = str(year)

        # DOI
        doi = paper.get("doi", "")
        if doi:
            ern = SubElement(record, "electronic-resource-num")
            ern.text = _xml_text(doi)

        # Abstract
        abstract = paper.get("abstract", "")
        if abstract:
            abs_el = SubElement(record, "abstract")
            abs_el.text = _xml_text(abstract)

    indent(xml_root, space="  ")
    tree = ElementTree(xml_root)
    tree.write(str(output_path), encoding="unicode", xml_declaration=True)
    logger.info("Exported %d entries to EndNote XML: %s", len(corpus), output_path)
    return output_path


# ── Report bundle ─────────────────────────────────────────────────────────


def _copy_synthesis_abstracts(config, output_dir: Path, copied_files: list[str]) -> None:
    """Copy synthesis abstracts without letting two sources share one filename.

    The pipeline abstract wins the ``synthesis_abstract.md`` name. When the
    cluster draft also exists, it is stored as ``abstract_draft.md``.
    """
    pipeline_abstract = config.output_dir / "synthesis_abstract.md"
    cluster_draft = config.clusters_dir / "abstract_draft.md"
    pipeline_exists = pipeline_abstract.exists()
    draft_exists = cluster_draft.exists()

    if pipeline_exists:
        try:
            shutil.copy2(pipeline_abstract, output_dir / "synthesis_abstract.md")
            copied_files.append("synthesis_abstract.md")
        except OSError as exc:
            logger.warning("Failed to copy %s: %s", pipeline_abstract, exc)
    elif draft_exists:
        try:
            shutil.copy2(cluster_draft, output_dir / "synthesis_abstract.md")
            copied_files.append("synthesis_abstract.md")
            logger.info(
                "Copied cluster abstract draft as synthesis_abstract.md; "
                "no pipeline synthesis abstract was present"
            )
        except OSError as exc:
            logger.warning("Failed to copy %s: %s", cluster_draft, exc)
        return

    if pipeline_exists and draft_exists:
        try:
            shutil.copy2(cluster_draft, output_dir / "abstract_draft.md")
            copied_files.append("abstract_draft.md")
            logger.info(
                "Kept cluster abstract draft as abstract_draft.md; "
                "synthesis_abstract.md already came from %s",
                pipeline_abstract,
            )
        except OSError as exc:
            logger.warning("Failed to copy %s: %s", cluster_draft, exc)


# Files the bundle writes besides the file map; only these names are cleared
# from a re-used bundle directory, so unrelated user files are kept.
_BUNDLE_README_TITLE = "# Sysys Report Bundle"
_BUNDLE_GENERATED_FILES = (
    "synthesis_abstract.md", "abstract_draft.md", "corpus.bib", "corpus.ris", "README.md",
)


def _bundle_file_map(config) -> list[tuple[Path, str]]:
    """(source, bundle filename) for every artifact copied into the bundle."""
    return [
        # PRISMA outputs
        (config.output_dir / "prisma_flow_diagram.png", "prisma_flow.png"),
        (config.output_dir / "prisma_flow_diagram.svg", "prisma_flow.svg"),
        (config.output_dir / "prisma_checklist.md", "prisma_checklist.md"),
        (config.output_dir / "prisma_checklist.csv", "prisma_checklist.csv"),
        # GRADE outputs
        (config.output_dir / "grade_sof_table.md", "grade_sof_table.md"),
        (config.output_dir / "grade_results.json", "grade_results.json"),
        # Meta-analysis plots
        (config.output_dir / "forest_plot.png", "forest_plot.png"),
        (config.output_dir / "funnel_plot.png", "funnel_plot.png"),
        # Co-occurrence network and optional neuroimaging coordinates
        (config.output_dir / "network_graph.svg", "network_graph.svg"),
        (config.output_dir / "coordinates.nimare.json", "coordinates.nimare.json"),
        (config.output_dir / "coordinates.sleuth.txt", "coordinates.sleuth.txt"),
        # Final source-linked review and its reproducible evidence ledger
        (config.output_dir / "final_review_report.md", "final_review_report.md"),
        (config.output_dir / "final_review_sources.json", "final_review_sources.json"),
        (config.output_dir / "final_review_evidence.md", "final_review_evidence.md"),
        (config.data_dir / "validation" / "verification_results.json", "verification_results.json"),
        # Robustness & synthesis
        (config.output_dir / "robustness_report.md", "robustness_report.md"),
        (config.clusters_dir / "synthesis_sources.md", "synthesis_sources.md"),
        (config.clusters_dir / "synthesis_sources.json", "synthesis_sources.json"),
        (config.clusters_dir / "synthesis_sources.bib", "synthesis_sources.bib"),
        # NeuroQuery
        (config.output_dir / "neuroquery_convergence.json", "neuroquery_convergence.json"),
        # Full results
        (config.output_dir / "full_results.json", "full_results.json"),
    ]


def _is_previous_bundle(output_dir: Path) -> bool:
    readme = output_dir / "README.md"
    try:
        return readme.is_file() and readme.read_text(
            encoding="utf-8", errors="replace").startswith(_BUNDLE_README_TITLE)
    except OSError:
        return False


def _remove_stale_bundle_files(config, output_dir: Path) -> None:
    """Delete files a previous bundle export wrote, so absent artifacts are not kept.

    Only a directory holding an earlier bundle README is cleared, and a
    bundle path that is also a source artifact is never deleted.
    """
    if not _is_previous_bundle(output_dir):
        return
    file_map = _bundle_file_map(config)
    sources = {src.resolve() for src, _dest in file_map}
    names = {dest for _src, dest in file_map} | set(_BUNDLE_GENERATED_FILES)
    for name in sorted(names):
        path = output_dir / name
        if path.is_file() and path.resolve() not in sources:
            try:
                path.unlink()
            except OSError as exc:
                logger.warning("Could not remove stale bundle file %s: %s", path, exc)


def export_report_bundle(config, output_dir: Path) -> Path:
    """One-click export of everything the pipeline has produced.

    Creates a directory with all available pipeline artifacts:
    PRISMA reports, GRADE results, plots, coordinates, bibliography,
    robustness report, synthesis abstract, and a README explaining each file.

    Skips files that don't exist (not all stages may have run). Files a
    previous export wrote into *output_dir* are removed first, so an
    artifact that no longer exists is not shipped as current.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_bundle_files(config, output_dir)

    copied_files: list[str] = []
    for src, dest_name in _bundle_file_map(config):
        if src.exists():
            try:
                shutil.copy2(src, output_dir / dest_name)
                if dest_name not in copied_files:
                    copied_files.append(dest_name)
            except OSError as exc:
                logger.warning("Failed to copy %s: %s", src, exc)

    _copy_synthesis_abstracts(config, output_dir, copied_files)

    # Export bibliography from corpus
    corpus = load_corpus_for_export(config)
    if corpus:
        try:
            export_bibtex(corpus, output_dir / "corpus.bib")
            copied_files.append("corpus.bib")
        except Exception as exc:
            logger.warning("BibTeX export failed: %s", exc)
        try:
            export_ris(corpus, output_dir / "corpus.ris")
            copied_files.append("corpus.ris")
        except Exception as exc:
            logger.warning("RIS export failed: %s", exc)

    # Generate README
    _write_bundle_readme(output_dir, copied_files, config)
    copied_files.append("README.md")

    logger.info(
        "Report bundle exported to %s (%d files)", output_dir, len(copied_files)
    )
    return output_dir


def _write_bundle_readme(output_dir: Path, files: list[str], config) -> None:
    """Write a README.md explaining the report bundle contents."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    descriptions = {
        "final_review_report.md": "Final source-linked review with internal numerical checks",
        "final_review_evidence.md": "Unabridged findings, methods and statistical evidence for every source",
        "final_review_sources.json": "Complete evidence ledger and input hashes for the final review",
        "verification_results.json": "Final review coverage checks and provenance",
        "prisma_flow.png": "PRISMA 2020 flow diagram (PNG)",
        "prisma_flow.svg": "PRISMA 2020 flow diagram (SVG, editable)",
        "prisma_checklist.md": "PRISMA 2020 checklist (Markdown)",
        "prisma_checklist.csv": "PRISMA 2020 checklist (CSV)",
        "grade_sof_table.md": "GRADE Summary of Findings table",
        "grade_results.json": "GRADE assessment results (JSON)",
        "forest_plot.png": "Forest plot — random-effects meta-analysis",
        "funnel_plot.png": "Contour-enhanced funnel plot for publication bias",
        "network_graph.svg": "Condition–entity co-occurrence network graph",
        "coordinates.nimare.json": "Neuroimaging coordinates in NiMARE format (Neuroimaging tab)",
        "coordinates.sleuth.txt": "Neuroimaging coordinates in GingerALE/Sleuth format (Neuroimaging tab)",
        "corpus.bib": "Full corpus bibliography (BibTeX)",
        "corpus.ris": "Full corpus bibliography (RIS)",
        "robustness_report.md": "Robustness analysis report",
        "synthesis_abstract.md": "Narrative synthesis abstract",
        "abstract_draft.md": "Cluster abstract draft (kept separate from the synthesis abstract)",
        "synthesis_sources.md": "Sources used in synthesis, grouped by cluster",
        "synthesis_sources.json": "Structured source map for synthesis citations",
        "synthesis_sources.bib": "Bibliography for sources used in synthesis",
        "neuroquery_convergence.json": "NeuroQuery convergence scores",
        "full_results.json": "Complete pipeline results",
    }

    lines = [
        _BUNDLE_README_TITLE,
        "",
        f"Generated: {timestamp}",
        f"LLM Provider: {config.llm_provider}",
        f"LLM Model: {config.llm_model}",
        "",
        "## Contents",
        "",
        "| File | Description |",
        "|------|-------------|",
    ]
    for fname in sorted(files):
        desc = descriptions.get(fname, "Pipeline output")
        lines.append(f"| `{fname}` | {desc} |")

    lines.extend([
        "",
        "## Pipeline Configuration",
        "",
        f"- Hypothesis: {config.hypothesis_text[:120]}...",
        f"- Relevance threshold: {config.relevance_threshold}",
        f"- Project root: {config.project_root}",
        "",
        "---",
        "*Generated by Sysys (Systematic Evidence Synthesis)*",
    ])

    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
