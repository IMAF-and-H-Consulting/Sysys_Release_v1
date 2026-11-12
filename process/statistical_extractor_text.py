"""Text candidate and paragraph/window helpers."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from core.doi import normalize_doi
from process.statistical_extractor_patterns import (
    TextBlock,
)


def article_id(paper: dict[str, Any], index: int = 0) -> str:
    """Return a stable best-effort article id."""
    for key in ("doi", "pmid", "s2_paper_id", "paperId", "openalex_id", "id"):
        value = str(paper.get(key, "") or "").strip()
        if value:
            return value
    title = str(paper.get("title", "") or "").strip()
    return title[:80] if title else f"paper_{index + 1}"


def text_candidates(paper: dict[str, Any]) -> list[tuple[str, str]]:
    """Return text fields in preferred extraction order."""
    candidates: list[tuple[str, str]] = []
    fields = (
        ("full_text", "full_text"),
        ("fulltext", "full_text"),
        ("extracted_text", "full_text"),
        ("text", "text"),
        ("abstract", "abstract"),
    )
    for key, label in fields:
        value = paper.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append((label, value.strip()))

    path_value = str(paper.get("full_text_path") or paper.get("text_path") or "").strip()
    if path_value:
        path = Path(path_value)
        if path.is_file():
            try:
                candidates.insert(0, ("full_text_file", path.read_text(encoding="utf-8")))
            except (OSError, UnicodeError):
                pass

    return candidates


def enrich_from_saved_fulltext(papers: list[dict], corpus_path: Path) -> tuple[list[dict], int]:
    """Attach available full text by stable identity, without changing inclusion.

    Work on copies; neither the relevance artifact nor the enriched corpus is
    rewritten. Do not match by title or append unretained corpus papers.
    """
    import json

    if not corpus_path.is_file():
        return papers, 0
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    if not isinstance(corpus, list):
        raise ValueError(f"Expected a list of papers in {corpus_path}")

    def keys(paper):
        doi = normalize_doi(paper.get("doi"))
        if doi:
            yield "doi", doi
        for name in ("pmid", "s2_paper_id", "paperId", "openalex_id"):
            if paper.get(name):
                yield name, str(paper[name]).strip()

    index = {}
    for source in corpus:
        texts = [(key, source.get(key)) for key in ("full_text", "fulltext", "extracted_text")]
        texts = [(key, value) for key, value in texts if isinstance(value, str) and value.strip()]
        if not texts:
            continue
        _, full_text = max(texts, key=lambda item: len(item[1]))
        for key in keys(source):
            previous = index.get(key)
            if previous is None or len(full_text) > len(previous[1]):
                index[key] = (source, full_text)

    enriched, count = [], 0
    for paper in papers:
        if any(isinstance(paper.get(k), str) and paper[k].strip()
               for k in ("full_text", "fulltext", "extracted_text")):
            enriched.append(paper)
            continue
        match = next((index[key] for key in keys(paper) if key in index), None)
        if match is None:
            enriched.append(paper)
            continue
        source, full_text = match
        # A matching secondary identifier must not override conflicting DOIs.
        left, right = normalize_doi(paper.get("doi")), normalize_doi(source.get("doi"))
        if left and right and left != right:
            enriched.append(paper)
            continue
        enriched.append({**paper, "full_text": full_text,
                         "statistical_text_source_path": str(corpus_path)})
        count += 1
    return enriched, count


def paragraph_blocks(text: str, source: str) -> list[TextBlock]:
    """Split text into reviewable paragraphs while keeping offsets."""
    blocks: list[TextBlock] = []
    if not text:
        return blocks

    matches = list(re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", text, re.DOTALL))
    if not matches:
        matches = [re.match(r".+", text, re.DOTALL)]  # type: ignore[list-item]

    section = ""
    for idx, match in enumerate(m for m in matches if m is not None):
        raw = match.group(0).strip()
        if not raw:
            continue
        if len(raw) <= 80 and re.match(r"^[A-Z][A-Za-z /-]{2,}:?$", raw):
            section = raw.rstrip(":")
            continue
        blocks.append(
            TextBlock(
                text=raw,
                source=source,
                paragraph_index=len(blocks),
                char_start=match.start(),
                section=section,
            )
        )

    if len(blocks) == 1 and len(blocks[0].text) > 1200:
        return sentence_window_blocks(blocks[0])
    return blocks


def sentence_window_blocks(block: TextBlock, max_sentences: int = 4) -> list[TextBlock]:
    """Group whole sentences without altering decimals or source offsets."""
    spans = sentence_spans(block.text)
    windows: list[TextBlock] = []
    for i in range(0, len(spans), max_sentences):
        chunk = spans[i:i + max_sentences]
        if not chunk:
            continue
        start, end = chunk[0][0], chunk[-1][1]
        text = block.text[start:end]
        if not text:
            continue
        windows.append(
            TextBlock(
                text=text,
                source=block.source,
                paragraph_index=len(windows),
                char_start=block.char_start + start,
                section=block.section,
            )
        )
    return windows or [block]


_ABBREVIATION_END = re.compile(r"\b(?:vs|al|e\.g|i\.e|Dr|Mr|Mrs|Fig|Eq)\Z", re.I)
# Only the text just before a period can end in an abbreviation. The longest is
# 3 characters; the extra context keeps ``\b`` exact at the start of the slice.
# Searching the whole preceding text made sentence_spans quadratic.
ABBREVIATION_TAIL_CHARS = 8


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Find sentence ends, excluding decimal points and common abbreviations.

    Spans refer to the original text, including PDF-spaced decimals. Never
    reconstruct sentences by joining tokens: that corrupts numbers and offsets.
    """
    spans = []
    start = 0
    for match in re.finditer(r'[.!?]+[\"\)\]]*(?=\s|$)', text):
        pos = match.start()
        if text[pos] == ".":
            before = text[:pos].rstrip()
            after = text[pos + 1:].lstrip()
            if after[:1].isdigit() and (before[-1:].isdigit() or before[-1:] in "=<>±−+-"):
                continue
            if _ABBREVIATION_END.search(before[-ABBREVIATION_TAIL_CHARS:]):
                continue
        end = match.end()
        while start < end and text[start].isspace():
            start += 1
        if start < end:
            spans.append((start, end))
        start = end
    while start < len(text) and text[start].isspace():
        start += 1
    if start < len(text):
        spans.append((start, len(text.rstrip())))
    return spans
