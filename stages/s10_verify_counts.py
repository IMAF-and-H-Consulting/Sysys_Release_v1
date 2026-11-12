"""Resolve abstract paper counts to corpus or saved synthesis clusters."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

_ORDINALS = {
    word: index for index, word in enumerate(
        ("first", "second", "third", "fourth", "fifth", "sixth", "seventh",
         "eighth", "ninth", "tenth"))
}
_ORDINAL = "(?:" + "|".join(_ORDINALS) + r"|\d+(?:st|nd|rd|th))"
_REFERENCE = re.compile(
    rf"\b(?:(?P<ordinal>{_ORDINAL})(?:\s+(?:cluster|group))?"
    r"|(?:cluster|group)\s+(?P<id>\d+|[a-z]))\b", re.IGNORECASE,
)
_CONNECTOR = re.compile(
    r"[\s(:,=]* (?:(?:contained|contains|included|includes|comprised|comprises|"
    r"had|has|with|of|consisted\s+of|composed\s+of|consisting\s+of|"
    r"was|were|n)\s*[=:]?\s*)* [\s(]*$", re.IGNORECASE | re.VERBOSE,
)


# Wording that makes a paper count a subgroup ("Only 4 studies used fMRI"),
# and wording that marks it as the corpus total ("We analyzed 705 papers").
_SUBSET_BEFORE = re.compile(
    r"\b(?:only|just|of\s+(?:these|which|them|those)|among\s+(?:these|them|those)|"
    r"another|a\s+further|further|additional|remaining|some|several|a\s+subset\s+of)"
    r"\s*,?\s*$", re.IGNORECASE)
_SUBSET_AFTER = re.compile(
    r"\s+(?:that|which|who|where|whose|"
    r"used|reported|found|showed|examined|investigated|assessed|measured|focused|"
    r"described|demonstrated|observed|employed|applied|linked|supported|suggested|"
    r"implicated|addressed|explored|evaluated|compared|tested|studied|targeted|"
    r"using|reporting|examining|investigating|assessing|measuring|focusing|"
    r"describing|demonstrating|employing|applying|linking|supporting|suggesting|"
    r"addressing|exploring|evaluating|comparing|testing|targeting)\b", re.IGNORECASE)
_TOTAL_CUE = re.compile(
    r"\b(?:analy[sz]ed|included|reviewed|synthesi[sz]ed|screened|retained|"
    r"total|corpus|overall|in\s+all)\b", re.IGNORECASE)


def is_subset_count(before: str, after: str) -> bool:
    """True when a paper count's own clause qualifies it as a subgroup."""
    if _TOTAL_CUE.search(before):
        return False
    return bool(_SUBSET_BEFORE.search(before) or _SUBSET_AFTER.match(after))


def _is_cluster_reference(text: str, before: str, match: re.Match, start: int) -> bool:
    """An ordinal must modify cluster/group, or be "the second" with an antecedent.

    A sentence-initial "Second, 20 papers ..." enumerates; it is not a cluster
    (audit 13-M3).
    """
    if match.group("id") or re.search(r"(?:cluster|group)$", match.group(0), re.IGNORECASE):
        return True
    has_determiner = re.search(r"\bthe\s+$", before[:match.start()], re.IGNORECASE)
    has_antecedent = re.search(r"\b(?:cluster|group)s?\b", text[:start], re.IGNORECASE)
    return bool(has_determiner and has_antecedent)


def load_clusters(path: Path | None, papers: list[dict]) -> tuple[list[dict] | None, str]:
    """Reject missing, malformed, or stale cluster references without failing a run."""
    if path is None or not path.exists():
        return None, "no clusters.json available"
    try:
        clusters = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(clusters, list) or not clusters:
            raise ValueError("expected a nonempty list of clusters")
        ids = set()
        for cluster in clusters:
            if not isinstance(cluster, dict):
                raise ValueError("invalid cluster record")
            cid, count = cluster.get("cluster_id"), cluster.get("paper_count")
            if (not isinstance(cid, (int, str)) or isinstance(cid, bool)
                    or str(cid) in ids):
                raise ValueError("missing or duplicate cluster IDs")
            if type(count) is not int or count < 0:
                raise ValueError("invalid cluster paper_count")
            ids.add(str(cid))
        if sum(c["paper_count"] for c in clusters) != len(papers):
            raise ValueError("cluster sizes do not sum to the current corpus size")
        # Stage 6 persists membership as titles. Check the multiset, including
        # duplicates, so equal totals from a different corpus cannot verify it.
        if any("paper_titles" in c for c in clusters):
            for cluster in clusters:
                titles = cluster.get("paper_titles")
                if (not isinstance(titles, list)
                        or len(titles) != cluster["paper_count"]
                        or not all(isinstance(t, str) for t in titles)):
                    raise ValueError("cluster membership does not match paper_count")
            if Counter(t for c in clusters for t in c["paper_titles"]) != Counter(
                    p.get("title", "") for p in papers):
                raise ValueError("cluster membership does not match the current corpus")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return None, f"unusable clusters.json: {exc}"
    return clusters, "clusters.json"


def paper_count_reference(text: str, item: dict, clusters: list[dict] | None,
                          cluster_note: str) -> tuple[int | None, str] | None:
    """Return a cluster count/note, or None for an ordinary corpus count.

    Resolve IDs or ordinals from wording, never by finding a cluster whose
    size happens to equal the claim. Unidentified cluster counts stay unverified.
    """
    start, end = item["span"]
    before = re.split(r"[.!?;\n]", text[:start])[-1]
    after = re.split(r"[.!?;\n]", text[end:])[0]
    # Do not borrow the next paper count's cluster reference.
    after = re.split(r"\b\d[\d,]*\s+(?:papers?|studies|articles|publications?)\b",
                     after, maxsplit=1, flags=re.IGNORECASE)[0]
    reference = None
    for match in _REFERENCE.finditer(before):
        if _CONNECTOR.fullmatch(before[match.end():]) and _is_cluster_reference(
                text, before, match, start):
            reference = match
    if reference is None:
        suffix = re.match(
            r"\s*(?:(?:were|was)\s+)?(?:(?:assigned|belonged)\s+)?"
            r"(?:in|to|from)\s+(?:the\s+)?", after, re.IGNORECASE,
        )
        if suffix:
            reference = _REFERENCE.match(after, suffix.end())
    if reference is None:
        # A nearby cluster noun is not enough to choose which cluster.
        # Restrict to the local clause so a previous cluster cannot turn a
        # subsequent explicit total into an unidentified subset.
        local = re.split(r",|\b(?:while|whereas|and|but)\b", before,
                         flags=re.IGNORECASE)[-1]
        if re.search(r"\b(?:cluster|group|subgroup|subset)s?\b", local, re.IGNORECASE):
            return None, "cluster/subgroup count: could not identify a reference group"
        if is_subset_count(before, after):
            return None, ("subset count: refers to a subgroup of papers, not the "
                          "corpus total; no reference data, not verified")
        return None
    if clusters is None:
        return None, f"cluster paper count: {cluster_note}"
    ordinal = reference.group("ordinal")
    if ordinal:
        ordinal = ordinal.lower()
        index = _ORDINALS.get(ordinal)
        if index is None:
            index = int(re.match(r"\d+", ordinal).group()) - 1
        cluster = clusters[index] if 0 <= index < len(clusters) else None
    else:
        cid = reference.group("id")
        cluster = next((c for c in clusters if str(c["cluster_id"]).lower() == cid.lower()), None)
    if cluster is None:
        return None, "cluster paper count: referenced cluster not found in clusters.json"
    return cluster["paper_count"], f"cluster {cluster['cluster_id']} (clusters.json)"
