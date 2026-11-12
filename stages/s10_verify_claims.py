"""Stage 9 verification: count claims, cluster-size lists and unclaimed numbers.

* Counts of conditions/mechanisms/clusters/entities accept digits and the
  number words one–twenty ("two clusters"); "X of Y conditions" keeps both.
* A list of sizes after a clusters mention ("sizes: 489 and 216",
  "(n = 489 and 216)", "two clusters of 489 and 216 papers") is checked
  against the saved cluster sizes.
* Every other number in the abstract that no checker claims becomes a WARN
  row (audit 13-M10), except obvious non-claims: citation brackets and
  author-year citations, section numbering, confidence levels ("95% CI"),
  labels such as "Cluster 0" / "Table 2", and digits inside identifiers
  ("Shank3", "COVID-19", "16p11.2", "5-HT", "1/f").
"""
from __future__ import annotations

import re
from collections import Counter

# ── Number words ────────────────────────────────────────────────────────

NUMBER_WORDS = {
    word: value for value, word in enumerate(
        ("one two three four five six seven eight nine ten eleven twelve "
         "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty").split(),
        start=1)
}
_COUNT = r"\d+|" + "|".join(NUMBER_WORDS)
# "one" after these is a determiner ("in one cluster"), not a count.
_DETERMINER_PREPOSITIONS = {"in", "within", "from", "for", "at", "on", "of", "by",
                            "to", "with", "the", "each", "any", "only"}
_COUNT_NOUN = r"conditions?|mechanisms?|clusters?|entit(?:y|ies)"
# Up to two modifiers between the number and the noun ("two co-occurring
# conditions"), but never a word that starts a different phrase.
_MODIFIER = (r"(?:(?!(?:of|and|or|in|on|with|from|to|than|papers?|studies|articles?)\b)"
             r"[A-Za-z][\w\-]*\s+)")
_GENERIC_COUNT_RE = re.compile(
    rf"\b(?P<num>{_COUNT})\s+(?:of\s+(?P<total>{_COUNT})\s+)?"
    rf"{_MODIFIER}{{0,2}}(?P<noun>{_COUNT_NOUN})\b",
    re.IGNORECASE,
)


def count_value(token: str) -> int:
    """Integer value of a digit string or number word."""
    return int(token) if token.isdigit() else NUMBER_WORDS[token.lower()]


def _is_determiner_one(text: str, match: re.Match) -> bool:
    if match.group("num").lower() != "one":
        return False
    previous = re.findall(r"[A-Za-z]+", text[:match.start()][-20:])
    return bool(previous) and previous[-1].lower() in _DETERMINER_PREPOSITIONS


def extract_generic_counts(text: str) -> list[dict]:
    """Counts of conditions/mechanisms/clusters/entities, digits or words.

    For "X of Y conditions" the item's value is the total Y; the subset X is
    kept as ``subset`` so it can be reported as unverified.
    """
    items = []
    for m in _GENERIC_COUNT_RE.finditer(text):
        if _is_determiner_one(text, m):
            continue
        item = {"value": count_value(m.group("num")), "raw": m.group(0),
                "span": m.span(), "noun": m.group("noun").lower()}
        if m.group("total"):
            item["subset"] = item["value"]
            item["value"] = count_value(m.group("total"))
        items.append(item)
    return items


# ── Cluster-size lists ──────────────────────────────────────────────────

_SIZE = r"\d{1,3}(?:,\d{3})+(?!\d)|\d+"
_ITEM = rf"(?:n\s*=\s*)?(?:{_SIZE})"
_SEP = r"(?:\s*,\s*(?:and\s+)?|\s*;\s*|\s*/\s*|\s*&\s*|\s+and\s+)"
_LIST = rf"(?P<list>{_ITEM}(?:{_SEP}{_ITEM})*)"
_SIZE_LIST_RES = (
    # "cluster sizes: 489, 216", "(sizes 489/216)", "sizes of 489 and 216"
    (re.compile(rf"\bsizes?\b\s*(?:of|were|was|are|is)?\s*[:=]?\s*\(?\s*{_LIST}",
                re.IGNORECASE), 1),
    # "(n = 489 and 216)", "(n = 489, n = 216)"
    (re.compile(rf"\(\s*n\s*=\s*{_LIST}", re.IGNORECASE), 2),
    # "two clusters of 489 and 216 papers"
    (re.compile(rf"\b(?:of|with|containing|comprising)\s+{_LIST}\s+"
                r"(?:papers?|studies|articles?|publications?)\b", re.IGNORECASE), 2),
)
_CLUSTER_MENTION = re.compile(r"\bclust\w*|\bk-means\b", re.IGNORECASE)
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)|\n")


def _sentence_prefix(text: str, end: int) -> str:
    start = max((m.end() for m in _SENTENCE_END.finditer(text, 0, end)), default=0)
    return text[start:end]


def _overlaps(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    return any(span[0] < e and s < span[1] for s, e in spans)


def extract_cluster_size_lists(text: str) -> list[dict]:
    """Size lists in a sentence that mentions clusters, as ``{values, raw, span}``."""
    items: list[dict] = []
    for pattern, min_items in _SIZE_LIST_RES:
        for m in pattern.finditer(text):
            if _overlaps(m.span(), [i["span"] for i in items]):
                continue
            values = [int(v.replace(",", "")) for v in re.findall(_SIZE, m.group("list"))]
            if len(values) < min_items:
                continue
            if not _CLUSTER_MENTION.search(_sentence_prefix(text, m.end())):
                continue
            items.append({"values": values, "raw": m.group(0).strip(), "span": m.span()})
    return sorted(items, key=lambda i: i["span"])


def drop_overlapping(items: list[dict], spans: list[tuple[int, int]]) -> list[dict]:
    """Items whose span does not overlap any of ``spans``."""
    return [i for i in items if not _overlaps(i["span"], spans)]


def check_cluster_sizes(item: dict, clusters: list[dict] | None,
                        cluster_note: str) -> list[dict]:
    """One row per listed size, compared as a multiset with clusters.json."""
    raw = item["raw"]
    if clusters is None:
        return [_row(raw, v, None, "WARN", f"cluster size: {cluster_note}")
                for v in item["values"]]
    remaining = Counter(c["paper_count"] for c in clusters)
    actual_sizes = [c["paper_count"] for c in clusters]
    rows = []
    for value in item["values"]:
        if remaining[value] > 0:
            remaining[value] -= 1
            rows.append(_row(raw, value, value, "PASS",
                             "cluster size (clusters.json): matches a cluster"))
        else:
            rows.append(_row(raw, value, actual_sizes, "FAIL",
                             f"cluster size (clusters.json): no cluster of {value} "
                             f"papers; sizes are {actual_sizes}"))
    if len(item["values"]) != len(clusters):
        rows.append(_row(raw, len(item["values"]), len(clusters), "FAIL",
                         f"cluster sizes: abstract lists {len(item['values'])} sizes "
                         f"but clusters.json has {len(clusters)} clusters"))
    return rows


def _row(claim: str, expected, actual, status: str, note: str) -> dict:
    return {"claim": claim, "expected": expected, "actual": actual,
            "status": status, "note": note}


# ── Unclaimed numbers ───────────────────────────────────────────────────

_NUMBER_TOKEN = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)*(?!\w)")
_CONFIDENCE_LEVEL = re.compile(
    r"\b\d{2}(?:\.\d+)?\s*%\s*(?:CI|CrI|confidence|credible)\b", re.IGNORECASE)
_BRACKET_GROUP = re.compile(r"\[[^\[\]]*\]")
_DECIMAL = re.compile(r"\d\.\d")
_YEAR = r"(?:19|20)\d{2}[a-z]?\b"
_AUTHOR_YEAR = re.compile(
    rf"(?:\bet\s+al\.?\s*,?\s*\(?\s*|\b[A-Z][A-Za-z'’\-]+\s*(?:,\s*|\(\s*)){_YEAR}")
_SECTION_NUMBER = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]*\d+(?:\.\d+)*\.?"
    r"|\d+(?:\.\d+)*[.)](?=[ \t]+\S)"
    r"|\d+\.\d+(?:\.\d+)*(?=[ \t]+[A-Z]))",
    re.MULTILINE)
_LABEL_NOUNS = (r"clusters?|groups?|subgroups?|themes?|topics?|components?|factors?|"
                r"tables?|figures?|figs?\.?|stages?|sections?|appendix|appendices|"
                r"steps?|phases?|types?|waves?|experiments?|study|aims?|"
                r"hypothes[ie]s|chapters?|panels?|models?|layers?|classes?|tiers?")
_LABEL_ID = r"S?\d+[a-z]?\b"
_LABEL = re.compile(
    rf"\b(?:{_LABEL_NOUNS})\s+#?\s*{_LABEL_ID}"
    rf"(?:\s*(?:,|and|&|or|-|–|/|to)\s*(?:(?:{_LABEL_NOUNS})\s+)?#?\s*{_LABEL_ID})*",
    re.IGNORECASE)


def _non_claim_spans(text: str) -> list[tuple[int, int]]:
    spans = [m.span() for m in _CONFIDENCE_LEVEL.finditer(text)]
    spans += [m.span() for m in _BRACKET_GROUP.finditer(text)
              if not _DECIMAL.search(m.group(0))]
    spans += [m.span() for m in _AUTHOR_YEAR.finditer(text)]
    spans += [m.span() for m in _SECTION_NUMBER.finditer(text)]
    spans += [m.span() for m in _LABEL.finditer(text)]
    return spans


def _inside_identifier(text: str, start: int, end: int) -> bool:
    """Digits that are part of a name: "COVID-19", "DSM-5", "5-HT", "1/f"."""
    before_hyphen = start >= 2 and text[start - 1] == "-" and text[start - 2].isalpha()
    joined_after = re.match(r"-[A-Z]|/[A-Za-z]", text[end:end + 2]) is not None
    return before_hyphen or joined_after


def _token_claim(text: str, match: re.Match) -> str:
    """The number plus the word right after it, as a readable claim."""
    tail = re.match(r"\s*(?:%|[A-Za-z][\w\-]*)", text[match.end():])
    return (match.group(0) + tail.group(0)) if tail else match.group(0)


def _number(token: str) -> int | float:
    plain = token.replace(",", "") if re.fullmatch(r"\d{1,3}(?:,\d{3})+", token) else token
    try:
        return int(plain)
    except ValueError:
        try:
            return float(plain)
        except ValueError:
            return float(plain.replace(",", "."))


def extract_unclaimed(text: str, claimed: list[tuple[int, int]]) -> list[dict]:
    """Numbers outside every claimed span and every non-claim pattern."""
    skip = claimed + _non_claim_spans(text)
    items = []
    for m in _NUMBER_TOKEN.finditer(text):
        span = m.span()
        if _overlaps(span, skip) or _inside_identifier(text, *span):
            continue
        items.append({"value": _number(m.group(0)), "raw": _token_claim(text, m),
                      "span": span})
    return items


def unclaimed_rows(items: list[dict]) -> list[dict]:
    return [_row(i["raw"], i["value"], None, "WARN",
                 "found but not verified: no check covers this number")
            for i in items]
