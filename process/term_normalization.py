"""Canonicalize structured categorical terms at their producer boundary.

The pipeline deliberately keeps this separate from prose cleanup.  Findings,
quotes, source text, methods descriptions, and statistical test text are source
evidence and must remain verbatim.  This module is only for fields whose values
are *labels* (for example conditions or outcome names).

Normalization is corpus-wide and conservative:

* Unicode/whitespace/case variants are equivalent.
* A terminal parenthetical is removed only when it is a valid initialism of the
  preceding phrase (for example ``Example condition (EC)``).
* A regular final-word plural is joined only when the corresponding singular is
  actually observed or configured.
* An acronym-only label is joined only when the corpus gives it one unambiguous
  expansion.
* Semantic synonyms require an explicit project vocabulary.

Active fields receive canonical labels.  Every producer record also receives a
raw-to-canonical ledger, and callers persist the returned corpus audit beside
their normal output.  Synthesis and meta-analysis therefore only count or group
already-canonical values; they never repair labels while reporting.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import re
import unicodedata
from typing import Any, Iterable, Mapping


NORMALIZATION_SCHEMA_VERSION = 1
NORMALIZATION_POLICY_ID = "structured-categorical-terms/v1"
TERM_NORMALIZATION_KEY = "term_normalization"

# These are categorical in the built-in claim schema.  Custom list fields must
# explicitly opt in through extraction.fields[].normalization or
# normalization.fields in the research configuration.
DEFAULT_CLAIM_CATEGORICAL_FIELDS = (
    "conditions",
    "mechanisms",
    "key_entities",
    "brain_regions",  # legacy name for key_entities
)
DEFAULT_STATISTICAL_CATEGORICAL_FIELDS = (
    "population",
    "outcome",
    "comparison",
)

_DASHES = str.maketrans({
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",
    "\u2014": "-", "\u2015": "-", "\u2212": "-",
    "\u2018": "'", "\u2019": "'", "\u02bc": "'",
    "\u201c": '"', "\u201d": '"',
})
_TERMINAL_PARENTHETICAL = re.compile(r"^(?P<base>.+?)\s*\((?P<suffix>[^()]{1,32})\)\s*$")
_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_INITIALISM_STOPWORDS = frozenset({
    "a", "an", "and", "at", "by", "for", "from", "in", "of", "on",
    "or", "the", "to", "with",
})


def clean_structured_term(value: Any) -> str:
    """Return a display-safe structured label without changing its meaning."""
    text = unicodedata.normalize("NFKC", str(value or "")).translate(_DASHES)
    return " ".join(text.split()).strip()


def _initialism(text: str) -> str:
    # English possessive 's is not an initialism word (Person's measure -> PM,
    # not PSM). Typographic apostrophes were unified by clean_structured_term.
    words = _WORD.findall(re.sub(r"(?i)(?<=\w)'s\b", "", text))
    meaningful = [word for word in words if word.casefold() not in _INITIALISM_STOPWORDS]
    selected = meaningful or words
    return "".join(word[0] for word in selected if word)


def _acronym_key(text: str) -> str:
    return "".join(ch for ch in text if ch.isalnum()).casefold()


def _split_valid_terminal_acronym(text: str) -> tuple[str, str | None]:
    match = _TERMINAL_PARENTHETICAL.fullmatch(text)
    if not match:
        return text, None
    base = clean_structured_term(match.group("base"))
    suffix = _acronym_key(match.group("suffix"))
    expected = _acronym_key(_initialism(base))
    if expected and suffix in {expected, expected + "s"}:
        return base, expected
    return text, None


def _comparison_key(text: str) -> str:
    base, _ = _split_valid_terminal_acronym(clean_structured_term(text))
    return base.casefold()


def _singular_candidate(key: str) -> str | None:
    """A conservative regular singular candidate, never applied on its own.

    The caller only joins it when that candidate is independently present (or
    configured).  This protects lexical endings such as diabetes, psychosis,
    stress, and multiple sclerosis.
    """
    head, separator, last = key.rpartition(" ")
    if not last:
        last = head
        head = ""
        separator = ""
    if len(last) <= 3 or last.endswith(("ss", "us", "is")):
        return None
    if last.endswith("ies") and len(last) > 4:
        singular = last[:-3] + "y"
    elif last.endswith(("ches", "shes", "xes", "zes")):
        singular = last[:-2]
    elif last.endswith("sses"):
        singular = last[:-2]
    elif last.endswith("s"):
        singular = last[:-1]
    else:
        return None
    return f"{head}{separator}{singular}" if head else singular


def _looks_like_acronym_only(text: str) -> bool:
    compact = "".join(ch for ch in text if ch.isalnum())
    return bool(compact) and len(compact) <= 12 and not any(ch.isspace() for ch in text) \
        and (compact.isupper() or len(compact) <= 4)


def _style_penalty(text: str) -> int:
    words = [word for word in _WORD.findall(text) if word]
    if not words:
        return 3
    if len(words) > 1 and all(word[:1].isupper() and word[1:].islower() for word in words):
        return 1  # Title Case is less natural for most report labels.
    if text.isupper() and len(text) > 4:
        return 2
    return 0


class _UnionFind:
    def __init__(self, values: Iterable[str]):
        self.parent = {value: value for value in values}

    def add(self, value: str) -> None:
        self.parent.setdefault(value, value)

    def find(self, value: str) -> str:
        self.add(value)
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        # Lexical root selection makes the groups independent of input order.
        if b < a:
            a, b = b, a
        self.parent[b] = a


@dataclass(frozen=True)
class _ObservedTerm:
    raw: str
    cleaned: str
    base: str
    key: str
    acronym: str | None


@dataclass
class CanonicalTermMap:
    """Resolved raw-label map plus a serializable audit."""

    entries: dict[str, dict[str, Any]]
    audit: dict[str, Any]

    def resolve(self, raw: Any) -> dict[str, Any]:
        text = str(raw or "")
        if text in self.entries:
            return self.entries[text]
        cleaned = clean_structured_term(text)
        return {
            "raw": text,
            "canonical": cleaned,
            "rules": ["late_unseen_value"],
            "changed": text != cleaned,
        }


def _normalization_section(config: Any) -> dict[str, Any]:
    research = getattr(config, "research_config", config if isinstance(config, dict) else {})
    if not isinstance(research, dict):
        return {}
    value = research.get("normalization", {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("normalization must be a mapping")
    if "schema_version" in value:
        version = value["schema_version"]
        if isinstance(version, bool) or str(version).strip() != str(
            NORMALIZATION_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported normalization.schema_version "
                f"{version!r}; this application supports version "
                f"{NORMALIZATION_SCHEMA_VERSION}"
            )
    return value


def _config_hash(config: Any) -> str:
    payload = json.dumps(
        _normalization_section(config), sort_keys=True, ensure_ascii=False, default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _field_config(config: Any, path: str) -> dict[str, Any]:
    fields = _normalization_section(config).get("fields", {})
    if fields is None:
        return {}
    if not isinstance(fields, dict):
        raise ValueError("normalization.fields must be a mapping")
    value = fields.get(path, {})
    if isinstance(value, str):
        return {"mode": value}
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"normalization.fields.{path} must be a mapping or mode string")
    return value


def _configured_vocabulary(config: Any, path: str) -> dict[str, dict[str, list[str]]]:
    section = _normalization_section(config)
    vocabularies = section.get("vocabularies", {})
    if vocabularies is None:
        return {}
    if not isinstance(vocabularies, dict):
        raise ValueError("normalization.vocabularies must be a mapping")
    field = _field_config(config, path)
    vocabulary_name = field.get("vocabulary") or path.split(".")[-1]
    raw_vocab = vocabularies.get(vocabulary_name, {})
    if raw_vocab is None:
        return {}
    if not isinstance(raw_vocab, dict):
        raise ValueError(f"normalization.vocabularies.{vocabulary_name} must be a mapping")
    parsed: dict[str, dict[str, list[str]]] = {}
    for raw_canonical, raw_rules in raw_vocab.items():
        canonical = clean_structured_term(raw_canonical)
        if not canonical:
            raise ValueError(f"normalization.vocabularies.{vocabulary_name} has an empty canonical term")
        if isinstance(raw_rules, list):
            aliases, patterns = raw_rules, []
        elif isinstance(raw_rules, dict):
            aliases = raw_rules.get("aliases", []) or []
            patterns = raw_rules.get("patterns", []) or []
        elif raw_rules is None:
            aliases, patterns = [], []
        else:
            raise ValueError(
                f"normalization.vocabularies.{vocabulary_name}.{canonical} must be a mapping or list"
            )
        if (not isinstance(aliases, list) or not all(isinstance(v, str) for v in aliases)
                or not isinstance(patterns, list) or not all(isinstance(v, str) for v in patterns)):
            raise ValueError(
                f"normalization.vocabularies.{vocabulary_name}.{canonical} aliases/patterns "
                "must be lists of strings"
            )
        for pattern in patterns:
            try:
                re.compile(pattern, re.IGNORECASE)
            except re.error as exc:
                raise ValueError(
                    f"Invalid regex in normalization.vocabularies.{vocabulary_name}.{canonical}: {exc}"
                ) from exc
        parsed[canonical] = {
            "aliases": [clean_structured_term(alias) for alias in aliases if clean_structured_term(alias)],
            "patterns": list(patterns),
        }
    return parsed


def _merge_vocabularies(*vocabularies: Mapping[str, Mapping[str, Iterable[str]]]) \
        -> dict[str, dict[str, list[str]]]:
    merged: dict[str, dict[str, list[str]]] = {}
    spelling_by_key: dict[str, str] = {}
    for vocabulary in vocabularies:
        for canonical, rules in vocabulary.items():
            spelling = clean_structured_term(canonical)
            identity = _comparison_key(spelling)
            previous = spelling_by_key.get(identity)
            if previous is not None and previous != spelling:
                # Later (project-explicit) vocabularies control display spelling,
                # while retaining the earlier pack's aliases/patterns.
                retained = merged.pop(previous)
                merged[spelling] = retained
            spelling_by_key[identity] = spelling
            target = merged.setdefault(spelling, {"aliases": [], "patterns": []})
            for key in ("aliases", "patterns"):
                for value in rules.get(key, []) or []:
                    if value not in target[key]:
                        target[key].append(value)
    return merged


def build_canonical_term_map(
    values: Iterable[Any],
    *,
    vocabulary: Mapping[str, Mapping[str, Iterable[str]]] | None = None,
) -> CanonicalTermMap:
    """Resolve a deterministic canonical label for every observed raw value."""
    raw_values = [str(value) for value in values if str(value or "").strip()]
    counts = Counter(raw_values)
    observed: dict[str, _ObservedTerm] = {}
    for raw in counts:
        cleaned = clean_structured_term(raw)
        base, acronym = _split_valid_terminal_acronym(cleaned)
        observed[raw] = _ObservedTerm(raw, cleaned, base, base.casefold(), acronym)

    vocabulary = _merge_vocabularies(vocabulary or {})
    keys = {item.key for item in observed.values()}
    acronym_only_keys = {
        item.key for item in observed.values()
        if item.acronym is None and _looks_like_acronym_only(item.base)
    }
    uf = _UnionFind(keys)
    plural_to_singular: dict[str, str] = {}
    for key in sorted(keys):
        singular = _singular_candidate(key)
        if singular and singular in keys:
            uf.union(key, singular)
            plural_to_singular[key] = singular

    # A validated long-form suffix establishes an acronym expansion.  Standalone
    # acronyms join only when exactly one expansion exists in this corpus.
    expansions: dict[str, set[str]] = defaultdict(set)
    for item in observed.values():
        if item.acronym:
            # Singular/plural long forms already joined above are one expansion,
            # not an ambiguous acronym (XYZ versus XYZs).
            expansions[item.acronym].add(uf.find(item.key))
    unique_acronym_links: dict[str, str] = {}
    for acronym, bases in sorted(expansions.items()):
        if len(bases) != 1:
            continue
        base = next(iter(bases))
        for key in sorted(keys):
            compact = _acronym_key(key)
            if compact in {acronym, acronym + "s"} and key in acronym_only_keys:
                uf.union(key, base)
                unique_acronym_links[key] = base

    configured_for_key: dict[str, set[str]] = defaultdict(set)
    configured_rule: dict[tuple[str, str], str] = {}
    for canonical, rules in vocabulary.items():
        canonical_key = _comparison_key(canonical)
        uf.add(canonical_key)
        aliases = [canonical, *(rules.get("aliases", []) or [])]
        for alias in aliases:
            alias_key = _comparison_key(alias)
            for key in keys:
                if key == alias_key:
                    configured_for_key[key].add(canonical)
                    configured_rule[(key, canonical)] = "configured_alias"
                elif _singular_candidate(key) == alias_key:
                    # A regular plural can join a singular explicitly named by
                    # the project even when that singular is not independently
                    # observed in this corpus.
                    configured_for_key[key].add(canonical)
                    configured_rule[(key, canonical)] = \
                        "configured_regular_plural_pair"
        for pattern in rules.get("patterns", []) or []:
            regex = re.compile(pattern, re.IGNORECASE)
            for key in keys:
                bases = {item.base for item in observed.values() if item.key == key}
                if any(regex.fullmatch(base) for base in bases):
                    configured_for_key[key].add(canonical)
                    configured_rule[(key, canonical)] = "configured_pattern"

    ambiguous: list[dict[str, Any]] = []
    configured_labels_by_root: dict[str, set[str]] = defaultdict(set)
    for key, labels in sorted(configured_for_key.items()):
        if len(labels) != 1:
            candidates = sorted(labels, key=str.casefold)
            raise ValueError(
                f"Structured-term configuration is ambiguous for {key!r}: "
                f"multiple canonical terms match ({', '.join(candidates)})"
            )
        canonical = next(iter(labels))
        canonical_key = _comparison_key(canonical)
        uf.union(key, canonical_key)

    # Rebuild configured roots after all unions.
    for key, labels in configured_for_key.items():
        if len(labels) == 1:
            configured_labels_by_root[uf.find(key)].update(labels)
    for root, labels in configured_labels_by_root.items():
        if len(labels) > 1:
            candidates = sorted(labels, key=str.casefold)
            raise ValueError(
                "Structured-term configuration assigns automatically equivalent "
                f"labels to conflicting canonical terms ({', '.join(candidates)}); "
                f"equivalence root {root!r} was not canonicalized"
            )

    by_root: dict[str, list[_ObservedTerm]] = defaultdict(list)
    for item in observed.values():
        by_root[uf.find(item.key)].append(item)

    canonical_by_root: dict[str, str] = {}
    for root, items in by_root.items():
        configured = configured_labels_by_root.get(root, set())
        if len(configured) == 1:
            canonical_by_root[root] = next(iter(configured))
            continue
        surface_counts: Counter[str] = Counter()
        surface_keys: dict[str, set[str]] = defaultdict(set)
        for item in items:
            surface_counts[item.base] += counts[item.raw]
            surface_keys[item.base].add(item.key)
        candidates = sorted(
            surface_counts,
            key=lambda surface: (
                min(1 if key in plural_to_singular else 0 for key in surface_keys[surface]),
                1 if _looks_like_acronym_only(surface) else 0,
                -surface_counts[surface],
                _style_penalty(surface),
                surface.casefold(),
                surface,
            ),
        )
        canonical_by_root[root] = candidates[0]

    entries: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw, item in observed.items():
        root = uf.find(item.key)
        canonical = canonical_by_root[root]
        rules: list[str] = []
        configured = configured_for_key.get(item.key, set())
        if len(configured) == 1:
            rules.append(configured_rule.get((item.key, next(iter(configured))), "configured_alias"))
        if item.acronym:
            rules.append("validated_terminal_acronym")
        if item.key in plural_to_singular:
            rules.append("observed_regular_plural_pair")
        if item.key in unique_acronym_links:
            rules.append("unique_acronym_expansion")
        if item.base.casefold() == canonical.casefold() and item.base != canonical:
            rules.append("casefold_equivalent")
        if item.cleaned != raw:
            rules.append("unicode_or_whitespace_cleanup")
        if not rules:
            rules.append("exact" if raw == canonical else "canonical_surface_selection")
        entry = {
            "raw": raw,
            "canonical": canonical,
            "rules": list(dict.fromkeys(rules)),
            "changed": raw != canonical,
        }
        entries[raw] = entry
        grouped[canonical].append({
            "raw": raw,
            "count": counts[raw],
            "rules": entry["rules"],
        })

    groups = []
    for canonical in sorted(grouped, key=lambda value: (value.casefold(), value)):
        variants = sorted(
            grouped[canonical], key=lambda item: (-item["count"], item["raw"].casefold(), item["raw"]),
        )
        groups.append({
            "canonical": canonical,
            "count": sum(item["count"] for item in variants),
            "variants": variants,
        })
    audit = {
        "input_occurrences": sum(counts.values()),
        "input_variants": len(counts),
        "canonical_terms": len(groups),
        "changed_occurrences": sum(counts[raw] for raw, entry in entries.items() if entry["changed"]),
        "groups": groups,
        "ambiguous": ambiguous,
    }
    return CanonicalTermMap(entries=entries, audit=audit)


def _mode_is_categorical(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        mode = value.strip().casefold().replace("_", "-")
        if mode in {"categorical", "canonical", "canonicalize", "canonicalise"}:
            return True
        if mode in {"none", "off", "verbatim", "prose", "raw"}:
            return False
    return None


def claim_categorical_fields(config: Any) -> list[str]:
    fields = list(DEFAULT_CLAIM_CATEGORICAL_FIELDS)
    research = getattr(config, "research_config", config if isinstance(config, dict) else {})
    extraction = research.get("extraction", {}) if isinstance(research, dict) else {}
    definitions = extraction.get("fields", []) if isinstance(extraction, dict) else []
    if isinstance(definitions, list):
        for definition in definitions:
            if not isinstance(definition, dict):
                continue
            name = str(definition.get("name") or "").strip()
            mode = _mode_is_categorical(definition.get("normalization"))
            if not name or mode is None:
                continue
            if mode and name not in fields:
                fields.append(name)
            elif not mode and name in fields:
                fields.remove(name)
    for name in tuple(dict.fromkeys(fields)):
        override = _mode_is_categorical(_field_config(config, f"claims.{name}").get("mode"))
        if override is True and name not in fields:
            fields.append(name)
        elif override is False and name in fields:
            fields.remove(name)
    # normalization.fields may opt in custom fields not present in defaults.
    configured_fields = _normalization_section(config).get("fields", {}) or {}
    if isinstance(configured_fields, dict):
        for path, spec in configured_fields.items():
            if not str(path).startswith("claims."):
                continue
            name = str(path).split(".", 1)[1]
            value = spec.get("mode") if isinstance(spec, dict) else spec
            if _mode_is_categorical(value) is True and name not in fields:
                fields.append(name)
    return fields


def _claim_seed_vocabulary(config: Any, logical_field: str) \
        -> dict[str, dict[str, list[str]]]:
    """Use heuristic canonical keys as spellings, never their broad text regexes.

    Detection patterns such as ``autism`` -> ``autism spectrum disorder`` are
    appropriate when scanning prose, but are too semantic for label identity.
    Alias/pattern identity decisions belong in ``normalization.vocabularies``.
    """
    if config is None:
        return {}
    try:
        from process.domain_patterns import extraction_patterns

        patterns = extraction_patterns(config)
    except (AttributeError, TypeError):
        return {}
    if logical_field == "conditions":
        # Condition detection regexes and keys were authored for scanning prose
        # and may encode broader relationships (for example an abbreviation as
        # its own graph node). Explicit normalization vocabularies decide identity.
        return {}
    source = patterns.get(logical_field) or []
    if not isinstance(source, list):
        return {}
    return {term: {"aliases": [], "patterns": []} for term in source}


def _raw_list(container: Mapping[str, Any], field: str) -> list[str]:
    marker = container.get(TERM_NORMALIZATION_KEY)
    if isinstance(marker, dict):
        saved = (marker.get("fields") or {}).get(field)
        if isinstance(saved, list) and all(isinstance(item, dict) and "raw" in item for item in saved):
            return [str(item["raw"]) for item in saved if str(item.get("raw") or "").strip()]
    value = container.get(field)
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item or "").strip()]
    return [str(value)] if str(value).strip() else []


def _record_marker(
    existing: Any,
    fields: dict[str, list[dict[str, Any]]],
    *,
    config_hash: str,
) -> dict[str, Any]:
    marker = dict(existing) if isinstance(existing, dict) else {}
    previous_fields = marker.get("fields") if isinstance(marker.get("fields"), dict) else {}
    marker.update({
        "schema_version": NORMALIZATION_SCHEMA_VERSION,
        "policy_id": NORMALIZATION_POLICY_ID,
        "status": "canonical",
        "config_hash": config_hash,
        "fields": {**previous_fields, **fields},
    })
    return marker


def normalize_claim_records(
    records: list[dict[str, Any]], config: Any = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Canonicalize categorical Stage 3 fields before ``claims.json`` is saved."""
    config_hash = _config_hash(config)
    fields = claim_categorical_fields(config)
    logical = {field: "key_entities" if field == "brain_regions" else field for field in fields}
    raw_by_record: list[dict[str, list[str]]] = []
    values_by_logical: dict[str, list[str]] = defaultdict(list)
    for record in records:
        claims = record.get("claims") if isinstance(record.get("claims"), dict) else {}
        raw_fields: dict[str, list[str]] = {}
        for field in fields:
            if field not in claims and not (
                field == "brain_regions" and "brain_regions" in claims
            ):
                continue
            raw_fields[field] = _raw_list(claims, field)
            values_by_logical[logical[field]].extend(raw_fields[field])
        raw_by_record.append(raw_fields)

    maps: dict[str, CanonicalTermMap] = {}
    for logical_field, values in values_by_logical.items():
        path = f"claims.{logical_field}"
        vocabulary = _merge_vocabularies(
            _claim_seed_vocabulary(config, logical_field),
            _configured_vocabulary(config, path),
        )
        maps[logical_field] = build_canonical_term_map(values, vocabulary=vocabulary)

    normalized: list[dict[str, Any]] = []
    duplicates_removed = Counter()
    for record, raw_fields in zip(records, raw_by_record, strict=True):
        claims = dict(record.get("claims") or {})
        ledger: dict[str, list[dict[str, Any]]] = {}
        for field, raw_values in raw_fields.items():
            resolver = maps[logical[field]]
            entries = [resolver.resolve(raw) for raw in raw_values]
            canonical: list[str] = []
            seen: set[str] = set()
            for entry in entries:
                value = entry["canonical"]
                if value in seen:
                    duplicates_removed[logical[field]] += 1
                    continue
                seen.add(value)
                canonical.append(value)
            claims[field] = canonical
            ledger[field] = entries
        claims[TERM_NORMALIZATION_KEY] = _record_marker(
            claims.get(TERM_NORMALIZATION_KEY), ledger,
            config_hash=config_hash,
        )
        normalized.append({**record, "claims": claims})

    audit_fields = {}
    for logical_field, resolver in maps.items():
        audit_fields[logical_field] = {
            **resolver.audit,
            "within_record_duplicates_removed": duplicates_removed[logical_field],
        }
    audit = {
        "schema_version": NORMALIZATION_SCHEMA_VERSION,
        "policy_id": NORMALIZATION_POLICY_ID,
        "boundary": "stage_3_structured_claim_output",
        "config_hash": config_hash,
        "record_count": len(records),
        "categorical_fields": fields,
        "fields": audit_fields,
    }
    return normalized, audit


def inspect_claim_normalization(
    records: list[dict[str, Any]], config: Any = None,
) -> dict[str, Any]:
    """Validate a claim artifact without changing the records.

    Current Stage 3 artifacts carry a per-record policy marker.  For an older
    artifact, the same deterministic resolver is run only as a diagnostic.  A
    downstream stage can then refuse a stale artifact instead of silently
    rewriting it while reporting.
    """
    relevant = []
    marked = 0
    policy_marked = 0
    config_mismatches = 0
    expected_config_hash = _config_hash(config)
    fields = claim_categorical_fields(config)
    for record in records:
        claims = record.get("claims") if isinstance(record.get("claims"), dict) else {}
        if not any(_raw_list(claims, field) for field in fields if field in claims):
            continue
        relevant.append(record)
        marker = claims.get(TERM_NORMALIZATION_KEY)
        current_policy = (
            isinstance(marker, dict)
            and marker.get("schema_version") == NORMALIZATION_SCHEMA_VERSION
            and marker.get("policy_id") == NORMALIZATION_POLICY_ID
            and marker.get("status") == "canonical"
        )
        if current_policy:
            policy_marked += 1
            if marker.get("config_hash") == expected_config_hash:
                marked += 1
            else:
                config_mismatches += 1
    if not relevant:
        return {
            "status": "no_categorical_terms",
            "policy_id": NORMALIZATION_POLICY_ID,
            "expected_config_hash": expected_config_hash,
            "records_with_terms": 0,
            "marked_records": 0,
            "policy_marked_records": 0,
            "config_mismatches": 0,
            "issues": {},
        }
    if marked == len(relevant):
        return {
            "status": "canonical",
            "policy_id": NORMALIZATION_POLICY_ID,
            "expected_config_hash": expected_config_hash,
            "records_with_terms": len(relevant),
            "marked_records": marked,
            "policy_marked_records": policy_marked,
            "config_mismatches": 0,
            "issues": {},
        }
    # Diagnostic only: normalize copies and retain just the audit summary.
    _unused, audit = normalize_claim_records(records, config)
    issues = {
        field: {
            "input_variants": details.get("input_variants", 0),
            "canonical_terms": details.get("canonical_terms", 0),
            "changed_occurrences": details.get("changed_occurrences", 0),
            "within_record_duplicates_removed": details.get(
                "within_record_duplicates_removed", 0,
            ),
        }
        for field, details in audit.get("fields", {}).items()
        if details.get("changed_occurrences", 0)
        or details.get("within_record_duplicates_removed", 0)
    }
    if issues:
        status = "stale"
    elif config_mismatches:
        status = "configuration_mismatch"
    elif marked:
        status = "mixed_markers"
    else:
        status = "legacy_unversioned"
    return {
        "status": status,
        "policy_id": NORMALIZATION_POLICY_ID,
        "expected_config_hash": expected_config_hash,
        "records_with_terms": len(relevant),
        "marked_records": marked,
        "policy_marked_records": policy_marked,
        "config_mismatches": config_mismatches,
        "issues": issues,
    }


def statistical_categorical_fields(config: Any) -> list[str]:
    fields = list(DEFAULT_STATISTICAL_CATEGORICAL_FIELDS)
    configured_fields = _normalization_section(config).get("fields", {}) or {}
    if isinstance(configured_fields, dict):
        for path, spec in configured_fields.items():
            if not str(path).startswith("statistics."):
                continue
            name = str(path).split(".", 1)[1]
            value = spec.get("mode") if isinstance(spec, dict) else spec
            mode = _mode_is_categorical(value)
            if mode is True and name not in fields:
                fields.append(name)
            elif mode is False and name in fields:
                fields.remove(name)
    return fields


def _statistical_containers(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    containers: list[dict[str, Any]] = []
    seen: set[int] = set()
    for record in records:
        for item in [*(record.get("results") or []), *(record.get("human_review") or [])]:
            if not isinstance(item, dict):
                continue
            for container in (item, item.get("llm_context_review")):
                if isinstance(container, dict) and id(container) not in seen:
                    seen.add(id(container))
                    containers.append(container)
    return containers


def normalize_statistical_records(
    records: list[dict[str, Any]],
    config: Any = None,
    *,
    normalization_scope: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Canonicalize Stage 2c labels in place before extraction artifacts are saved.

    ``comparison`` remains oriented: this function never sorts or reverses its
    groups, so ``A vs B`` and ``B vs A`` stay scientifically distinct.  When a
    producer creates only a pending subset, ``normalization_scope`` may provide
    the complete corpus used to resolve corpus-dependent acronym/plural rules;
    only ``records`` are mutated.
    """
    config_hash = _config_hash(config)
    fields = statistical_categorical_fields(config)
    containers = _statistical_containers(records)
    scope_containers = _statistical_containers(
        normalization_scope if normalization_scope is not None else records
    )
    scope_ids = {id(container) for container in scope_containers}
    scope_containers.extend(
        container for container in containers if id(container) not in scope_ids
    )

    # The built-in meta-analysis descriptors have a scalar schema.  Reject a
    # malformed list before mutating any sibling container.  Custom opted-in
    # categorical fields preserve list cardinality below.
    for container in scope_containers:
        for field in DEFAULT_STATISTICAL_CATEGORICAL_FIELDS:
            if field in fields and isinstance(container.get(field), (list, tuple, set)):
                raise ValueError(
                    f"Statistical field {field!r} must be a scalar label, not a list"
                )

    values_by_field: dict[str, list[str]] = defaultdict(list)
    for container in scope_containers:
        for field in fields:
            if field not in container:
                continue
            values_by_field[field].extend(_raw_list(container, field))

    maps = {
        field: build_canonical_term_map(
            values,
            vocabulary=_configured_vocabulary(config, f"statistics.{field}"),
        )
        for field, values in values_by_field.items()
    }
    raw_by_container: list[dict[str, list[str]]] = []
    for container in containers:
        raw_by_container.append({
            field: _raw_list(container, field)
            for field in fields if field in container
        })
    duplicates_removed = Counter()
    for container, raw_fields in zip(containers, raw_by_container, strict=True):
        ledger: dict[str, list[dict[str, Any]]] = {}
        for field, values in raw_fields.items():
            entries = [maps[field].resolve(raw) for raw in values]
            if isinstance(container.get(field), (list, tuple, set)):
                canonical_values: list[str] = []
                seen: set[str] = set()
                for entry in entries:
                    canonical = entry["canonical"]
                    if canonical in seen:
                        duplicates_removed[field] += 1
                        continue
                    seen.add(canonical)
                    canonical_values.append(canonical)
                container[field] = canonical_values
            else:
                # Built-in descriptors and scalar custom labels stay scalar.
                # Never invent a value for a failed/no-verdict review.
                container[field] = entries[0]["canonical"] if entries else ""
            ledger[field] = entries
        if ledger:
            container[TERM_NORMALIZATION_KEY] = _record_marker(
                container.get(TERM_NORMALIZATION_KEY), ledger,
                config_hash=config_hash,
            )
    audit_fields = {
        field: {
            **resolver.audit,
            "within_record_duplicates_removed": duplicates_removed[field],
        }
        for field, resolver in maps.items()
    }
    return {
        "schema_version": NORMALIZATION_SCHEMA_VERSION,
        "policy_id": NORMALIZATION_POLICY_ID,
        "boundary": "stage_2c_statistical_label_output",
        "config_hash": config_hash,
        "record_count": len(records),
        "normalization_scope_record_count": len(
            normalization_scope if normalization_scope is not None else records
        ),
        "categorical_fields": fields,
        "fields": audit_fields,
    }


def inspect_statistical_normalization(
    metrics: list[dict[str, Any]], config: Any = None,
) -> dict[str, Any]:
    """Report whether meta-analysis inputs already carry Stage 2c canonical labels.

    This performs no canonicalization.  Unmarked labels remain exact and will
    therefore form separate conservative groups rather than being silently
    merged during meta-analysis.
    """
    labeled = 0
    marked = 0
    policy_marked = 0
    config_mismatches = 0
    expected_config_hash = _config_hash(config)
    for entry in metrics:
        tests = entry.get("statistical_tests") if isinstance(entry, dict) else None
        if not isinstance(tests, list):
            continue
        for test in tests:
            if not isinstance(test, dict):
                continue
            review = test.get("llm_context_review")
            sources = [test, review] if isinstance(review, dict) else [test]
            if not any(
                str(source.get(field) or "").strip()
                for source in sources
                for field in DEFAULT_STATISTICAL_CATEGORICAL_FIELDS
            ):
                continue
            labeled += 1
            marker = test.get(TERM_NORMALIZATION_KEY)
            if not isinstance(marker, dict) and isinstance(review, dict):
                marker = review.get(TERM_NORMALIZATION_KEY)
            current_policy = (
                isinstance(marker, dict)
                and marker.get("schema_version") == NORMALIZATION_SCHEMA_VERSION
                and marker.get("policy_id") == NORMALIZATION_POLICY_ID
                and marker.get("status") == "canonical"
            )
            if current_policy:
                policy_marked += 1
                if marker.get("config_hash") == expected_config_hash:
                    marked += 1
                else:
                    config_mismatches += 1
    if not labeled:
        status = "no_structured_labels"
    elif marked == labeled:
        status = "canonical"
    elif config_mismatches:
        status = "configuration_mismatch"
    elif marked:
        status = "mixed"
    else:
        status = "legacy_unversioned"
    return {
        "status": status,
        "policy_id": NORMALIZATION_POLICY_ID,
        "expected_config_hash": expected_config_hash,
        "labeled_statistics": labeled,
        "marked_statistics": marked,
        "policy_marked_statistics": policy_marked,
        "config_mismatches": config_mismatches,
        "unmarked_statistics": labeled - marked,
        "action": (
            "consume-exact-canonical-labels"
            if status == "canonical" else
            "consume-exact-labels-without-downstream-normalization"
        ),
        "reason": (
            "Stage 2c canonicalized these labels and preserved their raw mappings."
            if status == "canonical" else
            "Meta-analysis does not repair unversioned or configuration-mismatched "
            "labels; rerun Stage 2c to create canonical labels and provenance under "
            "the active project vocabulary before pooling."
        ),
    }


def normalization_audit_path(output_path: Any) -> Any:
    """Sibling sidecar name without changing a producer's established schema."""
    return output_path.with_name(f"{output_path.stem}_term_normalization.json")
