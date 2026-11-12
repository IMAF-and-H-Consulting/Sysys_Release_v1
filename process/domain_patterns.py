"""Shared, validated vocabulary for configurable heuristic analyses."""
from __future__ import annotations

import re
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import yaml


def research_config(config) -> dict:
    value = getattr(config, "research_config", {})
    return value if isinstance(value, dict) else {}


def mapping(value, path: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping")
    return value


def terms(value, path: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(term, str) or not term.strip() for term in value
    ):
        raise ValueError(f"{path} must be a list of non-empty strings")
    return list(dict.fromkeys(term.strip() for term in value))


def literal_pattern(term: str) -> str:
    """Match a literal phrase, including punctuation, without substring hits.

    Any run of whitespace in the phrase matches any run in the text, so
    multi-word terms still match across line breaks and double spaces.
    """
    body = r"\s+".join(re.escape(word) for word in term.split())
    return rf"(?<!\w){body}(?!\w)"


def paper_text(paper: dict, key: str) -> str:
    """A paper's text field as a string; a missing or null value is empty."""
    value = paper.get(key)
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def contains_term(text: str, term: str) -> bool:
    return re.search(literal_pattern(term), text, re.IGNORECASE) is not None


def pattern_map(value, path: str) -> dict[str, list[str]]:
    result = {}
    for entity, regexes in mapping(value, path).items():
        if not isinstance(entity, str) or not entity.strip():
            raise ValueError(f"{path} entity names must be non-empty strings")
        terms(regexes, f"{path}.{entity}")  # Validate without altering regex whitespace.
        result[entity] = list(dict.fromkeys(regexes))
        for regex in result[entity]:
            try:
                re.compile(regex, re.IGNORECASE)
            except re.error as exc:
                raise ValueError(f"Invalid regex in {path}.{entity}: {exc}") from exc
    return result


@lru_cache(maxsize=2)
def _read_pack(name: str) -> dict:
    if name not in {"general", "neuroscience"}:
        raise ValueError("defaults.pattern_pack must be 'general' or 'neuroscience'")
    path = Path(__file__).resolve().parents[1] / "templates" / "domain_patterns" / f"{name}.yaml"
    pack = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(pack, dict):
        raise ValueError(f"pattern pack '{name}' must be a mapping ({path})")
    return pack


def load_pack(config) -> dict:
    defaults = mapping(research_config(config).get("defaults", {}), "defaults")
    name = defaults.get("pattern_pack", "general")
    if not isinstance(name, str):
        raise ValueError("defaults.pattern_pack must be a string")
    base = deepcopy(_read_pack("general"))
    if name != "general":
        for section, values in _read_pack(name).items():
            if isinstance(values, dict):
                base.setdefault(section, {}).update(deepcopy(values))
            elif isinstance(values, list) and isinstance(base.get(section), list):
                # Pack rules go first: method rules are first-match-wins.
                base[section] = deepcopy(values) + base[section]
            else:
                base[section] = deepcopy(values)
    return base


def condition_patterns(config) -> dict[str, list[str]]:
    graph = mapping(research_config(config).get("graph_analysis", {}), "graph_analysis")
    if "conditions" in graph:
        return pattern_map(graph["conditions"], "graph_analysis.conditions")
    conditions = terms(getattr(config, "co_occurring_conditions", []) or [],
                       "co_occurring_conditions")
    if conditions:
        return {name: [literal_pattern(name)] for name in conditions}
    return pattern_map(load_pack(config).get("graph_analysis", {}).get("conditions", {}),
                       "graph_analysis.conditions")


def extraction_patterns(config) -> dict:
    research = research_config(config)
    extraction = mapping(research.get("extraction", {}), "extraction")
    overrides = mapping(extraction.get("heuristics", {}), "extraction.heuristics")
    pack = load_pack(config)
    result = pack.get("extraction", {})
    # Graph/robustness conditions supplement the extraction pack; they do not
    # erase its primary-domain vocabulary.  A project that wants a replacement
    # (including an explicit empty set) uses extraction.heuristics.conditions.
    # Previously, merely configuring co-occurring graph conditions removed the
    # primary condition from Stage 3's heuristic path.
    graph = mapping(research.get("graph_analysis", {}), "graph_analysis")
    graph_conditions = (
        mapping(graph["conditions"], "graph_analysis.conditions")
        if "conditions" in graph else None
    )
    if "conditions" not in overrides:
        if graph_conditions == {}:
            # An explicit empty graph category is the established opt-out.
            result["conditions"] = {}
        elif graph_conditions is not None or getattr(config, "co_occurring_conditions", None):
            combined = deepcopy(result.get("conditions", {}))
            combined.update(condition_patterns(config))
            result["conditions"] = combined
    result.update(overrides)
    # "brain_regions" is the pre-rename key; packs and older configs still use it.
    if "brain_regions" in result:
        result.setdefault("key_entities", result.pop("brain_regions"))
    result["conditions"] = pattern_map(result.get("conditions", {}),
                                       "extraction.heuristics.conditions")
    for key in ("mechanisms", "key_entities", "methodology_hints", "finding_hints",
                "criticality_terms", "network_terms"):
        result[key] = terms(result.get(key, []), f"extraction.heuristics.{key}")
    return result


def method_rules(config) -> list[dict]:
    from process.method_review import ALLOWED_METHOD_TYPES

    section = mapping(research_config(config).get("method_review", {}), "method_review")
    if "heuristic_rules" in section:
        rules = section["heuristic_rules"]
    else:
        rules = load_pack(config)["method_review"]
    if not isinstance(rules, list):
        raise ValueError("method_review.heuristic_rules must be a list")
    result = []
    for i, rule in enumerate(rules):
        path = f"method_review.heuristic_rules[{i}]"
        rule = mapping(rule, path)
        kind = rule.get("study_method_type")
        if not isinstance(kind, str) or kind not in ALLOWED_METHOD_TYPES:
            raise ValueError(f"{path}.study_method_type must be a supported method type")
        confidence = rule.get("confidence")
        if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                or not 0 <= confidence <= 1):
            raise ValueError(f"{path}.confidence must be a number between 0 and 1")
        any_terms = terms(rule.get("any_terms", []), f"{path}.any_terms")
        all_terms = terms(rule.get("all_terms", []), f"{path}.all_terms")
        if not any_terms and not all_terms:
            raise ValueError(f"{path} must specify any_terms or all_terms")
        result.append({"study_method_type": kind, "confidence": confidence,
                       "any_terms": any_terms, "all_terms": all_terms})
    return result
