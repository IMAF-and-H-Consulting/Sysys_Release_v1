"""Project and Settings persistence for domain vocabulary and heuristic rules."""
from copy import deepcopy

_SECTIONS = ("defaults", "graph_analysis", "extraction", "method_review")


def domain_config_snapshot(config) -> dict:
    research = getattr(config, "research_config", {}) or {}
    return {
        **{key: deepcopy(research.get(key, {})) for key in _SECTIONS},
        "co_occurring_conditions": deepcopy(getattr(config, "co_occurring_conditions", [])),
    }


def restore_domain_config(config, snapshot) -> None:
    if not isinstance(snapshot, dict) or not snapshot:
        return  # Older saved projects continue using their research config file.
    research = deepcopy(getattr(config, "research_config", {}) or {})
    for key in _SECTIONS:
        if key in snapshot:
            research[key] = deepcopy(snapshot[key])
    config._research_config = research
    if "co_occurring_conditions" in snapshot:
        config.co_occurring_conditions = deepcopy(snapshot["co_occurring_conditions"])
