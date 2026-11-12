"""Helpers for applying persisted GUI settings to runtime config objects."""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from typing import Any


def migrate_legacy_settings(settings=None, legacy=None) -> None:
    """Copy preferences once after the Systes -> Sysys organization rename.

    Existing new-store values (including deliberate empty values) win. A
    completed migration is never repeated, so removed keys stay removed.
    """
    from PySide6.QtCore import QSettings

    if settings is None:
        settings = QSettings("Sysys", "HypothesisConsensusAnalyzer")
    if legacy is None:
        legacy = QSettings("Systes", "HypothesisConsensusAnalyzer")
    marker = "migration/systes_settings_v1"
    if settings.value(marker, False, type=bool):
        return
    remember_key = "keys/remember_keys"
    remember = settings.value(remember_key, legacy.value(remember_key, False, type=bool), type=bool)
    credentials = (
        set(API_KEY_QSETTINGS_KEYS.values()) | LEGACY_CREDENTIAL_QSETTINGS_KEYS
    ) - {"keys/pubmed_email"}
    for key in legacy.allKeys():
        if not settings.contains(key) and (key not in credentials or remember):
            settings.setValue(key, legacy.value(key))
    settings.setValue(marker, True)
    settings.sync()


API_KEY_QSETTINGS_KEYS: dict[str, str] = {
    "openai_api_key": "keys/openai_api_key",
    "custom_api_key": "keys/custom_api_key",
    "anthropic_api_key": "keys/anthropic_api_key",
    "s2_api_key": "keys/s2_api_key",
    "pubmed_api_key": "keys/pubmed_api_key",
    "pubmed_email": "keys/pubmed_email",
    "elicit_api_key": "keys/elicit_api_key",
    "scopus_api_key": "keys/scopus_api_key",
    "scopus_insttoken": "keys/scopus_insttoken",
    "wos_api_key": "keys/wos_api_key",
    "wos_researcher_api_key": "keys/wos_researcher_api_key",
}

LLM_QSETTINGS_KEYS: dict[str, str] = {
    "llm_provider": "llm/provider",
    "ollama_model": "llm/ollama_model",
    "ollama_base_url": "llm/ollama_url",
    "custom_api_format": "llm/custom_format",
    "custom_base_url": "llm/custom_url",
    "custom_model": "llm/custom_model",
    "openai_model": "llm/openai_model",
    "openai_base_url": "llm/openai_url",
    "anthropic_model": "llm/anthropic_model",
}

LLM_NUMERIC_QSETTINGS_KEYS: dict[str, str] = {
    "llm_context_tokens": "llm/context_tokens",
    "llm_max_output_tokens": "llm/max_output_tokens",
}

# Keys written before lmstudio/longcat were folded into the custom provider.
# They are read once through migrate_legacy_llm_fields and removed on save.
LEGACY_LLM_QSETTINGS_KEYS: dict[str, str] = {
    "lmstudio_model": "llm/lmstudio_model",
    "lmstudio_base_url": "llm/lmstudio_url",
    "lmstudio_api_key": "keys/lmstudio_api_key",
    "longcat_model": "llm/longcat_model",
    "longcat_base_url": "llm/longcat_url",
    "longcat_api_key": "keys/longcat_api_key",
}
LEGACY_CREDENTIAL_QSETTINGS_KEYS: set[str] = {
    qs_key for field, qs_key in LEGACY_LLM_QSETTINGS_KEYS.items()
    if field.endswith("_api_key")
}


def _saved_text(settings: Any, key: str) -> str:
    value = settings.value(key, "", type=str)
    return str(value or "").strip()


def restore_saved_settings(
    config: Any, settings: Any, *, keep_fields: Collection[str] = ()
) -> list[str]:
    """Layer saved QSettings values onto a Config-like object.

    API credentials keep the conservative env/config > QSettings priority, but
    LLM provider/model/base-url fields are user preferences and should override
    the non-empty dataclass defaults such as ``openai``. LLM fields named in
    ``keep_fields`` (values a loaded config file set explicitly) are left
    alone. Returns the LLM fields whose value a preference changed.
    """
    from core.config import migrate_legacy_llm_fields

    all_keys = {**API_KEY_QSETTINGS_KEYS, **LLM_QSETTINGS_KEYS, **LEGACY_LLM_QSETTINGS_KEYS}
    saved = {field: _saved_text(settings, qs_key) for field, qs_key in all_keys.items()}
    saved = migrate_legacy_llm_fields({k: v for k, v in saved.items() if v})

    for cfg_field in API_KEY_QSETTINGS_KEYS:
        if not hasattr(config, cfg_field) or not saved.get(cfg_field):
            continue
        current = getattr(config, cfg_field, "")
        if current and str(current).strip():
            continue
        setattr(config, cfg_field, saved[cfg_field])

    changed: list[str] = []
    for cfg_field in LLM_QSETTINGS_KEYS:
        if cfg_field in keep_fields or not hasattr(config, cfg_field) or not saved.get(cfg_field):
            continue
        if getattr(config, cfg_field) != saved[cfg_field]:
            changed.append(cfg_field)
        setattr(config, cfg_field, saved[cfg_field])
    for cfg_field, qs_key in LLM_NUMERIC_QSETTINGS_KEYS.items():
        if cfg_field in keep_fields or not hasattr(config, cfg_field):
            continue
        raw = settings.value(qs_key, 0)
        try:
            value = max(0, int(raw or 0))
        except (TypeError, ValueError):
            continue
        if value and getattr(config, cfg_field) != value:
            changed.append(cfg_field)
            setattr(config, cfg_field, value)
    return changed


def read_config_mapping(path: Path) -> dict:
    """Parse a YAML/JSON config file into its top-level mapping."""
    # core.config has no public reader yet; this is the loader load_config uses.
    from core.config import read_config_file

    return read_config_file(Path(path))


def explicit_config_fields(data: dict) -> frozenset[str]:
    """Config fields a config-file mapping sets, after legacy migration.

    Includes fields implied by research sections (``relevance.threshold``
    etc.) and ``run_name`` from ``name``. Values equal to the dataclass
    default count too, so loading them can reset a field (18-M8).
    """
    from core.config import (
        RESEARCH_SECTIONS,
        Config,
        fields_from_sections,
        migrate_legacy_llm_fields,
    )

    migrated = migrate_legacy_llm_fields(data)
    init_fields = {name for name, meta in Config.__dataclass_fields__.items() if meta.init}
    top = {key for key in migrated if key in init_fields and key != "project_root"}
    research = {k: v for k, v in migrated.items() if k in RESEARCH_SECTIONS}
    derived = set(fields_from_sections(research))
    if research.get("name"):
        derived.add("run_name")
    return frozenset(top | derived)
