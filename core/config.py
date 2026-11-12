"""
Pipeline configuration dataclass.

Full runtime configuration for the hypothesis consensus analyzer pipeline.
"""
from __future__ import annotations

import copy
import json
import logging
import os
from dataclasses import MISSING, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

DEFAULT_HAIKU_MODEL = "claude-3-haiku-20240307"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CUSTOM_BASE_URL = "http://localhost:1234/v1"
DEFAULT_CUSTOM_API_FORMAT = "openai"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = DEFAULT_HAIKU_MODEL
DEFAULT_CONCURRENT_WORKERS = 3
DEFAULT_STAGGER_DELAY = 0.5
DEFAULT_CHECKPOINT_EVERY = 1
DEFAULT_WORKER_TIMEOUT_SECONDS = 120.0

# Maps provider name → Config attribute that holds its model string.
# Used in llm_model and anywhere a provider→model field lookup is needed.
PROVIDER_MODEL_FIELD: dict[str, str] = {
    "ollama": "ollama_model",
    "custom": "custom_model",
    "openai": "openai_model",
    "anthropic": "anthropic_model",
}
SUPPORTED_LLM_PROVIDERS: set[str] = set(PROVIDER_MODEL_FIELD.keys())

# Wire protocols a "custom" endpoint can speak. LM Studio serves all three;
# hosted gateways usually speak one of the first two.
CUSTOM_API_FORMATS: dict[str, str] = {
    "openai": "OpenAI-compatible",
    "anthropic": "Anthropic-compatible",
    "lmstudio": "LM Studio REST",
}

# Named providers that were folded into "custom", with the API format each
# one spoke. Old configs and saved preferences migrate through this table.
LEGACY_CUSTOM_PROVIDERS: dict[str, str] = {
    "lmstudio": "lmstudio",
    "longcat": "openai",
}
_CUSTOM_ENDPOINT_SUFFIXES = ("base_url", "model", "api_key")
_LLM_WORKLOADS = ("abstract", "fulltext")


def _legacy_field_targets() -> dict[str, str]:
    """Every legacy lmstudio_*/longcat_* key -> the custom_* key it migrates to."""
    targets: dict[str, str] = {}
    for name in LEGACY_CUSTOM_PROVIDERS:
        for sfx in _CUSTOM_ENDPOINT_SUFFIXES:
            targets[f"{name}_{sfx}"] = f"custom_{sfx}"
        for workload in _LLM_WORKLOADS:
            targets[f"{workload}_{name}_model"] = f"{workload}_custom_model"
    return targets


_LEGACY_FIELD_TARGETS: dict[str, str] = _legacy_field_targets()

# LONGCAT_API_KEY is only ever sent to this host (over HTTPS); any other
# custom endpoint needs CUSTOM_LLM_API_KEY or an explicit key.
LONGCAT_API_HOST = "longcat.chat"

# Config-file environment variables for API keys, by Config field.
# custom_api_key is resolved separately (see _custom_api_key_from_env).
ENV_API_KEYS: dict[str, str] = {
    "s2_api_key": "S2_API_KEY",
    "pubmed_api_key": "PUBMED_API_KEY",
    "pubmed_email": "PUBMED_EMAIL",
    "openai_api_key": "OPENAI_API_KEY",
    "anthropic_api_key": "ANTHROPIC_API_KEY",
    "elicit_api_key": "ELICIT_API_KEY",
    "scopus_api_key": "SCOPUS_API_KEY",
    "wos_researcher_api_key": "WOS_RESEARCHER_API_KEY",
    "wos_api_key": "WOS_API_KEY",
}

# Top-level config-file sections stored in Config.research_config
# (research-domain data, not Config dataclass fields). Shared with every
# loader so none of them drops a section.
RESEARCH_SECTIONS: frozenset[str] = frozenset({
    "search", "relevance", "extraction", "synthesis",
    "method_review", "audit",
    "natural_language_question", "name", "description",
    "query_families", "defaults", "cheap_triage", "full_text",
    "axes", "overlap_rules", "tier_rules", "evidence_types",
    "stopping", "seeds", "dive", "graph", "graph_analysis",
    "llm_scoring", "committee_brief", "abct_tech_ai_infosheet",
    "robustness", "normalization",
})

_TRUE_WORDS = frozenset({"true", "yes", "on", "1"})
_FALSE_WORDS = frozenset({"false", "no", "off", "0"})

# Cross-encoder raters. These score papers directly from an embedding
# checkpoint instead of going through the LLM chat stack, so their "model"
# string is a checkpoint path and none of PROVIDER_MODEL_FIELD applies.
EMBEDDING_RATERS: frozenset[str] = frozenset(
    {"brt_bert", "brt-bert", "brt_irr", "brt-irr", "laya-bert", "bert-laya", "laya-only", "packaged-bert"}
)


DEFAULT_OA_LOOKUP_WORKERS = 4
DEFAULT_FULLTEXT_DOWNLOAD_WORKERS = 3
DEFAULT_AUDIT_SIMILARITY_THRESHOLD = 0.85
DEFAULT_CITATION_SNOWBALL_SEED_LIMIT = 12
DEFAULT_CITATION_SNOWBALL_RELATION_LIMIT = 20
# ``0`` means automatic throughout the LLM budget fields.  The loaded model's
# context is detected when the endpoint exposes it (LM Studio native REST), or
# supplied once by the user for providers that do not report it.  Positive
# response values are deliberate user overrides and are recorded as synthetic
# limits by the shared budget planner.
DEFAULT_LLM_CONTEXT_TOKENS = 0
# Independent provider/model completion ceiling. Zero means the endpoint did
# not report one and the user has not supplied one; it never means zero output.
DEFAULT_LLM_MAX_OUTPUT_TOKENS = 0
DEFAULT_SCORE_USE_THINKING = True
DEFAULT_SCORE_MAX_TOKENS = 0
DEFAULT_SCORE_BATCH_MAX_TOKENS = 0
DEFAULT_SCORE_ABSTRACT_CHARS = 0
DEFAULT_SCORE_FULLTEXT_CHARS = 0
DEFAULT_CLAIM_USE_THINKING = False
DEFAULT_CLAIM_MAX_TOKENS = 0
DEFAULT_SYNTHESIS_USE_THINKING = False
DEFAULT_SYNTHESIS_CLUSTER_MAX_TOKENS = 0
DEFAULT_SYNTHESIS_ABSTRACT_MAX_TOKENS = 0

LEGACY_PROVIDER_ALIASES: dict[str, str] = {
    name: "custom" for name in LEGACY_CUSTOM_PROVIDERS
}


def normalize_provider_name(provider: str) -> str:
    lower = provider.lower().strip()
    return LEGACY_PROVIDER_ALIASES.get(lower, lower)


def validate_llm_provider(provider: str) -> str:
    """Return a normalized provider name, raising for unsupported providers."""
    prov = normalize_provider_name(provider)
    if prov not in SUPPORTED_LLM_PROVIDERS:
        raise ValueError(
            f"Unsupported LLM provider: {provider!r}. "
            f"Supported providers are: {sorted(SUPPORTED_LLM_PROVIDERS)}"
        )
    return prov


def validate_custom_api_format(api_format: str) -> str:
    fmt = (api_format or DEFAULT_CUSTOM_API_FORMAT).lower().strip()
    if fmt not in CUSTOM_API_FORMATS:
        raise ValueError(
            f"Unsupported custom API format: {api_format!r}. "
            f"Supported formats are: {sorted(CUSTOM_API_FORMATS)}"
        )
    return fmt


def _legacy_source(data: dict) -> tuple[str | None, bool]:
    """The legacy provider to migrate from, and whether it is the active one."""
    provider = str(data.get("llm_provider") or "").lower().strip()
    if provider in LEGACY_CUSTOM_PROVIDERS:
        return provider, True
    for name in LEGACY_CUSTOM_PROVIDERS:
        if any(data.get(f"{name}_{sfx}") for sfx in _CUSTOM_ENDPOINT_SUFFIXES):
            return name, False
    return None, False


def _custom_endpoint_updates(data: dict) -> dict:
    """custom_* values to take from a legacy provider; never overwrites one.

    The active legacy provider fills each missing custom_* field and, unless
    one is set, the API format. An inactive one is used only when no
    custom_* value exists, so two endpoints are never mixed.
    """
    source, is_active = _legacy_source(data)
    has_custom = any(data.get(f"custom_{sfx}") for sfx in _CUSTOM_ENDPOINT_SUFFIXES)
    if source is None or (has_custom and not is_active):
        return {}
    updates = {
        f"custom_{sfx}": data[f"{source}_{sfx}"]
        for sfx in _CUSTOM_ENDPOINT_SUFFIXES
        if data.get(f"{source}_{sfx}") and not data.get(f"custom_{sfx}")
    }
    if not data.get("custom_api_format"):
        updates["custom_api_format"] = LEGACY_CUSTOM_PROVIDERS[source]
    return updates


def _workload_model_updates(data: dict) -> dict:
    """{workload}_custom_model from the first legacy workload model, if unset."""
    updates: dict = {}
    for legacy, target in _LEGACY_FIELD_TARGETS.items():
        if not legacy.endswith("_model") or target == "custom_model":
            continue
        if data.get(legacy) and not data.get(target) and target not in updates:
            updates[target] = data[legacy]
    return updates


def _log_dropped_legacy_fields(data: dict, out: dict) -> None:
    """Name (never the values of) legacy keys whose value was not carried over."""
    dropped = sorted(
        legacy for legacy, target in _LEGACY_FIELD_TARGETS.items()
        if data.get(legacy) and out.get(target) != data[legacy]
    )
    if dropped:
        logger.warning(
            "Legacy LLM settings dropped during migration because custom_* values "
            "already exist: %s", ", ".join(dropped))


def migrate_legacy_llm_fields(data: dict) -> dict:
    """Return a copy of ``data`` with lmstudio_*/longcat_* keys folded into custom_*.

    The legacy provider named by ``llm_provider`` wins; otherwise the first
    legacy provider with any saved value is used. Explicit custom_* values
    are never overwritten: the active legacy provider fills only missing
    fields. Legacy keys are dropped from the result, and any whose value was
    not carried over is named in a WARNING.
    """
    out = {k: v for k, v in data.items() if k not in _LEGACY_FIELD_TARGETS}
    out = {**out, **_custom_endpoint_updates(data)}
    out = {**out, **_workload_model_updates(data)}
    for key in ("llm_provider", "abstract_llm_provider", "fulltext_llm_provider"):
        if out.get(key):
            out[key] = normalize_provider_name(str(out[key]))
    _log_dropped_legacy_fields(data, out)
    return out


def _is_longcat_url(url: str) -> bool:
    parsed = urlparse(str(url or "").strip())
    host = (parsed.hostname or "").lower()
    is_longcat = host == LONGCAT_API_HOST or host.endswith(f".{LONGCAT_API_HOST}")
    return parsed.scheme == "https" and is_longcat


def _custom_api_key_from_env(base_url: str) -> str:
    """CUSTOM_LLM_API_KEY for any endpoint; LONGCAT_API_KEY only for LongCat.

    LONGCAT_API_KEY used to be sent to whatever custom endpoint was
    configured (a gateway, or LM Studio over plain-HTTP LAN).
    """
    key = os.environ.get("CUSTOM_LLM_API_KEY", "")
    if key:
        return key
    return os.environ.get("LONGCAT_API_KEY", "") if _is_longcat_url(base_url) else ""


def _parse_bool(text: str) -> bool:
    lowered = text.lower()
    if lowered in _TRUE_WORDS:
        return True
    if lowered in _FALSE_WORDS:
        return False
    raise ValueError(f"not a boolean: {text!r}")


def _parse_int(text: str) -> int:
    number = float(text)
    if not number.is_integer():
        raise ValueError(f"not an integer: {text!r}")
    return int(number)


_STRING_PARSERS: tuple[tuple[type, Any], ...] = (
    (bool, _parse_bool),  # before int: bool is an int subclass
    (int, _parse_int),
    (float, float),
)


def _coerce_field_value(name: str, value: Any, default: Any) -> Any:
    """Parse a quoted number/boolean (``'4'``, ``'false'``) for a typed field.

    Only strings are converted; a string that does not parse raises a
    ValueError naming the field instead of failing later inside a stage.
    """
    if not isinstance(value, str):
        return value
    for kind, parse in _STRING_PARSERS:
        if isinstance(default, kind):
            try:
                return parse(value.strip())
            except ValueError as exc:
                raise ValueError(
                    f"Config field {name!r} expects {kind.__name__}, got {value!r}"
                ) from exc
    return value


def _safe_run_name(name: str) -> str:
    """Sanitize a run name for use as a filesystem directory name."""
    import re
    clean = str(name or "").strip()
    clean = re.sub(r"[^\w\-. ]+", "_", clean)
    clean = re.sub(r"_+", "_", clean).strip("_. ")
    return clean


@dataclass
class Config:
    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    run_name: str = ""
    data_dir: Path = field(init=False)
    raw_dir: Path = field(init=False)
    corpus_dir: Path = field(init=False)
    claims_dir: Path = field(init=False)
    clusters_dir: Path = field(init=False)
    output_dir: Path = field(init=False)

    # API Keys
    s2_api_key: str = ""
    openalex_api_key: str = ""
    pubmed_api_key: str = ""
    pubmed_email: str = ""
    pubmed_tool: str = "hypothesis-consensus-analyzer"
    wos_api_key: str = ""
    wos_researcher_api_key: str = ""
    scopus_api_key: str = ""
    scopus_insttoken: str = ""
    elicit_api_key: str = ""
    openai_api_key: str = ""
    custom_api_key: str = ""
    anthropic_api_key: str = ""

    # LLM Provider Settings
    llm_provider: str = "openai"
    ollama_model: str = "qwen2.5:32b-instruct-q4_K_M"
    ollama_fallback_model: str = "llama3:8b-instruct-q8_0"
    ollama_base_url: str = "http://localhost:11434"
    custom_api_format: str = DEFAULT_CUSTOM_API_FORMAT
    custom_base_url: str = DEFAULT_CUSTOM_BASE_URL
    custom_model: str = ""
    openai_base_url: str = DEFAULT_OPENAI_BASE_URL
    openai_model: str = DEFAULT_OPENAI_MODEL
    anthropic_model: str = DEFAULT_ANTHROPIC_MODEL

    # Per-workload overrides
    abstract_llm_provider: str = ""
    abstract_ollama_model: str = ""
    abstract_custom_model: str = ""
    abstract_openai_model: str = ""
    abstract_anthropic_model: str = ""
    fulltext_llm_provider: str = ""
    fulltext_ollama_model: str = ""
    fulltext_custom_model: str = ""
    fulltext_openai_model: str = ""
    fulltext_anthropic_model: str = ""

    # Processing
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    relevance_model: str = "brt-bert"
    # Combined screening is the default; explicit legacy/comparison modes still load.
    relevance_screening_mode: str = "laya-bert"
    laya_screener_dir: str = ""  # blank resolves the shipped pw4 package beside main.py
    laya_device: str = "auto"  # CUDA when available, otherwise CPU
    laya_batch_size: int = 8
    laya_cut_bin: int = 4
    laya_brt_threshold: float = 0.236
    relevance_threshold: int = 3
    min_cluster_size: int = 5
    score_batch_size: int = 1
    # Stage 5: semantic similarity at which a flagged span is reclassified valid
    audit_similarity_threshold: float = DEFAULT_AUDIT_SIMILARITY_THRESHOLD
    # Stage 5: remove papers with no grounded span (default: flag them only)
    audit_exclude_ungrounded: bool = False
    # Meta-analysis: a result with no CI/SE/exact p/own n may borrow the paper's
    # single reported N. Off by default; the researcher decides (see statistics audit).
    meta_analysis_use_paper_sample_size: bool = False

    # Concurrency
    concurrent_workers: int = DEFAULT_CONCURRENT_WORKERS
    stagger_delay: float = DEFAULT_STAGGER_DELAY
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY
    worker_timeout_seconds: float = DEFAULT_WORKER_TIMEOUT_SECONDS
    oa_lookup_workers: int = DEFAULT_OA_LOOKUP_WORKERS
    fulltext_download_workers: int = DEFAULT_FULLTEXT_DOWNLOAD_WORKERS

    # Memory / full-text handling
    fulltext_streaming: bool = True
    fulltext_download_batch_size: int = 50
    paperqa_index_enabled: bool = True
    paperqa_index_batch_size: int = 100

    # Citation Snowballing
    citation_snowball_enabled: bool = True
    citation_snowball_seed_limit: int = DEFAULT_CITATION_SNOWBALL_SEED_LIMIT
    citation_snowball_relation_limit: int = DEFAULT_CITATION_SNOWBALL_RELATION_LIMIT

    # LLM Budget / Quality
    # 0 = auto-detect/use the context-aware policy.  A positive value is an
    # explicit override, never an undocumented application ceiling.
    llm_context_tokens: int = DEFAULT_LLM_CONTEXT_TOKENS
    llm_max_output_tokens: int = DEFAULT_LLM_MAX_OUTPUT_TOKENS
    score_use_thinking: bool = DEFAULT_SCORE_USE_THINKING
    score_max_tokens: int = DEFAULT_SCORE_MAX_TOKENS
    score_batch_max_tokens: int = DEFAULT_SCORE_BATCH_MAX_TOKENS
    score_abstract_chars: int = DEFAULT_SCORE_ABSTRACT_CHARS
    score_fulltext_chars: int = DEFAULT_SCORE_FULLTEXT_CHARS
    claim_use_thinking: bool = DEFAULT_CLAIM_USE_THINKING
    claim_max_tokens: int = DEFAULT_CLAIM_MAX_TOKENS
    synthesis_use_thinking: bool = DEFAULT_SYNTHESIS_USE_THINKING
    synthesis_cluster_max_tokens: int = DEFAULT_SYNTHESIS_CLUSTER_MAX_TOKENS
    synthesis_abstract_max_tokens: int = DEFAULT_SYNTHESIS_ABSTRACT_MAX_TOKENS

    # LLM response cache (prompt-hash → response text, under output_dir)
    llm_cache_enabled: bool = True
    llm_cache_dir: Path | None = None  # default: <output_dir>/llm_cache

    # Research / Hypothesis
    base_term_s2: str = ""
    base_term_pubmed: str = ""
    hypothesis_text: str = (
        "What does the current scholarly evidence say about the user's "
        "research question?"
    )
    co_occurring_conditions: list = field(default_factory=list)
    alignment_dimension: str = "Evidence alignment to the research question"
    _research_config: dict = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self):
        self.llm_provider = validate_llm_provider(self.llm_provider)
        self.custom_api_format = validate_custom_api_format(self.custom_api_format)
        self.llm_context_tokens = max(0, int(self.llm_context_tokens or 0))
        self.llm_max_output_tokens = max(0, int(self.llm_max_output_tokens or 0))
        self._resolve_paths()

        self._resolve_env_keys()

    def _resolve_env_keys(self) -> None:
        """Fill empty API key fields from the environment."""
        for attr, env_name in ENV_API_KEYS.items():
            if not getattr(self, attr):
                setattr(self, attr, os.environ.get(env_name, ""))
        if not self.custom_api_key:
            self.custom_api_key = _custom_api_key_from_env(self.custom_base_url)

    def _resolve_paths(self) -> None:
        """(Re-)compute all directory paths based on project_root and run_name."""
        safe = _safe_run_name(self.run_name)
        if safe:
            self.data_dir = self.project_root / "data" / "runs" / safe
            self.output_dir = self.project_root / "output" / "runs" / safe
        else:
            self.data_dir = self.project_root / "data"
            self.output_dir = self.project_root / "output"
        self.raw_dir = self.data_dir / "raw"
        self.corpus_dir = self.data_dir / "corpus"
        self.claims_dir = self.data_dir / "claims"
        self.clusters_dir = self.data_dir / "clusters"

    @property
    def llm_model(self) -> str:
        field = PROVIDER_MODEL_FIELD.get(self.llm_provider, "openai_model")
        return getattr(self, field)

    def llm_provider_for(self, workload: str = "abstract") -> str:
        override = getattr(self, f"{workload}_llm_provider", "")
        return validate_llm_provider(override) if override else self.llm_provider

    def llm_model_for(self, workload: str = "abstract") -> str:
        prov = self.llm_provider_for(workload)
        override = getattr(self, f"{workload}_{prov}_model", "")
        if override:
            return override
        return getattr(self, PROVIDER_MODEL_FIELD[prov])

    def runtime_config(self, workload: str = "abstract") -> Config:
        cfg = copy.deepcopy(self)
        cfg.llm_provider = self.llm_provider_for(workload)
        model = self.llm_model_for(workload)
        setattr(cfg, f"{cfg.llm_provider}_model", model)
        return cfg

    def ensure_dirs(self) -> None:
        for d in [self.raw_dir, self.corpus_dir, self.claims_dir,
                  self.clusters_dir, self.output_dir]:
            d.mkdir(parents=True, exist_ok=True)

    @property
    def research_config(self) -> dict:
        return self._research_config

    def get_search_config(self) -> dict:
        return self._research_config.get("search", {})

    def get_relevance_config(self) -> dict:
        return self._research_config.get("relevance", {})

    def get_synthesis_config(self) -> dict:
        return self._research_config.get("synthesis", {})

    def get_method_review_config(self) -> dict:
        return self._research_config.get("method_review", {})

    def get_natural_language_question(self) -> str:
        nlq = self._research_config.get("natural_language_question", "")
        if not nlq:
            nlq = self.get_relevance_config().get("research_question", "")
        return (nlq or "").strip()

    def get_full_text_config(self) -> dict:
        return self._research_config.get("full_text", {})

    def get_llm_scoring_config(self) -> dict:
        return self._research_config.get("llm_scoring", {})

    @classmethod
    def load_config(
        cls,
        config_path: Path | str | None = None,
        *,
        project_root: Path | str | None = None,
    ) -> Config:
        root = (
            Path(project_root).expanduser().resolve()
            if project_root else Path(__file__).parent.parent
        )
        if config_path is None:
            config_path = _find_config_file(root)
        if config_path is None:
            return cls(project_root=root)

        config_path = Path(config_path)
        data = _read_config_file(config_path)
        if project_root is None and data.get("project_root"):
            root = _configured_root(data["project_root"], config_path)
        return cls.from_mapping(data, project_root=root)

    @classmethod
    def from_mapping(cls, data: dict, *, project_root: Path | str) -> Config:
        """Build a Config from a parsed config file's top-level mapping.

        Applies legacy LLM migration, provider/format validation, typed
        field coercion and the section -> field mappings. ``RESEARCH_SECTIONS``
        go to ``research_config``; derived paths (``data_dir`` …) and unknown
        keys are ignored. Every loader (YAML, JSON, GUI) should come through
        here.
        """
        if not isinstance(data, dict):
            raise ValueError(
                f"Config must be a mapping at the top level, got {type(data).__name__}")
        cfg_fields, research = cls._split_mapping(migrate_legacy_llm_fields(data))
        cfg_fields = {**_fields_from_sections(research), **cfg_fields}
        if "run_name" not in cfg_fields and research.get("name"):
            cfg_fields["run_name"] = str(research["name"]).strip()
        cfg = cls(project_root=Path(project_root), **cls._coerce_fields(cfg_fields))
        cfg._research_config = research
        return cfg

    @classmethod
    def _split_mapping(cls, data: dict) -> tuple[dict, dict]:
        """(init-field values, research sections) from a config mapping."""
        init_fields = {name for name, meta in cls.__dataclass_fields__.items() if meta.init}
        cfg_fields, research = {}, {}
        for key, val in data.items():
            if key == "project_root":
                continue
            if key in RESEARCH_SECTIONS:
                research[key] = val
            elif key in init_fields:
                cfg_fields[key] = val
        return cfg_fields, research

    @classmethod
    def _coerce_fields(cls, cfg_fields: dict) -> dict:
        meta = cls.__dataclass_fields__
        return {
            name: _coerce_field_value(name, value, meta[name].default)
            if meta[name].default is not MISSING else value
            for name, value in cfg_fields.items()
        }

    def __repr__(self) -> str:
        run = f", run={self.run_name!r}" if self.run_name else ""
        return f"Config(provider={self.llm_provider!r}, model={self.llm_model!r}, project_root={self.project_root!r}{run})"


def _find_config_file(root: Path) -> Path | None:
    for name in ("research_config.yaml", "research_config.yml", "research_config.json"):
        path = root / name
        if path.exists():
            return path
    return None


def _read_config_file(config_path: Path) -> dict:
    """Parse a YAML/JSON config file; its top level must be a mapping."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    suffix = config_path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required for YAML config files") from exc
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    elif suffix == ".json":
        with open(config_path, encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {suffix}")
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Config file {config_path} must contain a mapping at the top level, "
            f"got {type(data).__name__}")
    return data


def _configured_root(value: Any, config_path: Path) -> Path:
    configured = Path(str(value)).expanduser()
    if configured.is_absolute():
        return configured.resolve()
    return (config_path.parent / configured).resolve()


def _fields_from_sections(research: dict) -> dict:
    """Config fields implied by research sections; explicit top-level keys win.

    ``defaults.per_source_limit`` is deliberately not mapped: it used to set
    ``citation_snowball_seed_limit`` (up to 500 seeds, each fetching its
    citations and references). Set ``citation_snowball_seed_limit`` directly.
    """
    fields: dict = {}
    llm_sc = research.get("llm_scoring") or {}
    if llm_sc.get("abstract_max_chars"):
        fields["score_abstract_chars"] = llm_sc["abstract_max_chars"]
    if llm_sc.get("fulltext_max_chars"):
        fields["score_fulltext_chars"] = llm_sc["fulltext_max_chars"]
    if llm_sc.get("max_output_tokens"):
        fields["score_max_tokens"] = llm_sc["max_output_tokens"]
    if "use_reasoning" in llm_sc:
        fields["score_use_thinking"] = bool(llm_sc["use_reasoning"])
    rel = research.get("relevance") or {}
    if rel.get("threshold") is not None:
        fields["relevance_threshold"] = rel["threshold"]
    syn = research.get("synthesis") or {}
    if syn.get("embedding_model"):
        fields["embedding_model"] = syn["embedding_model"]
    return fields


# Public names for callers outside core (GUI settings load/merge and per-stage
# endpoint keys); the underscored originals stay for existing internal callers.
read_config_file = _read_config_file
fields_from_sections = _fields_from_sections
is_longcat_url = _is_longcat_url
custom_api_key_from_env = _custom_api_key_from_env
