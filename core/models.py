"""
Data models for the Hypothesis Consensus Analyzer.

Contains enumerations and dataclasses used throughout the pipeline.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum, auto
from typing import Any

# Pipeline artifacts shared between their writer stage and the PRISMA reader.
SEARCH_MANIFEST_FILENAME = "search_manifest.json"  # in config.corpus_dir
SEARCH_MANIFEST_VERSION = 1
REMOVAL_LOG_FILENAME = "removal_log.json"  # in config.claims_dir (Stage 5)

# LLMScoringResult / EvidenceLink blending. No pipeline stage currently
# attaches an LLMScoringResult, so these apply only to callers that do.
SPAN_RELIABILITY_THRESHOLD = 0.6  # share of quote spans found (exact or close)
KEYWORD_SCORE_WEIGHT = 0.4
LLM_SCORE_WEIGHT = 0.6


class EvidenceDirection(Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"
    TANGENTIAL = "tangential"


class SourceType(Enum):
    PUBMED = "PubMed"
    SEMANTIC_SCHOLAR = "Semantic Scholar"
    OPENALEX = "OpenAlex"
    WEB_OF_SCIENCE = "Web of Science"
    SCOPUS = "Scopus"
    ELICIT = "Elicit"
    LOCAL_PDF = "Local PDF"


class SpanStatus(Enum):
    VERIFIED = "verified"
    CLOSE_MATCH = "close_match"
    NOT_FOUND = "not_found"
    UNCHECKED = "unchecked"


class LLMProvider(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    CUSTOM = "custom"
    ANTHROPIC = "anthropic"
    SENTENCE_TRANSFORMERS = "sentence_transformers"


class PipelineState(Enum):
    IDLE = auto()
    SEARCHING = auto()
    SCORING = auto()
    FULLTEXT_ACQUIRE = auto()
    EXTRACTING = auto()
    VALIDATING = auto()
    AUDITING = auto()
    SYNTHESIZING = auto()
    FULLTEXT_PROCESS = auto()
    INTERRATING = auto()
    ROBUSTNESS = auto()
    VERIFYING = auto()
    COMPARING = auto()
    ERROR = auto()
    CANCELLED = auto()

    @property
    def label(self) -> str:
        return self.name.replace("_", " ").title()


@dataclass
class HypothesisComponent:
    id: str
    label: str
    description: str
    keywords: list[str] = field(default_factory=list)
    weight: float = 1.0


@dataclass
class Article:
    id: str
    title: str
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    full_text: str = ""
    year: int | None = None
    journal: str = ""
    doi: str = ""
    source: SourceType = SourceType.PUBMED
    citations: int = 0
    url: str = ""
    pmid: str = ""
    keywords: list[str] = field(default_factory=list)
    has_full_text: bool = False
    open_access: bool = False
    full_text_url: str = ""

    @property
    def text_content(self) -> str:
        return self.full_text if self.full_text else self.abstract


@dataclass
class QuoteSpan:
    text: str
    component_id: str = ""
    direction: EvidenceDirection = EvidenceDirection.NEUTRAL
    relevance_note: str = ""
    status: SpanStatus = SpanStatus.UNCHECKED
    matched_text: str = ""
    match_similarity: float = 0.0
    char_offset_start: int = -1
    char_offset_end: int = -1


@dataclass
class LLMScoringResult:
    article_id: str
    component_id: str
    llm_relevance: float = 0.0
    llm_direction: EvidenceDirection = EvidenceDirection.NEUTRAL
    llm_reasoning: str = ""
    quote_spans: list[QuoteSpan] = field(default_factory=list)
    model_name: str = ""
    provider: LLMProvider = LLMProvider.OPENAI
    spans_verified: int = 0
    spans_close: int = 0
    spans_not_found: int = 0
    spans_total: int = 0
    verification_score: float = 0.0

    @property
    def is_reliable(self) -> bool:
        if self.spans_total == 0:
            return False
        verified = self.spans_verified + self.spans_close
        return verified / self.spans_total >= SPAN_RELIABILITY_THRESHOLD


@dataclass
class EvidenceLink:
    article_id: str
    component_id: str
    direction: EvidenceDirection = EvidenceDirection.NEUTRAL
    relevance_score: float = 0.0
    confidence: float = 0.0
    key_excerpts: list[str] = field(default_factory=list)
    matched_keywords: list[str] = field(default_factory=list)
    llm_result: LLMScoringResult | None = None

    @property
    def combined_score(self) -> float:
        if self.llm_result and self.llm_result.is_reliable:
            return (KEYWORD_SCORE_WEIGHT * self.relevance_score
                    + LLM_SCORE_WEIGHT * self.llm_result.llm_relevance)
        return self.relevance_score

    @property
    def combined_direction(self) -> EvidenceDirection:
        if self.llm_result and self.llm_result.is_reliable:
            return self.llm_result.llm_direction
        return self.direction


@dataclass
class AlignmentResult:
    component_id: str
    component_label: str
    support_score: float = 0.0
    evidence_count: int = 0
    supporting_count: int = 0
    contradicting_count: int = 0
    neutral_count: int = 0
    confidence: float = 0.0
    top_supporting: list[str] = field(default_factory=list)
    top_contradicting: list[str] = field(default_factory=list)


@dataclass
class TrendPoint:
    year: int
    article_count: int = 0
    avg_support_score: float = 0.0
    cumulative_citations: int = 0
    component_id: str = ""


@dataclass
class AnalysisSession:
    hypothesis_components: list[HypothesisComponent] = field(default_factory=list)
    articles: dict[str, Article] = field(default_factory=dict)
    evidence_links: list[EvidenceLink] = field(default_factory=list)
    alignment_results: list[AlignmentResult] = field(default_factory=list)
    trend_data: list[TrendPoint] = field(default_factory=list)
    total_articles_fetched: int = 0
    total_articles_analyzed: int = 0
    llm_results: dict[str, LLMScoringResult] = field(default_factory=dict)
    flagged_spans: list[QuoteSpan] = field(default_factory=list)
    corpus_json_path: str = ""
    fulltext_retrieval_results: list = field(default_factory=list)
    wordfish_model: object = None
    stylometry_profiles: dict = field(default_factory=dict)
    corpus_statistics: dict = field(default_factory=dict)
    pipeline_state: PipelineState = PipelineState.IDLE
    run_id: str = ""
    stage_timings: dict[str, float] = field(default_factory=dict)
    raw_search_counts: dict[str, int] = field(default_factory=dict)
    synthesis_clusters: list[dict[str, Any]] = field(default_factory=list)
    synthesis_abstract: str = ""


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class SearchRunRecord:
    """One Stage 1 search run as recorded in ``search_manifest.json``.

    The counts reconcile: ``corpus_after = corpus_before + records_retrieved
    - sum(records_removed_by_filters) - duplicates_removed - no_abstract_removed``.
    ``providers`` holds one entry per attempted source with ``source``,
    ``query``, ``hits`` (provider total, or None), ``retrieved``, ``error``
    and ``truncated`` (True/False, or None when unknown).
    """

    entry_point: str
    providers: list[dict[str, Any]]
    corpus_before: int
    records_retrieved: int
    records_removed_by_filters: dict[str, int]
    records_before_dedup: int
    duplicates_removed: int
    no_abstract_removed: int
    corpus_after: int
    filters: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_utc_timestamp)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def new_search_manifest(run: SearchRunRecord, baseline_records: int = 0) -> dict[str, Any]:
    """A manifest whose only run is *run*.

    *baseline_records* counts corpus records that predate any recorded run.
    """
    return {
        "version": SEARCH_MANIFEST_VERSION,
        "baseline_records": int(baseline_records),
        "runs": [run.to_dict()],
    }


_UNITEMISED_SOURCE = "earlier searches (not itemised)"


def _as_count(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


@dataclass(frozen=True)
class SearchManifestSummary:
    """Totals across every run of a search manifest (PRISMA identification)."""

    records_identified: int
    records_per_source: dict[str, int]
    duplicates_removed: int
    no_abstract_removed: int
    removed_by_filters: int
    failed_sources: tuple[str, ...]
    queries: dict[str, tuple[str, ...]]


def _manifest_runs(manifest: Any) -> list[dict]:
    if not isinstance(manifest, dict) or not isinstance(manifest.get("runs"), list):
        return []
    return [run for run in manifest["runs"] if isinstance(run, dict)]


def _provider_entries(runs: list[dict]) -> list[dict]:
    entries: list[dict] = []
    for run in runs:
        providers = run.get("providers")
        if isinstance(providers, list):
            entries.extend(e for e in providers if isinstance(e, dict))
    return entries


def _filter_removals(run: dict) -> int:
    removals = run.get("records_removed_by_filters")
    return sum(_as_count(v) for v in removals.values()) if isinstance(removals, dict) else 0


def summarize_search_manifest(manifest: Any) -> SearchManifestSummary | None:
    """Sum a ``search_manifest.json`` payload; None when it holds no runs.

    Per-source counts include only providers that answered; failed providers
    are listed separately. Records predating the first recorded run appear
    under one un-itemised source.
    """
    runs = _manifest_runs(manifest)
    if not runs:
        return None
    per_source: dict[str, int] = {}
    failed: list[str] = []
    queries: dict[str, list[str]] = {}
    for entry in _provider_entries(runs):
        source = str(entry.get("source") or "unknown")
        if entry.get("error"):
            if source not in failed:
                failed.append(source)
            continue
        per_source[source] = per_source.get(source, 0) + _as_count(entry.get("retrieved"))
        query = str(entry.get("query") or "").strip()
        known = queries.setdefault(source, [])
        if query and query not in known:
            known.append(query)
    baseline = _as_count(manifest.get("baseline_records"))
    if baseline:
        per_source[_UNITEMISED_SOURCE] = baseline
    return SearchManifestSummary(
        records_identified=baseline + sum(_as_count(r.get("records_retrieved")) for r in runs),
        records_per_source=per_source,
        duplicates_removed=sum(_as_count(r.get("duplicates_removed")) for r in runs),
        no_abstract_removed=sum(_as_count(r.get("no_abstract_removed")) for r in runs),
        removed_by_filters=sum(_filter_removals(r) for r in runs),
        failed_sources=tuple(failed),
        queries={source: tuple(q) for source, q in queries.items() if q},
    )
