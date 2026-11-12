"""Context-aware and auditable token planning for every chat workload.

The planner keeps three different concepts separate:

* the total context window actually available to the loaded model;
* a provider/model output limit, when one is known; and
* an application synthetic limit chosen for a particular task.

Task limits are proportions and per-item workload estimates, never fixed
application-wide token ceilings.  Structured tasks deliberately receive a
smaller share than open synthesis because extra prose makes their JSON less
reliable and adds latency without adding another decision.  Every such choice
is carried in :class:`TokenBudget` with a plain-language reason.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
import math
import threading
import time
from typing import Any, Iterable, Mapping

from process.custom_endpoint import (
    CustomEndpoint,
    ModelListError,
    loaded_model_context,
)

logger = logging.getLogger(__name__)

# A compatibility assumption only, not a claim about any particular model.
# ``ModelCapacity.confirmed`` and ``source`` make its uncertainty visible.
COMPATIBILITY_FALLBACK_CONTEXT_TOKENS = 8192
ESTIMATOR_CHARS_PER_TOKEN = 3.5
MESSAGE_FRAMING_TOKENS = 8
SAFETY_FRACTION = 0.04
CAPACITY_CACHE_SECONDS = 30.0
# Capacity discovery is advisory and precedes the actual model call. Keep its
# synchronous probe short so an older endpoint cannot add the normal chat
# timeout to every first request; failures are negatively cached below.
CAPACITY_DISCOVERY_TIMEOUT_SECONDS = 2.0


@dataclass(frozen=True)
class ModelCapacity:
    """Resolved model capacity and where it came from."""

    context_window_tokens: int
    max_output_tokens: int | None = None
    source: str = "user setting"
    provider: str = ""
    model: str = ""
    confirmed: bool = True
    detail: str = ""

    def __post_init__(self) -> None:
        if self.context_window_tokens < 1:
            raise ValueError("Model context window must be a positive token count")
        if self.max_output_tokens is not None and self.max_output_tokens < 1:
            raise ValueError("Model output limit must be positive when supplied")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_capacity_cache: dict[tuple[str, str, str], tuple[float, ModelCapacity]] = {}
_capacity_cache_lock = threading.Lock()
_fallback_warning_keys: set[tuple[str, str]] = set()


@dataclass(frozen=True)
class TaskBudgetProfile:
    """Context-relative output workload for one class of request."""

    name: str
    response_mode: str
    base_output_fraction: float
    per_item_output_fraction: float
    max_output_fraction: float
    synthetic_limit_reason: str
    overflow_action: str = "split-input"


@dataclass(frozen=True)
class TokenBudget:
    """One successful, fully auditable request-budget decision."""

    task: str
    context_window_tokens: int
    estimated_prompt_tokens: int
    safety_tokens: int
    max_output_tokens: int
    input_capacity_tokens: int
    expected_items: int
    required_output_tokens: int
    synthetic_limit: bool
    limit_reason: str
    action: str
    capacity_source: str
    estimator: str = "utf8-bytes/3.5 plus message framing"

    @property
    def planned_total_tokens(self) -> int:
        return self.estimated_prompt_tokens + self.safety_tokens + self.max_output_tokens

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def describe(self) -> str:
        limitation = "yes" if self.synthetic_limit else "no"
        return (
            f"Token action — {self.task}: context={self.context_window_tokens:,}; "
            f"estimated prompt={self.estimated_prompt_tokens:,}; "
            f"response allowance={self.max_output_tokens:,}; safety={self.safety_tokens:,}; "
            f"items={self.expected_items}; action={self.action}; "
            f"synthetic limit={limitation}; reason={self.limit_reason}"
        )


class BudgetPlanningError(ValueError):
    """A request cannot fit without dropping input or starving its reply."""

    def __init__(
        self,
        message: str,
        *,
        action: str,
        task: str,
        context_window_tokens: int,
        estimated_prompt_tokens: int,
        required_output_tokens: int,
        safety_tokens: int,
    ) -> None:
        super().__init__(message)
        self.action = action
        self.task = task
        self.context_window_tokens = context_window_tokens
        self.estimated_prompt_tokens = estimated_prompt_tokens
        self.required_output_tokens = required_output_tokens
        self.safety_tokens = safety_tokens

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "task": self.task,
            "context_window_tokens": self.context_window_tokens,
            "estimated_prompt_tokens": self.estimated_prompt_tokens,
            "required_output_tokens": self.required_output_tokens,
            "safety_tokens": self.safety_tokens,
            "error": str(self),
        }


_FIXED_SCHEMA_REASON = (
    "This request returns a bounded JSON decision. Additional prose does not add another "
    "decision and increases latency and invalid-JSON risk."
)
_PER_ITEM_REASON = (
    "The response allowance is derived from the number of required item decisions so every "
    "supplied ID can receive a complete answer without encouraging unrelated prose."
)
_NARRATIVE_REASON = (
    "The response uses a context-relative writing share; the remaining window is reserved for "
    "source evidence and framing so longer output cannot crowd out its grounding."
)


# Fractions scale with the loaded context.  ``max_output_fraction`` is the
# largest share this task may use before the correct action becomes splitting
# the workload.  It is not an absolute token ceiling.
TASK_PROFILES: dict[str, TaskBudgetProfile] = {
    "general": TaskBudgetProfile(
        "general", "open_synthesis", 0.24, 0.0, 0.45, _NARRATIVE_REASON),
    "relevance_single": TaskBudgetProfile(
        "relevance_single", "fixed_schema", 0.08, 0.0, 0.16, _FIXED_SCHEMA_REASON),
    "relevance_batch": TaskBudgetProfile(
        "relevance_batch", "per_item_schema", 0.04, 0.018, 0.40, _PER_ITEM_REASON),
    "claim_extraction": TaskBudgetProfile(
        "claim_extraction", "fixed_schema", 0.18, 0.02, 0.36,
        "Structured extraction can contain several findings, so its allowance grows with "
        "the evidence workload while remaining bounded to reduce schema drift."),
    "method_review": TaskBudgetProfile(
        "method_review", "fixed_schema", 0.08, 0.0, 0.16, _FIXED_SCHEMA_REASON),
    "statistical_review": TaskBudgetProfile(
        "statistical_review", "per_item_schema", 0.04, 0.035, 0.45, _PER_ITEM_REASON),
    "synthesis_cluster": TaskBudgetProfile(
        "synthesis_cluster", "open_synthesis", 0.18, 0.0015, 0.36, _NARRATIVE_REASON),
    "synthesis_abstract": TaskBudgetProfile(
        "synthesis_abstract", "bounded_narrative", 0.08, 0.005, 0.25,
        "The structured abstract has an explicit publication-format word target; extra context "
        "is used for evidence coverage rather than unbounded prose."),
    "meta_discovery": TaskBudgetProfile(
        "meta_discovery", "per_item_schema", 0.08, 0.00055, 0.45, _PER_ITEM_REASON),
    "meta_title": TaskBudgetProfile(
        "meta_title", "per_item_schema", 0.05, 0.012, 0.35, _PER_ITEM_REASON),
    "meta_characteristics": TaskBudgetProfile(
        "meta_characteristics", "per_item_schema", 0.10, 0.018, 0.45, _PER_ITEM_REASON),
    "meta_enhancement": TaskBudgetProfile(
        "meta_enhancement", "bounded_narrative", 0.20, 0.0, 0.36, _NARRATIVE_REASON),
    "interrater": TaskBudgetProfile(
        "interrater", "fixed_schema", 0.12, 0.0, 0.24, _FIXED_SCHEMA_REASON),
}


def estimate_text_tokens(text: str) -> int:
    """Conservative tokenizer-independent estimate from UTF-8 bytes."""
    return math.ceil(len(str(text or "").encode("utf-8")) / ESTIMATOR_CHARS_PER_TOKEN)


def estimate_messages_tokens(messages: Iterable[Mapping[str, Any]]) -> int:
    """Estimate content plus role/message framing for a chat request."""
    total = 0
    for message in messages:
        total += MESSAGE_FRAMING_TOKENS
        total += estimate_text_tokens(str(message.get("role") or ""))
        total += estimate_text_tokens(str(message.get("content") or ""))
    return total


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _profile(name: str) -> TaskBudgetProfile:
    try:
        return TASK_PROFILES[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown LLM budget task {name!r}; choose one of {sorted(TASK_PROFILES)}"
        ) from exc


def _required_output(profile: TaskBudgetProfile, context: int, items: int) -> int:
    fraction = profile.base_output_fraction + profile.per_item_output_fraction * items
    if fraction > profile.max_output_fraction:
        # The caller must split by items.  Capping here would recreate the exact
        # large-input/small-reply mismatch this module exists to prevent.
        return math.ceil(context * fraction)
    return max(1, math.ceil(context * fraction))


def plan_token_budget(
    task: str,
    capacity: ModelCapacity | int,
    *,
    messages: Iterable[Mapping[str, Any]] | None = None,
    prompt_text: str = "",
    expected_items: int = 1,
    explicit_output_tokens: int | None = None,
    explicit_reason: str = "",
) -> TokenBudget:
    """Plan one request without silently truncating either side of it.

    ``expected_items`` is mandatory workload accounting for per-item JSON
    tasks.  If the task-derived response or complete prompt cannot fit, a
    :class:`BudgetPlanningError` names the required action (normally
    ``split-input``).  An explicit output override is accepted only as an
    explicit synthetic limitation and is recorded as ``user-override``.
    """
    resolved = capacity if isinstance(capacity, ModelCapacity) else ModelCapacity(
        int(capacity), source="caller-supplied context window")
    context = resolved.context_window_tokens
    items = max(1, int(expected_items))
    profile = _profile(task)
    prompt_tokens = (
        estimate_messages_tokens(messages)
        if messages is not None
        else estimate_text_tokens(prompt_text)
    )
    safety = max(1, math.ceil(context * SAFETY_FRACTION))
    required = _required_output(profile, context, items)
    required_fraction = required / context
    if required_fraction > profile.max_output_fraction:
        raise BudgetPlanningError(
            f"{task} needs about {required:,} response tokens for {items:,} items, "
            f"more than its {profile.max_output_fraction:.0%} context-relative response "
            "share. Split the items; the planner did not cap or discard their answers.",
            action=profile.overflow_action,
            task=task,
            context_window_tokens=context,
            estimated_prompt_tokens=prompt_tokens,
            required_output_tokens=required,
            safety_tokens=safety,
        )

    output = required
    action = "synthetic-task-limit"
    reason = profile.synthetic_limit_reason
    if explicit_output_tokens is not None:
        if explicit_output_tokens < 1:
            raise ValueError("Explicit output token limit must be positive")
        output = int(explicit_output_tokens)
        action = "user-override"
        reason = explicit_reason.strip() or (
            "The caller supplied an explicit response ceiling; this overrides automatic "
            "task scaling for this request."
        )

    synthetic_limit = True
    if resolved.max_output_tokens is not None and output > resolved.max_output_tokens:
        requested_output = output
        output = resolved.max_output_tokens
        action = "provider-output-limit"
        synthetic_limit = False
        reason = (
            f"The provider/model has an independent {output:,}-token completion ceiling, "
            f"below this task's context-scaled {requested_output:,}-token allowance. This is "
            "a physical model constraint, not a SystS synthetic cap. If a complete reply "
            "reaches it, the caller must split and retry the affected workload."
        )

    total = prompt_tokens + output + safety
    if total > context:
        raise BudgetPlanningError(
            f"{task} does not fit: estimated prompt {prompt_tokens:,} + required response "
            f"{output:,} + safety {safety:,} = {total:,}, above the {context:,}-token "
            "context. Split the input; neither evidence nor response allowance was reduced.",
            action=profile.overflow_action,
            task=task,
            context_window_tokens=context,
            estimated_prompt_tokens=prompt_tokens,
            required_output_tokens=output,
            safety_tokens=safety,
        )

    decision = TokenBudget(
        task=task,
        context_window_tokens=context,
        estimated_prompt_tokens=prompt_tokens,
        safety_tokens=safety,
        max_output_tokens=output,
        input_capacity_tokens=context - output - safety,
        expected_items=items,
        required_output_tokens=required,
        synthetic_limit=synthetic_limit,
        limit_reason=reason,
        action=action,
        capacity_source=resolved.source,
    )
    logger.info("%s", decision.describe())
    return decision


def resolve_model_capacity(
    config: Any,
    *,
    requested_context_tokens: int | None = None,
    model: str | None = None,
    transport=None,
) -> ModelCapacity:
    """Resolve the active context window without confusing ceiling and load size.

    Resolution order is: per-run explicit value, configured user value, live LM
    Studio loaded-instance metadata, then a visibly unconfirmed compatibility
    fallback for providers that expose no capacity metadata.  The fallback keeps
    older non-GUI callers working, but its source/detail are never labelled as a
    detected model capability.
    """
    provider = str(getattr(config, "llm_provider", "") or "").strip().lower()
    selected_model = str(model or getattr(config, "llm_model", "") or "").strip()
    provider_output = _positive_int(getattr(config, "llm_max_output_tokens", None))

    requested = _positive_int(requested_context_tokens)
    if requested is not None:
        return ModelCapacity(
            requested, provider_output, "run-specific user context setting",
            provider, selected_model, True,
            "The user supplied this context window for the current task.",
        )

    configured = _positive_int(getattr(config, "llm_context_tokens", None))
    if configured is not None:
        return ModelCapacity(
            configured, provider_output, "configured model context setting",
            provider, selected_model, True,
            "This value is stored in the pipeline configuration.",
        )

    api_format = str(getattr(config, "custom_api_format", "") or "").strip().lower()
    is_native_lmstudio = provider == "lmstudio" or (
        provider == "custom" and api_format == "lmstudio"
    )
    native_cache_key: tuple[str, str, str] | None = None
    if is_native_lmstudio:
        if provider == "lmstudio":
            base_url = getattr(config, "lmstudio_base_url", "")
            api_key = getattr(config, "lmstudio_api_key", "")
        else:
            base_url = getattr(config, "custom_base_url", "")
            api_key = getattr(config, "custom_api_key", "")
        cache_key = (provider, str(base_url or "").rstrip("/"), selected_model)
        native_cache_key = cache_key
        if transport is None:
            with _capacity_cache_lock:
                cached = _capacity_cache.get(cache_key)
            if cached is not None and time.monotonic() - cached[0] <= CAPACITY_CACHE_SECONDS:
                return cached[1]
        try:
            detected = loaded_model_context(
                CustomEndpoint("lmstudio", str(base_url or ""), str(api_key or "")),
                selected_model,
                timeout=CAPACITY_DISCOVERY_TIMEOUT_SECONDS,
                transport=transport,
            )
        except ModelListError as exc:
            logger.warning("Could not detect LM Studio loaded context for %r: %s", selected_model, exc)
        else:
            if detected is not None:
                resolved = ModelCapacity(
                    detected.context_length,
                    provider_output,
                    detected.source,
                    provider,
                    selected_model,
                    True,
                    f"Loaded instance: {detected.instance_id}",
                )
                if transport is None:
                    with _capacity_cache_lock:
                        _capacity_cache[cache_key] = (time.monotonic(), resolved)
                return resolved

    detail = (
        "No loaded-model context metadata or explicit setting was available. "
        "This compatibility fallback is synthetic and should be replaced by the model's "
        "actual loaded context in Preferences."
    )
    warning_key = (provider or "unknown", selected_model or "unknown")
    with _capacity_cache_lock:
        first_warning = warning_key not in _fallback_warning_keys
        _fallback_warning_keys.add(warning_key)
    if first_warning:
        logger.warning(
            "Using unconfirmed %s-token context fallback for provider=%s model=%s. %s",
            COMPATIBILITY_FALLBACK_CONTEXT_TOKENS, provider or "unknown",
            selected_model or "unknown", detail,
        )
    fallback = ModelCapacity(
        COMPATIBILITY_FALLBACK_CONTEXT_TOKENS,
        provider_output,
        "unconfirmed compatibility fallback",
        provider,
        selected_model,
        False,
        detail,
    )
    # Older LM Studio servers may support chat without exposing the native
    # model metadata endpoint. Cache that negative discovery briefly too;
    # otherwise every request would pay another model-list timeout before the
    # actual completion call.
    if transport is None and native_cache_key is not None:
        with _capacity_cache_lock:
            _capacity_cache[native_cache_key] = (time.monotonic(), fallback)
    return fallback
