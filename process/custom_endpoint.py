"""User-supplied LLM endpoints: URL normalization and model discovery.

A "custom" provider is any server speaking one of three wire formats:
OpenAI-compatible (``/v1/chat/completions``), Anthropic-compatible
(``/v1/messages``), or LM Studio's native REST API (``/api/v1/chat``).
Users paste whatever URL their server's docs show, so every format
normalizes both bare hosts and full API paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit

ANTHROPIC_VERSION = "2023-06-01"
MODEL_LIST_TIMEOUT_SECONDS = 10.0
_ANTHROPIC_PAGE_LIMIT = 1000
_ANTHROPIC_MAX_PAGES = 20
_EMBEDDING_TYPES = {"embedding", "embeddings"}


class ModelListError(RuntimeError):
    """Model discovery failed; the message is safe to show in the GUI."""


@dataclass(frozen=True)
class CustomEndpoint:
    api_format: str
    base_url: str
    api_key: str = ""


@dataclass(frozen=True)
class LoadedModelContext:
    """Context capacity of one model instance that is loaded right now.

    ``max_context_length`` from LM Studio describes a model file's theoretical
    ceiling.  It is deliberately not represented here: requests have to fit the
    context length chosen when the current instance was loaded.
    """

    model: str
    instance_id: str
    context_length: int
    source: str = "LM Studio loaded-model metadata"


def _strip(url: str) -> str:
    return (url or "").strip().rstrip("/")


def server_root(url: str) -> str:
    """Server root for LM Studio's native REST API (drops API path suffixes)."""
    clean = _strip(url)
    for suffix in ("/api/v1", "/api/v0", "/v1", "/api"):
        if clean.endswith(suffix):
            return clean[: -len(suffix)].rstrip("/")
    return clean


def openai_base_url(url: str) -> str:
    """Base URL the OpenAI SDK expects (it appends ``/chat/completions``).

    A bare host such as ``http://localhost:1234`` gets ``/v1``; any explicit
    path (``/openai/v1``, ``/api/paas/v4``) is kept verbatim.
    """
    clean = _strip(url)
    if not urlsplit(clean).path.strip("/"):
        return f"{clean}/v1"
    return clean


def anthropic_base_url(url: str) -> str:
    """Base URL the Anthropic SDK expects (it appends ``/v1/messages``)."""
    clean = _strip(url)
    return clean[:-3].rstrip("/") if clean.endswith("/v1") else clean


def rank_models(model_ids: list[str]) -> list[str]:
    """Order discovered model IDs for display in the picker.

    TODO(user): decide what the picker shows first. Current behavior only
    drops blanks and duplicates, keeping server order.
    """
    seen: set[str] = set()
    ranked: list[str] = []
    for model_id in model_ids:
        clean = (model_id or "").strip()
        if clean and clean not in seen:
            seen.add(clean)
            ranked.append(clean)
    return ranked


def _entry_ids(entries: object, id_keys: tuple[str, ...]) -> list[str]:
    if not isinstance(entries, list):
        raise ModelListError("Server response did not contain a model list.")
    ids = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("type") or "").lower() in _EMBEDDING_TYPES:
            continue
        for key in id_keys:
            if entry.get(key):
                ids.append(str(entry[key]))
                break
    return ids


def _get_json(client, url: str, **kwargs) -> dict:
    import httpx

    try:
        response = client.get(url, **kwargs)
    except httpx.HTTPError as exc:
        raise ModelListError(f"Could not reach {url}: {exc}") from exc
    if response.status_code in (401, 403):
        raise ModelListError(
            f"{url} rejected the API key (HTTP {response.status_code})."
        )
    if response.status_code >= 400:
        raise ModelListError(
            f"{url} returned HTTP {response.status_code}: {response.text[:200]}"
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise ModelListError(f"{url} did not return JSON.") from exc
    if not isinstance(payload, dict):
        raise ModelListError(f"{url} returned an unexpected JSON shape.")
    return payload


def _list_openai(client, endpoint: CustomEndpoint) -> list[str]:
    headers = {}
    if endpoint.api_key:
        headers["Authorization"] = f"Bearer {endpoint.api_key}"
    url = f"{openai_base_url(endpoint.base_url)}/models"
    return _entry_ids(_get_json(client, url, headers=headers).get("data"), ("id",))


def _list_anthropic(client, endpoint: CustomEndpoint) -> list[str]:
    headers = {"anthropic-version": ANTHROPIC_VERSION}
    if endpoint.api_key:
        headers["x-api-key"] = endpoint.api_key
    url = f"{anthropic_base_url(endpoint.base_url)}/v1/models"
    ids: list[str] = []
    params: dict[str, object] = {"limit": _ANTHROPIC_PAGE_LIMIT}
    for _ in range(_ANTHROPIC_MAX_PAGES):
        payload = _get_json(client, url, headers=headers, params=params)
        ids.extend(_entry_ids(payload.get("data"), ("id",)))
        if not payload.get("has_more") or not payload.get("last_id"):
            break
        params = {"limit": _ANTHROPIC_PAGE_LIMIT, "after_id": payload["last_id"]}
    return ids


def _list_lmstudio(client, endpoint: CustomEndpoint) -> list[str]:
    headers = {}
    if endpoint.api_key:
        headers["Authorization"] = f"Bearer {endpoint.api_key}"
    root = server_root(endpoint.base_url)
    payload = _get_json(client, f"{root}/api/v1/models", headers=headers)
    # Current builds list ``models`` keyed by ``key``; some return an
    # OpenAI-style ``data`` list instead. One request serves both shapes.
    if isinstance(payload.get("models"), list):
        return _entry_ids(payload["models"], ("key", "id"))
    return _entry_ids(payload.get("data"), ("id",))


def _positive_int(value: object) -> int | None:
    """A positive integer from endpoint metadata, excluding booleans."""
    if isinstance(value, bool):
        return None
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _loaded_context_candidates(entry: dict, requested_model: str) -> list[LoadedModelContext]:
    """Loaded contexts in one LM Studio model entry that match ``requested_model``."""
    entry_ids = {
        str(entry.get(key) or "").strip()
        for key in ("key", "id", "selected_variant")
        if str(entry.get(key) or "").strip()
    }
    instances = entry.get("loaded_instances")
    matches: list[LoadedModelContext] = []
    if isinstance(instances, list):
        for instance in instances:
            if not isinstance(instance, dict):
                continue
            instance_id = str(instance.get("id") or "").strip()
            if requested_model not in entry_ids and requested_model != instance_id:
                continue
            config = instance.get("config")
            configured = config.get("context_length") if isinstance(config, dict) else None
            context_length = _positive_int(configured)
            if context_length is not None:
                matches.append(LoadedModelContext(
                    model=requested_model,
                    instance_id=instance_id or next(iter(entry_ids), requested_model),
                    context_length=context_length,
                ))
        return matches

    # Compatibility with older data-shaped listings.  Only accept a direct
    # context length when the entry explicitly says the model is loaded.  Never
    # substitute max_context_length, which is not the runtime allocation.
    state = str(entry.get("state") or entry.get("status") or "").strip().lower()
    if requested_model in entry_ids and state in {"loaded", "ready", "running"}:
        context_length = _positive_int(entry.get("context_length"))
        if context_length is not None:
            matches.append(LoadedModelContext(
                model=requested_model,
                instance_id=str(entry.get("id") or requested_model),
                context_length=context_length,
                source="LM Studio loaded-model metadata (legacy listing)",
            ))
    return matches


def loaded_model_context(
    endpoint: CustomEndpoint,
    model: str,
    *,
    timeout: float = MODEL_LIST_TIMEOUT_SECONDS,
    transport=None,
) -> LoadedModelContext | None:
    """Return the unambiguous context length of a loaded LM Studio model.

    The native ``/api/v1/models`` response exposes both the model's maximum
    supported context and each loaded instance's configured context.  Only the
    latter is safe for request planning.  ``None`` means the requested model is
    not loaded, its loaded instance does not report a context, or routing by a
    model key is ambiguous because several instances use different contexts.

    Raises :class:`ModelListError` for an invalid endpoint or an HTTP/protocol
    failure.  ``transport`` lets tests inject an ``httpx.MockTransport``.
    """
    import httpx

    if endpoint.api_format != "lmstudio":
        raise ModelListError(
            "Loaded context detection requires the LM Studio REST API format."
        )
    if not _strip(endpoint.base_url):
        raise ModelListError("Enter the endpoint's base URL first.")
    requested = str(model or "").strip()
    if not requested:
        raise ModelListError("Select a model before detecting its loaded context.")

    headers = {}
    if endpoint.api_key:
        headers["Authorization"] = f"Bearer {endpoint.api_key}"
    url = f"{server_root(endpoint.base_url)}/api/v1/models"
    with httpx.Client(timeout=timeout, transport=transport) as client:
        payload = _get_json(client, url, headers=headers)
    entries = payload.get("models")
    if not isinstance(entries, list):
        entries = payload.get("data")
    if not isinstance(entries, list):
        raise ModelListError("Server response did not contain a model list.")

    candidates = [
        candidate
        for entry in entries
        if isinstance(entry, dict)
        for candidate in _loaded_context_candidates(entry, requested)
    ]
    if not candidates:
        return None

    # An exact loaded-instance id identifies one allocation even when the model
    # key has several instances.  Routing by the model key is safe only when all
    # matching loaded instances agree on context length.
    exact_instances = [c for c in candidates if c.instance_id == requested]
    if len(exact_instances) == 1:
        return exact_instances[0]
    lengths = {candidate.context_length for candidate in candidates}
    if len(lengths) != 1:
        return None
    return candidates[0]


_LISTERS = {
    "openai": _list_openai,
    "anthropic": _list_anthropic,
    "lmstudio": _list_lmstudio,
}


def list_models(
    endpoint: CustomEndpoint,
    *,
    timeout: float = MODEL_LIST_TIMEOUT_SECONDS,
    transport=None,
) -> list[str]:
    """Fetch chat-capable model IDs from ``endpoint``.

    Raises :class:`ModelListError` with a user-facing message on any failure.
    ``transport`` lets tests inject an ``httpx.MockTransport``.
    """
    import httpx

    lister = _LISTERS.get(endpoint.api_format)
    if lister is None:
        raise ModelListError(f"Unknown API format: {endpoint.api_format!r}")
    if not _strip(endpoint.base_url):
        raise ModelListError("Enter the endpoint's base URL first.")
    with httpx.Client(timeout=timeout, transport=transport) as client:
        ids = rank_models(lister(client, endpoint))
    if not ids:
        raise ModelListError("The server returned no chat models.")
    return ids
