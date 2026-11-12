"""Module-level helpers shared by MainWindow and its tab/stage mixins.

Split from gui/main_window.py; re-exported there so existing imports keep working.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QLineEdit, QPushButton

from gui.model_picker import EndpointGetter, ModelFetchButton

LLM_PROVIDER_CHOICES = ["ollama", "custom", "openai", "anthropic"]


class PipelineState:
    """String-constant namespace for pipeline state tracking."""
    IDLE = "idle"
    SEARCHING = "searching"
    SCORING = "scoring"
    EXTRACTING = "extracting"
    VALIDATING = "validating"
    AUDITING = "auditing"
    SYNTHESIZING = "synthesizing"
    FULLTEXT = "fulltext"
    ROBUSTNESS = "robustness"
    VERIFYING = "verifying"
    COMPARING = "comparing"
    DONE = "done"


def original_question_for_model(host) -> str:
    """The frozen original research question that model-backed stages use.

    Model-backed stages can run before Stage 1 (e.g. against a loaded corpus).
    The fallback question is frozen on first non-empty use so later stages
    cannot silently drift to a different GUI value. Shared by MainWindow and
    its config-sync mixin so the mixin need not import MainWindow.
    """
    state = host._original_question_state
    original = state.question_for_model(host.sysys_nl_query.text())
    if original and not state.captured:
        host._capture_original_question_for_first_search(original)
    return original


def _paths_resolve_under_root(raw: str, project_root: Path) -> Path | None:
    """Resolve a user path; relative paths are taken under project_root."""
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (project_root / p).resolve()
        else:
            p = p.resolve()
        return p
    except OSError:
        return None


def _make_file_row(
    label: str,
    placeholder: str = "",
    browse_save: bool = False,
    file_filter: str = "All files (*)",
) -> tuple:
    """Create a reusable file-path input row (label + QLineEdit + Browse button)."""
    h = QHBoxLayout()
    lbl = QLabel(label)
    lbl.setMinimumWidth(100)
    edit = QLineEdit()
    edit.setPlaceholderText(placeholder)
    edit.setAccessibleName(label.replace(":", "").strip())
    lbl.setBuddy(edit)
    btn = QPushButton("Browse...")
    btn.setAccessibleName(f"Browse for {label.replace(':', '').strip()}")
    h.addWidget(lbl)
    h.addWidget(edit, 1)
    h.addWidget(btn)
    return h, edit, btn


def _make_llm_backend_row(
    default_provider: str = "ollama",
    default_model: str = "",
    extra_providers: list[str] | None = None,
    endpoint_getter: EndpointGetter | None = None,
) -> tuple:
    """Create a reusable LLM provider/model selector row.

    ``extra_providers`` appends non-LLM backends (e.g. "brt_bert") to the
    dropdown for stages that support them, without polluting every row.
    ``endpoint_getter`` adds a Fetch Models button, enabled while the row's
    provider is "custom".
    """
    h = QHBoxLayout()
    combo = QComboBox()
    combo.addItems(LLM_PROVIDER_CHOICES)
    if extra_providers:
        combo.addItems(extra_providers)
    idx = combo.findText(default_provider)
    if idx >= 0:
        combo.setCurrentIndex(idx)
    model_edit = QLineEdit()
    model_edit.setPlaceholderText(default_model or "model name")
    model_edit.setAccessibleName("LLM model")

    provider_lbl = QLabel("Provider:")
    provider_lbl.setBuddy(combo)
    h.addWidget(provider_lbl)
    h.addWidget(combo)

    model_lbl = QLabel("Model:")
    model_lbl.setBuddy(model_edit)
    h.addWidget(model_lbl)
    h.addWidget(model_edit, 1)

    if endpoint_getter is not None:
        fetch_btn = ModelFetchButton(model_edit, endpoint_getter)
        fetch_btn.setEnabled(combo.currentText() == "custom")
        combo.currentTextChanged.connect(
            lambda text: fetch_btn.setEnabled(text == "custom")
        )
        h.addWidget(fetch_btn)
        model_edit._fetch_button = fetch_btn

    # Track whether the user has hand-edited these fields. Once customized,
    # programmatic syncs (project open, Settings → Apply) must NOT overwrite
    # them — only user edits or a new preference default should change them.
    # Use user-initiated signals (activated / textEdited) so programmatic
    # setText()/setCurrentIndex() calls do not flip the flag.
    combo._user_customized = False
    combo.activated.connect(lambda *_: setattr(combo, "_user_customized", True))
    model_edit._user_customized = False
    model_edit.textEdited.connect(
        lambda *_: setattr(model_edit, "_user_customized", True)
    )
    return h, combo, model_edit
