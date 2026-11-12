"""Widget construction for SettingsDialog (mixin)."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.config import CUSTOM_API_FORMATS, DEFAULT_STAGGER_DELAY
from gui.model_picker import ModelFetchButton
from process.custom_endpoint import CustomEndpoint


class SettingsBuildMixin:
    """Builds the SettingsDialog widget tree; mixed into SettingsDialog."""

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        # Scrollable area for all settings
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(8)

        # --- Config file management ---
        layout.addWidget(self._build_config_group())

        # --- Appearance ---
        layout.addWidget(self._build_appearance_group())

        # --- Project root ---
        layout.addWidget(self._build_project_root_group())

        # --- Original research question ---
        layout.addWidget(self._build_original_question_group())

        # --- API Keys ---
        layout.addWidget(self._build_api_keys_group())

        # --- LLM Provider Settings ---
        layout.addWidget(self._build_llm_group())

        # --- Processing Parameters ---
        layout.addWidget(self._build_processing_group())

        layout.addStretch()
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

        # --- Bottom button row ---
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        apply_btn = QPushButton("Apply")
        apply_btn.setMinimumWidth(90)
        apply_btn.clicked.connect(self._on_apply)

        save_btn = QPushButton("Save")
        save_btn.setMinimumWidth(90)
        save_btn.clicked.connect(self._on_save)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setMinimumWidth(90)
        cancel_btn.clicked.connect(self.reject)

        btn_row.addWidget(apply_btn)
        btn_row.addWidget(save_btn)
        btn_row.addWidget(cancel_btn)
        outer.addLayout(btn_row)

    def _build_config_group(self) -> QGroupBox:
        grp = QGroupBox("Configuration File")
        layout = QVBoxLayout(grp)
        h = QHBoxLayout()

        load_btn = QPushButton("Load Config...")
        load_btn.clicked.connect(self._on_load_config)
        save_btn = QPushButton("Save Config...")
        save_btn.clicked.connect(self._on_save_config)
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self._on_reset_config)

        h.addWidget(load_btn)
        h.addWidget(save_btn)
        h.addWidget(reset_btn)
        h.addStretch()
        layout.addLayout(h)
        self.settings_domain_summary = QLabel()
        self.settings_domain_summary.setWordWrap(True)
        self.settings_domain_summary.setAccessibleName("Domain vocabulary configuration")
        self.settings_domain_summary.setToolTip(
            "Load a research configuration to customize graph entities, extraction "
            "vocabulary, and method classification rules. Save Config exports these settings."
        )
        layout.addWidget(self.settings_domain_summary)
        return grp

    def _build_appearance_group(self) -> QGroupBox:
        grp = QGroupBox("Appearance")
        form = QFormLayout(grp)

        self.settings_font_size = QSpinBox()
        self.settings_font_size.setRange(6, 24)
        self.settings_font_size.setValue(10)
        self.settings_font_size.setSuffix(" pt")
        self.settings_font_size.setToolTip(
            "Application font size (default 10pt). Changes take effect on Apply."
        )
        form.addRow("Font Size:", self.settings_font_size)

        return grp

    def _build_project_root_group(self) -> QGroupBox:
        grp = QGroupBox("Project Root")
        h = QHBoxLayout(grp)
        self.settings_project_root = QLineEdit()
        self.settings_project_root.setPlaceholderText("Project root directory")
        self.settings_project_root_browse = QPushButton("Browse...")
        self.settings_project_root_browse.clicked.connect(self._browse_project_root)
        h.addWidget(self.settings_project_root, 1)
        h.addWidget(self.settings_project_root_browse)
        return grp

    def _build_original_question_group(self) -> QGroupBox:
        group = QGroupBox("Original Research Question")
        self.original_question_group = group
        form = QFormLayout(group)
        self.settings_original_question = QLineEdit()
        self.settings_original_question.setAccessibleName(
            "Original research question"
        )
        self.settings_edit_original_question = QCheckBox(
            "Edit captured original question"
        )
        self.settings_edit_original_question.toggled.connect(
            self._on_edit_original_question_toggled
        )
        form.addRow("Original Question:", self.settings_original_question)
        form.addRow("", self.settings_edit_original_question)
        return group

    def _build_api_keys_group(self) -> QGroupBox:
        grp = QGroupBox("API Keys")
        form = QFormLayout(grp)

        self.settings_openai_key = QLineEdit()
        self.settings_openai_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.settings_openai_key.setPlaceholderText("sk-...")
        form.addRow("OpenAI API Key:", self.settings_openai_key)

        self.settings_custom_key = QLineEdit()
        self.settings_custom_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.settings_custom_key.setPlaceholderText("Optional for local servers")
        form.addRow("Custom Endpoint Key:", self.settings_custom_key)

        self.settings_anthropic_key = QLineEdit()
        self.settings_anthropic_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.settings_anthropic_key.setPlaceholderText("sk-ant-...")
        form.addRow("Anthropic API Key:", self.settings_anthropic_key)

        self.settings_s2_key = QLineEdit()
        self.settings_s2_key.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("Semantic Scholar Key:", self.settings_s2_key)

        self.settings_pubmed_key = QLineEdit()
        self.settings_pubmed_key.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("PubMed API Key:", self.settings_pubmed_key)

        self.settings_pubmed_email = QLineEdit()
        self.settings_pubmed_email.setPlaceholderText("user@example.com")
        form.addRow("PubMed Email:", self.settings_pubmed_email)

        self.settings_elicit_key = QLineEdit()
        self.settings_elicit_key.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("Elicit API Key:", self.settings_elicit_key)

        self.settings_scopus_key = QLineEdit()
        self.settings_scopus_key.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("Scopus API Key:", self.settings_scopus_key)

        self.settings_scopus_insttoken = QLineEdit()
        self.settings_scopus_insttoken.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("Scopus Inst Token:", self.settings_scopus_insttoken)

        self.settings_wos_key = QLineEdit()
        self.settings_wos_key.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("Web of Science Key:", self.settings_wos_key)
        self.settings_wos_researcher_key = QLineEdit()
        self.settings_wos_researcher_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.settings_wos_researcher_key.setToolTip("Clarivate WoS Researcher API key (X-ApiKey header)")
        form.addRow("WoS Researcher Key:", self.settings_wos_researcher_key)

        # Opt-in toggle: only persist keys to QSettings when explicitly requested
        self.settings_remember_keys = QCheckBox(
            "Remember API keys between sessions (stored in plaintext by QSettings)"
        )
        self.settings_remember_keys.setToolTip(
            "When checked, API keys are saved to the OS application settings store "
            "(registry on Windows, plist on macOS, ini file on Linux) in plaintext. "
            "Uncheck on shared or multi-user machines. "
            "If these fields are pre-filled from configuration or environment variables, "
            "checking this option may also save those displayed values to QSettings."
        )
        form.addRow("", self.settings_remember_keys)

        return grp

    def _build_llm_group(self) -> QGroupBox:
        grp = QGroupBox("LLM Provider Settings")
        form = QFormLayout(grp)

        self.settings_provider = QComboBox()
        self.settings_provider.addItems(["ollama", "custom", "openai", "anthropic"])
        form.addRow("Provider:", self.settings_provider)

        self.settings_ollama_model = QLineEdit()
        self.settings_ollama_model.setPlaceholderText("qwen2.5:32b-instruct-q4_K_M")
        form.addRow("Ollama Model:", self.settings_ollama_model)

        self.settings_ollama_url = QLineEdit()
        self.settings_ollama_url.setPlaceholderText("http://localhost:11434")
        form.addRow("Ollama Base URL:", self.settings_ollama_url)

        self.settings_custom_url = QLineEdit()
        self.settings_custom_url.setPlaceholderText("http://localhost:1234/v1")
        self.settings_custom_url.setToolTip(
            "Any OpenAI- or Anthropic-compatible server (LM Studio, vLLM, "
            "OpenRouter, LongCat, a gateway...) or LM Studio's REST API."
        )
        form.addRow("Custom Base URL:", self.settings_custom_url)

        self.settings_custom_format = QComboBox()
        for fmt, label in CUSTOM_API_FORMATS.items():
            self.settings_custom_format.addItem(label, fmt)
        form.addRow("Custom API Format:", self.settings_custom_format)

        custom_model_row = QHBoxLayout()
        self.settings_custom_model = QLineEdit()
        self.settings_custom_model.setPlaceholderText("Fetch or type a model ID")
        self.settings_custom_model.setAccessibleName("Custom endpoint model")
        self.settings_custom_fetch = ModelFetchButton(
            self.settings_custom_model, self._custom_endpoint_from_widgets, self
        )
        custom_model_row.addWidget(self.settings_custom_model, 1)
        custom_model_row.addWidget(self.settings_custom_fetch)
        form.addRow("Custom Model:", custom_model_row)

        self.settings_openai_model = QLineEdit()
        self.settings_openai_model.setPlaceholderText("gpt-4o-mini")
        form.addRow("OpenAI Model:", self.settings_openai_model)

        self.settings_openai_url = QLineEdit()
        self.settings_openai_url.setPlaceholderText("https://api.openai.com/v1")
        form.addRow("OpenAI Base URL:", self.settings_openai_url)

        self.settings_anthropic_model = QLineEdit()
        self.settings_anthropic_model.setPlaceholderText("claude-haiku-4-5-20251001")
        form.addRow("Anthropic Model:", self.settings_anthropic_model)

        self.settings_context_tokens = QSpinBox()
        self.settings_context_tokens.setRange(0, 2_147_483_647)
        self.settings_context_tokens.setSingleStep(4096)
        self.settings_context_tokens.setSpecialValueText("Auto-detect")
        self.settings_context_tokens.setToolTip(
            "Total loaded context (input plus output) for the selected model. "
            "Auto-detect reads the currently loaded value from LM Studio's native "
            "REST API. For providers that do not report it, enter the model's actual "
            "context here. SystS scales each task's input and response plan from this value."
        )
        form.addRow("Loaded Context Tokens:", self.settings_context_tokens)

        self.settings_max_output_tokens = QSpinBox()
        self.settings_max_output_tokens.setRange(0, 2_147_483_647)
        self.settings_max_output_tokens.setSingleStep(1024)
        self.settings_max_output_tokens.setSpecialValueText("Unknown / not reported")
        self.settings_max_output_tokens.setToolTip(
            "The provider/model's independent completion ceiling, if it has one. "
            "This is separate from the total context window. Leave at zero when the "
            "local endpoint has no separate ceiling or it is unknown; for hosted models, "
            "enter the documented maximum output. SystS records this physical constraint "
            "and splits item workloads instead of presenting it as a synthetic app cap."
        )
        form.addRow("Provider Output Limit:", self.settings_max_output_tokens)

        return grp

    def _custom_format_from_widget(self) -> str:
        return self.settings_custom_format.currentData() or "openai"

    def _custom_endpoint_from_widgets(self) -> CustomEndpoint:
        """Endpoint as currently typed, so Fetch works before Apply."""
        return CustomEndpoint(
            api_format=self._custom_format_from_widget(),
            base_url=self.settings_custom_url.text().strip(),
            api_key=self.settings_custom_key.text().strip(),
        )

    def _build_processing_group(self) -> QGroupBox:
        grp = QGroupBox("Processing Parameters")
        form = QFormLayout(grp)

        self.settings_threshold = QSpinBox()
        self.settings_threshold.setRange(1, 10)
        self.settings_threshold.setValue(3)
        form.addRow("Relevance Threshold:", self.settings_threshold)

        self.settings_workers = QSpinBox()
        self.settings_workers.setRange(1, 32)
        self.settings_workers.setValue(3)
        form.addRow("Concurrent Workers:", self.settings_workers)

        self.settings_batch_size = QSpinBox()
        self.settings_batch_size.setRange(1, 50)
        self.settings_batch_size.setValue(1)
        form.addRow("Score Batch Size:", self.settings_batch_size)

        self.settings_score_max_tokens = QSpinBox()
        self.settings_score_max_tokens.setRange(0, 2_147_483_647)
        self.settings_score_max_tokens.setSingleStep(512)
        self.settings_score_max_tokens.setValue(0)
        self.settings_score_max_tokens.setSpecialValueText("Automatic")
        self.settings_score_max_tokens.setToolTip(
            "Automatic uses the shared context-aware relevance policy. A positive value "
            "is an explicit synthetic response ceiling; SystS records it and may split "
            "work instead of silently truncating a reply."
        )
        form.addRow("Score Output Override:", self.settings_score_max_tokens)

        self.settings_stagger_delay = QDoubleSpinBox()
        self.settings_stagger_delay.setRange(0.0, 30.0)
        self.settings_stagger_delay.setSingleStep(0.5)
        self.settings_stagger_delay.setValue(DEFAULT_STAGGER_DELAY)
        form.addRow("Stagger Delay (s):", self.settings_stagger_delay)

        self.settings_timeout = QDoubleSpinBox()
        self.settings_timeout.setRange(10.0, 600.0)
        self.settings_timeout.setSingleStep(10.0)
        self.settings_timeout.setValue(120.0)
        form.addRow("Worker Timeout (s):", self.settings_timeout)

        self.settings_fulltext_streaming = QCheckBox(
            "Stream Stage 7 results to resumable JSONL"
        )
        self.settings_fulltext_streaming.setChecked(True)
        self.settings_fulltext_streaming.setToolTip(
            "Process one paper at a time and preserve completed JSONL records "
            "so a cancelled or interrupted Stage 7 run can resume."
        )
        form.addRow("Stage 7 Streaming:", self.settings_fulltext_streaming)

        self.settings_fulltext_batch_size = QSpinBox()
        self.settings_fulltext_batch_size.setRange(1, 10000)
        self.settings_fulltext_batch_size.setValue(50)
        self.settings_fulltext_batch_size.setToolTip(
            "Maximum papers retained by the legacy buffered Stage 7 path. "
            "This is disabled while streaming is enabled."
        )
        form.addRow("Buffered Download Batch:", self.settings_fulltext_batch_size)

        self.settings_paperqa_index_enabled = QCheckBox(
            "Build the optional PaperQA index during corpus enrichment"
        )
        self.settings_paperqa_index_enabled.setChecked(True)
        self.settings_paperqa_index_enabled.setToolTip(
            "Index extracted full texts during corpus full-text enrichment. "
            "Disable this to avoid loading and checkpointing the PaperQA store."
        )
        form.addRow("PaperQA Indexing:", self.settings_paperqa_index_enabled)

        self.settings_paperqa_batch_size = QSpinBox()
        self.settings_paperqa_batch_size.setRange(1, 10000)
        self.settings_paperqa_batch_size.setValue(100)
        self.settings_paperqa_batch_size.setToolTip(
            "Checkpoint the PaperQA document store after this many input rows."
        )
        form.addRow("PaperQA Checkpoint Batch:", self.settings_paperqa_batch_size)

        self.settings_fulltext_streaming.toggled.connect(
            lambda enabled: self.settings_fulltext_batch_size.setDisabled(enabled)
        )
        self.settings_paperqa_index_enabled.toggled.connect(
            self.settings_paperqa_batch_size.setEnabled
        )
        self.settings_fulltext_batch_size.setDisabled(
            self.settings_fulltext_streaming.isChecked()
        )

        # Stage 2 advanced-LLM toggle. Default: off, which locks Stage 2 to
        # the bespoke HouseCatBERT relevance model and hides the provider /
        # model dropdowns. Enable to expose Ollama / LM Studio / OpenAI /
        # Anthropic options for relevance scoring.
        self.settings_show_score_advanced = QCheckBox(
            "Show advanced LLM options for Stage 2 (relevance scoring)"
        )
        self.settings_show_score_advanced.setToolTip(
            "When unchecked (default), Stage 2 always uses the bespoke "
            "HouseCatBERT-SciBERT model. Check this to expose the LLM "
            "provider / model dropdowns and override the default."
        )
        form.addRow("", self.settings_show_score_advanced)

        return grp

    # ── Populate from config ───────────────────────────────────────────
