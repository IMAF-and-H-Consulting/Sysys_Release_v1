"""Keep tabs, stage paths, query components, and per-stage LLM config in sync with settings.

Split from gui/main_window.py; mixed into MainWindow.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import cast

from PySide6.QtWidgets import QComboBox, QLineEdit, QTableWidgetItem

from core.config import (
    PROVIDER_MODEL_FIELD,
    Config,
    custom_api_key_from_env,
    is_longcat_url,
    normalize_provider_name,
)
from core.models import Article
from gui.main_window_common import (
    _paths_resolve_under_root,
    original_question_for_model,
)
from process.custom_endpoint import CustomEndpoint

_AUTO_PATH_ATTR = "_auto_path_text"

STAGE_PATH_FIELDS = (
    "score_corpus_path", "stat_extract_input", "stat_extract_output",
    "extract_corpus", "extract_output", "validate_claims", "validate_output",
    "audit_input", "audit_output", "synth_input", "synth_output",
    "ft_output_path", "sysys_ft_corpus", "robust_input", "robust_output",
    "verify_claims", "compare_run_a", "compare_run_b", "compare_output",
)


def _first_existing(*paths: Path) -> Path | None:
    return next((p for p in paths if p.exists()), None)


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return -1.0


def _score_corpus_target(cfg: Config) -> Path:
    """Stage 2 input: the enriched corpus only while it is newer than the search."""
    corpus = cfg.corpus_dir / "corpus.json"
    enriched = cfg.corpus_dir / "corpus_fulltext.json"
    return enriched if _mtime(enriched) > _mtime(corpus) else corpus


def _custom_key_for_endpoint(cfg: Config) -> str:
    """custom_api_key, minus a LONGCAT_API_KEY resolved for an earlier base URL.

    The env key is resolved once, when Config is built; changing the custom
    base URL away from LongCat afterwards must not send it to the new host.
    """
    longcat_key = os.environ.get("LONGCAT_API_KEY", "")
    if longcat_key and cfg.custom_api_key == longcat_key and not is_longcat_url(cfg.custom_base_url):
        return custom_api_key_from_env(cfg.custom_base_url)
    return cfg.custom_api_key


def _coerce_like(value: object, default: object) -> object:
    """``value`` converted to ``default``'s type, or ``default`` when it cannot be."""
    if isinstance(default, bool):
        return value if isinstance(value, bool) else default
    if isinstance(default, (int, float)) and not isinstance(value, bool):
        try:
            return type(default)(cast(float, value))
        except (TypeError, ValueError):
            return default
    if isinstance(default, str):
        return value if isinstance(value, str) else default
    return value


def _article_from_record(art_id: str, p: dict) -> Article:
    return Article(
        id=art_id,
        title=p.get("title") or "",
        authors=p.get("authors") if isinstance(p.get("authors"), list) else [],
        abstract=p.get("abstract") or "",
        year=p.get("year"),
        doi=p.get("doi") or "",
        pmid=p.get("pmid") or "",
        citations=p.get("citationCount", 0) or p.get("citations", 0) or p.get("citation_count", 0) or 0,
        open_access=bool(
            p.get("open_access")
            or p.get("openalex_is_oa")
            or p.get("s2_open_access_pdf_url")
        ),
        full_text_url=(
            p.get("openalex_best_oa_pdf_url")
            or p.get("openalex_oa_url")
            or p.get("s2_open_access_pdf_url")
            or ""
        ),
    )


def articles_from_corpus(corpus: list) -> dict[str, Article]:
    """Article per corpus record; records without a DOI/PMID get unique ids (18-L1)."""
    articles: dict[str, Article] = {}
    for index, p in enumerate(corpus):
        if not isinstance(p, dict):
            continue
        art_id = str(p.get("doi") or p.get("pmid") or "")
        if not art_id:
            art_id = str(p.get("title") or "")[:60] or f"record-{index + 1}"
            if art_id in articles:
                art_id = f"{art_id} #{index + 1}"
        articles[art_id] = _article_from_record(art_id, p)
    return articles


def stage_path_targets(cfg: Config) -> list[tuple[str, Path | None]]:
    """(field name, standard file) for every auto-filled stage path field.

    Inputs name the newest upstream output that exists, mirroring the
    launchers' defaults, so Run All chains each stage to the previous one.
    """
    claims_dir = cfg.claims_dir
    relevant = claims_dir / "relevant.json"
    claims = claims_dir / "claims.json"
    filtered = claims_dir / "claims_filtered.json"
    synth_in = _first_existing(filtered, claims)
    analysed = _first_existing(filtered, claims, relevant)
    return [
        ("score_corpus_path", _score_corpus_target(cfg)),
        ("stat_extract_input", relevant),
        ("stat_extract_output", claims_dir / "statistical_extractions.json"),
        ("extract_corpus", relevant),
        ("extract_output", claims),
        ("validate_claims", _first_existing(claims, relevant)),
        ("validate_output", cfg.data_dir / "method_review" / "method_profiles_validated.json"),
        ("audit_input", claims),
        ("audit_output", filtered),
        ("synth_input", synth_in),
        ("synth_output", cfg.clusters_dir / "clusters.json"),
        ("ft_output_path", cfg.data_dir / "fulltext" / "auto_extracted_metrics.json"),
        ("sysys_ft_corpus", synth_in),
        ("robust_input", analysed),
        ("robust_output", cfg.data_dir / "validation" / "robustness_results.json"),
        ("verify_claims", analysed),
    ]


class ConfigSyncMixin:
    """Keep tabs, stage paths, query components, and per-stage LLM config in sync with settings."""

    # ── Hypothesis component management ───────────────────────────────

    def _refresh_component_list(self):
        from analysis.scoring import (
            build_boolean_query_plan, describe_boolean_query_plan, quote_boolean_term,
        )

        self.component_list.blockSignals(True)
        self.component_list.setRowCount(len(self.hypothesis_components))
        for i, comp in enumerate(self.hypothesis_components):
            self.component_list.setItem(i, 0, QTableWidgetItem(comp.id))
            self.component_list.setItem(i, 1, QTableWidgetItem(comp.label))
            self.component_list.setItem(i, 2, QTableWidgetItem(", ".join(comp.keywords)))
            self.component_list.setItem(i, 3, QTableWidgetItem(str(comp.weight)))
            terms = [quote_boolean_term(term) for term in (comp.keywords or [comp.label])]
            terms = [term for term in terms if term]
            self.component_list.setItem(i, 4, QTableWidgetItem(" OR ".join(terms[:4])))
        self.component_list.blockSignals(False)
        if hasattr(self, "boolean_query_preview"):
            plan = build_boolean_query_plan(self.hypothesis_components)
            self.boolean_query_preview.setPlainText(describe_boolean_query_plan(plan))

    def _sync_query_components_from_question(self, query: str):
        """Create or refresh weighted hypothesis/query components."""
        from analysis.scoring import build_hypothesis_components, merge_component_edits

        if self.stat_extract_llm_context.isChecked():
            cfg = self._apply_config_overrides_for_stage(
                self.stat_extract_provider, self.stat_extract_model)
        else:
            cfg = self.pipeline_config
        components = build_hypothesis_components(
            hypothesis=getattr(cfg, "hypothesis_text", "") or query,
            research_config=getattr(cfg, "_research_config", {}) or {},
            co_occurring_conditions=getattr(cfg, "co_occurring_conditions", []) or [],
            query=query,
        )
        # Keep the user's weights, keywords and descriptions across a rebuild.
        components = merge_component_edits(components, self.hypothesis_components)

        if components:
            self.hypothesis_components = components
            self._refresh_component_list()
            self._update_search_workflow_buttons()
            self._log(
                f"Hypothesis parts prepared for keyword weighting: {len(components)} components"
            )

    def _update_search_workflow_buttons(self):
        """Gate Search controls around the search → download → analysis workflow."""
        # Fetch Articles is a download-only action, so it needs an existing
        # corpus (in memory or a saved corpus.json), not just a question.
        has_corpus = bool(self._pipeline_corpus) or bool(self.articles)
        if not has_corpus and getattr(self, "pipeline_config", None) is not None:
            try:
                has_corpus = (self.pipeline_config.corpus_dir / "corpus.json").exists()
            except Exception:
                has_corpus = False
        if hasattr(self, "fetch_btn"):
            self.fetch_btn.setEnabled(has_corpus)
        if hasattr(self, "analysis_btn"):
            has_results = bool(self.articles or self._pipeline_corpus)
            worker = getattr(self, "_analysis_worker", None)
            analysis_running = worker is not None and worker.isRunning()
            self.analysis_btn.setEnabled(
                bool(self._keyword_analysis_ready and has_results and not analysis_running)
            )

    def _edit_component(self):
        row = self.component_list.currentRow()
        if row < 0 or row >= len(self.hypothesis_components):
            return
        from gui.dialogs import ComponentEditor
        comp = self.hypothesis_components[row]
        dlg = ComponentEditor(comp, parent=self)
        if dlg.exec():
            self._refresh_component_list()

    def _populate_components_from_config(self):
        """Build HypothesisComponent objects from pipeline_config fields."""
        from core.research_components import build_hypothesis_components_from_config

        components = build_hypothesis_components_from_config(self.pipeline_config)

        if components:
            self.hypothesis_components = components
            self._refresh_component_list()
            self._update_search_workflow_buttons()
            self._log(f"Loaded {len(components)} hypothesis components from config")

    def _reset_components(self):
        """Replace the components with exactly what the config defines (18-L10)."""
        from core.research_components import build_hypothesis_components_from_config

        self.hypothesis_components = build_hypothesis_components_from_config(self.pipeline_config)
        self._refresh_component_list()
        self._update_search_workflow_buttons()
        count = len(self.hypothesis_components)
        self._log(
            f"Components reset from config: {count}" if count
            else "Components cleared: the config defines none"
        )

    # ── Persisted keys ────────────────────────────────────────────────────

    def _restore_saved_keys(self, file_fields: frozenset[str] = frozenset(), source: str = ""):
        """Restore API keys and LLM settings from QSettings into pipeline_config.

        API keys only fill empty fields after Config.__post_init__ so env vars
        keep priority. LLM settings are user preferences, so saved provider/model
        choices override non-empty dataclass defaults such as OpenAI. When a
        preference replaces a value ``source`` set explicitly (``file_fields``),
        that is logged (18-M3).
        """
        from PySide6.QtCore import QSettings

        from gui.settings_persistence import restore_saved_settings

        s = QSettings("Sysys", "HypothesisConsensusAnalyzer")
        changed = restore_saved_settings(self.pipeline_config, s)
        overridden = [name for name in changed if name in file_fields]
        if overridden:
            values = ", ".join(f"{n}={getattr(self.pipeline_config, n)!r}" for n in overridden)
            self._log(
                f"Saved LLM preferences override {source or 'the config file'}: {values}"
            )

    # ── Sync settings → tab widgets ────────────────────────────────────────

    # Stage file-path fields. A field is "auto-filled" while its text equals
    # the last value the sync wrote (or is empty); anything else — typed,
    # chosen via Browse..., or restored from project state — is a user
    # override. Auto-filled fields always follow the newest standard output,
    # so stages chain to the previous stage's actual result.

    def _stage_path_edits(self) -> list[QLineEdit]:
        return [getattr(self, name) for name in STAGE_PATH_FIELDS if hasattr(self, name)]

    @staticmethod
    def _is_auto_stage_path(edit: QLineEdit) -> bool:
        cur = edit.text().strip()
        return not cur or cur == getattr(edit, _AUTO_PATH_ATTR, None)

    @staticmethod
    def _set_auto_stage_path(edit: QLineEdit, text: str) -> None:
        edit.setText(text)
        setattr(edit, _AUTO_PATH_ATTR, text.strip())

    def _reset_stage_path_fields(self) -> list[str]:
        """Clear every stage path field; return the user overrides dropped."""
        dropped = [
            edit.text().strip() for edit in self._stage_path_edits()
            if not self._is_auto_stage_path(edit)
        ]
        for edit in self._stage_path_edits():
            self._set_auto_stage_path(edit, "")
        return dropped

    def _sync_stage_path_line_edit(self, edit: QLineEdit, path: Path | None) -> None:
        """Point an auto-filled field at ``path`` (cleared when it does not exist).

        User overrides are left alone; project switches and Run All reset
        them explicitly via ``_reset_stage_path_fields``.
        """
        if not self._is_auto_stage_path(edit):
            return
        try:
            exists = path is not None and path.exists()
        except OSError:
            exists = False
        self._set_auto_stage_path(edit, str(path) if exists else "")

    def _sync_compare_run_paths(self, cfg: Config) -> None:
        """Pre-fill manual compare inputs from the current data layout."""
        claims_p = cfg.claims_dir / "claims.json"
        filt_p = cfg.claims_dir / "claims_filtered.json"
        rel_p = cfg.corpus_dir / "relevant.json"
        out_md = cfg.output_dir / "run_compare_report.md"

        run_a = _first_existing(claims_p, rel_p)
        self._sync_stage_path_line_edit(self.compare_run_a, run_a)
        run_b = _first_existing(filt_p, claims_p, rel_p)
        if run_b == claims_p and self._compare_run_a_is(claims_p, cfg):
            run_b = None
        self._sync_stage_path_line_edit(self.compare_run_b, run_b)

        if self._is_auto_stage_path(self.compare_output):
            self._set_auto_stage_path(self.compare_output, str(out_md))

    def _compare_run_a_is(self, path: Path, cfg: Config) -> bool:
        a_res = _paths_resolve_under_root(self.compare_run_a.text(), cfg.project_root)
        try:
            return bool(a_res) and a_res.resolve() == path.resolve()
        except OSError:
            return False

    def _sync_stage_path_fields(self, cfg: Config) -> None:
        """Align every auto-filled stage path field with the newest outputs."""
        for name, target in stage_path_targets(cfg):
            if hasattr(self, name):
                self._sync_stage_path_line_edit(getattr(self, name), target)
        if hasattr(self, "compare_run_a"):
            self._sync_compare_run_paths(cfg)

    def _sync_all_stage_paths(self, cfg: Config) -> None:
        """Align stage file path fields, then load existing run data from disk."""
        self._sync_stage_path_fields(cfg)
        self._load_existing_run_data(cfg)

    def _load_existing_run_data(self, cfg) -> None:
        """Load previous pipeline outputs from disk into the GUI state.

        Checks for corpus.json, relevant.json, claims.json etc. and populates
        the articles dict, glance panel, corpus browser, and result displays.
        A file already loaded and unchanged since is not parsed again, so one
        project open reads each file once (18-M7).
        """
        corpus_path = cfg.corpus_dir / "corpus.json"
        if corpus_path.exists() and not self._corpus_already_loaded(corpus_path):
            if self._load_run_corpus(corpus_path):
                # The browser was rebuilt; relevance marks must be redrawn.
                self._run_data_seen().clear()
                self._run_file_unchanged("corpus", corpus_path)

        self._load_run_artifact("relevant", cfg.claims_dir / "relevant.json", self._show_run_relevant)
        self._load_existing_human_relevance_review(cfg)
        self._load_run_artifact(
            "stats", cfg.claims_dir / "statistical_extractions.json", self._show_run_stats)
        self._load_run_artifact("claims", cfg.claims_dir / "claims.json", self._show_run_claims)
        self._load_run_artifact(
            "filtered", cfg.claims_dir / "claims_filtered.json", self._show_run_filtered)
        self._load_run_artifact(
            "robustness", cfg.data_dir / "validation" / "robustness_results.json",
            self._show_run_robustness)
        self._refresh_pipeline_progress_from_disk(cfg)

    def _run_data_seen(self) -> dict:
        seen = getattr(self, "_run_data_markers", None)
        if seen is None:
            seen = {}
            self._run_data_markers = seen
        return seen

    def _run_file_unchanged(self, key: str, path: Path) -> bool:
        """True when ``path`` is unchanged since the last load under ``key``."""
        seen = self._run_data_seen()
        try:
            stat = path.stat()
        except OSError:
            seen.pop(key, None)
            return False
        marker = (str(path), stat.st_mtime_ns, stat.st_size)
        if seen.get(key) == marker:
            return True
        seen[key] = marker
        return False

    def _corpus_already_loaded(self, path: Path) -> bool:
        unchanged = self._run_file_unchanged("corpus", path)
        return unchanged and self._pipeline_corpus is getattr(self, "_run_data_corpus", None)

    def _read_run_json(self, path: Path) -> object:
        """Parsed JSON, or None (logged) when the file cannot be read."""
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._log(f"Could not load {path.name}: {exc}", "WARNING")
            return None

    def _load_run_corpus(self, corpus_path: Path) -> bool:
        corpus = self._read_run_json(corpus_path)
        if not isinstance(corpus, list) or not corpus:
            return False
        # Build into a local first: a bad record must not leave the window
        # with a replaced corpus and a half-built articles dict.
        articles = articles_from_corpus(corpus)
        self._pipeline_corpus = corpus
        self._run_data_corpus = corpus
        self.articles = articles
        self._update_glance_corpus(len(corpus))
        self._set_tab_badge(0, "Search", len(corpus))
        if hasattr(self, '_populate_corpus_browser'):
            self._populate_corpus_browser(corpus)
        self._log(f"Loaded existing corpus: {len(corpus)} papers from {corpus_path}")
        return True

    def _load_run_artifact(self, key: str, path: Path, show) -> None:
        if not path.exists() or self._run_file_unchanged(key, path):
            return
        data = self._read_run_json(path)
        if data is not None:
            show(data, path)

    def _show_run_relevant(self, relevant: object, path: Path) -> None:
        if not isinstance(relevant, list):
            return
        if hasattr(self, "_set_corpus_stage_relevance"):
            self._set_corpus_stage_relevance("stage2", relevant)
        self._update_glance_relevant(len(relevant))
        self._set_tab_badge(1, "Score && Filter", len(relevant))
        self._update_glance_stage("Score done")
        self._log(f"Loaded existing scored papers: {len(relevant)} from {path}")

    def _show_run_stats(self, stats_artifact: object, _path: Path) -> None:
        if not isinstance(stats_artifact, dict):
            return
        summary = stats_artifact.get("summary") or {}
        self._log(
            "Loaded statistical review: "
            f"{summary.get('results', 0)} results, "
            f"{summary.get('human_review_items', 0)} review items")

    def _show_run_claims(self, claims: object, path: Path) -> None:
        if isinstance(claims, list):
            self._log(f"Loaded existing claims: {len(claims)} from {path}")
            self._update_glance_stage("Extract done")

    def _show_run_filtered(self, filtered: object, path: Path) -> None:
        if isinstance(filtered, list):
            self._log(f"Loaded existing filtered claims: {len(filtered)} from {path}")
            self._update_glance_stage("Audit done")

    def _show_run_robustness(self, robustness: object, _path: Path) -> None:
        if isinstance(robustness, dict) and robustness.get("n_papers"):
            self._log(f"Loaded robustness results: {robustness.get('n_papers')} papers analysed")
            self._update_glance_stage("Robustness done")

    def _sync_fulltext_controls_to_config(self, *_args) -> None:
        """Persist visible full-text controls into the active Config object."""
        cfg = self.pipeline_config
        if hasattr(self, "sysys_ft_workers"):
            cfg.fulltext_download_workers = self.sysys_ft_workers.value()
        if hasattr(self, "sysys_ft_streaming"):
            cfg.fulltext_streaming = self.sysys_ft_streaming.isChecked()
        if hasattr(self, "sysys_ft_batch_size"):
            cfg.fulltext_download_batch_size = self.sysys_ft_batch_size.value()
        if hasattr(self, "sysys_ft_paperqa_enabled"):
            cfg.paperqa_index_enabled = self.sysys_ft_paperqa_enabled.isChecked()
        if hasattr(self, "sysys_ft_paperqa_batch_size"):
            cfg.paperqa_index_batch_size = self.sysys_ft_paperqa_batch_size.value()

    def _restore_screening_project_config(self, state: dict, defaults: Config | None = None) -> bool:
        """Restore screening settings without carrying them over from another project."""
        saved = state.get("config") or {}
        if not isinstance(saved, dict):
            saved = {}
        changed = False
        for name in (
            "relevance_screening_mode", "laya_screener_dir", "laya_device",
            "laya_batch_size", "laya_cut_bin", "laya_brt_threshold",
        ):
            default = (
                getattr(defaults, name) if defaults is not None
                else Config.__dataclass_fields__[name].default
            )
            value = _coerce_like(saved.get(name, default), default)
            changed |= getattr(self.pipeline_config, name) != value
            setattr(self.pipeline_config, name, value)
        return changed

    def _restore_fulltext_project_config(self, state: dict) -> bool:
        """Restore project-scoped full-text settings saved in project_state."""
        saved = state.get("config") or {}
        if not isinstance(saved, dict):
            return False
        restored = False
        bool_fields = ("fulltext_streaming", "paperqa_index_enabled")
        int_fields = (
            "fulltext_download_workers",
            "fulltext_download_batch_size",
            "paperqa_index_batch_size",
        )
        for field in bool_fields:
            value = saved.get(field)
            if isinstance(value, bool):
                setattr(self.pipeline_config, field, value)
                restored = True
        for field in int_fields:
            value = saved.get(field)
            if value is None:
                continue
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if parsed >= 1:
                setattr(self.pipeline_config, field, parsed)
                restored = True
        return restored

    def _sync_settings_to_tabs(self, force: bool = False):
        """Push pipeline_config values into all per-stage tab widgets."""
        cfg = self.pipeline_config
        provider = cfg.llm_provider

        # Determine the model string for the active provider
        model_for_provider = {
            "ollama": cfg.ollama_model,
            "custom": cfg.custom_model,
            "openai": cfg.openai_model,
            "anthropic": cfg.anthropic_model,
        }
        model = model_for_provider.get(provider, "")

        # Helper to set a provider combo + model edit pair.
        # Normal project/open syncs preserve hand-edited stage rows. Preferences
        # Apply/Save passes force=True so the chosen provider/model becomes the
        # new baseline everywhere, including rows previously edited by hand.
        def _set_llm(combo: QComboBox, model_edit: QLineEdit):
            if force or not getattr(combo, "_user_customized", False):
                idx = combo.findText(provider)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
                if force:
                    combo._user_customized = False
            if force or (model and not getattr(model_edit, "_user_customized", False)):
                model_edit.setText(model)
                if force:
                    model_edit._user_customized = False

        # Tab 0: Search — natural language question from research config
        nlq = cfg.get_natural_language_question()
        if nlq and not self.sysys_nl_query.text().strip():
            self.sysys_nl_query.setText(nlq)

        # Tab 0: Search — full-text preference from research config
        ft_cfg = cfg.get_full_text_config()
        if ft_cfg.get("keep_only_full_text"):
            self.fulltext_only.setChecked(True)
        if ft_cfg.get("enabled"):
            self.fetch_fulltext.setChecked(True)

        # Tab 1: Score & Filter
        relevance_model = (getattr(cfg, "relevance_model", "") or "").strip()
        if not force and relevance_model.lower().startswith(("brt", "housecatbert")):
            if not getattr(self.score_provider, "_user_customized", False):
                idx = self.score_provider.findText("brt-bert")
                if idx >= 0:
                    self.score_provider.setCurrentIndex(idx)
            if not getattr(self.score_model, "_user_customized", False):
                self.score_model.setText(relevance_model)
        else:
            _set_llm(self.score_provider, self.score_model)
        if hasattr(self, "score_relevance_model") and (
            force or not getattr(self.score_relevance_model, "_user_customized", False)
        ):
            self.score_relevance_model.setCurrentText(
                getattr(cfg, "relevance_model", "") or "brt-bert"
            )
            if force:
                self.score_relevance_model._user_customized = False

        self.score_threshold.setValue(cfg.relevance_threshold)
        if hasattr(self, "citation_snowball"):
            self.citation_snowball.setChecked(bool(getattr(cfg, "citation_snowball_enabled", True)))
        if hasattr(self, "score_screening_mode"):
            self.score_screening_mode.setCurrentIndex(max(0, self.score_screening_mode.findData(
                getattr(cfg, "relevance_screening_mode", "off"))))
        self.score_batch.setValue(
            cfg.score_batch_size if getattr(cfg, "relevance_screening_mode", "off") == "off"
            else cfg.laya_batch_size)
        if hasattr(self, "score_screening_mode"):
            self._on_screening_strategy_changed()
        if hasattr(self, "stat_extract_provider"):
            _set_llm(self.stat_extract_provider, self.stat_extract_model)

        # Tab 2: Extract & Validate
        _set_llm(self.extract_provider, self.extract_model)
        self.extract_max_tokens.setValue(getattr(cfg, "claim_max_tokens", 0) or 0)
        llm_sc = cfg.get_llm_scoring_config()
        if llm_sc.get("use_reasoning") is not None:
            self.extract_thinking.setChecked(bool(llm_sc["use_reasoning"]))
        _set_llm(self.validate_provider, self.validate_model)

        # Tab 3: Audit & Synthesize
        _set_llm(self.synth_provider, self.synth_model)
        syn_cfg = cfg.get_synthesis_config()
        if syn_cfg.get("n_clusters"):
            self.synth_min_cluster.setValue(syn_cfg["n_clusters"])
        if hasattr(self, "synth_max_tokens"):
            self.synth_max_tokens.setValue(
                getattr(cfg, "synthesis_cluster_max_tokens", 0) or 0
            )
        if hasattr(self, "meta_context_tokens"):
            self.meta_context_tokens.setValue(
                getattr(cfg, "llm_context_tokens", 0) or 0
            )

        # Tab 4: Fulltext
        ft_workers = cfg.fulltext_download_workers
        ft_streaming = bool(cfg.fulltext_streaming)
        ft_batch_size = cfg.fulltext_download_batch_size
        paperqa_enabled = bool(cfg.paperqa_index_enabled)
        paperqa_batch_size = cfg.paperqa_index_batch_size
        if hasattr(self, 'sysys_ft_workers'):
            self.sysys_ft_workers.setValue(ft_workers)
        if hasattr(self, "sysys_ft_streaming"):
            self.sysys_ft_streaming.setChecked(ft_streaming)
        if hasattr(self, "sysys_ft_batch_size"):
            self.sysys_ft_batch_size.setValue(ft_batch_size)
            self.sysys_ft_batch_size.setDisabled(ft_streaming)
        if hasattr(self, "sysys_ft_paperqa_enabled"):
            self.sysys_ft_paperqa_enabled.setChecked(paperqa_enabled)
        if hasattr(self, "sysys_ft_paperqa_batch_size"):
            self.sysys_ft_paperqa_batch_size.setValue(paperqa_batch_size)
            self.sysys_ft_paperqa_batch_size.setEnabled(paperqa_enabled)

        # Tab 5: Robustness & Verify has no LLM settings (stats only).

        # All tabs: default paths from current project_root / data layout
        self._sync_all_stage_paths(cfg)

        # Stage 2: apply advanced-LLM visibility from QSettings
        self._apply_score_advanced_mode_from_settings()

        self._log("Settings synced to all stage tabs")

    def _apply_score_advanced_mode_from_settings(self):
        """Read QSettings and toggle Stage 2 advanced-LLM widgets accordingly."""
        from PySide6.QtCore import QSettings
        s = QSettings("Sysys", "HypothesisConsensusAnalyzer")
        show_advanced = cast(bool, s.value("score/show_advanced_llm", False, type=bool))
        self._apply_score_advanced_mode(show_advanced)

    def _apply_score_advanced_mode(self, show_advanced: bool):
        """Show or hide Stage 2's provider/model controls.

        Without a packaged screening strategy, hidden controls select the
        existing BRT-BERT scorer. The cascade selector is always available.
        """
        if hasattr(self, "score_advanced_box"):
            self.score_advanced_box.setVisible(show_advanced)
        if hasattr(self, "score_simple_label"):
            self.score_simple_label.setVisible(
                not show_advanced and getattr(self.pipeline_config, "relevance_screening_mode", "off") == "off")

    # ── Generic pipeline stage infrastructure ─────────────────────────

    def _custom_endpoint(self) -> CustomEndpoint:
        """The custom endpoint from Settings, used by every stage's Fetch button."""
        cfg = self.pipeline_config
        return CustomEndpoint(cfg.custom_api_format, cfg.custom_base_url, _custom_key_for_endpoint(cfg))

    def _apply_config_overrides_for_stage(
        self, provider_combo: QComboBox, model_edit: QLineEdit
    ) -> Config:
        import copy

        cfg = copy.deepcopy(self.pipeline_config)
        custom_key = _custom_key_for_endpoint(cfg)
        if custom_key != cfg.custom_api_key:
            self._log(
                "LONGCAT_API_KEY is only sent to LongCat; set CUSTOM_LLM_API_KEY or a "
                f"key in Settings for {cfg.custom_base_url}", "WARNING")
            cfg.custom_api_key = custom_key
        original = original_question_for_model(self)
        if original:
            # Route the original question through `_research_config` /
            # `get_natural_language_question()` only. Setting
            # `cfg.hypothesis_text` here would leak it into
            # `build_hypothesis_components` (via `_sync_query_components_
            # from_question`) and silently change search behavior whenever
            # LLM Context Check is enabled.
            research_config = dict(getattr(cfg, "_research_config", {}) or {})
            research_config["natural_language_question"] = original
            cfg._research_config = research_config
        prov = provider_combo.currentText()
        model = model_edit.text().strip()
        if prov in ("brt-bert", "brt-irr", "housecatbert-scibert-releevance"):
            cfg.relevance_model = (
                model
                if model.lower().startswith(("brt", "housecatbert"))
                else prov
            )
            return cfg
        prov = normalize_provider_name(prov)
        cfg.llm_provider = prov
        if model:
            setattr(cfg, PROVIDER_MODEL_FIELD.get(prov, "openai_model"), model)
        return cfg
