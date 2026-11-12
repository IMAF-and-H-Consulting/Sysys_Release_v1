"""
Help content for the SystS GUI.

``HELP_CONTENT`` maps each help key (the ``_add_help_button`` /
``_section_help_button`` keys) to its entry; the entries live in per-tab
modules. ``WELCOME_CONTENT`` feeds the welcome dialog. No project imports
outside ``gui.help_content_*`` to avoid circular dependencies.
"""
from gui.help_content_analysis import ANALYSIS_HELP
from gui.help_content_results import RESULTS_HELP
from gui.help_content_screening import SCREENING_HELP

HELP_CONTENT = {**SCREENING_HELP, **ANALYSIS_HELP, **RESULTS_HELP}


# ── Welcome dialog content ─────────────────────────────────────────────

WELCOME_CONTENT = {
    "title": "Welcome to Sysys",
    "overview": (
        "Sysys (Systematic Evidence Synthesis) is a desktop application "
        "that orchestrates a multi-stage research pipeline for assessing "
        "how well published academic literature supports a specific "
        "hypothesis. It combines automated literature searching, LLM-"
        "assisted relevance scoring, structured data extraction, evidence "
        "auditing, and robustness analysis into a single reproducible "
        "workflow\u2014all wrapped in a classic Windows 98 interface."
    ),
    "pipeline_stages": [
        "Search \u2013 Query multiple databases and de-duplicate results (Stage 1)",
        "Score & Filter \u2013 Relevance screening and statistical evidence "
        "extraction (Stages 2\u20132c)",
        "Extract & Validate \u2013 Structured claim extraction and study design "
        "classification (Stages 3\u20134)",
        "Audit \u2013 Check that quoted evidence is grounded in each abstract (Stage 5)",
        "Synthesis \u2013 Cluster narratives and abstract draft (Stage 6)",
        "Fulltext \u2013 Full-text retrieval and metric extraction (Stage 7)",
        "Robustness & Verify \u2013 Robustness checks, abstract verification and "
        "meta-analysis (Stages 8\u20139)",
        "Text Scaling \u2013 Wordfish scaling and stylometric profiling",
        "Evidence Map \u2013 Article \u00d7 component evidence table",
        "Results Dashboard \u2013 Support scores and GRADE certainty per component",
        "Neuroimaging \u2013 Coordinate extraction, NeuroQuery convergence, and network visualization",
        "Compare Runs \u2013 Run-to-run stability and living-review snapshots (Stage 10)",
        "Report & Export \u2013 PRISMA 2020 flow diagram, checklist, and AI transparency reporting",
        "Corpus \u2013 Browse, filter and export the corpus",
        "Log window (File menu) \u2013 Real-time execution log",
    ],
    "getting_started": [
        "Open File \u2192 Settings (Ctrl+,) to configure your API keys and LLM provider.",
        "Edit your hypothesis components in the Search tab, then click \u2018Run Search\u2019.",
        "Work through the tabs left-to-right\u2014each stage builds on the previous one.",
        "Click the \u2018?\u2019 button in any tab to learn what that stage does and why it matters.",
        "Check the templates/ directory for pre-built configurations: "
        "abct_tech_ai_infosheet.yaml (ABCT AI subcommittee info-sheet), "
        "abct_ai_digital_mental_health.yaml (ABCT evidence brief), "
        "lead_developmental_cognition.yaml (developmental cognition starter), "
        "autism_ei_balance.yaml (archived autism/E-I example), "
        "generic_neuroscience.yaml (broader neuroscience), or "
        "blank_template.yaml (start from scratch). Load a template via "
        "File \u2192 Settings and the config file browser.",
    ],
}
