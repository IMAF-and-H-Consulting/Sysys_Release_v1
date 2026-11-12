"""Help text for the Evidence Map, Results Dashboard, Neuroimaging, Compare Runs
and Report & Export tabs.

Part of ``gui.help_content``; no project imports.
"""

EVIDENCE_MAP_HELP = {
    "title": "Evidence Map",
    "icon": "\U0001f5fa️",
    "what": (
        "The evidence table lists, for each article and hypothesis component, "
        "the score, the evidence direction (supports, contradicts, neutral or "
        "tangential) and an excerpt. Load Evidence Table reads the live keyword "
        "analysis when one has run in this session, otherwise the Stage 5 "
        "claims (claims_filtered.json) or the Stage 3 claims (claims.json). "
        "Filter by direction or component to focus on part of the evidence."
    ),
    "why": (
        "An evidence map shows where the literature is dense and where it is "
        "sparse. Gaps—components with little or no evidence—are as "
        "important as what is known: they tell you where future research is "
        "needed."
    ),
    "outputs": (
        "The status line names the source the table was loaded from. Click a "
        "row to see the full evidence text in the detail panel below the table."
    ),
    "reading": [
        {
            "text": "Miake-Lye et al. (2016) – What is an evidence map?",
            "url": "https://doi.org/10.1186/s13643-016-0204-x",
        },
        {
            "text": "Cochrane Handbook Ch. 12.2: Approaches to synthesis without meta-analysis",
            "url": "https://training.cochrane.org/handbook/current/chapter-12#section-12-2",
        },
        {
            "text": "Snilstveit et al. (2016) – Evidence & gap maps: a tool for evidence synthesis",
            "url": "https://doi.org/10.4073/cmdp.2016.3",
        },
    ],
}

RESULTS_DASHBOARD_HELP = {
    "title": "Results Dashboard & GRADE Assessment",
    "icon": "\U0001f4ca",
    "what": (
        "The dashboard shows each hypothesis component’s support score from "
        "the keyword analysis, from −1 (strong contradiction) through 0 to "
        "+1 (strong support). The cards show total and analysed articles, "
        "supporting and contradicting evidence, and the overall score (the mean "
        "component score). Without alignment results they show — rather "
        "than a misleading 0.\n\n"
        "Run GRADE Assessment rates the certainty of the evidence for each "
        "component across five domains: risk of bias, inconsistency, "
        "indirectness, imprecision and publication bias. It reads the files the "
        "stages wrote (claims, method profiles and robustness results)."
    ),
    "why": (
        "Support and certainty are different questions. A high support score "
        "with Low GRADE certainty means the literature agrees but the evidence "
        "may not be trustworthy—for example, agreement from many case "
        "reports. Certainty starts at High for bodies of evidence dominated by "
        "RCTs or systematic reviews and Low for observational studies; each "
        "domain can downgrade it by one or two levels, giving High, Moderate, "
        "Low or Very Low."
    ),
    "outputs": (
        "After GRADE runs, a certainty badge appears beside each component bar: "
        "green (High), amber (Moderate), orange (Low) or red (Very Low). The "
        "Summary of Findings table colours each domain: green for no concern, "
        "yellow for serious, red for very serious and grey for a domain that "
        "could not be assessed (no judgement, not ‘fine’). Click a row "
        "to see the rationale for each domain rating."
    ),
    "reading": [
        {
            "text": "Cochrane Handbook Ch. 14: Completing ‘Summary of findings’ tables",
            "url": "https://training.cochrane.org/handbook/current/chapter-14",
        },
        {
            "text": "Guyatt et al. (2011) – GRADE guidelines: rating quality of evidence",
            "url": "https://doi.org/10.1016/j.jclinepi.2010.07.015",
        },
        {
            "text": "GRADE Working Group – Official website and resources",
            "url": "https://www.gradeworkinggroup.org/",
        },
    ],
}

NEUROIMAGING_HELP = {
    "title": "Neuroimaging Analysis",
    "icon": "\U0001f9e0",
    "what": (
        "(1) Coordinate Extraction scans the full-text articles for brain "
        "coordinates reported in MNI or Talairach space (Talairach peaks are "
        "converted to MNI) and exports them for coordinate-based meta-analysis. "
        "(2) NeuroQuery Cross-Reference queries the NeuroQuery engine with each "
        "component’s keywords and compares the predicted regions with the "
        "regions observed in your claims. Observed regions come only from the "
        "resolved claims file (Stage 5 claims_filtered.json when present), so "
        "papers the audit removed do not count. (3) Network Visualization draws "
        "condition–brain-region co-occurrence. Extraction and NeuroQuery "
        "run in the background, so the window stays responsive."
    ),
    "why": (
        "Text-based reviews capture which regions are mentioned but lose "
        "spatial precision. Coordinate-based meta-analysis (ALE, MKDA) uses the "
        "reported stereotaxic peaks to test whether activations converge beyond "
        "chance. The NeuroQuery check adds an independent, data-driven prior: if "
        "the literature implicates regions NeuroQuery also predicts for your "
        "terms, confidence in the spatial claim increases."
    ),
    "outputs": (
        "The coordinate table lists each peak (article, x, y, z, space, region, "
        "statistic, sample size); export as NiMARE JSON, GingerALE/Sleuth text "
        "or CSV. The convergence table has one row per component: the score "
        "(0–1) is the share of NeuroQuery-predicted regions observed in "
        "your corpus, and scores of 0.3 or more suggest meaningful agreement. "
        "‘n/a’ means NeuroQuery gave no usable prediction, which is "
        "not the same as no agreement. Select a row for the matched, "
        "predicted-only and observed-only regions. The network graph shows "
        "condition nodes and brain-region nodes with edge thickness "
        "proportional to co-occurrence; click a node to highlight its links."
    ),
    "reading": [
        {
            "text": "Eickhoff et al. (2012) – ALE meta-analysis",
            "url": "https://doi.org/10.1016/j.neuroimage.2011.09.017",
        },
        {
            "text": "Dockès et al. (2020) – NeuroQuery",
            "url": "https://doi.org/10.7554/eLife.53385",
        },
        {
            "text": "Salo et al. (2022) – NiMARE",
            "url": "https://doi.org/10.55458/neurolibre.00007",
        },
    ],
}

COMPARE_RUNS_HELP = {
    "title": "Stage 10: Compare Runs & Living Reviews",
    "icon": "\U0001f504",
    "what": (
        "Manual Run Comparison takes the claims files of two runs (for example "
        "with a different model, threshold or search) and measures how stable "
        "the pipeline is: overlap of the paper sets (Jaccard), agreement on "
        "relevance scores (weighted kappa, exact and within-1 agreement), "
        "agreement on support for the research question (kappa) and overlap of "
        "the extracted conditions, mechanisms, findings and entities. No LLM is "
        "used.\n\n"
        "Living Review Snapshots freeze the current pipeline state (corpus, "
        "claim counts, stage completion, file hashes). Select two snapshots and "
        "click ‘Compare Selected’ to see what changed: articles added "
        "or removed, relevance score changes, claim counts and GRADE changes."
    ),
    "why": (
        "Conclusions should not depend on one arbitrary configuration. If "
        "switching models changes which papers are kept or whether they support "
        "the question, the evidence base may be too thin or ambiguous for firm "
        "conclusions (compare ‘multiverse analysis’). Living reviews "
        "must document what changed at each update; snapshots record that."
    ),
    "outputs": (
        "The comparison report lists per-run summaries, the pairwise metrics "
        "above and an overall stability score with its interpretation; it is "
        "saved to the Output file. The snapshot diff is shown in the tab and "
        "saved as snapshot_diff_report.md in the output folder; if saving fails "
        "the status line and log say so."
    ),
    "reading": [
        {
            "text": "Steegen et al. (2016) – Increasing transparency through a multiverse analysis",
            "url": "https://doi.org/10.1177/1745691616658637",
        },
        {
            "text": "Page et al. (2021) – PRISMA 2020 statement: Item 24, sensitivity analyses",
            "url": "https://doi.org/10.1136/bmj.n71",
        },
        {
            "text": "Elliott et al. (2017) – Living systematic reviews",
            "url": "https://doi.org/10.1371/journal.pmed.1002260",
        },
    ],
}

REPORT_EXPORT_HELP = {
    "title": "Report & Export",
    "icon": "\U0001f4cb",
    "what": (
        "Generate PRISMA Report builds a PRISMA 2020 flow diagram and checklist "
        "from the stages’ output files: records identified (from the search "
        "manifest), screened, excluded by Stage 2 and, when exclusion is on, by "
        "the Stage 5 audit, and included. Boxes that do not reconcile are "
        "flagged. The checklist covers the 27 PRISMA 2020 items plus four "
        "PRISMA-trAIce items on AI use, roles, human oversight and performance.\n\n"
        "Export Bibliography writes the corpus as BibTeX (.bib), RIS (.ris) or "
        "EndNote XML (.xml). Export Full Report Bundle copies every artifact the "
        "pipeline produced into one folder with a README. Committee Brief and "
        "Info-Sheet exports produce committee-ready summaries."
    ),
    "why": (
        "PRISMA compliance is a de facto requirement for publishing systematic "
        "reviews, and the 2020 update asks for reporting of automation tools. "
        "PRISMA-trAIce adds disclosure of which AI models were used, what they "
        "decided, where humans intervened and how AI performance was checked. "
        "Taking the counts from each stage’s files avoids hand-tallying."
    ),
    "outputs": (
        "Checklist items the pipeline can fill are marked ‘auto’; the "
        "rest are ‘manual’ with editable cells. Text you type in "
        "manual cells is kept when you regenerate the report. Export PRISMA "
        "Report saves the flow diagram (PNG and SVG) and the checklist "
        "(Markdown and CSV). BibTeX keys are the first author’s surname "
        "plus year, with a, b… suffixes for duplicates."
    ),
    "reading": [
        {
            "text": "Page et al. (2021) – PRISMA 2020 statement",
            "url": "https://doi.org/10.1136/bmj.n71",
        },
        {
            "text": "PRISMA-trAIce – Transparent Reporting of AI in Evidence Synthesis",
            "url": "https://ai.jmir.org/2025/1/e80247",
        },
        {
            "text": "PRISMA 2020 Flow Diagram",
            "url": "http://www.prisma-statement.org/PRISMAStatement/FlowDiagram",
        },
    ],
}

RESULTS_HELP = {
    "evidence_map": EVIDENCE_MAP_HELP,
    "results_dashboard": RESULTS_DASHBOARD_HELP,
    "neuroimaging": NEUROIMAGING_HELP,
    "compare_runs": COMPARE_RUNS_HELP,
    "report_export": REPORT_EXPORT_HELP,
}
