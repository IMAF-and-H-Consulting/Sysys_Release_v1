"""Help text for the Synthesis, Fulltext, Robustness & Verify and Text Scaling tabs.

Part of ``gui.help_content``; no project imports. Stage numbers follow the
section headers in the GUI.
"""

SYNTHESIS_HELP = {
    "title": "Stage 6: Synthesis",
    "icon": "\U0001f9e9",
    "what": (
        "Stage 6 embeds the audited claims (claims_filtered.json, or claims.json "
        "when Stage 5 has not run), groups papers into clusters (automatically "
        "by silhouette score, or up to the number you set) and asks the LLM for "
        "a narrative per cluster. Each narrative is built from every paper in "
        "the cluster, not a sample. The LLM then drafts a structured abstract "
        "from the cluster narratives. Evidence and response allowances scale "
        "from the loaded model context; oversized clusters are split into "
        "complete requests rather than silently shortened. The 2-3 paragraph "
        "cluster narrative and 350-word abstract are deliberate writing-format "
        "constraints: they keep the synthesis focused and publication-ready, "
        "not artificial limits on how much source evidence may be considered. "
        "Every completed context batch and cluster narrative is saved immediately. "
        "If a later request fails or reaches an output boundary, the stage is marked "
        "PARTIAL and a compatible rerun resumes the saved work instead of repeating or "
        "hiding it."
    ),
    "why": (
        "Narrative synthesis turns a long list of individual claims into a "
        "readable summary of what groups of studies say, similar to the "
        "‘Summary of findings’ tables recommended by GRADE."
    ),
    "outputs": (
        "Completed runs publish clusters.json, narratives.json, k_selection.json and "
        "abstract_draft.md. In-progress or failed runs write clusters.partial.json, "
        "narratives.partial.json, k_selection.partial.json, synthesis_partial.md and "
        "synthesis_progress.json without replacing the last completed run. "
        "hypothesis_explanation.json/.md and evidence_network.json are written "
        "to the clusters folder (or beside the Clusters Output file you choose). "
        "The cluster table lists each cluster with its paper count and key "
        "terms; the sub-tabs show the abstract, the ‘Why This "
        "Hypothesis’ provenance report, per-claim source citations and the "
        "evidence network. The progress artifact records the request fingerprint, "
        "completed clusters, saved context batches, token capacity, and unresolved model "
        "actions. Empty LLM replies or weak cluster structure are reported as warnings "
        "in the log and status line."
    ),
    "reading": [
        {
            "text": "Cochrane Handbook Ch. 12: Synthesizing and presenting findings using other methods",
            "url": "https://training.cochrane.org/handbook/current/chapter-12",
        },
        {
            "text": "Popay et al. (2006) – Guidance on narrative synthesis",
            "url": "https://doi.org/10.13140/2.1.1018.4643",
        },
    ],
}

FULLTEXT_HELP = {
    "title": "Stage 7: Full-Text Enrichment",
    "icon": "\U0001f4da",
    "what": (
        "Retrieve Full Texts checks the loaded articles against Unpaywall (a "
        "contact email must be set in Settings) and downloads open-access copies "
        "to a folder named in the log. Stage 7 enriches retained papers with full "
        "text and extracted metrics. Streaming checkpoints preserve completed "
        "work across cancellation or restart."
    ),
    "why": (
        "Full text provides methods, limitations, and effect-size details "
        "that may be absent from abstracts. Relevance filtering is handled "
        "in Stage 2 by the combined Laya and BRT-BERT screener."
    ),
    "outputs": (
        "The retrieval panel shows article status and errors. Stage 7 saves "
        "auto_extracted_metrics.jsonl and download_log.jsonl checkpoints, "
        "then produces deduplicated JSON arrays for downstream stages. "
        "Run All continues to Stage 8 (Robustness) after full-text enrichment."
    ),
    "reading": [
        {
            "text": "Piwowar et al. (2018) - The state of OA: a large-scale analysis",
            "url": "https://doi.org/10.7717/peerj.4375",
        },
    ],
}

_ROBUSTNESS_WHAT = (
    "Stage 8 (Robustness) measures how stable one outcome is: the share of "
    "papers whose extracted claims support the research question "
    "(supports_primary_question, or the field set in robustness.outcome_field). "
    "It runs bootstrap resampling, split-half reliability of the per-condition "
    "support profile, sensitivity across relevance thresholds 1–5, a "
    "permutation test of whether the co_occurring_conditions ranking tracks "
    "support rates, a condition network, and a per-condition sensitivity check. "
    "Cancel stops it between steps. Stage 9 generates the final review from saved "
    "findings and abstracts, with source references, thematic agreements and conflicts, "
    "available saved robustness and pooled results, and explicit limitations. "
    "Internal checks validate coverage numbers and source references. The report and "
    "complete evidence ledger are saved as final_review_report.md and final_review_sources.json. "
    "A synthesis abstract is not required.\n\n"
    "Neither Stage 8 nor Stage 9 calls an LLM; both run locally and nothing is "
    "sent to any endpoint. When an analysis cannot run (too few papers, no "
    "configured conditions, no abstract, no threshold with papers) the log, the "
    "status line and the report say SKIPPED, not evaluable or NOTHING WAS "
    "VERIFIED instead of showing placeholder numbers.\n\n"
    "Meta-Analytic Synthesis pools the statistics Stage 2c extracted from the "
    "retained papers. Each effect needs a measure of uncertainty, taken from "
    "the first of: its confidence interval, a reported standard error, the "
    "p-value next to the statistic, the result's own n or group ns, r(df), or "
    "(only when meta_analysis_use_paper_sample_size is set in the research "
    "config) the paper-level N. Effects are pooled within compatible groups "
    "(the groups you set in Review Effects & Groups…, then model groups that "
    "passed both semantic gates, otherwise the same measure, outcome, comparison "
    "and population), one paper-level estimate per group (multiple selected "
    "within-paper estimates require explicit subgroup/composite justification), with a "
    "DerSimonian–Laird random-effects model. Ratios and correlations are also "
    "reported on their natural scale. Every run also writes a source-linked "
    "evidence report, including papers without poolable estimates. "
    "Re-check statistics and group effects with model (on by default) asks the "
    "Preferences model about statistics Stage 2c left without a verdict, then "
    "discovers cross-paper candidate groups. It first checks exact paper titles "
    "for research-question relevance and comparability, then reports and checks "
    "construct, measurement definition, population, subgroup independence, "
    "comparison/exposure, design/timepoint, effect direction and scientific "
    "plausibility. Only a candidate that passes every gate is pooled; the full "
    "paper-title/statistic audit is saved with the meta-analysis evidence. "
    "The optional Enhance report with model control uses the Preferences model to "
    "write context-aware sections. Auto-detect reads the actual loaded context from "
    "LM Studio native REST; otherwise set it once in Preferences or enter an override. "
    "Each request records its evidence, response, and safety allocation. Intentional "
    "concision limits state their reason; incomplete responses are evicted and split or "
    "repaired. Completed sections are saved progressively; Stop model work keeps "
    "the local report and completed sections. No final whole-corpus request is made."
)

_ROBUSTNESS_WHY = (
    "A review whose conclusions change when one or two papers are removed is "
    "fragile; the robustness analyses quantify that. The permutation test "
    "gives a p-value for ‘does the condition ranking track support more "
    "than chance?’, and the condition sensitivity check shows whether one "
    "condition’s literature drives the pooled result.\n\n"
    "A random-effects model allows between-study variance (τ²), so the "
    "pooled estimate accounts for heterogeneity. I² describes how much of "
    "the variability reflects real differences between studies: below 25% is "
    "low, 25–75% moderate, above 75% high.\n\n"
    "With 10 or more studies in a group, publication bias is assessed with "
    "Egger’s regression test (funnel asymmetry), Duval & Tweedie "
    "trim-and-fill (estimated missing studies and the adjusted effect) and "
    "Rosenthal’s fail-safe N. With fewer studies these tests have little "
    "power and the summary says so."
)

_ROBUSTNESS_OUTPUTS = (
    "robustness_report.md names the outcome measured and reports the bootstrap "
    "95% confidence interval, the Spearman–Brown corrected split-half "
    "reliability, support rates per relevance threshold, the permutation "
    "test’s Spearman r and p-value with the per-condition gradient, and a "
    "per-condition sensitivity table. Verification results appear as PASS / "
    "FAIL / WARN per number in the abstract.\n\n"
    "For meta-analysis, choose a pooled group to see its forest plot (each "
    "study’s effect and confidence interval, with the pooled diamond) and "
    "contour-enhanced funnel plot. The summary panel lists coverage (papers "
    "assessed, statistics found and usable), then for each group the pooled "
    "effect and 95% CI, Z and p, I² and τ², and the publication "
    "bias verdict. meta_analysis_results.json, meta_analysis_report.md, "
    "meta_analysis_paper_audit.csv (why each paper was or was not pooled) and "
    "statistics_audit.csv are saved to the output folder."
)

ROBUSTNESS_VERIFY_HELP = {
    "title": "Stages 8–9: Robustness, Verification & Meta-Analysis",
    "icon": "\U0001f9ea",
    "what": _ROBUSTNESS_WHAT,
    "why": _ROBUSTNESS_WHY,
    "outputs": _ROBUSTNESS_OUTPUTS,
    "reading": [
        {
            "text": "Cochrane Handbook Ch. 10.14: Sensitivity analyses",
            "url": "https://training.cochrane.org/handbook/current/chapter-10#section-10-14",
        },
        {
            "text": "Efron & Tibshirani (1993) – An Introduction to the Bootstrap",
            "url": "https://doi.org/10.1201/9780429246593",
        },
        {
            "text": "Mathur & VanderWeele (2020) – Sensitivity analysis for publication bias",
            "url": "https://doi.org/10.1177/0962280218820575",
        },
        {
            "text": "DerSimonian & Laird (1986) – Meta-analysis in clinical trials",
            "url": "https://doi.org/10.1016/0197-2456(86)90046-2",
        },
        {
            "text": "Cochrane Handbook Ch. 10.4: Detecting reporting biases",
            "url": "https://training.cochrane.org/handbook/current/chapter-10#section-10-4",
        },
        {
            "text": "Duval & Tweedie (2000) – Trim and fill: a simple funnel-plot-based method",
            "url": "https://doi.org/10.1111/j.0006-341X.2000.00455.x",
        },
    ],
}

TEXT_SCALING_HELP = {
    "title": "Text Scaling",
    "icon": "\U0001f4d0",
    "what": (
        "Text scaling applies the Wordfish algorithm to position each article "
        "on a latent thematic dimension estimated from word frequencies. "
        "Articles that use similar vocabulary sit together; those using "
        "different terminology are placed further apart. A complementary "
        "stylometric analysis computes per-article features such as average "
        "sentence length, type–token ratio, passive-voice frequency and "
        "hedging-word density."
    ),
    "why": (
        "Wordfish was developed for scaling legislative texts but is equally "
        "useful for mapping the terminology of a scientific literature. It can "
        "reveal whether your corpus separates into research traditions that "
        "frame the hypothesis differently. Stylometric features help identify "
        "outliers—a paper with very high hedging language may be a "
        "speculative commentary rather than an empirical study."
    ),
    "outputs": (
        "The Wordfish panel lists articles on the estimated dimension (theta "
        "values) and shows the most discriminating words for each pole. The "
        "convergence indicator reports whether the Poisson model met its "
        "likelihood tolerance. The stylometry table lists per-article features "
        "and highlights differences in hedging, certainty language, reading "
        "grade and lexical diversity. Select a row for additional details."
    ),
    "reading": [
        {
            "text": "Slapin & Proksch (2008) – A scaling model for estimating time-series party positions",
            "url": "https://doi.org/10.1111/j.1540-5907.2008.00338.x",
        },
        {
            "text": "Grimmer & Stewart (2013) – Text as data: the promise and pitfalls",
            "url": "https://doi.org/10.1093/pan/mps028",
        },
        {
            "text": "Stamatatos (2009) – A survey of modern authorship attribution methods",
            "url": "https://doi.org/10.1002/asi.21001",
        },
    ],
}

ANALYSIS_HELP = {
    "synthesis": SYNTHESIS_HELP,
    "fulltext": FULLTEXT_HELP,
    "robustness_verify": ROBUSTNESS_VERIFY_HELP,
    "text_scaling": TEXT_SCALING_HELP,
}
