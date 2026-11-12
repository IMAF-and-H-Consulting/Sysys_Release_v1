"""Help text for the Search, Score & Filter, Extract & Validate and Audit tabs.

Part of ``gui.help_content``; no project imports. Stage numbers follow the
section headers in the GUI.
"""

SEARCH_HELP = {
    "title": "Stage 1: Literature Search",
    "icon": "\U0001f50d",
    "what": (
        "This stage searches the selected academic databases (PubMed, Semantic "
        "Scholar, OpenAlex, Scopus, Web of Science and Elicit) with queries built "
        "from your research question and hypothesis components. Results are "
        "de-duplicated using exact identifiers (DOI, PMID, PMCID and provider IDs) "
        "plus fuzzy title similarity, so each article appears once in the corpus. "
        "When duplicate records disagree, SystS keeps the copy with full text "
        "first, then the copy with richer identifiers and bibliographic metadata, "
        "and merges missing fields from the other copies.\n\n"
        "Min Year / Max Year drop newly found papers outside the range (papers "
        "with no year are kept and counted). Full text only keeps papers whose "
        "provider reports freely available full text. Citation snowballing adds "
        "papers that cite, or are cited by, the most-cited Semantic Scholar "
        "results (up to about 480 extra records with the default limits)."
    ),
    "why": (
        "Comprehensive, reproducible searching is the foundation of any "
        "systematic review. Searching only one database risks missing relevant "
        "studies: PubMed emphasises biomedical literature while Semantic Scholar "
        "and OpenAlex have broader coverage of computational and "
        "interdisciplinary work. The Cochrane Handbook recommends searching at "
        "least two databases and documenting the search strategy so that it can "
        "be replicated. Automated de-duplication reduces the screening burden and "
        "keeps your PRISMA flow-diagram counts accurate."
    ),
    "outputs": (
        "The merged corpus is saved as corpus.json and shown on the Corpus tab "
        "(#, Title, Authors, Year, Source, Score, DOI). search_manifest.json "
        "records each provider's hits, retrievals and errors, the records removed "
        "by de-duplication and by the year / full-text filters, and the snowball "
        "records; PRISMA reads its identification counts from this file. The "
        "number of unique articles is your initial screening population."
    ),
    "reading": [
        {
            "text": "Cochrane Handbook Ch. 4: Searching for and selecting studies",
            "url": "https://training.cochrane.org/handbook/current/chapter-04",
        },
        {
            "text": "Lefebvre et al. (2022) – Searching for and selecting studies",
            "url": "https://doi.org/10.1002/14651858.ED000169",
        },
        {
            "text": "Page et al. (2021) – PRISMA 2020 statement",
            "url": "https://doi.org/10.1136/bmj.n71",
        },
    ],
}

SCORE_FILTER_HELP = {
    "title": "Stage 2: Score & Filter",
    "icon": "\U0001f3af",
    "what": (
        "Stage 2 screens abstracts (or a full-text excerpt) against the "
        "Original Research Question from Preferences. The default Laya → "
        "BRT-BERT strategy keeps a paper only when Laya passes (bin >= 4, "
        "P >= 0.6) and the bundled BRT-BERT model also passes (P >= 0.236). "
        "Single-model and legacy scorers remain available for comparison."
    ),
    "why": (
        "Both models contribute to the filtering decision in one stage. "
        "BRT-BERT runs only on papers that pass Laya. Human review remains "
        "available to inspect and correct model decisions. Evaluate screening "
        "quality against human-labelled papers, including rejected papers."
    ),
    "outputs": (
        "relevant.json contains retained papers; scored_all.json contains "
        "all scored papers, raw model probabilities and gate decisions. "
        "screening_report.json records counts, settings and timings. "
        "For the combined filter, Model Score is 0 (reject) or 5 (keep); "
        "the ordinary score threshold does not apply. Human ratings and "
        "human/model agreement are saved under data/validation. Model scores "
        "you correct by hand are counted separately: the agreement summary "
        "also reports agreement without them."
    ),
    "reading": [
        {
            "text": "Cochrane Handbook Ch. 4.6: Selecting studies",
            "url": "https://training.cochrane.org/handbook/current/chapter-04#section-4-6",
        },
        {
            "text": "Alshami et al. (2023) – LLMs for systematic review screening",
            "url": "https://doi.org/10.1016/j.jbi.2023.104519",
        },
        {
            "text": "Cohen et al. (2006) – Reducing workload in systematic reviews",
            "url": "https://doi.org/10.1197/jamia.M1929",
        },
    ],
}

STATISTICAL_EXTRACTION_HELP = {
    "title": "Stage 2c: Statistical Evidence Extraction",
    "icon": "\U0001f4ca",
    "what": (
        "Stage 2c scans the retained papers (full text when Stage 2b saved it, "
        "otherwise the abstract) for reported statistics: test names, test "
        "statistics, p-values, sample sizes, effect sizes and confidence "
        "intervals, each linked to its source sentence. An optional LLM context "
        "check reviews only the text window around each statistic. Items with "
        "missing or ambiguous fields are listed for human review. LLM replies are "
        "cached, so after a cancel the next run with the same model only asks about "
        "the windows not yet answered.\n\n"
        "Bundle override normally stays Automatic: SystS detects or uses the configured "
        "loaded model context, then packs a paper's statistics while reserving response "
        "space for every verdict and a safety margin. A positive override is an explicit "
        "synthetic request limit and is recorded. Scope limit normally stays All "
        "candidates; a positive value is a separately recorded user runtime/cost limit. "
        "The reply must answer every statistic. If it misses an ID or is malformed/cut "
        "off, valid member verdicts are retained and only the affected IDs are evicted "
        "from cache, split, and retried in smaller complete requests during the same run. "
        "A singleton that still fails stays unchecked for human review. Every raw attempt, "
        "budget, split, recovery, terminal failure, and any unreviewed scope is saved in "
        "statistical_extractions.json and summarized in the log.\n\n"
        "Review Effects & Groups… opens a table of every effect found or "
        "converted. Tick Include to pool an effect, set its Group (effects in the "
        "same group are pooled together; suggested groups are never imposed) and, "
        "where needed, its Direction (+ or −). Effects reported directly with "
        "their own uncertainty are included by default. Effects converted from a "
        "test statistic (t→d, F(1,df)→d, χ²(1)→φ, z→r, p→r) are off until you "
        "include them; F, χ² and p-value-only conversions carry no sign, so they "
        "also need a direction."
    ),
    "why": (
        "Meta-analysis and GRADE depend on accurate numeric inputs. A reviewable "
        "table makes extraction errors visible before they reach a pooled "
        "estimate, and keeps a human in charge of which results are pooled "
        "together and of every conversion that needs an assumption."
    ),
    "outputs": (
        "statistical_extractions.json (all results with source context), "
        "statistical_review_required.json (items for human review), "
        "statistical_meta_analysis_input.json (the meta-analysis input) and "
        "statistics_audit.csv (every statistic, whether it can be used and, if "
        "not, why) are written to the claims folder. The log summarises how many "
        "statistics were found and how many are usable."
    ),
    "reading": [
        {
            "text": "Cochrane Handbook Ch. 6: Choosing effect measures and computing estimates",
            "url": "https://training.cochrane.org/handbook/current/chapter-06",
        },
    ],
}

EXTRACT_VALIDATE_HELP = {
    "title": "Stage 3 & 4: Claim Extraction & Method Review",
    "icon": "\U0001f4cb",
    "what": (
        "Stage 3 asks the LLM to pull structured claims from each retained "
        "paper (findings, populations, conditions, mechanisms, key entities, the "
        "evidence direction and a supporting quote), with a heuristic fallback "
        "when the reply is unusable. Stage 4 (Method Review) classifies each "
        "paper's study design into a fixed list (randomized controlled trial, "
        "cohort, case-control, cross-sectional, longitudinal, systematic review / "
        "meta-analysis, case report, preclinical, imaging classifier and others) "
        "and checks that the quoted evidence spans occur in the abstract."
    ),
    "why": (
        "Structured extraction gives every paper the same fields, so evidence "
        "can be compared and aggregated. Study-design labels feed GRADE's risk "
        "of bias. Checking quotes against the source catches hallucinated "
        "evidence before synthesis."
    ),
    "outputs": (
        "Stage 3 writes claims.json to the claims folder. Stage 4 always writes "
        "method_profiles_validated.json, method_review_manual.csv (papers that "
        "need a human check, including every design assigned by the keyword "
        "fallback) and method_type_summary.json to data/method_review; the "
        "Validated Output field only shows that location. The status line and "
        "log report how many papers were validated and how many were flagged."
    ),
    "reading": [
        {
            "text": "Cochrane Handbook Ch. 5: Collecting data",
            "url": "https://training.cochrane.org/handbook/current/chapter-05",
        },
        {
            "text": "Higgins et al. (2019) – Data collection forms for intervention reviews",
            "url": "https://doi.org/10.1002/jrsm.1355",
        },
        {
            "text": "Wang et al. (2023) – GPT-based data extraction for systematic reviews",
            "url": "https://doi.org/10.1101/2023.10.05.23296583",
        },
    ],
}

AUDIT_HELP = {
    "title": "Stage 5: Span Audit",
    "icon": "\U0001f50e",
    "what": (
        "The span audit checks that each paper's quoted evidence (its key and "
        "relevance quotes and validated spans) is grounded in the paper's "
        "abstract. It compares each quote with the abstract by token overlap, "
        "TF-IDF similarity and, when sentence-transformers is installed, "
        "semantic similarity; quotes that first look weak are re-checked by "
        "fuzzy matching. A quote at or above the Similarity Threshold slider "
        "(default 0.85) is reclassified as grounded. A paper is grounded when "
        "at least one of its quotes is rescued or at least half of a quote's "
        "words occur in the abstract. No LLM is used.\n\n"
        "Papers with no grounded quote are flagged, not removed. Tick "
        "‘Exclude papers with no grounded quote’ to remove them."
    ),
    "why": (
        "An extracted claim is only as good as its link to the source. Quotes "
        "that do not occur in the paper usually mean the extraction paraphrased "
        "or invented evidence. Flagging them lets you check those papers by hand; "
        "excluding them is a protocol decision you should report."
    ),
    "outputs": (
        "claims_filtered.json (the papers kept, used by later stages), "
        "audit_flags.json (papers with no grounded quote, for review) and "
        "removal_log.json (papers removed, only when the exclusion box is ticked; "
        "PRISMA reports these as exclusions) are written to the claims folder, "
        "plus three audit CSVs in data/method_review/review. The log reports "
        "papers kept, removed, rescued and flagged."
    ),
    "reading": [
        {
            "text": "Reimers & Gurevych (2019) – Sentence-BERT for semantic similarity",
            "url": "https://doi.org/10.18653/v1/D19-1410",
        },
    ],
}

SCREENING_HELP = {
    "search": SEARCH_HELP,
    "score_filter": SCORE_FILTER_HELP,
    "statistical_extraction": STATISTICAL_EXTRACTION_HELP,
    "extract_validate": EXTRACT_VALIDATE_HELP,
    "audit": AUDIT_HELP,
}
