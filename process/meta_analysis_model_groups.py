"""Audited semantic grouping for meta-analysis effect estimates.

Free-text outcome labels are useful for finding *candidate* matches, but they
are not evidence that two estimates are scientifically poolable. This module
separates recall-oriented candidate discovery from two mandatory gates:

1. paper-title and research-question relevance; and
2. estimate-level scientific compatibility.

Only candidates that pass both gates become ``model_validated`` proposals.
Every candidate, rejection reason, paper title, statistic and gate verdict is
retained in ``model_groups.json`` and in a readable Markdown audit.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from concurrent.futures import CancelledError
from pathlib import Path
from typing import Any

from core.io import atomic_write_json, atomic_write_text
from process.llm_budget import (
    BudgetPlanningError,
    ModelCapacity,
    TokenBudget,
    plan_token_budget,
    resolve_model_capacity,
)
from process.llm_context import ContextWindowExceededError, OutputLimitExceededError
from process.meta_analysis_effects import STATUS_CALCULABLE, STATUS_CONVERTED
from process.meta_analysis_groups import _family, suggest_groups
from process.statistical_extractor_llm import _parse_bool, reply_json_candidates

logger = logging.getLogger(__name__)

PROPOSALS_FILE = "model_groups.json"
GROUPING_REVIEW_FILE = "model_grouping_review.md"
MIN_PAPERS = 2
OVERLAP_FRACTION = 0.25
FAMILY_NAMES = {"smd": "mean difference", "ratio": "ratio", "correlation": "correlation"}

REQUIRED_CHARACTERISTICS = (
    "outcome_construct",
    "measurement_definition",
    "population",
    "subgroup_independence",
    "research_question_fit",
    "comparison_or_exposure",
    "study_design_and_timepoint",
    "effect_direction",
    "scientific_plausibility",
)

DISCOVERY_PROMPT = """You are finding CANDIDATE matches for a meta-analysis.
Research question: {question}
Effect family: {family}

Each line is one extracted estimate item. It includes the exact paper title,
outcome, comparison/exposure, population, scale, and extracted statistic text.
This is a recall-oriented discovery pass, not permission to pool.

Return every scientifically conceivable cross-paper candidate set, including
sets that may later fail on construct, population, comparison, independence,
or design. A candidate must contain estimates from at least two different
paper IDs. Do not group estimates merely because they share words, and do not
return a set containing only one paper. Preserve Glu, GABA+, Glx, and a
Glu/GABA+ ratio as distinct constructs unless a later review explicitly
justifies equivalence. Several candidates may overlap; later gates resolve
them. Account for EVERY supplied estimate ID: place it in at least one
candidate, or put it in unmatched_ids when there is no conceivable cross-paper
match in this request. An ID cannot be both a candidate member and unmatched.

Reply with JSON only:
{{"candidate_groups": [{{
  "label": "short candidate description",
  "candidate_reason": "why these papers might estimate a comparable quantity",
  "members": [{{"id": "E1", "reversed": false}}]
}}], "unmatched_ids": ["E3", "E4"]}}

Estimates:
{lines}
"""

TITLE_REVIEW_PROMPT = """First compatibility gate for a proposed meta-analysis.
Research question: {question}

Use ONLY the supplied exact paper titles for this first gate. For each paper,
decide (a) whether it addresses the project's research question and (b)
whether, from the title, there is a reasonable scientific route for comparing
it with at least one other paper in this candidate. A shared keyword is not
enough. Mark an ambiguous title false rather than inventing study details.

Candidate: {label}
Candidate reason: {candidate_reason}
Papers:
{papers}

Reply with JSON only. Copy each title exactly:
{{
  "paper_reviews": [{{
    "paper_id": "P1",
    "title": "exact supplied title",
    "research_question_relevant": true,
    "comparable_in_principle": true,
    "reason": "title-grounded reason"
  }}],
  "eligible_paper_ids": ["P1", "P2"],
  "group_pass": true,
  "reason": "why the eligible papers pass or fail the title gate"
}}
"""

CHARACTERISTIC_REVIEW_PROMPT = """Second compatibility gate for a meta-analysis.
Research question: {question}
Effect family: {family}

The title/relevance gate has already retained these candidate papers. Review
the exact extracted estimates below. You may exclude members or papers that
do not belong, but the final compatible set must contain at least two distinct
papers. Do not combine distinct biological constructs merely to create a
larger group: in particular, Glu, GABA+, Glx, and Glu/GABA+ ratio are not
interchangeable. ASD, dyslexia, typically developing, and generic healthy
samples are not automatically compatible. Multiple subgroup estimates from
one paper are not independent studies and may be retained together only when
their within-paper combination is scientifically valid and explicitly
explained.

For EVERY listed characteristic, state what matched, what differed, why that
difference is or is not acceptable, and an explicit pass boolean. The final
verdict can be "compatible", "incompatible", or "needs_human_review". Only
"compatible" with every characteristic passing will be pooled. When more than
one selected estimate comes from the same paper, subgroup_independence must
include a nonempty within_paper_handling entry keyed by that paper ID. Account
for every eligible estimate ID exactly once in either members or
excluded_members; every exclusion needs a concrete scientific reason.

Candidate: {label}
Title-gate result:
{title_result}

Estimates (title and abstract are supplied only as evidence; do not infer
unstated methods):
{estimates}

Reply with JSON only:
{{
  "group_label": "precise shared construct and comparison",
  "reference_comparison": "direction used for the pooled effect",
  "members": [{{"id": "E1", "reversed": false}}],
  "excluded_members": [{{"id": "E3", "reason": "specific incompatibility with the final group"}}],
  "characteristics": {{
    "outcome_construct": {{"pass": true, "matched": "...", "differences": "...", "reason": "..."}},
    "measurement_definition": {{"pass": true, "matched": "...", "differences": "...", "reason": "..."}},
    "population": {{"pass": true, "matched": "...", "differences": "...", "reason": "..."}},
    "subgroup_independence": {{"pass": true, "matched": "...", "differences": "...", "reason": "...",
      "within_paper_handling": {{"P2": "why these selected estimates can form one paper-level composite"}}}},
    "research_question_fit": {{"pass": true, "matched": "...", "differences": "...", "reason": "..."}},
    "comparison_or_exposure": {{"pass": true, "matched": "...", "differences": "...", "reason": "..."}},
    "study_design_and_timepoint": {{"pass": true, "matched": "...", "differences": "...", "reason": "..."}},
    "effect_direction": {{"pass": true, "matched": "...", "differences": "...", "reason": "..."}},
    "scientific_plausibility": {{"pass": true, "matched": "...", "differences": "...", "reason": "..."}}
  }},
  "final_verdict": "compatible",
  "review_summary": "concise explanation of the final decision"
}}
"""


def _clip(text: Any, limit: int | None = None) -> str:
    """Normalize model evidence without silently discarding its tail.

    ``limit`` remains in the signature for compatibility with the semantic
    gating rewrite, but is deliberately not applied.  Request sizing now
    happens through context-aware batching; the saved evidence and model audit
    retain the complete title, statistic, abstract, label, and explanation.
    """
    return " ".join(str(text or "").split())


def _json_object(reply: str, required_key: str) -> dict:
    for candidate in reply_json_candidates(reply or ""):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and required_key in data:
            return data
    raise ValueError(f"reply holds no JSON object with a '{required_key}' field")


def _discovery_in(reply: str, expected_ids: set[str]) -> tuple[list, set[str]]:
    """Read a discovery reply and require accounting for every supplied ID."""
    for candidate in reply_json_candidates(reply or ""):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        groups = data.get("candidate_groups")
        if not isinstance(groups, list):
            groups = data.get("groups")
        unmatched = data.get("unmatched_ids")
        if not isinstance(groups, list) or not isinstance(unmatched, list):
            continue
        unmatched_ids = {str(value) for value in unmatched}
        unknown_unmatched = unmatched_ids - expected_ids
        if unknown_unmatched:
            raise ValueError(
                "unmatched_ids contains unknown IDs: " + ", ".join(sorted(unknown_unmatched))
            )
        member_ids: set[str] = set()
        for group in groups:
            if not isinstance(group, dict):
                continue
            for member in group.get("members") or []:
                raw = member if isinstance(member, dict) else {"id": member}
                member_id = str(raw.get("id") or "")
                if member_id in expected_ids:
                    member_ids.add(member_id)
        overlap = member_ids & unmatched_ids
        if overlap:
            raise ValueError(
                "IDs cannot be both candidate members and unmatched: " + ", ".join(sorted(overlap))
            )
        missing = expected_ids - member_ids - unmatched_ids
        if missing:
            raise ValueError(
                "discovery reply did not account for estimate IDs: " + ", ".join(sorted(missing))
            )
        return groups, unmatched_ids
    raise ValueError(
        "reply holds no JSON object with a 'candidate_groups' and 'unmatched_ids' list"
    )


def _paper_context(config) -> dict[str, dict[str, str]]:
    """Best-effort title/abstract lookup for the current run's retained corpus."""
    try:
        from process.paths import load_paper_list, resolve_claims_corpus
        from process.statistical_extractor_text import article_id

        path = resolve_claims_corpus(config)
        if path is None:
            return {}
        papers = load_paper_list(path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        logger.warning("Could not load paper context for semantic grouping: %s", exc)
        return {}

    context: dict[str, dict[str, str]] = {}
    for index, paper in enumerate(papers):
        if not isinstance(paper, dict):
            continue
        key = str(article_id(paper, index))
        entry = {
            "title": _clip(paper.get("title") or key),
            "abstract": _clip(paper.get("abstract")),
        }
        context[key] = entry
        for alias in ("doi", "pmid", "s2_paper_id", "paperId", "openalex_id", "id"):
            value = str(paper.get(alias) or "").strip()
            if value:
                context.setdefault(value, entry)
    return context


def _statistic_text(row: dict) -> str:
    parts = []
    for key in ("test_text", "statistic_text", "effect_used", "route", "analysis_scale"):
        value = _clip(row.get(key))
        if value and value not in parts:
            parts.append(value)
    return _clip("; ".join(parts))


def _items(rows: list[dict], paper_context: dict[str, dict[str, str]] | None = None) -> dict[str, list[dict]]:
    """One separately reviewable item per extracted effect.

    Identically labelled effects within one paper must remain separate here:
    they may be different subgroups, timepoints, regions, or constructs.  The
    semantic independence gate needs to see and select them individually.
    """
    paper_context = paper_context or {}
    papers: dict[str, str] = {}
    by_effect_id: dict[str, dict] = {}
    families: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("status") not in (STATUS_CALCULABLE, STATUS_CONVERTED) or not _clip(row.get("outcome")):
            continue
        article = str(row.get("article_id") or "").strip()
        effect_id = str(row.get("effect_id") or "").strip()
        if not article or not effect_id:
            continue
        family = _family(row)
        labels = tuple(_clip(row.get(k)) for k in ("outcome", "comparison", "population"))
        scale = _clip(row.get("analysis_scale") or row.get("effect_measure"), 80)
        if effect_id not in by_effect_id:
            paper = papers.setdefault(article, f"P{len(papers) + 1}")
            context = paper_context.get(article, {})
            title = _clip(row.get("title") or context.get("title") or article)
            abstract = _clip(row.get("abstract") or context.get("abstract"))
            by_effect_id[effect_id] = {
                "id": f"E{len(by_effect_id) + 1}", "paper": paper, "article_id": article,
                "title": title, "abstract": abstract, "labels": labels,
                "analysis_scale": scale, "effect_ids": [effect_id], "statistics": [],
            }
            families[family].append(by_effect_id[effect_id])
        item = by_effect_id[effect_id]
        statistic = _statistic_text(row)
        if statistic and statistic not in item["statistics"]:
            item["statistics"].append(statistic)
    return families


def _line(item: dict) -> str:
    outcome, comparison, population = item["labels"]
    statistics = " || ".join(item.get("statistics") or []) or "statistic text not available"
    return (
        f"{item['id']} [{item['paper']}] title={json.dumps(item['title'], ensure_ascii=False)} | "
        f"outcome={json.dumps(outcome, ensure_ascii=False)} | "
        f"comparison={json.dumps(comparison or 'not stated', ensure_ascii=False)} | "
        f"population={json.dumps(population or 'not stated', ensure_ascii=False)} | "
        f"scale={json.dumps(item.get('analysis_scale') or 'not stated', ensure_ascii=False)} | "
        f"statistic={json.dumps(_clip(statistics), ensure_ascii=False)}"
    )


def _discovery_messages(question: str, family: str, items: list[dict]) -> list[dict[str, str]]:
    return [{"role": "user", "content": DISCOVERY_PROMPT.format(
        question=question,
        family=FAMILY_NAMES[family],
        lines="\n".join(_line(item) for item in items),
    )}]


def _discovery_budget(
    capacity: ModelCapacity,
    question: str,
    family: str,
    items: list[dict],
) -> TokenBudget:
    """Reserve prompt *and* every-ID response capacity for one discovery view."""
    return plan_token_budget(
        "meta_discovery",
        capacity,
        messages=_discovery_messages(question, family, items),
        expected_items=len(items),
    )


def _candidate_gate_budgets_fit(
    capacity: ModelCapacity,
    question: str,
    family: str,
    items: list[dict],
) -> None:
    """Ensure a discovery view cannot create a candidate too large to audit."""
    papers: dict[str, dict] = {}
    for item in items:
        papers.setdefault(item["paper"], item)
    title_prompt = TITLE_REVIEW_PROMPT.format(
        question=question,
        label="candidate discovered in this request",
        candidate_reason="candidate discovery result",
        papers="\n".join(
            f"{paper_id} | {item['title']}" for paper_id, item in sorted(papers.items())
        ),
    )
    title_budget = plan_token_budget(
        "meta_title",
        capacity,
        messages=[{"role": "user", "content": title_prompt}],
        expected_items=len(papers),
    )
    shown: set[str] = set()
    estimates = []
    for item in items:
        abstract = item.get("abstract") or "abstract not available"
        if item["paper"] in shown:
            abstract = f"see complete {item['paper']} abstract above"
        shown.add(item["paper"])
        outcome, comparison, population = item["labels"]
        estimates.append(
            f"{item['id']} [{item['paper']}] title={json.dumps(item['title'], ensure_ascii=False)}\n"
            f"  abstract={json.dumps(abstract, ensure_ascii=False)}\n"
            f"  outcome={json.dumps(outcome, ensure_ascii=False)}; "
            f"comparison/exposure={json.dumps(comparison or 'not stated', ensure_ascii=False)}; "
            f"population={json.dumps(population or 'not stated', ensure_ascii=False)}; "
            f"scale={json.dumps(item.get('analysis_scale') or 'not stated', ensure_ascii=False)}; "
            f"statistics={json.dumps(' || '.join(item.get('statistics') or []) or 'not available', ensure_ascii=False)}"
        )
    # The characteristic prompt includes the title response. Reserve its full
    # planned allowance rather than assuming a short answer.
    title_result_reserve = "x" * int(title_budget.max_output_tokens * 3.5)
    characteristic_prompt = CHARACTERISTIC_REVIEW_PROMPT.format(
        question=question,
        family=FAMILY_NAMES[family],
        label="candidate discovered in this request",
        title_result=title_result_reserve,
        estimates="\n".join(estimates),
    )
    plan_token_budget(
        "meta_characteristics",
        capacity,
        messages=[{"role": "user", "content": characteristic_prompt}],
        expected_items=len(items),
    )


def _chunks(
    items: list[dict],
    capacity: ModelCapacity,
    question: str,
    family: str,
    *,
    sort_key=None,
    planning_holds: list[dict] | None = None,
) -> list[list[dict]]:
    """Pack by complete prompt plus item-scaled reply, never input size alone."""
    sort_key = sort_key or (lambda i: (i["labels"][0].casefold(), i["title"].casefold()))
    chunks: list[list[dict]] = []
    current: list[dict] = []
    for item in sorted(items, key=sort_key):
        proposed = [*current, item]
        try:
            _discovery_budget(capacity, question, family, proposed)
            _candidate_gate_budgets_fit(capacity, question, family, proposed)
        except BudgetPlanningError as exc:
            if not current:
                if planning_holds is not None:
                    planning_holds.append({
                        "family": family,
                        "family_name": FAMILY_NAMES[family],
                        "estimate_id": item["id"],
                        "article_id": item["article_id"],
                        "title": item["title"],
                        "outcome": item["labels"][0],
                        "error": str(exc),
                        "planning_error": exc.as_dict(),
                        "action": "singleton held for human review; other estimates continued",
                    })
                current = []
                continue
            chunks.append(current)
            current = [item]
            try:
                _discovery_budget(capacity, question, family, current)
                _candidate_gate_budgets_fit(capacity, question, family, current)
            except BudgetPlanningError as singleton_exc:
                if planning_holds is not None:
                    planning_holds.append({
                        "family": family,
                        "family_name": FAMILY_NAMES[family],
                        "estimate_id": item["id"],
                        "article_id": item["article_id"],
                        "title": item["title"],
                        "outcome": item["labels"][0],
                        "error": str(singleton_exc),
                        "planning_error": singleton_exc.as_dict(),
                        "action": "singleton held for human review; other estimates continued",
                    })
                current = []
        else:
            current = proposed
    if current:
        chunks.append(current)
    return chunks


def _bridge_chunks(
    chunks: list[list[dict]],
    capacity: ModelCapacity,
    question: str,
    family: str,
) -> list[list[dict]]:
    """Overlap adjacent boundaries while retaining complete response capacity."""
    bridges: list[list[dict]] = []
    for left, right in zip(chunks, chunks[1:]):
        target = max(2, int(max(len(left), len(right)) * OVERLAP_FRACTION))
        candidates = left[-target:] + right[:target]
        bridge: list[dict] = []
        for item in candidates:
            proposed = [*bridge, item]
            try:
                _discovery_budget(capacity, question, family, proposed)
                _candidate_gate_budgets_fit(capacity, question, family, proposed)
            except BudgetPlanningError:
                break
            bridge = proposed
        if len({item["article_id"] for item in bridge}) >= MIN_PAPERS:
            bridges.append(bridge)
    return bridges


def _discovery_batches(
    items: list[dict],
    capacity: ModelCapacity,
    question: str,
    family: str,
    planning_holds: list[dict] | None = None,
) -> list[list[dict]]:
    """Primary, boundary, and alternate views planned against total context."""
    primary = _chunks(
        items, capacity, question, family, planning_holds=planning_holds
    )
    batches = primary + _bridge_chunks(primary, capacity, question, family)
    if len(primary) > 1:
        alternate = _chunks(
            items,
            capacity,
            question,
            family,
            sort_key=lambda i: (
                i["labels"][1].casefold(), i["labels"][2].casefold(),
                i["labels"][0].casefold(), i["title"].casefold(),
            ),
            planning_holds=planning_holds,
        )
        batches.extend(alternate)
        batches.extend(_bridge_chunks(alternate, capacity, question, family))
    unique: list[list[dict]] = []
    seen: set[tuple[str, ...]] = set()
    for batch in batches:
        signature = tuple(sorted(item["id"] for item in batch))
        if signature and signature not in seen:
            seen.add(signature)
            unique.append(batch)
    return unique


def _split_discovery_batch(items: list[dict]) -> list[list[dict]]:
    """Smaller complete views plus a cross-boundary bridge after a failed reply."""
    if len(items) < 3:
        return []
    middle = len(items) // 2
    left, right = items[:middle], items[middle:]
    target = max(1, int(max(len(left), len(right)) * OVERLAP_FRACTION))
    bridge = left[-target:] + right[:target]
    pieces = [left, right]
    if len({item["article_id"] for item in bridge}) >= MIN_PAPERS:
        pieces.append(bridge)
    return [piece for piece in pieces if piece]


def _public_item(item: dict, reversed_: bool | None = None) -> dict:
    outcome, comparison, population = item["labels"]
    result = {
        "id": item["id"], "paper_id": item["paper"], "article_id": item["article_id"],
        "title": item["title"], "outcome": outcome, "comparison": comparison,
        "population": population, "analysis_scale": item.get("analysis_scale", ""),
        "abstract": item.get("abstract", ""),
        "effect_ids": list(item.get("effect_ids") or []),
        "statistics": list(item.get("statistics") or []),
    }
    if reversed_ is not None:
        result["reversed"] = bool(reversed_)
    return result


def _candidate_from_group(group: Any, item_map: dict[str, dict], family: str, request: int) -> dict:
    audit = {
        "audit_id": "", "family": family, "family_name": FAMILY_NAMES[family],
        "discovery_request": request, "candidate_label": "", "candidate_reason": "",
        "status": "structurally_rejected", "rejection_reasons": [], "members": [],
        "title_review": None, "characteristic_review": None,
    }
    if not isinstance(group, dict):
        audit["rejection_reasons"].append("candidate is not a JSON object")
        return audit
    audit["candidate_label"] = _clip(group.get("label"))
    audit["candidate_reason"] = _clip(group.get("candidate_reason"), 600)
    seen: set[str] = set()
    for raw_member in group.get("members") or []:
        member = raw_member if isinstance(raw_member, dict) else {"id": raw_member}
        member_id = str(member.get("id") or "")
        item = item_map.get(member_id)
        if item is None:
            audit["rejection_reasons"].append(f"unknown estimate ID {member_id or '<blank>'}")
            continue
        if member_id in seen:
            audit["rejection_reasons"].append(f"duplicate estimate ID {member_id}")
            continue
        seen.add(member_id)
        audit["members"].append(_public_item(item, _parse_bool(member.get("reversed")) is True))
    if not audit["candidate_label"]:
        audit["rejection_reasons"].append("candidate has no label")
    papers = {member["article_id"] for member in audit["members"]}
    if len(papers) < MIN_PAPERS:
        audit["rejection_reasons"].append(
            f"candidate contains {len(papers)} paper; at least {MIN_PAPERS} independent papers are required"
        )
    if not audit["rejection_reasons"]:
        audit["status"] = "awaiting_title_review"
    return audit


def _title_papers(audit: dict) -> dict[str, dict]:
    papers: dict[str, dict] = {}
    for member in audit["members"]:
        papers.setdefault(member["paper_id"], {
            "paper_id": member["paper_id"], "article_id": member["article_id"],
            "title": member["title"],
        })
    return papers


def _validate_title_review(
    data: dict,
    audit: dict,
) -> tuple[list[str], list[str], list[str]]:
    """Return eligible IDs, response-contract errors, and scientific failures."""
    expected = _title_papers(audit)
    reviews = data.get("paper_reviews")
    contract: list[str] = []
    scientific: list[str] = []
    eligible: list[str] = []
    if not isinstance(reviews, list):
        return [], ["title review has no paper_reviews list"], []
    by_id: dict[str, dict] = {}
    for review in reviews:
        if not isinstance(review, dict):
            contract.append("title review contains a non-object paper review")
            continue
        paper_id = str(review.get("paper_id") or "")
        if paper_id not in expected:
            contract.append(f"title review contains unknown paper ID {paper_id or '<blank>'}")
            continue
        if paper_id in by_id:
            contract.append(f"title review duplicates {paper_id}")
            continue
        by_id[paper_id] = review
        if str(review.get("title") or "").strip() != expected[paper_id]["title"]:
            contract.append(f"title review did not copy the exact title for {paper_id}")
            continue
        relevant = _parse_bool(review.get("research_question_relevant"))
        comparable = _parse_bool(review.get("comparable_in_principle"))
        if relevant is None:
            contract.append(f"title review has no boolean research-question verdict for {paper_id}")
        if comparable is None:
            contract.append(f"title review has no boolean comparability verdict for {paper_id}")
        reason = _clip(review.get("reason"), 800)
        if not reason:
            contract.append(f"title review supplied no reason for {paper_id}")
        if relevant is True and comparable is True and reason:
            eligible.append(paper_id)
        elif relevant is not None and comparable is not None and reason:
            scientific.append(f"title gate did not retain {paper_id}: {reason}")
    missing = sorted(set(expected) - set(by_id))
    if missing:
        contract.append("title review omitted " + ", ".join(missing))
    declared = data.get("eligible_paper_ids")
    if not isinstance(declared, list):
        contract.append("title review has no eligible_paper_ids list")
    else:
        declared_set = {str(value) for value in declared}
        if declared_set != set(eligible):
            contract.append("eligible_paper_ids does not match the paper-level verdicts")
    group_pass = _parse_bool(data.get("group_pass"))
    if group_pass is None:
        contract.append("title review has no boolean group_pass verdict")
    elif group_pass is False:
        scientific.append("title review group_pass is not true")
    if not _clip(data.get("reason"), 800):
        contract.append("title review supplied no group-level reason")
    if len(eligible) < MIN_PAPERS:
        scientific.append(
            f"title gate retained {len(eligible)} paper; at least {MIN_PAPERS} are required"
        )
    return eligible, contract, scientific


def _estimate_block(audit: dict, eligible_papers: set[str], context_by_article: dict[str, dict]) -> str:
    lines = []
    abstract_shown: set[str] = set()
    for member in audit["members"]:
        if member["paper_id"] not in eligible_papers:
            continue
        context = context_by_article.get(member["article_id"], {})
        abstract = _clip(member.get("abstract") or context.get("abstract")) \
            or "abstract not available"
        if member["paper_id"] in abstract_shown:
            abstract = f"see complete {member['paper_id']} abstract above"
        else:
            abstract_shown.add(member["paper_id"])
        lines.append(
            f"{member['id']} [{member['paper_id']}] title={json.dumps(member['title'], ensure_ascii=False)}\n"
            f"  abstract={json.dumps(abstract, ensure_ascii=False)}\n"
            f"  outcome={json.dumps(member['outcome'], ensure_ascii=False)}; "
            f"comparison/exposure={json.dumps(member['comparison'] or 'not stated', ensure_ascii=False)}; "
            f"population={json.dumps(member['population'] or 'not stated', ensure_ascii=False)}; "
            f"scale={json.dumps(member['analysis_scale'] or 'not stated', ensure_ascii=False)}; "
            f"statistics={json.dumps(' || '.join(member['statistics']) or 'not available', ensure_ascii=False)}"
        )
    return "\n".join(lines)


def _validate_characteristic_review(
    data: dict,
    audit: dict,
    eligible_papers: set[str],
) -> tuple[list[dict], list[str], list[str]]:
    """Separate an incomplete response contract from a scientific no-match."""
    contract: list[str] = []
    scientific: list[str] = []
    original = {
        member["id"]: member for member in audit["members"]
        if member["paper_id"] in eligible_papers
    }
    selected: list[dict] = []
    seen: set[str] = set()
    members = data.get("members")
    if not isinstance(members, list):
        contract.append("characteristic review has no members list")
    else:
        for raw_member in members:
            member = raw_member if isinstance(raw_member, dict) else {"id": raw_member}
            member_id = str(member.get("id") or "")
            if member_id not in original:
                contract.append(f"characteristic review selected ineligible or unknown ID {member_id or '<blank>'}")
                continue
            if member_id in seen:
                contract.append(f"characteristic review duplicates {member_id}")
                continue
            seen.add(member_id)
            selected.append({**original[member_id], "reversed": _parse_bool(member.get("reversed")) is True})
    excluded = data.get("excluded_members")
    excluded_ids: set[str] = set()
    if not isinstance(excluded, list):
        contract.append("characteristic review has no excluded_members list")
    else:
        for raw_exclusion in excluded:
            exclusion = raw_exclusion if isinstance(raw_exclusion, dict) else {"id": raw_exclusion}
            member_id = str(exclusion.get("id") or "")
            if member_id not in original:
                contract.append(
                    f"characteristic review excluded ineligible or unknown ID {member_id or '<blank>'}"
                )
                continue
            if member_id in seen or member_id in excluded_ids:
                contract.append(f"characteristic review duplicates selected/excluded ID {member_id}")
                continue
            excluded_ids.add(member_id)
            if not _clip(exclusion.get("reason"), 1200):
                contract.append(f"characteristic review supplied no exclusion reason for {member_id}")
    unaccounted = set(original) - seen - excluded_ids
    if unaccounted:
        contract.append(
            "characteristic review did not select or exclude IDs: "
            + ", ".join(sorted(unaccounted))
        )
    papers = {member["article_id"] for member in selected}
    if len(papers) < MIN_PAPERS:
        scientific.append(f"final selection contains {len(papers)} paper; at least {MIN_PAPERS} are required")
    characteristics = data.get("characteristics")
    if not isinstance(characteristics, dict):
        contract.append("characteristic review has no characteristics object")
    else:
        for name in REQUIRED_CHARACTERISTICS:
            verdict = characteristics.get(name)
            if not isinstance(verdict, dict):
                contract.append(f"missing characteristic verdict: {name}")
                continue
            passed = _parse_bool(verdict.get("pass"))
            if passed is None:
                contract.append(f"characteristic has no boolean pass verdict: {name}")
            elif passed is False:
                scientific.append(f"characteristic did not pass: {name}")
            if not _clip(verdict.get("matched"), 1000):
                contract.append(f"characteristic has no matched description: {name}")
            if not _clip(verdict.get("differences"), 1000):
                contract.append(f"characteristic has no differences description: {name}")
            if not _clip(verdict.get("reason"), 1200):
                contract.append(f"characteristic has no reason: {name}")
        subgroup = characteristics.get("subgroup_independence")
        subgroup = subgroup if isinstance(subgroup, dict) else {}
        handling = subgroup.get("within_paper_handling")
        if handling is None:
            handling = {}
        if not isinstance(handling, dict):
            contract.append("subgroup_independence within_paper_handling is not an object")
            handling = {}
        selected_by_paper: dict[str, int] = defaultdict(int)
        for member in selected:
            selected_by_paper[member["paper_id"]] += 1
        for paper_id, count in selected_by_paper.items():
            if count > 1 and not _clip(handling.get(paper_id), 1200):
                contract.append(
                    f"subgroup_independence does not explain the {count} selected estimates from {paper_id}"
                )
    final_verdict = str(data.get("final_verdict") or "").strip().casefold()
    if final_verdict not in {"compatible", "incompatible", "needs_human_review"}:
        contract.append("final verdict is not compatible, incompatible, or needs_human_review")
    elif final_verdict != "compatible":
        scientific.append(f"final verdict is {final_verdict.replace('_', ' ')}")
    if not _clip(data.get("group_label")):
        contract.append("characteristic review has no group_label")
    if not _clip(data.get("review_summary"), 1200):
        contract.append("characteristic review has no review_summary")
    return selected, contract, scientific


def _forget(
    forget_cached_reply,
    config,
    messages: list[dict],
    budget: TokenBudget,
) -> None:
    forget_cached_reply(
        config.llm_model,
        messages,
        temperature=0,
        think=False,
        task=budget.task,
        expected_items=budget.expected_items,
        context_tokens=budget.context_window_tokens,
        token_budget=budget,
    )


def _review_with_one_repair(
    *,
    config,
    chat,
    forget_cached_reply,
    capacity: ModelCapacity,
    task: str,
    expected_items: int,
    base_prompt: str,
    required_key: str,
    validate,
    cancel_event=None,
    progress=None,
) -> tuple[dict | None, Any, list[str], list[str], list[dict]]:
    """Run a structured gate and repair an incomplete response exactly once.

    Scientific failures are returned immediately and are never retried.  A
    parse/schema/every-ID contract failure is evicted from the cache and gets
    one explicit repair request; a second incomplete response is held for
    human review instead of being mislabeled as scientific rejection.
    """
    progress = progress or (lambda message: None)
    contract_errors: list[str] = []
    scientific_failures: list[str] = []
    payload: Any = None
    data: dict | None = None
    attempts: list[dict] = []
    for attempt_number in (1, 2):
        prompt = base_prompt
        if attempt_number == 2:
            prompt += (
                "\n\nCONTRACT REPAIR REQUIRED. The previous response could not be used because:\n- "
                + "\n- ".join(contract_errors)
                + "\nReturn a complete replacement JSON object. Account for every supplied ID and "
                  "include every required field; do not discuss the repair."
            )
        messages = [{"role": "user", "content": prompt}]
        reply = ""
        budget: TokenBudget | None = None
        error: Exception | None = None
        try:
            budget = plan_token_budget(
                task,
                capacity,
                messages=messages,
                expected_items=expected_items,
            )
            progress(budget.describe())
            reply = chat(
                config.llm_model,
                messages,
                temperature=0,
                think=False,
                cancel_event=cancel_event,
                task=task,
                expected_items=expected_items,
                context_tokens=capacity.context_window_tokens,
                token_budget=budget,
            )
            data = _json_object(reply, required_key)
            payload, contract_errors, scientific_failures = validate(data)
        except Exception as exc:  # noqa: BLE001 - preserved in the gate audit
            if isinstance(exc, CancelledError):
                raise
            error = exc
            contract_errors = [str(exc)]
            scientific_failures = []
        record = {
            "attempt": attempt_number,
            "budget": budget.as_dict() if budget is not None else None,
            "status": "complete" if not contract_errors else "contract_incomplete",
            "contract_errors": list(contract_errors),
            "scientific_failures": list(scientific_failures),
            "response": reply,
        }
        if error is not None:
            record["error_type"] = type(error).__name__
        attempts.append(record)
        if not contract_errors:
            record["action"] = (
                "scientific verdict retained" if scientific_failures
                else "complete response accepted"
            )
            return data, payload, scientific_failures, [], attempts
        if reply and budget is not None:
            _forget(forget_cached_reply, config, messages, budget)
        if isinstance(error, BudgetPlanningError):
            record["action"] = "request cannot fit; held for human review"
            break
        if attempt_number == 1:
            record["action"] = "cached reply evicted; one contract repair queued"
            progress(
                "Structured gate response was incomplete; its cached reply was removed and "
                "one complete-response repair is being requested."
            )
            continue
        record["action"] = "repair remained incomplete; held for human review"
    return data, payload, scientific_failures, contract_errors, attempts


def propose_groups(config, rows: list[dict], *, context_tokens: int | None,
                   progress=None, cancel_event=None) -> tuple[dict[str, dict], dict]:
    """Return only two-gate-validated proposals plus a complete candidate audit."""
    from process.llm import chat, forget_cached_reply, set_config

    progress = progress or (lambda message: None)
    capacity = resolve_model_capacity(
        config,
        requested_context_tokens=context_tokens if context_tokens and context_tokens > 0 else None,
        model=config.llm_model,
    )
    question = config.get_natural_language_question() or "Not specified."
    paper_context = _paper_context(config)
    families = _items(rows, paper_context)
    discovery_planning_holds: list[dict] = []
    requests = [
        {"family": family, "items": chunk, "parent_attempt": None, "repair": 0}
        for family, items in families.items()
        for chunk in _discovery_batches(
            items, capacity, question, family, discovery_planning_holds
        )
    ]
    initial_discovery_requests = len(requests)
    summary: dict[str, Any] = {
        "model": config.llm_model,
        "model_capacity": capacity.as_dict(),
        "items_considered": sum(len(items) for items in families.values()),
        "papers_considered": len({item["article_id"] for items in families.values() for item in items}),
        "initial_discovery_requests": initial_discovery_requests,
        "discovery_requests": 0, "review_requests": 0,
        "discovery_attempts": [], "token_actions": [], "holds": [],
        "discovery_planning_holds": discovery_planning_holds,
        "corpus_wide_candidates_seeded": 0,
        "requests": 0, "groups": [], "audits": [], "errors": [],
        "family_accounting": {
            family: {
                "items": len(items),
                "papers": len({item["article_id"] for item in items}),
                "discovery_batches": sum(1 for request in requests if request["family"] == family),
                "candidates_returned": 0, "structurally_rejected": 0,
                "title_rejected": 0, "characteristic_rejected": 0,
                "title_incomplete": 0, "characteristic_incomplete": 0,
                "conflicted": 0, "accepted": 0,
            }
            for family, items in families.items()
        },
    }
    set_config(config)

    candidate_audits: list[dict] = []
    candidate_signatures: set[tuple[str, frozenset[str]]] = set()
    item_by_id = {item["id"]: item for items in families.values() for item in items}
    successful_views: dict[str, int] = defaultdict(int)
    unmatched_views: dict[str, int] = defaultdict(int)
    all_partner_ids: dict[str, set[str]] = {}
    successful_partner_ids: dict[str, set[str]] = defaultdict(set)
    failed_partner_ids: dict[str, set[str]] = defaultdict(set)
    failed_view_attempts: dict[str, set[int]] = defaultdict(set)
    for family_items in families.values():
        for item in family_items:
            all_partner_ids[item["id"]] = {
                other["id"] for other in family_items
                if other["article_id"] != item["article_id"]
            }
    # The model's context windows must not be the only way two papers can meet.
    # Seed every corpus-wide label-similarity suggestion into the same strict
    # semantic gates. This is recall support, never automatic compatibility.
    suggested = suggest_groups(rows)
    seeded: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for family, items in families.items():
        for item in items:
            labels = {suggested[effect_id] for effect_id in item["effect_ids"]
                      if effect_id in suggested}
            for label in labels:
                seeded[(family, label)].append(item)
    for (family, label), members in seeded.items():
        # A corpus-wide seed must pass through the same context-aware gates as a
        # model candidate. Split it into overlapping audited views rather than
        # letting a large seed become an unauditable hidden exception.
        for seed_members in _discovery_batches(
            members, capacity, question, family, discovery_planning_holds
        ):
            raw_group = {
                "label": label,
                "candidate_reason": (
                    "corpus-wide outcome/population label similarity seed; all semantic gates "
                    "still required; this is one context-aware audited view"
                ),
                "members": [{"id": item["id"], "reversed": False} for item in seed_members],
            }
            item_map = {item["id"]: item for item in families[family]}
            audit = _candidate_from_group(raw_group, item_map, family, 0)
            member_ids = frozenset(member["id"] for member in audit["members"])
            signature = (family, member_ids)
            if member_ids and signature in candidate_signatures:
                continue
            candidate_signatures.add(signature)
            audit["audit_id"] = f"A{len(candidate_audits) + 1}"
            candidate_audits.append(audit)
            summary["corpus_wide_candidates_seeded"] += 1
            accounting = summary["family_accounting"][family]
            accounting["candidates_returned"] += 1
            if audit["status"] == "structurally_rejected":
                accounting["structurally_rejected"] += 1
    successful_discovery = 0
    discovery_number = 0
    while requests:
        if cancel_event is not None and cancel_event.is_set():
            break
        request = requests.pop(0)
        family = request["family"]
        chunk = request["items"]
        discovery_number += 1
        summary["discovery_requests"] = discovery_number
        progress(
            f"Discovering semantic candidates: attempt {discovery_number} "
            f"({len(chunk)} {FAMILY_NAMES[family]} estimates; "
            f"{len(requests)} planned attempts remain)"
        )
        messages = _discovery_messages(question, family, chunk)
        reply = ""
        budget: TokenBudget | None = None
        groups: list = []
        unmatched_ids: set[str] = set()
        failure: Exception | None = None
        try:
            budget = _discovery_budget(capacity, question, family, chunk)
            progress(budget.describe())
            reply = chat(
                config.llm_model,
                messages,
                temperature=0,
                think=False,
                cancel_event=cancel_event,
                task="meta_discovery",
                expected_items=len(chunk),
                context_tokens=capacity.context_window_tokens,
                token_budget=budget,
            )
            expected_ids = {item["id"] for item in chunk}
            groups, unmatched_ids = _discovery_in(reply, expected_ids)
            for item_id in expected_ids:
                successful_views[item_id] += 1
            for item_id in unmatched_ids:
                unmatched_views[item_id] += 1
            for item in chunk:
                successful_partner_ids[item["id"]].update(
                    other["id"] for other in chunk
                    if other["article_id"] != item["article_id"]
                )
            successful_discovery += 1
        except Exception as exc:  # noqa: BLE001 - preserve partial audited state
            failure = exc
        attempt: dict[str, Any] = {
            "attempt": discovery_number,
            "parent_attempt": request.get("parent_attempt"),
            "repair": request.get("repair", 0),
            "family": family,
            "family_name": FAMILY_NAMES[family],
            "estimate_ids": [item["id"] for item in chunk],
            "estimate_count": len(chunk),
            "budget": budget.as_dict() if budget is not None else None,
        }
        if failure is not None:
            logger.warning("Model candidate discovery attempt %d failed: %s", discovery_number, failure)
            for item in chunk:
                failed_view_attempts[item["id"]].add(discovery_number)
                failed_partner_ids[item["id"]].update(
                    other["id"] for other in chunk
                    if other["article_id"] != item["article_id"]
                )
            if isinstance(failure, ValueError) and budget is not None:
                _forget(forget_cached_reply, config, messages, budget)
            attempt.update({
                "status": "incomplete",
                "error": str(failure),
                "response": reply,
            })
            retriable = isinstance(failure, (
                ValueError,
                BudgetPlanningError,
                ContextWindowExceededError,
                OutputLimitExceededError,
            ))
            pieces = _split_discovery_batch(chunk) if retriable else []
            if pieces:
                attempt["action"] = "evicted unusable reply and split discovery batch"
                attempt["child_estimate_counts"] = [len(piece) for piece in pieces]
                requests[0:0] = [
                    {"family": family, "items": piece,
                     "parent_attempt": discovery_number, "repair": 0}
                    for piece in pieces
                ]
                progress(
                    f"Discovery attempt {discovery_number} was incomplete; its cached reply was "
                    f"removed when present and the {len(chunk)} estimates were split into "
                    + ", ".join(str(len(piece)) for piece in pieces)
                    + " estimate views."
                )
            elif retriable and request.get("repair", 0) < 1:
                attempt["action"] = "evicted unusable reply and queued one contract repair"
                requests.insert(0, {
                    "family": family,
                    "items": chunk,
                    "parent_attempt": discovery_number,
                    "repair": 1,
                })
                progress(
                    f"Discovery attempt {discovery_number} could not be split further; "
                    "one complete-response repair was queued."
                )
            else:
                attempt["action"] = "terminal failure retained for audit"
                summary["errors"].append(
                    f"{FAMILY_NAMES[family]} discovery attempt {discovery_number}: {failure}"
                )
            summary["discovery_attempts"].append(attempt)
            summary["token_actions"].append({
                "phase": "discovery",
                "attempt": discovery_number,
                "action": attempt["action"],
                "reason": str(failure),
                "budget": attempt["budget"],
            })
            continue
        attempt.update({
            "status": "complete",
            "action": "complete response accepted",
            "candidate_groups": len(groups),
            "unmatched_ids": sorted(unmatched_ids),
        })
        summary["discovery_attempts"].append(attempt)
        summary["token_actions"].append({
            "phase": "discovery",
            "attempt": discovery_number,
            "action": attempt["action"],
            "budget": attempt["budget"],
        })
        item_map = {item["id"]: item for item in chunk}
        for raw_group in groups:
            audit = _candidate_from_group(raw_group, item_map, family, discovery_number)
            member_ids = frozenset(member["id"] for member in audit["members"])
            signature = (family, member_ids)
            if member_ids and signature in candidate_signatures:
                continue
            candidate_signatures.add(signature)
            audit["audit_id"] = f"A{len(candidate_audits) + 1}"
            candidate_audits.append(audit)
            accounting = summary["family_accounting"][family]
            accounting["candidates_returned"] += 1
            if audit["status"] == "structurally_rejected":
                accounting["structurally_rejected"] += 1

    for index, audit in enumerate(candidate_audits, 1):
        if cancel_event is not None and cancel_event.is_set():
            break
        if audit["status"] != "awaiting_title_review":
            continue
        family = audit["family"]
        progress(
            f"Reviewing semantic candidate {index} of {len(candidate_audits)}: "
            f"title and research-question gate"
        )
        papers = _title_papers(audit)
        title_prompt = TITLE_REVIEW_PROMPT.format(
            question=question, label=audit["candidate_label"],
            candidate_reason=audit["candidate_reason"] or "not supplied",
            papers="\n".join(
                f"{paper_id} | {paper['title']}" for paper_id, paper in sorted(papers.items())
            ),
        )
        title_data, eligible_papers, title_scientific, title_contract, title_attempts = (
            _review_with_one_repair(
                config=config,
                chat=chat,
                forget_cached_reply=forget_cached_reply,
                capacity=capacity,
                task="meta_title",
                expected_items=len(papers),
                base_prompt=title_prompt,
                required_key="paper_reviews",
                validate=lambda data: _validate_title_review(data, audit),
                cancel_event=cancel_event,
                progress=progress,
            )
        )
        audit["title_attempts"] = title_attempts
        summary["review_requests"] += len(title_attempts)
        for attempt in title_attempts:
            summary["token_actions"].append({
                "phase": "title_review",
                "audit_id": audit["audit_id"],
                "attempt": attempt["attempt"],
                "action": attempt.get("action", attempt["status"]),
                "budget": attempt.get("budget"),
                "contract_errors": attempt.get("contract_errors", []),
            })
        audit["title_review"] = title_data
        if title_contract:
            audit["status"] = "title_review_incomplete"
            audit["rejection_reasons"].extend(title_contract)
            summary["family_accounting"][family]["title_incomplete"] += 1
            summary["holds"].append(
                f"{audit['audit_id']} title response incomplete after one repair: "
                + "; ".join(title_contract)
            )
            continue
        if title_scientific:
            audit["status"] = "title_rejected"
            audit["rejection_reasons"].extend(title_scientific)
            summary["family_accounting"][family]["title_rejected"] += 1
            continue

        progress(
            f"Reviewing semantic candidate {index} of {len(candidate_audits)}: "
            f"all compatibility characteristics"
        )
        eligible_set = set(eligible_papers)
        characteristic_prompt = CHARACTERISTIC_REVIEW_PROMPT.format(
            question=question, family=FAMILY_NAMES[family], label=audit["candidate_label"],
            title_result=json.dumps(title_data, ensure_ascii=False),
            estimates=_estimate_block(audit, eligible_set, paper_context),
        )
        eligible_estimates = sum(
            member["paper_id"] in eligible_set for member in audit["members"]
        )
        characteristic_data, selected, characteristic_scientific, characteristic_contract, characteristic_attempts = (
            _review_with_one_repair(
                config=config,
                chat=chat,
                forget_cached_reply=forget_cached_reply,
                capacity=capacity,
                task="meta_characteristics",
                expected_items=max(1, eligible_estimates),
                base_prompt=characteristic_prompt,
                required_key="characteristics",
                validate=lambda data: _validate_characteristic_review(
                    data, audit, eligible_set
                ),
                cancel_event=cancel_event,
                progress=progress,
            )
        )
        audit["characteristic_attempts"] = characteristic_attempts
        summary["review_requests"] += len(characteristic_attempts)
        for attempt in characteristic_attempts:
            summary["token_actions"].append({
                "phase": "characteristic_review",
                "audit_id": audit["audit_id"],
                "attempt": attempt["attempt"],
                "action": attempt.get("action", attempt["status"]),
                "budget": attempt.get("budget"),
                "contract_errors": attempt.get("contract_errors", []),
            })
        audit["characteristic_review"] = characteristic_data
        audit["selected_members"] = selected or []
        if characteristic_contract:
            audit["status"] = "characteristic_review_incomplete"
            audit["rejection_reasons"].extend(characteristic_contract)
            summary["family_accounting"][family]["characteristic_incomplete"] += 1
            summary["holds"].append(
                f"{audit['audit_id']} characteristic response incomplete after one repair: "
                + "; ".join(characteristic_contract)
            )
            continue
        if characteristic_scientific:
            final_verdict = str((characteristic_data or {}).get("final_verdict") or "").casefold()
            if final_verdict == "needs_human_review":
                audit["status"] = "characteristic_needs_human_review"
                summary["holds"].append(
                    f"{audit['audit_id']} model requested human review: "
                    + "; ".join(characteristic_scientific)
                )
            else:
                audit["status"] = "characteristic_rejected"
                summary["family_accounting"][family]["characteristic_rejected"] += 1
            audit["rejection_reasons"].extend(characteristic_scientific)
            continue
        if not selected:
            audit["status"] = "characteristic_review_incomplete"
            audit["rejection_reasons"].append("complete characteristic response selected no estimates")
            summary["family_accounting"][family]["characteristic_incomplete"] += 1
            summary["holds"].append(
                f"{audit['audit_id']} complete response selected no estimates; held for human review"
            )
            continue
        if len({member["article_id"] for member in selected}) < MIN_PAPERS:
            audit["status"] = "characteristic_rejected"
            audit["rejection_reasons"].append(
                f"final selection contains fewer than {MIN_PAPERS} independent papers"
            )
            summary["family_accounting"][family]["characteristic_rejected"] += 1
            continue
        audit["status"] = "accepted"

    # An estimate cannot silently join two incompatible semantic groups.
    accepted = [audit for audit in candidate_audits if audit["status"] == "accepted"]
    memberships: dict[str, list[dict]] = defaultdict(list)
    for audit in accepted:
        for member in audit.get("selected_members") or []:
            memberships[member["id"]].append(audit)
    conflicted_audits = {
        audit["audit_id"] for audits in memberships.values() if len(audits) > 1 for audit in audits
    }
    for audit in accepted:
        if audit["audit_id"] in conflicted_audits:
            audit["status"] = "conflict_needs_review"
            audit["rejection_reasons"].append(
                "one or more estimates also passed in another semantic group; no automatic group was chosen"
            )
            summary["family_accounting"][audit["family"]]["conflicted"] += 1

    proposals: dict[str, dict] = {}
    used_labels: set[str] = set()
    for audit in candidate_audits:
        if audit["status"] != "accepted":
            continue
        review = audit["characteristic_review"]
        base_label = _clip(review.get("group_label"))
        labelled = f"{base_label} ({FAMILY_NAMES[audit['family']]})"
        label = labelled
        suffix = 2
        while label in used_labels:
            label = f"{labelled} [set {suffix}]"
            suffix += 1
        used_labels.add(label)
        audit["accepted_label"] = label
        for member in audit.get("selected_members") or []:
            for effect_id in member["effect_ids"]:
                proposals[effect_id] = {
                    "group": label, "reversed": bool(member.get("reversed")),
                    "audit_id": audit["audit_id"], "validation": "semantic_pass",
                }
        group_summary = {
            "audit_id": audit["audit_id"], "label": label,
            "reference_comparison": _clip(review.get("reference_comparison")),
            "papers": len({member["article_id"] for member in audit["selected_members"]}),
            "paper_titles": sorted({member["title"] for member in audit["selected_members"]}),
            "effects": sum(len(member["effect_ids"]) for member in audit["selected_members"]),
            "review_summary": _clip(review.get("review_summary"), 1200),
        }
        summary["groups"].append(group_summary)
        summary["family_accounting"][audit["family"]]["accepted"] += 1

    summary["audits"] = candidate_audits
    summary["requests"] = summary["discovery_requests"] + summary["review_requests"]
    summary["candidates_returned"] = len(candidate_audits)
    summary["groups_accepted"] = len(summary["groups"])
    summary["groups_rejected"] = sum(1 for audit in candidate_audits if audit["status"] != "accepted")
    summary["effects_assigned"] = len(proposals)
    candidate_item_ids = {
        member["id"] for audit in candidate_audits for member in audit["members"]
    }
    without_candidate_ids = sorted(set(item_by_id) - candidate_item_ids)
    explicit_unmatched_ids = []
    unresolved_coverage_ids = []
    for item_id in without_candidate_ids:
        expected_partners = all_partner_ids.get(item_id, set())
        observed_partners = successful_partner_ids.get(item_id, set())
        complete_opportunity_coverage = expected_partners <= observed_partners
        explicitly_unmatched = (
            successful_views[item_id] > 0
            and unmatched_views[item_id] == successful_views[item_id]
        )
        if explicitly_unmatched and complete_opportunity_coverage:
            explicit_unmatched_ids.append(item_id)
        else:
            unresolved_coverage_ids.append(item_id)
    summary["items_without_any_candidate"] = len(without_candidate_ids)
    summary["items_explicitly_unmatched_in_all_views"] = len(explicit_unmatched_ids)
    summary["items_unresolved_due_to_incomplete_comparison_coverage"] = len(
        unresolved_coverage_ids
    )
    summary["unmatched_items"] = [
        {**_public_item(item_by_id[item_id]),
         "successful_discovery_views": successful_views[item_id],
         "unmatched_views": unmatched_views[item_id],
         "cross_paper_opportunities": len(all_partner_ids.get(item_id, set())),
         "cross_paper_opportunities_observed": len(
             successful_partner_ids.get(item_id, set())
         ),
         "unseen_partner_ids": sorted(
             all_partner_ids.get(item_id, set())
             - successful_partner_ids.get(item_id, set())
         ),
         "failed_view_attempts": sorted(failed_view_attempts.get(item_id, set())),
         "failed_view_partner_ids": sorted(
             failed_partner_ids.get(item_id, set())
             - successful_partner_ids.get(item_id, set())
         ),
         "accounting": (
             "model explicitly returned unmatched after complete cross-paper opportunity coverage"
             if item_id in explicit_unmatched_ids
             else "unresolved due to split, failed, or incomplete cross-paper opportunity views"
         )}
        for item_id in without_candidate_ids
    ]
    if unresolved_coverage_ids:
        summary["holds"].append(
            f"{len(unresolved_coverage_ids)} estimate item(s) remain unresolved because "
            "split, failed, or incomplete discovery views did not cover every cross-paper "
            "comparison opportunity. They were not classified as definitive no-matches."
        )
        summary["token_actions"].append({
            "phase": "discovery_accounting",
            "attempt": "coverage",
            "action": "held unresolved estimates; no definitive unmatched verdict",
            "reason": "cross-paper opportunity coverage incomplete",
            "budget": None,
        })
    unique_planning_holds: list[dict] = []
    held_ids: set[tuple[str, str]] = set()
    for hold in discovery_planning_holds:
        key = (str(hold.get("family") or ""), str(hold.get("estimate_id") or ""))
        if key in held_ids:
            continue
        held_ids.add(key)
        unique_planning_holds.append(hold)
        summary["holds"].append(
            f"{hold.get('estimate_id')} ({hold.get('title') or hold.get('article_id')}): "
            f"{hold.get('error')}"
        )
        summary["token_actions"].append({
            "phase": "discovery_planning",
            "attempt": hold.get("estimate_id"),
            "action": hold.get("action"),
            "reason": hold.get("error"),
            "budget": None,
        })
    summary["discovery_planning_holds"] = unique_planning_holds
    cancelled = cancel_event is not None and cancel_event.is_set()
    summary["status"] = (
        "cancelled" if cancelled else "failed"
        if initial_discovery_requests and successful_discovery == 0 else "partial"
        if summary["errors"] or summary["holds"] else "complete"
    )
    return proposals, summary


def save_proposals(path: Path, proposals: dict[str, dict], summary: dict) -> Path:
    atomic_write_json(Path(path), {"summary": summary, "proposals": proposals})
    return Path(path)


def render_grouping_review(summary: dict) -> str:
    """Render the complete model grouping audit without hiding failed candidates."""
    lines = [
        "# Meta-analysis Semantic Grouping Review", "",
        "Candidate discovery is recall-oriented. A candidate is pooled only after "
        "the title/research-question gate and every scientific compatibility "
        "characteristic pass.", "",
        f"- Model: {summary.get('model') or 'not recorded'}",
        f"- Context window used: "
        f"{(summary.get('model_capacity') or {}).get('context_window_tokens', 'not recorded')}",
        f"- Context source: {(summary.get('model_capacity') or {}).get('source', 'not recorded')}",
        f"- Estimate items considered: {summary.get('items_considered', 0)}",
        f"- Papers considered: {summary.get('papers_considered', 0)}",
        f"- Discovery requests: {summary.get('discovery_requests', 0)}",
        f"- Candidate groups returned: {summary.get('candidates_returned', 0)}",
        f"- Groups accepted: {summary.get('groups_accepted', 0)}",
        f"- Groups rejected or conflicted: {summary.get('groups_rejected', 0)}",
        f"- Estimate items never placed in any candidate: {summary.get('items_without_any_candidate', 0)}",
        f"- Explicitly unmatched after complete cross-paper opportunity coverage: "
        f"{summary.get('items_explicitly_unmatched_in_all_views', 0)}",
        f"- Unresolved because cross-paper opportunity coverage was incomplete: "
        f"{summary.get('items_unresolved_due_to_incomplete_comparison_coverage', 0)}",
        "", "## Token budgets and recovery actions", "",
        "Every request budget below was derived from the resolved model context. "
        "Structured response limits are explicit synthetic limits: they scale with the required "
        "item decisions because extra prose does not add another scientific decision. "
        "Incomplete replies are retained in model_groups.json, removed from the response cache, "
        "and split or repaired as recorded here.", "",
        "| Phase | Request | Action | Prompt estimate | Response allowance | Safety | Reason |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for action in summary.get("token_actions") or []:
        budget = action.get("budget") or {}
        reason = action.get("reason") or budget.get("limit_reason") or "not recorded"
        clean = lambda value: str(value or "").replace("|", "\\|").replace("\n", " ")
        request = action.get("audit_id") or action.get("attempt") or ""
        lines.append(
            f"| {clean(action.get('phase'))} | {clean(request)} | {clean(action.get('action'))} | "
            f"{budget.get('estimated_prompt_tokens', '')} | {budget.get('max_output_tokens', '')} | "
            f"{budget.get('safety_tokens', '')} | {clean(reason)} |"
        )
    holds = summary.get("holds") or []
    if holds:
        lines.extend(["", "### Responses held for human review", ""])
        lines.extend(f"- {hold}" for hold in holds)
    lines.extend([
        "", "## Family accounting", "",
        "| Family | Items | Papers | Discovery batches | Candidates | Structural rejects | Title rejects | Title incomplete | Characteristic rejects | Characteristic incomplete | Conflicts | Accepted |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for family, counts in (summary.get("family_accounting") or {}).items():
        lines.append(
            f"| {FAMILY_NAMES.get(family, family)} | {counts.get('items', 0)} | "
            f"{counts.get('papers', 0)} | {counts.get('discovery_batches', 0)} | "
            f"{counts.get('candidates_returned', 0)} | {counts.get('structurally_rejected', 0)} | "
            f"{counts.get('title_rejected', 0)} | {counts.get('title_incomplete', 0)} | "
            f"{counts.get('characteristic_rejected', 0)} | "
            f"{counts.get('characteristic_incomplete', 0)} | "
            f"{counts.get('conflicted', 0)} | {counts.get('accepted', 0)} |"
        )
    unmatched_items = summary.get("unmatched_items") or []
    if unmatched_items:
        lines.extend([
            "", "## Estimate items without a candidate", "",
            "These estimates were not silently discarded. The accounting column distinguishes "
            "an explicit no-match verdict with complete cross-paper opportunity coverage from "
            "split, failed, or otherwise incomplete comparison views.", "",
            "| ID | Paper title | Outcome | Comparison/exposure | Population | Statistic | Opportunity coverage | Failed attempts | Accounting |",
            "|---|---|---|---|---|---|---:|---|---|",
        ])
        for item in unmatched_items:
            def unmatched_cell(value):
                return str(value or "not stated").replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {unmatched_cell(item.get('id'))} | {unmatched_cell(item.get('title'))} | "
                f"{unmatched_cell(item.get('outcome'))} | {unmatched_cell(item.get('comparison'))} | "
                f"{unmatched_cell(item.get('population'))} | "
                f"{unmatched_cell(' / '.join(item.get('statistics') or []))} | "
                f"{item.get('cross_paper_opportunities_observed', 0)}/"
                f"{item.get('cross_paper_opportunities', 0)} | "
                f"{unmatched_cell(', '.join(map(str, item.get('failed_view_attempts') or [])))} | "
                f"{unmatched_cell(item.get('accounting'))} |"
            )
    audits = summary.get("audits") or []
    if not audits:
        lines.extend(["", "## Candidate audits", "", "The model returned no candidate groups."])
    for audit in audits:
        label = audit.get("accepted_label") or audit.get("candidate_label") or "Unlabelled candidate"
        lines.extend([
            "", f"## {audit.get('audit_id', 'Candidate')}: {label}", "",
            f"- Status: **{str(audit.get('status') or 'unknown').replace('_', ' ')}**",
            f"- Effect family: {audit.get('family_name') or audit.get('family') or 'unknown'}",
            f"- Discovery rationale: {audit.get('candidate_reason') or 'Not supplied'}",
        ])
        reasons = audit.get("rejection_reasons") or []
        if reasons:
            lines.append("- Rejection/hold reasons: " + "; ".join(str(reason) for reason in reasons))
        lines.extend([
            "", "### Matching papers and extracted statistics", "",
            "| Paper ID | Paper title | Outcome | Comparison/exposure | Population | Scale | Extracted statistic |",
            "|---|---|---|---|---|---|---|",
        ])
        for member in audit.get("members") or []:
            def cell(value):
                return str(value or "not stated").replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {cell(member.get('paper_id'))} | {cell(member.get('title'))} | "
                f"{cell(member.get('outcome'))} | {cell(member.get('comparison'))} | "
                f"{cell(member.get('population'))} | {cell(member.get('analysis_scale'))} | "
                f"{cell(' / '.join(member.get('statistics') or []))} |"
            )
        title_review = audit.get("title_review")
        lines.extend(["", "### Gate 1: title and research-question relevance", ""])
        if not isinstance(title_review, dict):
            lines.append("No usable title review was returned.")
        else:
            lines.extend([
                "| Paper | Exact title returned | RQ relevant | Comparable in principle | Reason |",
                "|---|---|---|---|---|",
            ])
            for review in title_review.get("paper_reviews") or []:
                if not isinstance(review, dict):
                    continue
                lines.append(
                    f"| {review.get('paper_id', '')} | {str(review.get('title') or '').replace('|', '\\|')} | "
                    f"{review.get('research_question_relevant')} | {review.get('comparable_in_principle')} | "
                    f"{str(review.get('reason') or '').replace('|', '\\|')} |"
                )
            lines.extend(["", "Group-level title verdict: " + str(title_review.get("reason") or "not supplied")])
        characteristic_review = audit.get("characteristic_review")
        lines.extend(["", "### Gate 2: scientific compatibility", ""])
        if not isinstance(characteristic_review, dict):
            lines.append("No usable characteristic review was returned.")
        else:
            lines.extend([
                "| Characteristic | Pass | Matched | Differences | Reason |",
                "|---|---|---|---|---|",
            ])
            characteristics = characteristic_review.get("characteristics") or {}
            for name in REQUIRED_CHARACTERISTICS:
                verdict = characteristics.get(name) if isinstance(characteristics, dict) else None
                verdict = verdict if isinstance(verdict, dict) else {}
                def clean(value):
                    return str(value or "not supplied").replace("|", "\\|").replace("\n", " ")
                lines.append(
                    f"| {name.replace('_', ' ')} | {clean(verdict.get('pass'))} | "
                    f"{clean(verdict.get('matched'))} | {clean(verdict.get('differences'))} | "
                    f"{clean(verdict.get('reason'))} |"
                )
            subgroup = characteristics.get("subgroup_independence", {}) \
                if isinstance(characteristics, dict) else {}
            handling = subgroup.get("within_paper_handling", {}) \
                if isinstance(subgroup, dict) else {}
            if handling:
                lines.extend([
                    "",
                    "Within-paper handling: "
                    + "; ".join(f"{paper_id}: {reason}" for paper_id, reason in handling.items()),
                ])
            exclusions = characteristic_review.get("excluded_members") or []
            if exclusions:
                titles_by_id = {
                    member.get("id"): member.get("title") for member in audit.get("members") or []
                }
                lines.extend(["", "Excluded candidate members:"])
                for exclusion in exclusions:
                    if not isinstance(exclusion, dict):
                        continue
                    effect_id = exclusion.get("id") or "unknown ID"
                    title = titles_by_id.get(effect_id) or "unknown paper"
                    lines.append(
                        f"- {effect_id} — {title}: {exclusion.get('reason') or 'no reason supplied'}"
                    )
            lines.extend([
                "", f"Final verdict: **{characteristic_review.get('final_verdict') or 'not supplied'}**", "",
                str(characteristic_review.get("review_summary") or "No review summary supplied."),
            ])
    lines.append("")
    return "\n".join(lines)


def save_grouping_review(path: Path, summary: dict) -> Path:
    atomic_write_text(Path(path), render_grouping_review(summary))
    return Path(path)


def with_proposals(decisions: dict[str, dict], proposals: dict[str, dict]) -> dict[str, dict]:
    """Apply only proposals that carry a completed two-gate semantic audit."""
    merged = dict(decisions)
    for effect_id, proposal in proposals.items():
        if proposal.get("validation") != "semantic_pass" or not proposal.get("audit_id"):
            continue
        decision = merged.get(effect_id, {})
        if not str(decision.get("group") or "").strip():
            merged[effect_id] = {
                **decision, "group": proposal["group"], "reversed": proposal["reversed"],
                "group_source": "model_validated", "group_audit_id": proposal["audit_id"],
            }
    return merged
