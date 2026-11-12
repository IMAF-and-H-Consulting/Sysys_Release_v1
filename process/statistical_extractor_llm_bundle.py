"""Bundled Stage 2c LLM context checks: several statistics from one paper per call.

A bundle shares one copy of the instructions, title and abstract, and lists each
statistic under an ID (S1, S2, ...). The reply must be a JSON object holding
exactly one verdict per ID. Each ID's verdict is filed under that ID, so when
some IDs are missing or lack a verdict, the valid verdicts are applied and only
the others are rejected. A reply whose IDs cannot be trusted (unparseable or cut
off, an ID repeated or not asked about) is rejected as a whole. Rejected
statistics are flagged for human review with the raw reply and the model's
answer for them. A bundle of one statistic uses the single-window prompt unchanged.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

from process.llm_budget import (
    BudgetPlanningError,
    ModelCapacity,
    TokenBudget,
    plan_token_budget,
)
from process.statistical_extractor_llm import (
    SOURCE_RULES,
    VERDICT_FIELDS,
    VERDICT_RULES,
    candidate_json,
    failed_review,
    reply_json_candidates,
    review_from_data,
    review_verdict,
)

# Real Stage 2c prompts run about 4.2 characters per token; counting 3.5
# over-estimates tokens a little so a bundle stays inside the model's context.
CHARS_PER_TOKEN = 3.5
_SHOWN_KEY_CHARS = 20  # unexpected keys are quoted in the rejection reason, truncated
_STATISTIC_ID = re.compile(r"S\d+")

BUNDLE_PROMPT = (
    "You are validating {count} local evidence windows from one scientific paper.\n\n"
    "For each statistic below, decide separately whether it represents actual study data "
    "that can be used in evidence synthesis or meta-analysis, judging it from its own local "
    "context window. " + SOURCE_RULES + """

Paper title:
{title}

Additional source context (complete abstract when available):
{article_context}

Statistics to validate:

{statistics}

Return ONLY a JSON object whose keys are exactly these {count} statistic IDs: {id_list}. Include every ID exactly once and no other keys. The value for each ID is the verdict for that statistic, a JSON object with these fields:
""" + VERDICT_FIELDS + "\n\nApply these rules to each statistic separately:\n" + VERDICT_RULES
)

_STATISTIC_SECTION = (
    "=== {sid} ===\nDeterministic extraction candidate:\n{candidate}\n\n"
    "Local context window:\n{context}"
)

Target = tuple[dict[str, Any], dict[str, Any], str]  # (record, candidate, context)


def statistic_id(position: int) -> str:
    return f"S{position + 1}"


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def build_bundle_prompt(record: dict[str, Any], members: list[tuple[dict[str, Any], str]]) -> str:
    """Prompt for several (candidate, context) statistics of one paper."""
    ids = [statistic_id(position) for position in range(len(members))]
    sections = "\n\n".join(
        _STATISTIC_SECTION.format(sid=sid, candidate=candidate_json(candidate), context=context or "")
        for sid, (candidate, context) in zip(ids, members, strict=True)
    )
    return BUNDLE_PROMPT.format(
        count=len(ids),
        title=str(record.get("title") or ""),
        article_context=str(record.get("review_context") or "") or "Not supplied.",
        statistics=sections,
        id_list=", ".join(f'"{sid}"' for sid in ids),
    )


@dataclass(frozen=True)
class Bundle:
    indices: tuple[int, ...]  # target positions, in target order
    prompt: str
    token_budget: TokenBudget | None
    planning_error: dict[str, Any] | None = None

    @property
    def size(self) -> int:
        return len(self.indices)

    @property
    def max_tokens(self) -> int:
        """Planned response allowance; zero means the complete request cannot fit."""
        return self.token_budget.max_output_tokens if self.token_budget is not None else 0

    @property
    def estimated_tokens(self) -> int:
        """Complete request estimate: prompt, response, and safety reserve."""
        if self.token_budget is not None:
            return self.token_budget.planned_total_tokens
        return estimate_tokens(self.prompt)

    def budget_dict(self) -> dict[str, Any]:
        if self.token_budget is not None:
            return self.token_budget.as_dict()
        return {"planning_error": dict(self.planning_error or {})}


def make_bundle(
    targets: list[Target],
    single_prompts: list[str],
    indices: list[int] | tuple[int, ...],
    *,
    capacity: ModelCapacity,
    explicit_output_tokens: int | None = None,
) -> Bundle:
    """Build and budget one complete request without dropping prompt material."""
    indices = list(indices)
    if len(indices) == 1:
        prompt = single_prompts[indices[0]]
    else:
        record = targets[indices[0]][0]
        prompt = build_bundle_prompt(record, [(targets[i][1], targets[i][2]) for i in indices])
    messages = [{"role": "user", "content": prompt}]
    try:
        budget = plan_token_budget(
            "statistical_review",
            capacity,
            messages=messages,
            expected_items=len(indices),
            explicit_output_tokens=explicit_output_tokens,
            explicit_reason=(
                "The caller supplied an explicit Stage 2c response allowance. This is a "
                "synthetic limit and replaces automatic schema-and-item scaling for this request."
            ),
        )
    except BudgetPlanningError as exc:
        return Bundle(tuple(indices), prompt, None, exc.as_dict())
    return Bundle(tuple(indices), prompt, budget)


def _by_paper(targets: list[Target]) -> list[list[int]]:
    """Target positions grouped by paper, keeping target order."""
    groups: list[list[int]] = []
    for index in range(len(targets)):
        if groups and targets[groups[-1][-1]][0] is targets[index][0]:
            groups[-1].append(index)
        else:
            groups.append([index])
    return groups


def plan_bundles(
    targets: list[Target],
    single_prompts: list[str],
    *,
    capacity: ModelCapacity,
    explicit_output_tokens: int | None = None,
) -> list[Bundle]:
    """Pack each paper into the largest complete requests that fit ``capacity``.

    The fit test uses the entire prompt, the context-scaled response required
    for every statistic ID, and the planner's safety reserve. Papers are never
    mixed. A singleton that cannot fit remains in the plan with its explicit
    planning error so the runner can produce a terminal human-review verdict
    instead of aborting or silently shortening it.
    """
    bundles: list[Bundle] = []
    for paper in _by_paper(targets):
        current = make_bundle(
            targets, single_prompts, [paper[0]], capacity=capacity,
            explicit_output_tokens=explicit_output_tokens,
        )
        for index in paper[1:]:
            if current.planning_error is not None:
                bundles.append(current)
                current = make_bundle(
                    targets, single_prompts, [index], capacity=capacity,
                    explicit_output_tokens=explicit_output_tokens,
                )
                continue
            grown = make_bundle(
                targets, single_prompts, [*current.indices, index], capacity=capacity,
                explicit_output_tokens=explicit_output_tokens,
            )
            if grown.planning_error is None:
                current = grown
                continue
            bundles.append(current)
            current = make_bundle(
                targets, single_prompts, [index], capacity=capacity,
                explicit_output_tokens=explicit_output_tokens,
            )
        bundles.append(current)
    return bundles


class _RepeatedKeys(ValueError):
    pass


def _without_repeats(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Refuse a repeated statistic ID; a field repeated inside one verdict keeps
    its last value, as it would in a single-statistic reply."""
    keys = [key for key, _value in pairs if _STATISTIC_ID.fullmatch(key)]
    repeated = sorted({key for key in keys if keys.count(key) > 1})
    if repeated:
        raise _RepeatedKeys(", ".join(repeated))
    return dict(pairs)


def _parse_reply(text: str) -> tuple[Any, str | None]:
    """Return (parsed JSON, None) or (None, why it could not be read)."""
    if not (text or "").strip():
        return None, "empty reply"
    for candidate in reply_json_candidates(text):
        try:
            return json.loads(candidate, object_pairs_hook=_without_repeats), None
        except _RepeatedKeys as exc:
            return None, f"repeated key {exc}"
        except json.JSONDecodeError:
            continue
    return None, "reply is not valid JSON (cut off or malformed)"


def _reply_problems(data: Any, ids: list[str]) -> list[str]:
    if not isinstance(data, dict):
        return ["reply is not a JSON object keyed by statistic ID"]
    problems = []
    missing = [sid for sid in ids if sid not in data]
    unexpected = [str(key)[:_SHOWN_KEY_CHARS] for key in data if key not in ids]
    no_verdict = [
        sid for sid in ids
        if sid in data and (not isinstance(data[sid], dict) or review_verdict(data[sid]) is None)
    ]
    if missing:
        problems.append("missing " + ", ".join(missing))
    if unexpected:
        problems.append("unexpected " + ", ".join(unexpected))
    if no_verdict:
        problems.append("no valid verdict for " + ", ".join(no_verdict))
    return problems


def _rejected_review(reason: str, sid: str, answer: Any, bundle: Bundle) -> dict[str, Any]:
    return {
        **failed_review(f"incomplete bundle reply ({reason})"),
        "bundle_rejected": True,
        "bundle": {"statistic_id": sid, "statistics": bundle.size, "reject_reason": reason},
        "model_answer": answer,  # what the model said for this statistic; not applied
    }


def _has_verdict(answer: Any) -> bool:
    return isinstance(answer, dict) and review_verdict(answer) is not None


def bundle_reviews(reply: str, bundle: Bundle) -> list[dict[str, Any]]:
    """One review per statistic in ``bundle``.

    Valid verdicts are applied; a missing or verdict-less ID rejects only that
    statistic. An unreadable reply or unexpected IDs reject the whole bundle.
    """
    ids = [statistic_id(position) for position in range(bundle.size)]
    data, unreadable = _parse_reply(reply)
    problems = [unreadable] if unreadable else _reply_problems(data, ids)
    if not problems:
        return [review_from_data(data[sid]) for sid in ids]
    reason = "; ".join(problems)
    answers = data if isinstance(data, dict) else {}
    trusted = not unreadable and not any(p.startswith("unexpected") for p in problems)
    return [review_from_data(answers[sid]) if trusted and _has_verdict(answers.get(sid))
            else _rejected_review(reason, sid, answers.get(sid), bundle) for sid in ids]


@dataclass
class BundleStats:
    """Auditable request planning, recovery, and terminal-failure accounting."""

    planned: int = 0
    requests: int = 0
    replies: int = 0
    budgets: list[dict[str, Any]] = field(default_factory=list)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    rejections: list[dict[str, Any]] = field(default_factory=list)
    split_events: list[dict[str, Any]] = field(default_factory=list)
    recovery_events: list[dict[str, Any]] = field(default_factory=list)
    terminal_events: list[dict[str, Any]] = field(default_factory=list)
    planning: dict[str, Any] = field(default_factory=dict)

    def record_plan(self, bundles: list[Bundle], planning: dict[str, Any]) -> None:
        self.planned = len(bundles)
        self.planning = dict(planning)

    def record_attempts(
        self,
        attempts: list[dict[str, Any]],
        *,
        record: dict[str, Any],
    ) -> None:
        """Keep every raw attempt and its exact budget, including recovered ones."""
        title = str(record.get("title") or "")
        article_id = str(record.get("article_id") or "")
        for raw in attempts:
            number = len(self.attempts) + 1
            attempt = {"attempt": number, "title": title, "article_id": article_id, **raw}
            self.attempts.append(attempt)
            if attempt.get("sent"):
                self.requests += 1
            if attempt.get("reply") is not None:
                self.replies += 1
            budget = attempt.get("budget")
            if isinstance(budget, dict):
                self.budgets.append({"attempt": number, "indices": attempt.get("indices"), **budget})
            rejected = int(attempt.get("rejected_statistics") or 0)
            if rejected:
                self.rejections.append({
                    "rejection": len(self.rejections) + 1,
                    "attempt": number,
                    "title": title,
                    "article_id": article_id,
                    "statistics": rejected,
                    "reason": str(attempt.get("reason") or "invalid bundle reply"),
                    "reply": attempt.get("reply"),
                    "action": str(attempt.get("action") or "retry-smaller"),
                })

    def record_events(
        self,
        *,
        splits: list[dict[str, Any]],
        recoveries: list[dict[str, Any]],
        terminals: list[dict[str, Any]],
        record: dict[str, Any],
    ) -> None:
        identity = {
            "title": str(record.get("title") or ""),
            "article_id": str(record.get("article_id") or ""),
        }
        self.split_events.extend({**identity, **event} for event in splits)
        self.recovery_events.extend({**identity, **event} for event in recoveries)
        self.terminal_events.extend({**identity, **event} for event in terminals)

    def as_dict(self) -> dict[str, Any]:
        return {
            "planned": self.planned,
            "requests": self.requests,
            "replies": self.replies,
            "rejected": len(self.rejections),
            "rejections": list(self.rejections),
            "budgets": list(self.budgets),
            "attempts": list(self.attempts),
            "splits": len(self.split_events),
            "split_events": list(self.split_events),
            "recoveries": len(self.recovery_events),
            "recovered_statistics": sum(int(event.get("statistics") or 0)
                                        for event in self.recovery_events),
            "recovery_events": list(self.recovery_events),
            "terminal_failures": sum(int(event.get("statistics") or 0)
                                     for event in self.terminal_events),
            "terminal_events": list(self.terminal_events),
            "planning": dict(self.planning),
        }
