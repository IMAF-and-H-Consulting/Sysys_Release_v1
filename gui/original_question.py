"""Project-scoped original research question lifecycle."""
from __future__ import annotations

from dataclasses import dataclass


def _clean(question: object) -> str:
    return str(question or "").strip()


@dataclass(slots=True)
class OriginalQuestionState:
    value: str = ""
    captured: bool = False

    @classmethod
    def from_window_state(cls, window_state: object) -> OriginalQuestionState:
        if not isinstance(window_state, dict):
            window_state = {}
        stored = _clean(window_state.get("original_research_question"))
        if stored:
            captured = bool(
                window_state.get("original_question_captured", True)
            )
            return cls(stored, captured)
        legacy_query = _clean(window_state.get("search_query"))
        return cls(legacy_query, bool(legacy_query))

    def set_prospective(self, question: str) -> str:
        if not self.captured:
            self.value = _clean(question)
        return self.value

    def capture_first_search(self, question: str) -> str:
        if not self.captured:
            self.value = _clean(question) or self.value
            if not self.value:
                raise ValueError("Original research question cannot be empty")
            self.captured = True
        return self.value

    def edit_captured(self, question: str) -> str:
        replacement = _clean(question)
        if not replacement:
            raise ValueError("Original research question cannot be empty")
        self.value = replacement
        self.captured = True
        return self.value

    def question_for_model(self, current_query: str) -> str:
        return self.value or _clean(current_query)

    def to_window_state(self) -> dict[str, object]:
        return {
            "original_research_question": self.value,
            "original_question_captured": self.captured,
        }
