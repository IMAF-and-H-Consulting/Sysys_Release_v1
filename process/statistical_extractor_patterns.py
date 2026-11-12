"""Shared constants, regular expressions, and the TextBlock record."""
from __future__ import annotations

import re
from dataclasses import dataclass

_SIGN = "[-+\u2212\u2013]"
_DECIMAL_POINT = "[.\u00b7]"
_SUPERSCRIPT_DIGITS = "[\u2070\u00b9\u00b2\u00b3\u2074-\u2079]"
_EXPONENT = (
    rf"(?:[eE]{_SIGN}?\d+"
    rf"|\s*[\u00d7xX]\s*10\s*(?:\^\s*{_SIGN}?\s*\d+|{_SIGN}\s*\d+"
    rf"|[\u207a\u207b]?{_SUPERSCRIPT_DIGITS}+))"
)
# The trailing lookahead stops a match from backtracking to an integer prefix
# (``0·45`` -> 0, ``1.2e-5`` -> 1) when the full number cannot be matched.
NUMBER_RE = (
    rf"{_SIGN}?(?:\d+\s*{_DECIMAL_POINT}\s*\d+|{_DECIMAL_POINT}\s*\d+|\d+)"
    rf"(?:{_EXPONENT})?(?!\d|{_DECIMAL_POINT}\d|[eE]{_SIGN}?\d)"
)
# Whole-number counts, allowing grouped thousands (1,234 / 12 345) but not a
# list such as "12, 15". The left boundary stops a match inside "1,234".
# Counts accept comma and typographic-space thousands grouping ("1,234", "1\u2009234").
# A plain space only groups digits after an explicit "N =" (COUNT_SPACED_RE), so
# "Table 2 300 patients" is not read as 2,300.
COUNT_RE = r"(?<![\d.,\u00b7])(?:\d{1,3}(?:[,\u2009\u202f\u00a0]\d{3})+|\d{1,6})(?!\d|[.\u00b7]\d)"
COUNT_SPACED_RE = r"(?<![\d.,\u00b7])(?:\d{1,3}(?:[,\u2009\u202f\u00a0 ]\d{3})+|\d{1,6})(?!\d|[.\u00b7]\d)"
CI_LABEL_RE = r"(?P<ci_level>9[059])\s*%\s*(?i:CI|confidence\s+intervals?)"
RATIO_RE = re.compile(
    rf"\b(?P<effect_type>aOR|OR|RR|HR|(?i:odds[ -]ratio|risk[ -]ratio|hazard[ -]ratio))"
    rf"\s*(?:\((?:OR|RR|HR)\))?\s*(?:,\s*{CI_LABEL_RE}\s*)?[:=,]\s*(?P<stat>{NUMBER_RE})(?!\s*%)"
)


@dataclass(frozen=True)
class TextBlock:
    text: str
    source: str
    paragraph_index: int
    char_start: int
    section: str

    @property
    def char_end(self) -> int:
        return self.char_start + len(self.text)


TEST_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "t_test",
        re.compile(
            rf"\bt\s*\(\s*(?P<df>{NUMBER_RE})\s*\)\s*=\s*(?P<stat>{NUMBER_RE})",
            re.IGNORECASE,
        ),
    ),
    (
        "f_test",
        re.compile(
            rf"\bF\s*\(\s*(?P<df1>{NUMBER_RE})\s*,\s*(?P<df2>{NUMBER_RE})\s*\)"
            rf"\s*=\s*(?P<stat>{NUMBER_RE})",
            re.IGNORECASE,
        ),
    ),
    (
        "chi_square",
        re.compile(
            f"(?:chi[- ]?square|x\\^?2|\u03c7\\s*(?:2|\\^2|\u00b2))"
            rf"\s*(?:\(\s*(?P<df>{NUMBER_RE})\s*(?:,\s*N\s*=\s*{COUNT_RE}\s*)?\))?"
            rf"\s*=\s*(?P<stat>{NUMBER_RE})",
            re.IGNORECASE,
        ),
    ),
    (
        "z_test",
        re.compile(rf"\bz\s*=\s*(?P<stat>{NUMBER_RE})", re.IGNORECASE),
    ),
    (
        "correlation",
        re.compile(
            rf"\b(?:r|rho|rs)\s*(?:\(\s*(?P<df>{NUMBER_RE})\s*\))?\s*=\s*(?P<stat>{NUMBER_RE})",
            re.IGNORECASE,
        ),
    ),
    (
        "regression_coefficient",
        re.compile(
            rf"(?<!\w)(?:beta|β|b|coef(?:ficient)?)\s*=\s*(?P<stat>{NUMBER_RE})",
            re.IGNORECASE,
        ),
    ),
    (
        "effect_size",
        re.compile(
            rf"\b(?:Cohen['']?s\s+)?(?P<effect_type>d|g)\s*=\s*(?P<stat>{NUMBER_RE})",
            re.IGNORECASE,
        ),
    ),
    (
        "odds_or_risk_ratio",
        RATIO_RE,
    ),
)

P_VALUE_RE = re.compile(
    rf"\bp(?:[- ]?values?)?\s*(?P<operator><=|>=|[<>=\u2264\u2265])\s*(?P<value>{NUMBER_RE})"
    rf"(?:\b|(?=\s))",
    re.IGNORECASE,
)
CI_RE = re.compile(
    rf"(?P<level>9[059])\s*%?\s*(?:CI|confidence\s+intervals?)\s*\)?\s*[:=,]?\s*[\[(]?"
    rf"\s*(?P<lower>{NUMBER_RE})\s*(?:,|;|\bto\b|[-–—])\s*(?P<upper>{NUMBER_RE})",
    re.IGNORECASE,
)
# An interval whose level label was already consumed (e.g. "HR, 95% CI: est, lo to hi").
UNLABELLED_CI_RE = re.compile(
    rf"\s*[,;]?\s*[\[(]?\s*(?P<lower>{NUMBER_RE})\s*(?:,|;|\bto\b|[-–—])\s*(?P<upper>{NUMBER_RE})",
    re.IGNORECASE,
)
SAMPLE_RE = re.compile(
    r"(?:\b(?:total\s+of|included|enrolled|sample(?:\s+of)?)\s+)?"
    rf"(?:\b[Nn]\s*=\s*)?(?P<n>{COUNT_RE})\s+"
    r"(?P<label>participants|patients|subjects|children|adults|adolescents|"
    r"students|controls|cases|mice|rats|animals|dyads|families|respondents|"
    r"households|schools|firms|plots|sites|countries|teachers|employees)",
    re.IGNORECASE,
)
N_EQUALS_RE = re.compile(rf"\b[Nn]\s*=\s*(?P<n>{COUNT_SPACED_RE})\b")
AGE_RE = re.compile(
    rf"\bage\s*(?:mean|M)?\s*[=:]?\s*(?P<age>{NUMBER_RE})"
    rf"(?:\s*(?:years|yrs?))?(?:\s*,?\s*SD\s*=\s*(?P<sd>{NUMBER_RE}))?",
    re.IGNORECASE,
)
COMPARISON_RE = re.compile(
    r"\b(?P<a>[A-Za-z][A-Za-z0-9 /-]{2,40})\s+"
    r"(?:vs\.?|versus|compared with|compared to)\s+"
    r"(?P<b>[A-Za-z][A-Za-z0-9 /-]{2,40})",
    re.IGNORECASE,
)

SUBJECT_WORDS = (
    "participants", "patients", "subjects", "children", "adults", "adolescents",
    "students", "controls", "cases", "mice", "rats", "animals", "caregivers",
    "families", "dyads", "cohort", "sample", "respondents", "households",
    "schools", "firms", "plots", "sites", "countries", "teachers", "employees",
)
