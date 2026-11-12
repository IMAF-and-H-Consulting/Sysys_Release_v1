"""Enumerations and data models for the GRADE certainty assessment."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class GradeConcern(Enum):
    NO_CONCERN = "no_concern"
    SERIOUS = "serious"
    VERY_SERIOUS = "very_serious"


class GradeCertainty(Enum):
    HIGH = "High"
    MODERATE = "Moderate"
    LOW = "Low"
    VERY_LOW = "Very Low"


@dataclass
class DomainRating:
    domain: str  # risk_of_bias / inconsistency / indirectness / imprecision / publication_bias
    concern: GradeConcern = GradeConcern.NO_CONCERN
    rationale: str = ""
    downgrade_levels: int = 0  # 0, 1, or 2
    data_points: dict = field(default_factory=dict)


@dataclass
class ComponentGrade:
    component_id: str
    component_label: str
    n_studies: int = 0
    starting_certainty: str = "Low"  # "High" or "Low"
    domains: list = field(default_factory=list)  # list[DomainRating]
    overall_certainty: GradeCertainty = GradeCertainty.VERY_LOW
    summary: str = ""
