"""
models/rubric.py
────────────────
Pydantic model for a structured job-requirement scoring rubric.
Produced by the Requirement Parser Node from free-text user input.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ScoringWeights(BaseModel):
    """Relative weights for each scoring dimension.  Must sum to 1.0."""

    skills: float = Field(0.5, ge=0.0, le=1.0)
    experience: float = Field(0.3, ge=0.0, le=1.0)
    domain: float = Field(0.2, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "ScoringWeights":
        total = round(self.skills + self.experience + self.domain, 6)
        if abs(total - 1.0) > 0.01:
            # Auto-normalise rather than hard-fail — better UX for LLM output
            s, e, d = self.skills, self.experience, self.domain
            self.skills = round(s / total, 4)
            self.experience = round(e / total, 4)
            self.domain = round(1 - self.skills - self.experience, 4)
        return self


SeniorityLevel = Literal[
    "intern", "junior", "mid", "senior", "staff", "principal", "lead", "manager", "director", "vp", "any"
]


class ScoringRubric(BaseModel):
    """
    Structured representation of what the recruiter is looking for.
    This drives all downstream scoring logic.
    """

    # Core requirements
    required_skills: list[str] = Field(
        default_factory=list,
        description="Must-have technical skills, tools, languages",
    )
    preferred_skills: list[str] = Field(
        default_factory=list,
        description="Nice-to-have skills — contribute to score but not mandatory",
    )

    # Experience
    min_years_experience: int = Field(
        0,
        ge=0,
        le=50,
        description="Minimum total years of professional experience required",
    )
    seniority_level: SeniorityLevel = Field(
        "any",
        description="Target seniority level",
    )

    # Domain
    domain: str = Field(
        "",
        description="Industry / technical domain (e.g. 'backend engineering', 'data science')",
    )
    additional_requirements: list[str] = Field(
        default_factory=list,
        description="Any other requirements that don't fit neatly into skills/experience",
    )

    # Weights
    weights: ScoringWeights = Field(default_factory=ScoringWeights)

    # Meta
    raw_requirement: str = Field(
        "",
        description="Original user input — stored for traceability",
    )
    parser_confidence: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="How confident the parser was in extracting this rubric",
    )

    def to_query_string(self) -> str:
        """
        Build a human-readable query string for embedding-based retrieval.
        Combines required skills, preferred skills, domain, and seniority.
        """
        parts: list[str] = []
        if self.domain:
            parts.append(f"Domain: {self.domain}.")
        if self.required_skills:
            parts.append(f"Required skills: {', '.join(self.required_skills)}.")
        if self.preferred_skills:
            parts.append(f"Preferred skills: {', '.join(self.preferred_skills)}.")
        if self.min_years_experience:
            parts.append(f"Minimum {self.min_years_experience} years of experience.")
        if self.seniority_level != "any":
            parts.append(f"Seniority: {self.seniority_level}.")
        if self.additional_requirements:
            parts.append(
                f"Additional requirements: {', '.join(self.additional_requirements)}."
            )
        return " ".join(parts) or self.raw_requirement