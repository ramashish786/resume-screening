from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ScoringWeights(BaseModel):
    skills: float = Field(0.5, ge=0.0, le=1.0)
    experience: float = Field(0.3, ge=0.0, le=1.0)
    domain: float = Field(0.2, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "ScoringWeights":
        total = round(self.skills + self.experience + self.domain, 6)
        if abs(total - 1.0) > 0.01:
            # LLM sometimes returns weights that don't sum to 1, normalise rather than crash
            s, e, d = self.skills, self.experience, self.domain
            self.skills = round(s / total, 4)
            self.experience = round(e / total, 4)
            self.domain = round(1 - self.skills - self.experience, 4)
        return self


SeniorityLevel = Literal[
    "intern", "junior", "mid", "senior", "staff",
    "principal", "lead", "manager", "director", "vp", "any"
]


class ScoringRubric(BaseModel):
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    min_years_experience: int = Field(0, ge=0, le=50)
    seniority_level: SeniorityLevel = Field("any")
    domain: str = Field("")
    additional_requirements: list[str] = Field(default_factory=list)
    weights: ScoringWeights = Field(default_factory=ScoringWeights)
    raw_requirement: str = Field("")
    parser_confidence: float = Field(1.0, ge=0.0, le=1.0)

    def to_query_string(self) -> str:
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
            parts.append(f"Additional: {', '.join(self.additional_requirements)}.")
        return " ".join(parts) or self.raw_requirement