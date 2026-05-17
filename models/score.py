from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, computed_field


class MatchLevel(str, Enum):
    STRONG = "strong"
    GOOD = "good"
    PARTIAL = "partial"
    WEAK = "weak"
    NO_MATCH = "no_match"


class ResumeDocument(BaseModel):
    candidate_name: str = Field("Unknown")
    raw_text: str
    file_name: str
    file_hash: str
    page_count: int = Field(1, ge=1)
    parse_warnings: list[str] = Field(default_factory=list)
    word_count: int = Field(0, ge=0)
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None

    @computed_field  # type: ignore[misc]
    @property
    def is_sufficient(self) -> bool:
        return self.word_count >= 80


class CandidateScore(BaseModel):
    candidate_name: str
    file_name: str
    email: Optional[str] = None
    phone: Optional[str] = None

    skills_score: float = Field(0.0, ge=0.0, le=100.0)
    experience_score: float = Field(0.0, ge=0.0, le=100.0)
    domain_score: float = Field(0.0, ge=0.0, le=100.0)
    overall_score: float = Field(0.0, ge=0.0, le=100.0)

    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    matched_preferred_skills: list[str] = Field(default_factory=list)
    experience_years_found: Optional[int] = None
    seniority_detected: Optional[str] = None

    justification: str = ""
    key_strengths: list[str] = Field(default_factory=list)
    key_gaps: list[str] = Field(default_factory=list)

    confidence: float = Field(1.0, ge=0.0, le=1.0)
    retrieved_chunks: list[str] = Field(default_factory=list)
    error: Optional[str] = None

    @computed_field
    @property
    def match_level(self) -> MatchLevel:
        s = self.overall_score
        if s >= 75:
            return MatchLevel.STRONG
        if s >= 55:
            return MatchLevel.GOOD
        if s >= 35:
            return MatchLevel.PARTIAL
        if s > 0:
            return MatchLevel.WEAK
        return MatchLevel.NO_MATCH

    @computed_field
    @property
    def match_level_emoji(self) -> str:
        return {
            MatchLevel.STRONG: "🟢",
            MatchLevel.GOOD: "🔵",
            MatchLevel.PARTIAL: "🟡",
            MatchLevel.WEAK: "🟠",
            MatchLevel.NO_MATCH: "🔴",
        }[self.match_level]

    @computed_field
    @property
    def needs_manual_review(self) -> bool:
        return self.confidence < 0.3 or self.error is not None


class RankedResult(BaseModel):
    candidates: list[CandidateScore] = Field(default_factory=list)
    comparative_summary: str = ""
    rubric_used: str = ""
    total_processed: int = 0
    total_failed: int = 0
    no_match_count: int = 0

    @property
    def top_candidate(self) -> Optional[CandidateScore]:
        return self.candidates[0] if self.candidates else None

    @property
    def has_strong_match(self) -> bool:
        return any(c.match_level == MatchLevel.STRONG for c in self.candidates)