"""
models/score.py
───────────────
Pydantic models for per-candidate scoring output and the final ranked result.
All scores are 0–100 floats.  The overall_score is the weighted aggregate.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, computed_field


class MatchLevel(str, Enum):
    STRONG = "strong"       # >= 75
    GOOD = "good"           # >= 55
    PARTIAL = "partial"     # >= 35
    WEAK = "weak"           # >= min_threshold
    NO_MATCH = "no_match"   # below threshold


class ResumeDocument(BaseModel):
    """
    Raw parsed resume before embedding.
    Produced by the File Parsing Node.
    """

    candidate_name: str = Field("Unknown", description="Extracted or inferred name")
    raw_text: str = Field(..., description="Full extracted text from the resume")
    file_name: str = Field(..., description="Original upload filename")
    file_hash: str = Field(..., description="MD5 hash for deduplication")
    page_count: int = Field(1, ge=1)
    parse_warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal issues encountered during parsing",
    )
    word_count: int = Field(0, ge=0)

    @computed_field  # type: ignore[misc]
    @property
    def is_sufficient(self) -> bool:
        """Flag resumes with too little text for reliable scoring."""
        return self.word_count >= 80


class CandidateScore(BaseModel):
    """
    Scoring result for a single candidate against the job rubric.
    Produced by the Scoring Node.
    """

    # Identity
    candidate_name: str
    file_name: str

    # Dimension scores (0–100)
    skills_score: float = Field(0.0, ge=0.0, le=100.0)
    experience_score: float = Field(0.0, ge=0.0, le=100.0)
    domain_score: float = Field(0.0, ge=0.0, le=100.0)

    # Aggregate
    overall_score: float = Field(0.0, ge=0.0, le=100.0)

    # Evidence
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    matched_preferred_skills: list[str] = Field(default_factory=list)
    experience_years_found: Optional[int] = Field(
        None, description="Total years of experience found in the resume"
    )
    seniority_detected: Optional[str] = Field(
        None, description="Seniority level inferred from the resume"
    )

    # Explanation
    justification: str = Field(
        "", description="2-3 sentence GPT-generated explanation of the score"
    )
    key_strengths: list[str] = Field(default_factory=list)
    key_gaps: list[str] = Field(default_factory=list)

    # Meta
    confidence: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="How much evidence was found in the resume (0=none, 1=abundant)",
    )
    retrieved_chunks: list[str] = Field(
        default_factory=list,
        description="Raw resume passages used as evidence for scoring",
    )
    error: Optional[str] = Field(
        None, description="Set if scoring failed for this candidate"
    )

    @computed_field  # type: ignore[misc]
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

    @computed_field  # type: ignore[misc]
    @property
    def match_level_emoji(self) -> str:
        mapping = {
            MatchLevel.STRONG: "🟢",
            MatchLevel.GOOD: "🔵",
            MatchLevel.PARTIAL: "🟡",
            MatchLevel.WEAK: "🟠",
            MatchLevel.NO_MATCH: "🔴",
        }
        return mapping[self.match_level]

    @computed_field  # type: ignore[misc]
    @property
    def needs_manual_review(self) -> bool:
        return self.confidence < 0.3 or self.error is not None


class RankedResult(BaseModel):
    """
    Final output from the Ranking & Aggregation Node.
    Contains the full ranked list plus a comparative summary.
    """

    candidates: list[CandidateScore] = Field(
        default_factory=list,
        description="Candidates sorted by overall_score descending",
    )
    comparative_summary: str = Field(
        "", description="GPT-generated narrative comparing top candidates"
    )
    rubric_used: str = Field("", description="Human-readable rubric summary")
    total_processed: int = Field(0)
    total_failed: int = Field(0)
    no_match_count: int = Field(0)

    @property
    def top_candidate(self) -> Optional[CandidateScore]:
        return self.candidates[0] if self.candidates else None

    @property
    def has_strong_match(self) -> bool:
        return any(c.match_level == MatchLevel.STRONG for c in self.candidates)