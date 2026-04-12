"""
agent/nodes/ranker.py
──────────────────────
Ranking & Aggregation Node — the final step in the pipeline.

Responsibilities:
  1. Sort candidate scores by overall_score descending.
  2. Apply configurable min_score_threshold to flag non-matches.
  3. Generate a GPT-4o comparative summary narrative.
  4. Build and return the final RankedResult object.
"""

from __future__ import annotations

from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings
from models.rubric import ScoringRubric
from models.score import CandidateScore, MatchLevel, RankedResult


# ── Comparative summary prompt ────────────────────────────────────────────────

SUMMARY_SYSTEM = """You are a senior HR analyst presenting hiring recommendations.
Write a concise, honest, data-driven comparative analysis of the evaluated candidates.
Focus on who to interview first and why. Mention key differentiators.
Keep the summary to 3-5 sentences. Be direct and professional."""

SUMMARY_HUMAN = """
Job requirement: {requirement}

Candidate scores (ranked):
{scores_text}

Write a 3-5 sentence comparative summary recommending who to prioritize and why.
"""


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5), reraise=False)
def _generate_summary(
    rubric: ScoringRubric,
    top_candidates: list[CandidateScore],
) -> str:
    """Generate a GPT-4o comparative narrative for the top candidates."""
    if not top_candidates:
        return "No candidates were successfully evaluated."

    try:
        llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0.3,
            api_key=settings.openai_api_key,
        )

        scores_lines: list[str] = []
        for i, c in enumerate(top_candidates[:5], 1):  # top 5 only
            line = (
                f"{i}. {c.candidate_name} — Overall: {c.overall_score:.1f}/100 "
                f"({c.match_level.value}). "
                f"Skills: {c.skills_score:.1f}, Experience: {c.experience_score:.1f}, "
                f"Domain: {c.domain_score:.1f}. "
                f"Matched: {', '.join(c.matched_skills[:5]) or 'none'}. "
                f"Missing: {', '.join(c.missing_skills[:3]) or 'none'}."
            )
            scores_lines.append(line)

        prompt = ChatPromptTemplate.from_messages([
            ("system", SUMMARY_SYSTEM),
            ("human", SUMMARY_HUMAN),
        ])
        chain = prompt | llm
        response = chain.invoke({
            "requirement": rubric.raw_requirement,
            "scores_text": "\n".join(scores_lines),
        })
        return response.content.strip()

    except Exception as e:
        logger.warning(f"Summary generation failed: {e}")
        # Fallback: construct a basic summary from the data
        top = top_candidates[0]
        return (
            f"The top-ranked candidate is {top.candidate_name} with an overall score of "
            f"{top.overall_score:.1f}/100. "
            f"Key matched skills: {', '.join(top.matched_skills[:4]) or 'none found'}. "
            f"Please review the detailed scores below for a complete comparison."
        )


# ── Main node function ────────────────────────────────────────────────────────

def ranking_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: sort, filter, and summarise candidate scores.

    Input state keys:  candidate_scores, scoring_rubric, parse_errors, scoring_errors
    Output state keys: ranked_result, status
    """
    candidate_scores: list[CandidateScore] = state.get("candidate_scores", [])
    rubric: ScoringRubric | None = state.get("scoring_rubric")
    parse_errors: dict[str, str] = state.get("parse_errors", {})
    scoring_errors: dict[str, str] = state.get("scoring_errors", {})

    logger.info(f"Ranking Node: aggregating {len(candidate_scores)} score(s)")

    # Sort by overall_score descending, with confidence as tiebreaker
    sorted_candidates = sorted(
        candidate_scores,
        key=lambda c: (c.overall_score, c.confidence),
        reverse=True,
    )

    # Count outcome categories
    no_match_count = sum(
        1 for c in sorted_candidates
        if c.overall_score < settings.min_score_threshold
    )

    total_failed = len(parse_errors) + len(scoring_errors)

    # Generate comparative summary
    if rubric:
        eligible = [
            c for c in sorted_candidates
            if c.overall_score >= settings.min_score_threshold and not c.error
        ]
        comparative_summary = _generate_summary(rubric, eligible or sorted_candidates)
    else:
        comparative_summary = "Rubric was not available — comparative summary skipped."

    # Build human-readable rubric summary
    rubric_summary = ""
    if rubric:
        parts = []
        if rubric.required_skills:
            parts.append(f"Required: {', '.join(rubric.required_skills)}")
        if rubric.preferred_skills:
            parts.append(f"Preferred: {', '.join(rubric.preferred_skills)}")
        if rubric.min_years_experience:
            parts.append(f"{rubric.min_years_experience}+ years experience")
        if rubric.seniority_level != "any":
            parts.append(f"Seniority: {rubric.seniority_level}")
        if rubric.domain:
            parts.append(f"Domain: {rubric.domain}")
        rubric_summary = " | ".join(parts)

    ranked_result = RankedResult(
        candidates=sorted_candidates,
        comparative_summary=comparative_summary,
        rubric_used=rubric_summary,
        total_processed=len(candidate_scores) + total_failed,
        total_failed=total_failed,
        no_match_count=no_match_count,
    )

    logger.success(
        f"Ranking complete: {len(sorted_candidates)} candidates ranked, "
        f"{no_match_count} below threshold, {total_failed} failed"
    )

    return {
        "ranked_result": ranked_result,
        "status": "done",
    }