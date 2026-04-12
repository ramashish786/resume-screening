"""
agent/nodes/scorer.py
──────────────────────
Scoring Node — the core intelligence of the pipeline.

Uses GPT-4o as an expert HR evaluator ("LLM-as-Judge") to score each
candidate against the structured rubric using their retrieved resume chunks
as evidence.

Output: one CandidateScore per candidate, with dimension scores,
matched/missing skills, justification, and confidence.

Self-consistency: if settings.scoring_runs > 1, the LLM is called multiple
times and scores are averaged to reduce variance (recommended for production).
"""

from __future__ import annotations

import json
import statistics
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings
from models.rubric import ScoringRubric
from models.score import CandidateScore, ResumeDocument


# ── Scoring Prompt ────────────────────────────────────────────────────────────

SCORING_SYSTEM = """You are a senior technical recruiter and hiring manager with 15 years of experience.
You evaluate candidates objectively based on evidence found in their resumes.

Scoring rules:
- Score ONLY based on evidence in the resume excerpts provided.
- Do NOT assume skills that aren't mentioned.
- Partial matches score proportionally (e.g., Python experience when Java required = partial credit).
- Related/adjacent skills get partial credit (e.g., GCP when AWS required = 60% credit).
- Be calibrated: 100 = perfect match, 0 = no evidence at all.
- Confidence reflects how much evidence is present (0 = no relevant content, 1 = abundant evidence).

Always respond with ONLY valid JSON. No markdown, no preamble, no explanation outside the JSON.
"""

SCORING_HUMAN = """
# Job Requirement Rubric
Required skills: {required_skills}
Preferred skills: {preferred_skills}
Minimum experience: {min_years_experience} years
Seniority level: {seniority_level}
Domain: {domain}
Additional requirements: {additional_requirements}

# Candidate Resume Excerpts
Candidate name: {candidate_name}

{resume_chunks}

# Task
Score this candidate against the rubric above.

Respond with this exact JSON structure:
{{
  "skills_score": 0-100,
  "experience_score": 0-100,
  "domain_score": 0-100,
  "matched_skills": ["list of required/preferred skills found in resume"],
  "missing_skills": ["list of required skills NOT found"],
  "matched_preferred_skills": ["preferred skills found"],
  "experience_years_found": null_or_integer,
  "seniority_detected": "null or detected level",
  "justification": "2-3 sentence explanation of the overall score",
  "key_strengths": ["up to 3 specific strengths"],
  "key_gaps": ["up to 3 specific gaps"],
  "confidence": 0.0-1.0
}}
"""


# ── Single scoring call ───────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    reraise=True,
)
def _score_once(
    rubric: ScoringRubric,
    candidate_name: str,
    resume_chunks: list[str],
) -> dict:
    """
    Call GPT-4o once and return raw score dict.
    Decorated with tenacity retry for transient API errors.
    """
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.openai_api_key,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SCORING_SYSTEM),
        ("human", SCORING_HUMAN),
    ])

    chain = prompt | llm

    # Format chunks with separators for clarity
    formatted_chunks = "\n\n---\n\n".join(
        f"[Excerpt {i+1}]\n{chunk}" for i, chunk in enumerate(resume_chunks)
    )

    response = chain.invoke({
        "required_skills": ", ".join(rubric.required_skills) or "Not specified",
        "preferred_skills": ", ".join(rubric.preferred_skills) or "None",
        "min_years_experience": rubric.min_years_experience or "Not specified",
        "seniority_level": rubric.seniority_level,
        "domain": rubric.domain or "Not specified",
        "additional_requirements": ", ".join(rubric.additional_requirements) or "None",
        "candidate_name": candidate_name,
        "resume_chunks": formatted_chunks,
    })

    raw = response.content.strip()
    # Strip markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw)


def _average_scores(score_dicts: list[dict]) -> dict:
    """Average numeric fields across multiple scoring runs for self-consistency."""
    if len(score_dicts) == 1:
        return score_dicts[0]

    numeric_fields = ["skills_score", "experience_score", "domain_score", "confidence"]
    result = score_dicts[0].copy()  # use first run's list fields

    for field in numeric_fields:
        values = [d[field] for d in score_dicts if isinstance(d.get(field), (int, float))]
        if values:
            result[field] = round(statistics.mean(values), 2)

    if "experience_years_found" in result:
        years = [d.get("experience_years_found") for d in score_dicts if d.get("experience_years_found") is not None]
        result["experience_years_found"] = int(statistics.mean(years)) if years else None

    return result


# ── Main node function ────────────────────────────────────────────────────────

def scoring_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: score each candidate using LLM-as-Judge.

    Input state keys:  parsed_documents, scoring_rubric, retrieved_chunks_map
    Output state keys: candidate_scores, scoring_errors, status
    """
    parsed_documents: list[ResumeDocument] = state.get("parsed_documents", [])
    rubric: ScoringRubric | None = state.get("scoring_rubric")
    retrieved_chunks_map: dict[str, list[str]] = state.get("retrieved_chunks_map", {})

    if rubric is None:
        return {"status": "error", "fatal_error": "No rubric available for scoring."}

    logger.info(f"Scoring Node: scoring {len(parsed_documents)} candidate(s)")

    candidate_scores: list[CandidateScore] = []
    scoring_errors: dict[str, str] = {}

    for doc in parsed_documents:
        logger.info(f"Scoring: '{doc.candidate_name}' ({doc.file_name})")

        # Get retrieved chunks for this candidate
        chunks = retrieved_chunks_map.get(doc.file_hash, [])
        if not chunks:
            chunks = [doc.raw_text[:2000]]  # last resort fallback

        try:
            # Run scoring N times for self-consistency (N=1 for POC)
            raw_scores: list[dict] = []
            for run in range(settings.scoring_runs):
                score_dict = _score_once(rubric, doc.candidate_name, chunks)
                raw_scores.append(score_dict)
                if settings.scoring_runs > 1:
                    logger.debug(f"  Run {run+1}/{settings.scoring_runs} complete")

            averaged = _average_scores(raw_scores)

            # Compute weighted overall score
            w = rubric.weights
            overall = round(
                averaged["skills_score"] * w.skills
                + averaged["experience_score"] * w.experience
                + averaged["domain_score"] * w.domain,
                2,
            )

            # Handle low-confidence resumes
            confidence = averaged.get("confidence", 1.0)
            if not doc.is_sufficient:
                confidence = min(confidence, 0.3)

            score = CandidateScore(
                candidate_name=doc.candidate_name,
                file_name=doc.file_name,
                skills_score=averaged.get("skills_score", 0.0),
                experience_score=averaged.get("experience_score", 0.0),
                domain_score=averaged.get("domain_score", 0.0),
                overall_score=overall,
                matched_skills=averaged.get("matched_skills", []),
                missing_skills=averaged.get("missing_skills", []),
                matched_preferred_skills=averaged.get("matched_preferred_skills", []),
                experience_years_found=averaged.get("experience_years_found"),
                seniority_detected=averaged.get("seniority_detected"),
                justification=averaged.get("justification", ""),
                key_strengths=averaged.get("key_strengths", []),
                key_gaps=averaged.get("key_gaps", []),
                confidence=confidence,
                retrieved_chunks=chunks,
            )

            candidate_scores.append(score)
            logger.success(
                f"Scored '{doc.candidate_name}': {overall:.1f}/100 "
                f"({score.match_level.value}) confidence={confidence:.2f}"
            )

        except Exception as e:
            logger.error(f"Scoring failed for '{doc.candidate_name}': {e}")
            scoring_errors[doc.file_name] = str(e)

            # Add a zero-score entry so the candidate still appears in results
            candidate_scores.append(
                CandidateScore(
                    candidate_name=doc.candidate_name,
                    file_name=doc.file_name,
                    overall_score=0.0,
                    confidence=0.0,
                    error=str(e),
                    justification="Scoring failed due to an API error.",
                )
            )

    return {
        "candidate_scores": candidate_scores,
        "scoring_errors": scoring_errors,
        "status": "ranking",
    }