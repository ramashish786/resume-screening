"""
agent/nodes/requirement.py
───────────────────────────
Requirement Parser Node.

Uses GPT-4o via LangChain to convert the user's free-text job requirement
into a structured ScoringRubric Pydantic object.

If parsing fails or produces low-confidence output, the node sets
rubric_error in state and the pipeline can either abort or use
a fallback keyword-based rubric.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings
from models.rubric import ScoringRubric, ScoringWeights


# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert HR analyst and technical recruiter.
Your job is to parse a job requirement written in natural language and extract
a structured scoring rubric that can be used to evaluate candidates.

Be precise and realistic:
- Only list skills that are EXPLICITLY mentioned or strongly implied.
- Do NOT invent requirements that weren't stated.
- If experience years aren't mentioned, set min_years_experience to 0.
- If seniority isn't mentioned, set seniority_level to "any".
- Distribute weights sensibly: if skills are the main focus, give them more weight.
- Set parser_confidence to 0.5–0.8 if the requirement is vague, 0.9–1.0 if clear.

Always respond with ONLY valid JSON matching the schema below. No markdown, no preamble.
"""

USER_PROMPT = """Job requirement:
\"\"\"
{user_requirement}
\"\"\"

Output JSON schema:
{{
  "required_skills": ["list of must-have skills"],
  "preferred_skills": ["list of nice-to-have skills"],
  "min_years_experience": 0,
  "seniority_level": "any|intern|junior|mid|senior|staff|principal|lead|manager|director|vp",
  "domain": "technical domain or industry",
  "additional_requirements": ["any other explicit requirements"],
  "weights": {{
    "skills": 0.5,
    "experience": 0.3,
    "domain": 0.2
  }},
  "raw_requirement": "{user_requirement}",
  "parser_confidence": 0.9
}}"""


# ── LLM call with retry ───────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _call_llm(user_requirement: str) -> str:
    """Call GPT-4o and return raw JSON string."""
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0.0,  # Deterministic for structured extraction
        api_key=settings.openai_api_key,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT),
    ])

    chain = prompt | llm
    response = chain.invoke({"user_requirement": user_requirement})
    return response.content


# ── Fallback rubric from keywords ─────────────────────────────────────────────

def _keyword_fallback_rubric(user_requirement: str) -> ScoringRubric:
    """
    Very basic fallback: extract capitalized words as potential skills.
    Used when the LLM call fails completely.
    """
    import re
    words = re.findall(r"\b[A-Z][a-zA-Z+#.]+\b", user_requirement)
    skills = list(set(w for w in words if len(w) > 2))

    return ScoringRubric(
        required_skills=skills[:10],
        preferred_skills=[],
        min_years_experience=0,
        seniority_level="any",
        domain="",
        weights=ScoringWeights(skills=0.6, experience=0.2, domain=0.2),
        raw_requirement=user_requirement,
        parser_confidence=0.3,
    )


# ── Main node function ────────────────────────────────────────────────────────

def requirement_parser_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: parse user_requirement into a ScoringRubric.

    Input state keys:  user_requirement
    Output state keys: scoring_rubric, rubric_error, status
    """
    user_requirement: str = state.get("user_requirement", "").strip()
    logger.info("Requirement Parser Node: parsing job requirement")

    if not user_requirement:
        return {
            "rubric_error": "No job requirement provided.",
            "status": "error",
            "fatal_error": "Job requirement is empty.",
        }

    rubric: ScoringRubric | None = None
    rubric_error: str | None = None

    try:
        raw_json = _call_llm(user_requirement)

        # Strip markdown fences if present
        raw_json = raw_json.strip()
        if raw_json.startswith("```"):
            raw_json = raw_json.split("```")[1]
            if raw_json.startswith("json"):
                raw_json = raw_json[4:]

        data = json.loads(raw_json)
        data["raw_requirement"] = user_requirement  # ensure it's set
        rubric = ScoringRubric(**data)

        logger.success(
            f"Rubric parsed: {len(rubric.required_skills)} required skills, "
            f"{len(rubric.preferred_skills)} preferred, "
            f"confidence={rubric.parser_confidence:.2f}"
        )

        if rubric.parser_confidence < 0.5:
            rubric_error = (
                "The job requirement was vague — the rubric may not capture all requirements. "
                "Consider providing more specific skills, experience, and domain information."
            )

    except Exception as e:
        logger.error(f"Rubric parsing failed: {e}")
        rubric_error = f"Could not parse requirement automatically: {e}. Using keyword fallback."
        rubric = _keyword_fallback_rubric(user_requirement)
        logger.warning("Using keyword-based fallback rubric")

    return {
        "scoring_rubric": rubric,
        "rubric_error": rubric_error,
        "status": "scoring",
    }