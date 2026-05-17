from __future__ import annotations

from typing import Annotated, Any, Optional
from typing_extensions import TypedDict

from models.rubric import ScoringRubric
from models.score import CandidateScore, RankedResult, ResumeDocument


def _merge_dicts(a: dict, b: dict) -> dict:
    return {**a, **b}


def _append_list(a: list, b: list) -> list:
    return a + b


class AgentState(TypedDict, total=False):
    session_id: str
    uploaded_files: list[dict[str, Any]]
    user_requirement: str
    parsed_documents: Annotated[list[ResumeDocument], _append_list]
    parse_errors: Annotated[dict[str, str], _merge_dicts]
    indexed_file_hashes: Annotated[list[str], _append_list]
    collection_name: str
    scoring_rubric: Optional[ScoringRubric]
    rubric_error: Optional[str]
    candidate_scores: Annotated[list[CandidateScore], _append_list]
    scoring_errors: Annotated[dict[str, str], _merge_dicts]
    ranked_result: Optional[RankedResult]
    status: str
    fatal_error: Optional[str]