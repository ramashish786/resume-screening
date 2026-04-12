"""
agent/state.py
───────────────
LangGraph AgentState definition.

The state is the single source of truth passed between all graph nodes.
LangGraph merges partial state updates returned by each node — any key
not returned by a node is left unchanged.

Design notes:
  • All fields are Optional so the state can be built incrementally.
  • session_id scopes ChromaDB collections to avoid cross-run contamination.
  • errors dict accumulates per-candidate errors without stopping the pipeline.
"""

from __future__ import annotations

from typing import Annotated, Any, Optional
from typing_extensions import TypedDict

from models.rubric import ScoringRubric
from models.score import CandidateScore, RankedResult, ResumeDocument


def _merge_dicts(a: dict, b: dict) -> dict:
    """LangGraph reducer: merge two dicts (b overwrites a on key conflicts)."""
    return {**a, **b}


def _append_list(a: list, b: list) -> list:
    """LangGraph reducer: append items from b to a."""
    return a + b


class AgentState(TypedDict, total=False):
    """
    Central state object for the Resume Scoring Agent graph.

    Nodes read from and write to this state.  LangGraph handles merging.
    """

    # ── Session ──────────────────────────────────────────────────────────
    session_id: str
    """Unique ID for this scoring run — used to namespace ChromaDB collections."""

    # ── Raw inputs ───────────────────────────────────────────────────────
    uploaded_files: list[dict[str, Any]]
    """
    List of raw upload dicts from Streamlit:
        {"file_bytes": bytes, "file_name": str}
    """

    user_requirement: str
    """Free-text job requirement entered by the user."""

    # ── Parsed documents ─────────────────────────────────────────────────
    parsed_documents: Annotated[list[ResumeDocument], _append_list]
    """ResumeDocument objects produced by the File Parsing Node."""

    parse_errors: Annotated[dict[str, str], _merge_dicts]
    """file_name → error message for files that failed to parse."""

    # ── Indexing ─────────────────────────────────────────────────────────
    indexed_file_hashes: Annotated[list[str], _append_list]
    """Hashes of successfully indexed resumes (for dedup tracking)."""

    collection_name: str
    """ChromaDB collection name for this session."""

    # ── Rubric ───────────────────────────────────────────────────────────
    scoring_rubric: Optional[ScoringRubric]
    """Structured scoring rubric extracted from user_requirement."""

    rubric_error: Optional[str]
    """Set if rubric parsing fails."""

    # ── Scoring ──────────────────────────────────────────────────────────
    candidate_scores: Annotated[list[CandidateScore], _append_list]
    """CandidateScore objects produced by the Scoring Node (one per resume)."""

    scoring_errors: Annotated[dict[str, str], _merge_dicts]
    """file_name → error message for resumes that failed to score."""

    # ── Final output ─────────────────────────────────────────────────────
    ranked_result: Optional[RankedResult]
    """Final ranked result — set by the Ranking Node."""

    # ── Control flow ─────────────────────────────────────────────────────
    status: str
    """
    Pipeline status:
      "idle" | "parsing" | "indexing" | "parsing_rubric" |
      "retrieving" | "scoring" | "ranking" | "done" | "error"
    """

    fatal_error: Optional[str]
    """Set only if a non-recoverable error halts the entire pipeline."""