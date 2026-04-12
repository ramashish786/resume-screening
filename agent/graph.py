"""
agent/graph.py
───────────────
LangGraph StateGraph definition for the Resume Scoring Agent.

Pipeline:
  START
    → file_parsing        (parse uploaded files)
    → indexing            (chunk + embed into ChromaDB)
    → requirement_parsing (parse free-text job req into ScoringRubric)
    → retrieval           (semantic retrieval of top-k chunks per candidate)
    → scoring             (LLM-as-Judge scoring per candidate)
    → ranking             (sort + aggregate + comparative summary)
  END

Conditional edge after file_parsing:
  - If no documents were parsed successfully → END with fatal error
  - Otherwise → indexing

Conditional edge after requirement_parsing:
  - If fatal_error is set → END
  - Otherwise → retrieval
"""

from __future__ import annotations

import uuid
from typing import Any

from langgraph.graph import END, START, StateGraph
from loguru import logger

from agent.nodes import (
    file_parsing_node,
    indexing_node,
    ranking_node,
    requirement_parser_node,
    retrieval_node,
    scoring_node,
)
from agent.state import AgentState


# ── Conditional edge functions ────────────────────────────────────────────────

def route_after_parsing(state: AgentState) -> str:
    """Route to indexing if at least one document was parsed, else END."""
    parsed = state.get("parsed_documents", [])
    fatal = state.get("fatal_error")

    if fatal:
        logger.error(f"Fatal error after parsing: {fatal}")
        return END

    if not parsed:
        logger.error("No documents parsed — aborting pipeline")
        return END

    return "indexing"


def route_after_rubric(state: AgentState) -> str:
    """Route to retrieval if rubric is available, else END."""
    if state.get("fatal_error"):
        return END
    if state.get("scoring_rubric") is None:
        return END
    return "retrieval"


def route_after_scoring(state: AgentState) -> str:
    """Route to ranking if we have any scores, else END."""
    scores = state.get("candidate_scores", [])
    if not scores:
        return END
    return "ranking"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph() -> Any:
    """
    Construct and compile the LangGraph StateGraph.
    Returns a compiled graph ready for .invoke() calls.
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("file_parsing", file_parsing_node)
    graph.add_node("indexing", indexing_node)
    graph.add_node("requirement_parsing", requirement_parser_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("scoring", scoring_node)
    graph.add_node("ranking", ranking_node)

    # Edges
    graph.add_edge(START, "file_parsing")

    graph.add_conditional_edges(
        "file_parsing",
        route_after_parsing,
        {
            "indexing": "indexing",
            END: END,
        },
    )

    # indexing and requirement_parsing can run in sequence
    # (requirement parsing doesn't depend on indexing being complete,
    #  but I have kept it sequential for simplicity)
    graph.add_edge("indexing", "requirement_parsing")

    graph.add_conditional_edges(
        "requirement_parsing",
        route_after_rubric,
        {
            "retrieval": "retrieval",
            END: END,
        },
    )

    graph.add_edge("retrieval", "scoring")

    graph.add_conditional_edges(
        "scoring",
        route_after_scoring,
        {
            "ranking": "ranking",
            END: END,
        },
    )

    graph.add_edge("ranking", END)

    return graph.compile()


# ── Public run function ───────────────────────────────────────────────────────

def run_agent(
    uploaded_files: list[dict[str, Any]],
    user_requirement: str,
    session_id: str | None = None,
) -> AgentState:
    """
    Run the full resume scoring pipeline.

    Args:
        uploaded_files:   List of {"file_bytes": bytes, "file_name": str}
        user_requirement: Free-text job requirement from the user
        session_id:       Optional session identifier (auto-generated if None)

    Returns:
        Final AgentState with ranked_result populated.
    """
    if session_id is None:
        session_id = uuid.uuid4().hex[:12]

    collection_name = f"resumes_{session_id}"

    initial_state: AgentState = {
        "session_id": session_id,
        "collection_name": collection_name,
        "uploaded_files": uploaded_files,
        "user_requirement": user_requirement,
        "parsed_documents": [],
        "parse_errors": {},
        "indexed_file_hashes": [],
        "scoring_rubric": None,
        "rubric_error": None,
        "candidate_scores": [],
        "scoring_errors": {},
        "ranked_result": None,
        "status": "idle",
        "fatal_error": None,
    }

    logger.info(
        f"Starting agent | session={session_id} | "
        f"files={len(uploaded_files)} | requirement='{user_requirement[:60]}...'"
    )

    app = build_graph()

    try:
        final_state = app.invoke(initial_state)
        logger.success(f"Agent completed | session={session_id} | status={final_state.get('status')}")
        return final_state
    except Exception as e:
        logger.error(f"Agent failed | session={session_id}: {e}")
        initial_state["status"] = "error"
        initial_state["fatal_error"] = str(e)
        return initial_state
