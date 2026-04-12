"""
agent/nodes/retriever.py
─────────────────────────
Semantic Retrieval Node.

For each parsed candidate, embeds the scoring rubric query string and
retrieves the top-k most relevant resume chunks from ChromaDB.

Returns a dict mapping file_hash → list[chunk_text] for the Scoring Node.

This node is designed to run BEFORE scoring so the scorer has rich
evidence passages rather than operating on the full raw text blindly.
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from openai import OpenAI

from config import settings
from models.rubric import ScoringRubric
from models.score import ResumeDocument
from vector_store.chroma_client import query_collection


def _embed_query(query: str) -> list[float]:
    """Embed a single query string using OpenAI embeddings."""
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=[query],
    )
    return response.data[0].embedding


def retrieval_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: retrieve top-k relevant chunks per candidate.

    Input state keys:  parsed_documents, scoring_rubric, collection_name
    Output state keys: retrieved_chunks_map, status

    retrieved_chunks_map: dict[file_hash, list[str]]
    """
    parsed_documents: list[ResumeDocument] = state.get("parsed_documents", [])
    rubric: ScoringRubric | None = state.get("scoring_rubric")
    collection_name: str = state.get("collection_name", "resumes_default")

    if rubric is None:
        return {
            "status": "error",
            "fatal_error": "No scoring rubric available for retrieval.",
        }

    logger.info(
        f"Retrieval Node: fetching chunks for {len(parsed_documents)} candidate(s)"
    )

    # Build the query string from the rubric and embed it once
    query_string = rubric.to_query_string()
    logger.debug(f"Query string: {query_string[:120]}...")

    try:
        query_embedding = _embed_query(query_string)
    except Exception as e:
        logger.error(f"Failed to embed query: {e}")
        return {
            "status": "error",
            "fatal_error": f"Embedding API error: {e}",
        }

    retrieved_chunks_map: dict[str, list[str]] = {}

    for doc in parsed_documents:
        try:
            chunks = query_collection(
                collection_name=collection_name,
                query_embedding=query_embedding,
                top_k=settings.top_k_retrieval,
                file_hash=doc.file_hash,
            )

            if not chunks:
                # Fallback: use first 600 words of raw text as evidence
                logger.warning(
                    f"No chunks retrieved for '{doc.candidate_name}' — using raw text fallback"
                )
                words = doc.raw_text.split()
                chunks = [" ".join(words[:600])]

            retrieved_chunks_map[doc.file_hash] = chunks
            logger.debug(
                f"Retrieved {len(chunks)} chunks for '{doc.candidate_name}'"
            )

        except Exception as e:
            logger.error(f"Retrieval failed for '{doc.candidate_name}': {e}")
            # Fallback to raw text snippet
            words = doc.raw_text.split()
            retrieved_chunks_map[doc.file_hash] = [" ".join(words[:600])]

    return {
        "retrieved_chunks_map": retrieved_chunks_map,
        "status": "scoring",
    }