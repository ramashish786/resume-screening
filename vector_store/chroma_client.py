"""
vector_store/chroma_client.py
──────────────────────────────
Thin wrapper around ChromaDB for the resume agent.

Design decisions:
  • Each upload SESSION gets its own collection prefix to avoid cross-session bleed.
  • Each CANDIDATE within a session gets their own sub-collection keyed by file_hash.
  • Collections are created on first upsert and deleted on session reset.
  • In production, swap chromadb.Client() for chromadb.HttpClient() pointing at
    a hosted Chroma or replace entirely with Pinecone / Weaviate.
"""

from __future__ import annotations

import os
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger

from config import settings


def _get_client() -> chromadb.ClientAPI:
    """Return a persistent ChromaDB client."""
    os.makedirs(settings.chroma_persist_dir, exist_ok=True)
    return chromadb.PersistentClient(
        path=settings.chroma_persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_or_create_collection(collection_name: str) -> chromadb.Collection:
    """Get or create a ChromaDB collection by name."""
    client = _get_client()
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},  # use cosine similarity
    )
    logger.debug(f"Collection ready: {collection_name} ({collection.count()} docs)")
    return collection


def upsert_chunks(
    collection_name: str,
    chunks: list[str],
    embeddings: list[list[float]],
    candidate_name: str,
    file_name: str,
    file_hash: str,
) -> None:
    """
    Upsert embedded resume chunks into ChromaDB.

    IDs are deterministic: {file_hash}_{chunk_index}
    so re-uploading the same file is idempotent.
    """
    collection = get_or_create_collection(collection_name)

    ids = [f"{file_hash}_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "candidate_name": candidate_name,
            "file_name": file_name,
            "file_hash": file_hash,
            "chunk_index": i,
        }
        for i in range(len(chunks))
    ]

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    logger.info(f"Upserted {len(chunks)} chunks for '{candidate_name}' → {collection_name}")


def query_collection(
    collection_name: str,
    query_embedding: list[float],
    top_k: int = 8,
    file_hash: Optional[str] = None,
) -> list[str]:
    """
    Retrieve the top-k most relevant chunks from a collection.

    If file_hash is provided, results are filtered to that candidate only.
    Returns a list of document strings (the raw resume chunks).
    """
    collection = get_or_create_collection(collection_name)

    if collection.count() == 0:
        logger.warning(f"Collection '{collection_name}' is empty — no results.")
        return []

    where_filter = {"file_hash": {"$eq": file_hash}} if file_hash else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        where=where_filter,
        include=["documents", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]

    # Filter by minimum similarity (cosine distance < 0.7 means similarity > 0.3)
    MIN_DISTANCE = 0.7
    filtered = [
        doc for doc, dist in zip(docs, distances) if dist <= MIN_DISTANCE
    ]

    logger.debug(
        f"Query returned {len(docs)} chunks, {len(filtered)} above similarity threshold"
    )
    return filtered


def delete_collection(collection_name: str) -> None:
    """Delete a collection (used for session cleanup)."""
    try:
        client = _get_client()
        client.delete_collection(collection_name)
        logger.info(f"Deleted collection: {collection_name}")
    except Exception as e:
        logger.warning(f"Could not delete collection '{collection_name}': {e}")


def list_collections() -> list[str]:
    """List all existing collection names."""
    client = _get_client()
    return [c.name for c in client.list_collections()]