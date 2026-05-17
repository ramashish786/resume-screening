from __future__ import annotations

import os
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger

from config import settings


def _get_client() -> chromadb.ClientAPI:
    os.makedirs(settings.chroma_persist_dir, exist_ok=True)
    return chromadb.PersistentClient(
        path=settings.chroma_persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_or_create_collection(collection_name: str) -> chromadb.Collection:
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
    try:
        client = _get_client()
        client.delete_collection(collection_name)
        logger.info(f"Deleted collection: {collection_name}")
    except Exception as e:
        logger.warning(f"Could not delete collection '{collection_name}': {e}")


def list_collections() -> list[str]:
    client = _get_client()
    return [c.name for c in client.list_collections()]