"""
agent/nodes/indexer.py
───────────────────────
LlamaIndex Document Ingestion Node.

Responsibilities:
  1. Receive parsed ResumeDocuments from state.
  2. Chunk each resume's text using LlamaIndex's SentenceSplitter.
  3. Embed chunks using OpenAI text-embedding-3-small.
  4. Upsert into ChromaDB under the session's collection.

Idempotent: re-uploading the same file (same hash) skips re-embedding.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from config import settings
from models.score import ResumeDocument
from vector_store.chroma_client import upsert_chunks


def _chunk_text(text: str) -> list[str]:
    """
    Split resume text into overlapping chunks using LlamaIndex SentenceSplitter.
    Falls back to naive word-based splitting if LlamaIndex is unavailable.
    """
    try:
        from llama_index.core.node_parser import SentenceSplitter

        splitter = SentenceSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        # SentenceSplitter.split_text returns list of strings
        chunks = splitter.split_text(text)
        return [c.strip() for c in chunks if c.strip()]

    except Exception as e:
        logger.warning(f"LlamaIndex splitter failed ({e}), falling back to naive chunker")
        return _naive_chunk(text, settings.chunk_size, settings.chunk_overlap)


def _naive_chunk(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Simple word-based fallback chunker.
    chunk_size and overlap are in approximate tokens (words as proxy).
    """
    words = text.split()
    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def _embed_chunks(chunks: list[str]) -> list[list[float]]:
    """
    Embed a list of text chunks using OpenAI text-embedding-3-small.
    Batches in groups of 100 to respect API limits.
    """
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    embeddings: list[list[float]] = []
    batch_size = 100

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        response = client.embeddings.create(
            model=settings.embedding_model,
            input=batch,
        )
        embeddings.extend([item.embedding for item in response.data])

    return embeddings


def indexing_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: chunk, embed, and store resume documents in ChromaDB.

    Input state keys:  parsed_documents, collection_name, indexed_file_hashes
    Output state keys: indexed_file_hashes, status
    """
    parsed_documents: list[ResumeDocument] = state.get("parsed_documents", [])
    collection_name: str = state.get("collection_name", "resumes_default")
    already_indexed: list[str] = state.get("indexed_file_hashes", [])

    logger.info(f"Indexing Node: {len(parsed_documents)} document(s) to index")

    indexed_hashes: list[str] = []

    for doc in parsed_documents:
        # Skip already-indexed documents (idempotency)
        if doc.file_hash in already_indexed:
            logger.debug(f"Skipping already-indexed: {doc.file_name}")
            continue

        try:
            # 1. Chunk
            chunks = _chunk_text(doc.raw_text)
            if not chunks:
                logger.warning(f"No chunks produced for {doc.file_name} — skipping")
                continue

            logger.debug(f"Chunked '{doc.candidate_name}': {len(chunks)} chunks")

            # 2. Embed
            embeddings = _embed_chunks(chunks)

            # 3. Upsert into ChromaDB
            upsert_chunks(
                collection_name=collection_name,
                chunks=chunks,
                embeddings=embeddings,
                candidate_name=doc.candidate_name,
                file_name=doc.file_name,
                file_hash=doc.file_hash,
            )

            indexed_hashes.append(doc.file_hash)
            logger.success(
                f"Indexed '{doc.candidate_name}': {len(chunks)} chunks → {collection_name}"
            )

        except Exception as e:
            logger.error(f"Indexing failed for {doc.file_name}: {e}")
            # Don't abort — other documents can still be indexed

    logger.info(
        f"Indexing complete: {len(indexed_hashes)} new, "
        f"{len(already_indexed)} already indexed"
    )

    return {
        "indexed_file_hashes": indexed_hashes,
        "status": "parsing_rubric",
    }