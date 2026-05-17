from __future__ import annotations

from typing import Any

from loguru import logger

from config import settings
from models.score import ResumeDocument
from vector_store.chroma_client import upsert_chunks


def _chunk_text(text: str) -> list[str]:
    try:
        from llama_index.core.node_parser import SentenceSplitter
        splitter = SentenceSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        chunks = splitter.split_text(text)
        return [c.strip() for c in chunks if c.strip()]
    except Exception as e:
        logger.warning(f"LlamaIndex splitter failed ({e}), using fallback")
        return _naive_chunk(text, settings.chunk_size, settings.chunk_overlap)


def _naive_chunk(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def _embed_chunks(chunks: list[str]) -> list[list[float]]:
    from openai import OpenAI
    client = OpenAI(api_key=settings.openai_api_key)
    embeddings: list[list[float]] = []

    for i in range(0, len(chunks), 100):
        batch = chunks[i : i + 100]
        response = client.embeddings.create(model=settings.embedding_model, input=batch)
        embeddings.extend([item.embedding for item in response.data])

    return embeddings


def indexing_node(state: dict[str, Any]) -> dict[str, Any]:
    parsed_documents: list[ResumeDocument] = state.get("parsed_documents", [])
    collection_name: str = state.get("collection_name", "resumes_default")
    already_indexed: list[str] = state.get("indexed_file_hashes", [])

    logger.info(f"Indexing {len(parsed_documents)} document(s)")
    indexed_hashes: list[str] = []

    for doc in parsed_documents:
        if doc.file_hash in already_indexed:
            continue

        try:
            chunks = _chunk_text(doc.raw_text)
            if not chunks:
                logger.warning(f"No chunks for {doc.file_name}, skipping")
                continue

            embeddings = _embed_chunks(chunks)
            upsert_chunks(
                collection_name=collection_name,
                chunks=chunks,
                embeddings=embeddings,
                candidate_name=doc.candidate_name,
                file_name=doc.file_name,
                file_hash=doc.file_hash,
            )
            indexed_hashes.append(doc.file_hash)
            logger.info(f"Indexed '{doc.candidate_name}': {len(chunks)} chunks")

        except Exception as e:
            logger.error(f"Indexing failed for {doc.file_name}: {e}")

    return {
        "indexed_file_hashes": indexed_hashes,
        "status": "parsing_rubric",
    }