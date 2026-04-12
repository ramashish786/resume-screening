"""
agent/nodes/parser.py
──────────────────────
File Parsing Node for the LangGraph agent.

Responsibilities:
  1. Iterate over uploaded_files in the state.
  2. Detect file type and route to the correct parser.
  3. Attempt to extract the candidate's name from the text.
  4. Deduplicate by file hash.
  5. Return parsed ResumeDocument objects + any parse errors.

This node is intentionally fault-tolerant: a single corrupt file
does not abort the pipeline — it's added to parse_errors and skipped.
"""

from __future__ import annotations

import re
from typing import Any

from loguru import logger

from models.score import ResumeDocument
from parsers import is_duplicate, parse_resume


# ── Candidate name extraction helpers ────────────────────────────────────────

def _extract_candidate_name(text: str, file_name: str) -> str:
    """
    Attempt to extract the candidate's name from the resume text.

    Strategy (in order of preference):
      1. First non-empty line of text (most resumes start with name).
      2. Line matching "Name: ..." pattern.
      3. Fall back to the file name without extension.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    if not lines:
        return _name_from_filename(file_name)

    # Try pattern: "Name: John Doe" or "Name – John Doe"
    for line in lines[:10]:
        m = re.match(r"(?:Name|Candidate)[:\-–\s]+(.+)", line, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            if 2 <= len(candidate.split()) <= 5:
                return candidate.title()

    # First line heuristic — must look like a name (2-4 words, no digits, not too long)
    first = lines[0]
    words = first.split()
    if (
        2 <= len(words) <= 4
        and len(first) <= 50
        and not any(c.isdigit() for c in first)
        and not any(kw in first.lower() for kw in ("resume", "cv", "curriculum", "vitae", "profile"))
    ):
        return first.title()

    return _name_from_filename(file_name)


def _name_from_filename(file_name: str) -> str:
    """Derive a human-readable name from the filename."""
    import os
    stem = os.path.splitext(file_name)[0]
    # Convert snake_case / kebab-case / CamelCase to Title Case
    stem = re.sub(r"[_\-]+", " ", stem)
    stem = re.sub(r"([a-z])([A-Z])", r"\1 \2", stem)
    return stem.strip().title() or "Unknown Candidate"


# ── Main node function ────────────────────────────────────────────────────────

def file_parsing_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: parse all uploaded files into ResumeDocument objects.

    Input state keys:  uploaded_files, session_id
    Output state keys: parsed_documents, parse_errors, status
    """
    uploaded_files: list[dict] = state.get("uploaded_files", [])
    logger.info(f"File Parsing Node: processing {len(uploaded_files)} file(s)")

    parsed_documents: list[ResumeDocument] = []
    parse_errors: dict[str, str] = {}
    seen_hashes: set[str] = set()

    for upload in uploaded_files:
        file_bytes: bytes = upload["file_bytes"]
        file_name: str = upload["file_name"]

        # Deduplication check
        is_dup, file_hash = is_duplicate(file_bytes, seen_hashes)
        if is_dup:
            logger.warning(f"Duplicate file skipped: {file_name}")
            parse_errors[file_name] = "Duplicate file — already uploaded in this session."
            continue
        seen_hashes.add(file_hash)

        try:
            result = parse_resume(file_bytes, file_name)

            candidate_name = _extract_candidate_name(result["raw_text"], file_name)

            doc = ResumeDocument(
                candidate_name=candidate_name,
                raw_text=result["raw_text"],
                file_name=file_name,
                file_hash=file_hash,
                page_count=result["page_count"],
                parse_warnings=result["warnings"],
                word_count=result["word_count"],
            )

            if not doc.is_sufficient:
                parse_errors[file_name] = (
                    f"Insufficient text ({doc.word_count} words) — "
                    "may be image-only. Results will be marked as low-confidence."
                )
                # Still include it — scorer will flag low confidence

            parsed_documents.append(doc)
            logger.success(
                f"Parsed: '{candidate_name}' from {file_name} ({doc.word_count} words)"
            )

        except Exception as e:
            logger.error(f"Failed to parse {file_name}: {e}")
            parse_errors[file_name] = str(e)

    logger.info(
        f"Parsing complete: {len(parsed_documents)} successful, {len(parse_errors)} errors"
    )

    return {
        "parsed_documents": parsed_documents,
        "parse_errors": parse_errors,
        "status": "indexing",
    }