from __future__ import annotations

import os
import re
from typing import Any

from loguru import logger

from models.score import ResumeDocument
from parsers import is_duplicate, parse_resume


def _extract_email(text: str) -> str | None:
    match = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
    return match.group(0).lower() if match else None


def _extract_phone(text: str) -> str | None:
    match = re.search(
        r"(\+?\d{1,3}[\s\-.]?)?(\(?\d{3}\)?[\s\-.]?)?\d{3}[\s\-.]?\d{4,}",
        text,
    )
    if match:
        number = re.sub(r"\s+", " ", match.group(0).strip())
        # reject short matches that are just years or zip codes
        if len(re.sub(r"\D", "", number)) >= 7:
            return number
    return None


def _extract_linkedin(text: str) -> str | None:
    match = re.search(
        r"(?:https?://)?(?:www\.)?linkedin\.com/in/[a-zA-Z0-9_\-%.]+",
        text,
        re.IGNORECASE,
    )
    if match:
        return match.group(0).rstrip("/")
    match = re.search(r"linkedin[:\s]+([a-zA-Z0-9_\-%.]+)", text, re.IGNORECASE)
    if match:
        return f"linkedin.com/in/{match.group(1).strip()}"
    return None


def _extract_candidate_name(text: str, file_name: str) -> str:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return _name_from_filename(file_name)

    for line in lines[:10]:
        m = re.match(r"(?:Name|Candidate)[:\-–\s]+(.+)", line, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            if 2 <= len(candidate.split()) <= 5:
                return candidate.title()

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
    stem = os.path.splitext(file_name)[0]
    stem = re.sub(r"[_\-]+", " ", stem)
    stem = re.sub(r"([a-z])([A-Z])", r"\1 \2", stem)
    return stem.strip().title() or "Unknown Candidate"


def file_parsing_node(state: dict[str, Any]) -> dict[str, Any]:
    uploaded_files: list[dict] = state.get("uploaded_files", [])
    logger.info(f"Parsing {len(uploaded_files)} file(s)")

    parsed_documents: list[ResumeDocument] = []
    parse_errors: dict[str, str] = {}
    seen_hashes: set[str] = set()

    for upload in uploaded_files:
        file_bytes: bytes = upload["file_bytes"]
        file_name: str = upload["file_name"]

        is_dup, file_hash = is_duplicate(file_bytes, seen_hashes)
        if is_dup:
            logger.warning(f"Duplicate skipped: {file_name}")
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
                email=_extract_email(result["raw_text"]),
                phone=_extract_phone(result["raw_text"]),
                linkedin=_extract_linkedin(result["raw_text"]),
            )

            if not doc.is_sufficient:
                parse_errors[file_name] = (
                    f"Only {doc.word_count} words extracted — "
                    "may be image-only. Score confidence will be low."
                )

            parsed_documents.append(doc)
            logger.info(f"Parsed '{candidate_name}' from {file_name} ({doc.word_count} words)")

        except Exception as e:
            logger.error(f"Failed to parse {file_name}: {e}")
            parse_errors[file_name] = str(e)

    return {
        "parsed_documents": parsed_documents,
        "parse_errors": parse_errors,
        "status": "indexing",
    }