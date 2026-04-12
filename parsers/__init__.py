"""
parsers/__init__.py
────────────────────
Unified entry point for resume file parsing.
Detects file type from MIME type / extension and routes to the correct parser.
"""

from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path

from loguru import logger

from parsers.docx_parser import parse_docx
from parsers.pdf_parser import parse_pdf
from parsers.pptx_parser import parse_pptx


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".pptx", ".ppt"}

# MIME type → parser mapping
MIME_MAP = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/msword": "docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "application/vnd.ms-powerpoint": "pptx",
}


def parse_resume(file_bytes: bytes, file_name: str) -> dict:
    """
    Route the uploaded file to the correct parser based on extension / MIME type.

    Returns a standardised dict:
        raw_text, page_count, word_count, warnings, file_hash
    
    Raises:
        ValueError for unsupported file types or corrupt files.
    """
    suffix = Path(file_name).suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: '{suffix}'. "
            f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    # Determine parser from extension (more reliable than MIME sniffing for uploads)
    if suffix == ".pdf":
        return parse_pdf(file_bytes, file_name)
    elif suffix in (".docx", ".doc"):
        return parse_docx(file_bytes, file_name)
    elif suffix in (".pptx", ".ppt"):
        return parse_pptx(file_bytes, file_name)
    else:
        raise ValueError(f"No parser registered for extension '{suffix}'")


def is_duplicate(file_bytes: bytes, seen_hashes: set[str]) -> tuple[bool, str]:
    """
    Check if a file has already been uploaded in the current session.
    Returns (is_duplicate, file_hash).
    """
    file_hash = hashlib.md5(file_bytes).hexdigest()
    return file_hash in seen_hashes, file_hash