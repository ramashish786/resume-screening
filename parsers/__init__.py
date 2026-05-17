from __future__ import annotations

import hashlib
from pathlib import Path

from parsers.docx_parser import parse_docx
from parsers.pdf_parser import parse_pdf
from parsers.pptx_parser import parse_pptx


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".pptx", ".ppt"}


def parse_resume(file_bytes: bytes, file_name: str) -> dict:
    suffix = Path(file_name).suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    if suffix == ".pdf":
        return parse_pdf(file_bytes, file_name)
    elif suffix in (".docx", ".doc"):
        return parse_docx(file_bytes, file_name)
    elif suffix in (".pptx", ".ppt"):
        return parse_pptx(file_bytes, file_name)


def is_duplicate(file_bytes: bytes, seen_hashes: set[str]) -> tuple[bool, str]:
    """
    Check if a file has already been uploaded in the current session.
    Returns (is_duplicate, file_hash).
    """
    file_hash = hashlib.md5(file_bytes).hexdigest()
    return file_hash in seen_hashes, file_hash