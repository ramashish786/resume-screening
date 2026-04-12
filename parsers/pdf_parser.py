"""
parsers/pdf_parser.py
──────────────────────
Extracts text from PDF resumes using pdfplumber.
Handles multi-column layouts, tables, and scanned-PDF warnings.
"""

from __future__ import annotations

import hashlib
import io
from typing import Union

from loguru import logger

try:
    import pdfplumber
except ImportError:
    pdfplumber = None  # type: ignore


def parse_pdf(file_bytes: bytes, file_name: str) -> dict:
    """
    Extract text from a PDF file.

    Args:
        file_bytes: Raw bytes of the uploaded PDF.
        file_name:  Original filename (for logging / metadata).

    Returns:
        dict with keys: raw_text, page_count, word_count, warnings, file_hash
    """
    if pdfplumber is None:
        raise ImportError("pdfplumber is not installed. Run: pip install pdfplumber")

    warnings: list[str] = []
    pages_text: list[str] = []

    file_hash = hashlib.md5(file_bytes).hexdigest()

    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            page_count = len(pdf.pages)

            for i, page in enumerate(pdf.pages):
                try:
                    # Extract words with layout-aware ordering
                    page_text = page.extract_text(
                        x_tolerance=3,
                        y_tolerance=3,
                        layout=True,
                        x_density=7.25,
                        y_density=13,
                    )

                    # Also try to extract tables and append as structured text
                    tables = page.extract_tables()
                    table_texts: list[str] = []
                    for table in tables:
                        for row in table:
                            if row:
                                row_text = " | ".join(
                                    cell.strip() if cell else ""
                                    for cell in row
                                    if cell
                                )
                                if row_text.strip():
                                    table_texts.append(row_text)

                    combined = (page_text or "") + (
                        "\n" + "\n".join(table_texts) if table_texts else ""
                    )
                    pages_text.append(combined)

                except Exception as e:
                    warnings.append(f"Page {i + 1}: extraction warning — {str(e)[:80]}")
                    pages_text.append("")

    except Exception as e:
        logger.error(f"PDF parse failed for {file_name}: {e}")
        raise ValueError(f"Could not open PDF '{file_name}': {e}") from e

    raw_text = "\n\n".join(p for p in pages_text if p.strip())
    word_count = len(raw_text.split())

    if word_count < 30:
        warnings.append(
            "Very little text extracted — this may be a scanned/image-only PDF. "
            "Consider using OCR (Azure Document Intelligence) for better results."
        )

    logger.info(
        f"PDF parsed: {file_name} | pages={page_count} | words={word_count} | warnings={len(warnings)}"
    )

    return {
        "raw_text": raw_text,
        "page_count": page_count,
        "word_count": word_count,
        "warnings": warnings,
        "file_hash": file_hash,
    }