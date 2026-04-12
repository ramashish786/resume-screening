"""
resume_store.py
────────────────
Persistent resume storage for the POC.

Resumes are saved to disk under ./resume_library/ so they survive
across Streamlit sessions and server restarts.

Storage layout:
  resume_library/
    registry.json          ← index of all stored resumes
    files/
      <sha256_hash>.bin    ← raw file bytes
      <sha256_hash>.meta   ← JSON metadata (name, filename, size, etc.)

The registry is a flat JSON list — small enough that reading it on every
render is fast (< 1ms for hundreds of resumes).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Storage paths ─────────────────────────────────────────────────────────────

LIBRARY_DIR  = Path("./resume_library")
FILES_DIR    = LIBRARY_DIR / "files"
REGISTRY_PATH = LIBRARY_DIR / "registry.json"

LIBRARY_DIR.mkdir(exist_ok=True)
FILES_DIR.mkdir(exist_ok=True)


# ── Registry I/O ──────────────────────────────────────────────────────────────

def _load_registry() -> list[dict]:
    """Load the resume registry from disk. Returns [] if not yet created."""
    if not REGISTRY_PATH.exists():
        return []
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _save_registry(registry: list[dict]) -> None:
    """Atomically write the registry to disk."""
    tmp = REGISTRY_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(REGISTRY_PATH)


# ── Public API ────────────────────────────────────────────────────────────────

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def add_resume(file_bytes: bytes, file_name: str) -> dict:
    """
    Save a resume to the persistent library.

    If the exact same file (by SHA-256) is already stored, returns the
    existing record without duplicating the file.

    Returns the registry entry dict.
    """
    file_hash = sha256(file_bytes)
    registry  = _load_registry()

    # Dedup check
    existing = next((r for r in registry if r["file_hash"] == file_hash), None)
    if existing:
        return existing

    # Write raw bytes
    bin_path  = FILES_DIR / f"{file_hash}.bin"
    bin_path.write_bytes(file_bytes)

    # Build registry entry
    now = datetime.now(timezone.utc).isoformat()
    ext = Path(file_name).suffix.lower().lstrip(".")
    entry: dict = {
        "file_hash":    file_hash,
        "file_name":    file_name,
        "file_ext":     ext,
        "size_bytes":   len(file_bytes),
        "added_at":     now,
        "candidate_name": _infer_name(file_name),
    }

    registry.append(entry)
    _save_registry(registry)
    return entry


def remove_resume(file_hash: str) -> bool:
    """
    Remove a resume from the library by its SHA-256 hash.

    Deletes both the raw file and the registry entry.
    Returns True if found and removed, False if not found.
    """
    registry = _load_registry()
    before   = len(registry)
    registry = [r for r in registry if r["file_hash"] != file_hash]

    if len(registry) == before:
        return False

    _save_registry(registry)

    # Delete raw file
    bin_path = FILES_DIR / f"{file_hash}.bin"
    if bin_path.exists():
        bin_path.unlink()

    return True


def remove_all_resumes() -> int:
    """Remove every resume from the library. Returns count deleted."""
    registry = _load_registry()
    count    = len(registry)

    if FILES_DIR.exists():
        shutil.rmtree(FILES_DIR)
        FILES_DIR.mkdir(exist_ok=True)

    _save_registry([])
    return count


def get_all_resumes() -> list[dict]:
    """Return the full registry (sorted newest first)."""
    return sorted(_load_registry(), key=lambda r: r["added_at"], reverse=True)


def get_resume_bytes(file_hash: str) -> Optional[bytes]:
    """Load the raw bytes for a stored resume. Returns None if not found."""
    bin_path = FILES_DIR / f"{file_hash}.bin"
    if not bin_path.exists():
        return None
    return bin_path.read_bytes()


def get_resume_count() -> int:
    return len(_load_registry())


def get_selected_file_dicts(file_hashes: list[str]) -> list[dict]:
    """
    Build the uploaded_files list expected by run_agent() for a selection
    of stored resumes.
    """
    registry = {r["file_hash"]: r for r in _load_registry()}
    result   = []
    for fh in file_hashes:
        entry = registry.get(fh)
        if not entry:
            continue
        raw = get_resume_bytes(fh)
        if raw is None:
            continue
        result.append({
            "file_bytes": raw,
            "file_name":  entry["file_name"],
        })
    return result


def update_candidate_name(file_hash: str, name: str) -> bool:
    """Allow the user to rename a candidate in the library."""
    registry = _load_registry()
    for entry in registry:
        if entry["file_hash"] == file_hash:
            entry["candidate_name"] = name.strip()
            _save_registry(registry)
            return True
    return False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _infer_name(file_name: str) -> str:
    """Derive a human-readable candidate name from the filename."""
    import re
    stem = Path(file_name).stem
    stem = re.sub(r"[_\-]+", " ", stem)
    stem = re.sub(r"([a-z])([A-Z])", r"\1 \2", stem)
    stem = re.sub(r"\d+", "", stem).strip()
    return stem.title() or Path(file_name).stem


def format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / 1024 ** 2:.1f} MB"


def format_age(added_at: str) -> str:
    """Return a human-friendly relative time string."""
    try:
        dt    = datetime.fromisoformat(added_at)
        delta = datetime.now(timezone.utc) - dt
        secs  = int(delta.total_seconds())
        if secs < 60:       return "just now"
        if secs < 3600:     return f"{secs // 60}m ago"
        if secs < 86400:    return f"{secs // 3600}h ago"
        if secs < 604800:   return f"{secs // 86400}d ago"
        return dt.strftime("%b %d, %Y")
    except Exception:
        return ""