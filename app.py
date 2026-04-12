"""
app.py
───────
Streamlit UI for the Resume Scoring Agent — with Persistent Resume Library.

Screens:
  1. Sidebar  — scoring weights, model info, resume library (select/remove/rename)
  2. Upload   — new file upload + library selection + requirement input
  3. Running  — live progress while agent runs
  4. Results  — ranked leaderboard, score cards, evidence viewer

Persistence:
  • Every uploaded resume is saved to ./resume_library/ on disk.
  • Resumes survive browser refreshes and server restarts.
  • Users can remove individual resumes or clear the entire library.
  • Previously stored resumes can be re-selected for any new scoring run
    without re-uploading — just check them in the sidebar or the Library tab.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Resume Scoring Agent",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

from dotenv import load_dotenv
load_dotenv()

from agent.graph import run_agent
from config import settings
from models.score import CandidateScore, MatchLevel, RankedResult
from resume_store import (
    add_resume,
    format_age,
    format_size,
    get_all_resumes,
    get_resume_count,
    get_selected_file_dicts,
    remove_all_resumes,
    remove_resume,
    update_candidate_name,
)
from vector_store.chroma_client import delete_collection


# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main-header{font-size:2.2rem;font-weight:700;margin-bottom:.2rem}
  .sub-header{color:#888;font-size:1rem;margin-bottom:2rem}
  .ext-badge{display:inline-block;padding:1px 7px;border-radius:4px;font-size:.72rem;
             font-weight:600;letter-spacing:.03em;flex-shrink:0}
  .ext-pdf {background:#fee2e2;color:#991b1b}
  .ext-docx{background:#dbeafe;color:#1e40af}
  .ext-doc {background:#dbeafe;color:#1e40af}
  .ext-pptx{background:#fef3c7;color:#92400e}
  .ext-ppt {background:#fef3c7;color:#92400e}
  .resume-name{font-weight:500;font-size:.93rem}
  .resume-meta{font-size:.78rem;color:#888}
  .lib-empty{text-align:center;padding:1.5rem 0;color:#9ca3af;font-size:.9rem}
  .pill-green{background:#dcfce7;color:#166534;display:inline-block;
              padding:2px 10px;border-radius:99px;font-size:.82rem;margin:2px}
  .pill-red  {background:#fee2e2;color:#991b1b;display:inline-block;
              padding:2px 10px;border-radius:99px;font-size:.82rem;margin:2px}
  .pill-blue {background:#dbeafe;color:#1e40af;display:inline-block;
              padding:2px 10px;border-radius:99px;font-size:.82rem;margin:2px}
  .pill-gray {background:#f3f4f6;color:#374151;display:inline-block;
              padding:2px 10px;border-radius:99px;font-size:.82rem;margin:2px}
  .warning-box{background:#fffbeb;border:1px solid #fbbf24;border-radius:8px;
               padding:.75rem 1rem;font-size:.9rem}
  .error-box  {background:#fef2f2;border:1px solid #fca5a5;border-radius:8px;
               padding:.75rem 1rem;font-size:.9rem}
  .info-box   {background:#eff6ff;border:1px solid #93c5fd;border-radius:8px;
               padding:.75rem 1rem;font-size:.9rem}
  hr.thin{margin:3px 0;border:none;border-top:1px solid #f0f0f0}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────

def _init():
    defaults: dict[str, Any] = {
        "session_id":          uuid.uuid4().hex[:12],
        "agent_result":        None,
        "is_running":          False,
        "run_count":           0,
        "weight_skills":       0.5,
        "weight_experience":   0.3,
        "weight_domain":       0.2,
        "lib_selected":        set(),      # set of file_hashes selected for scoring
        "_confirm_remove":     {},         # {file_hash: bool} — awaiting second click
        "_confirm_clear":      False,      # global clear-all confirm
        "_renaming":           {},         # {file_hash: bool}
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _badge(ext: str) -> str:
    css = f"ext-{ext}" if ext in {"pdf","docx","doc","pptx","ppt"} else "ext-pdf"
    return f'<span class="ext-badge {css}">{ext.upper()}</span>'

def _pill(text: str, kind: str = "green") -> str:
    return f'<span class="pill-{kind}">{text}</span>'


# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — library + weights + session
# ═════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🎯 Resume Scoring Agent")
        st.caption("Persistent Resume Library Edition")
        st.divider()

        _render_sidebar_library()

        st.divider()
        st.markdown("### ⚖️ Scoring Weights")
        st.caption("Adjust how each dimension contributes to the overall score.")

        ws = st.slider("Skills",     0.0, 1.0, st.session_state.weight_skills,     0.05, key="w_s")
        we = st.slider("Experience", 0.0, 1.0, st.session_state.weight_experience, 0.05, key="w_e")
        wd = st.slider("Domain",     0.0, 1.0, st.session_state.weight_domain,     0.05, key="w_d")
        total = round(ws + we + wd, 4)
        st.session_state.weight_skills     = ws
        st.session_state.weight_experience = we
        st.session_state.weight_domain     = wd
        if abs(total - 1.0) > 0.01:
            st.warning(f"Weights sum to {total:.2f} — will be auto-normalised.")
        else:
            st.success("✓ Weights sum to 1.00")

        st.divider()
        st.markdown("### 🤖 Model")
        st.code(
            f"LLM:       {settings.llm_model}\n"
            f"Embedding: {settings.embedding_model}\n"
            f"Chunk:     {settings.chunk_size} tokens\n"
            f"Top-K:     {settings.top_k_retrieval}",
            language=None,
        )

        st.divider()
        st.markdown("### 🔄 Session")
        st.caption(f"ID: `{st.session_state.session_id}` · Runs: {st.session_state.run_count}")
        if st.button("Reset Session", use_container_width=True):
            try:
                delete_collection(f"resumes_{st.session_state.session_id}")
            except Exception:
                pass
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

        st.divider()
        st.caption("LangGraph · LlamaIndex · OpenAI · ChromaDB")


def _render_sidebar_library():
    """Full resume library panel in the sidebar."""
    resumes = get_all_resumes()
    count   = len(resumes)

    # Header
    h1, h2 = st.columns([3, 1])
    with h1:
        st.markdown("### 📚 Resume Library")
    with h2:
        st.markdown(
            f"<p style='text-align:right;padding-top:22px;font-size:.82rem;"
            f"color:#6b7280'>{count} saved</p>",
            unsafe_allow_html=True,
        )

    if count == 0:
        st.markdown(
            '<div class="lib-empty">No resumes yet.<br>'
            'Upload files on the main screen to add them.</div>',
            unsafe_allow_html=True,
        )
        return

    # Bulk selection
    bc1, bc2 = st.columns(2)
    with bc1:
        if st.button("All", key="sel_all_sb", use_container_width=True):
            st.session_state.lib_selected = {r["file_hash"] for r in resumes}
            st.rerun()
    with bc2:
        if st.button("None", key="sel_none_sb", use_container_width=True):
            st.session_state.lib_selected = set()
            st.rerun()

    sel_n = len(st.session_state.lib_selected)
    if sel_n:
        st.caption(f"✓ {sel_n} selected for scoring")

    st.markdown("<br>", unsafe_allow_html=True)

    # Per-resume rows
    for resume in resumes:
        _sidebar_resume_row(resume)

    st.markdown("<br>", unsafe_allow_html=True)

    # Clear-all
    with st.expander("⚠️ Danger zone"):
        st.caption("Permanently removes all saved resumes.")
        if st.session_state._confirm_clear:
            st.warning("This cannot be undone. Confirm?")
            ca, cb = st.columns(2)
            with ca:
                if st.button("Yes, clear all", type="primary", use_container_width=True, key="do_clear"):
                    n = remove_all_resumes()
                    st.session_state.lib_selected    = set()
                    st.session_state._confirm_clear  = False
                    st.toast(f"Removed {n} resume(s)", icon="✅")
                    st.rerun()
            with cb:
                if st.button("Cancel", use_container_width=True, key="cancel_clear"):
                    st.session_state._confirm_clear = False
                    st.rerun()
        else:
            if st.button("🗑️ Clear Entire Library", use_container_width=True, key="ask_clear"):
                st.session_state._confirm_clear = True
                st.rerun()


def _sidebar_resume_row(resume: dict):
    """One row in the sidebar library: checkbox · badge · name · age · ✏ · ✕"""
    fh   = resume["file_hash"]
    name = resume.get("candidate_name") or resume["file_name"]
    ext  = resume.get("file_ext", "")
    age  = format_age(resume.get("added_at", ""))
    size = format_size(resume.get("size_bytes", 0))

    # Rename mode
    if st.session_state._renaming.get(fh):
        new_name = st.text_input(
            "Rename", value=name, key=f"ren_inp_{fh}", label_visibility="collapsed"
        )
        ra, rb = st.columns(2)
        with ra:
            if st.button("Save", key=f"ren_save_{fh}", use_container_width=True):
                update_candidate_name(fh, new_name)
                st.session_state._renaming[fh] = False
                st.rerun()
        with rb:
            if st.button("Cancel", key=f"ren_cancel_{fh}", use_container_width=True):
                st.session_state._renaming[fh] = False
                st.rerun()
        return

    # Normal row: [ chk ][ info ][ ✏ ][ ✕/✓ ]
    c1, c2, c3, c4 = st.columns([0.08, 0.68, 0.12, 0.12])

    with c1:
        checked = st.checkbox(
            "", value=fh in st.session_state.lib_selected,
            key=f"sb_chk_{fh}", label_visibility="collapsed"
        )
        if checked != (fh in st.session_state.lib_selected):
            if checked:
                st.session_state.lib_selected.add(fh)
            else:
                st.session_state.lib_selected.discard(fh)
            st.rerun()

    with c2:
        st.markdown(
            f'{_badge(ext)} <span class="resume-name" title="{resume["file_name"]}">'
            f'{name}</span><br>'
            f'<span class="resume-meta">{age} · {size}</span>',
            unsafe_allow_html=True,
        )

    with c3:
        if st.button("✏️", key=f"sb_edit_{fh}", help="Rename"):
            st.session_state._renaming[fh] = True
            st.rerun()

    with c4:
        awaiting = st.session_state._confirm_remove.get(fh, False)
        if awaiting:
            if st.button("✓", key=f"sb_del_ok_{fh}", help="Confirm remove", type="primary"):
                remove_resume(fh)
                st.session_state.lib_selected.discard(fh)
                st.session_state._confirm_remove.pop(fh, None)
                st.toast(f"Removed: {name}", icon="🗑️")
                st.rerun()
        else:
            if st.button("✕", key=f"sb_del_{fh}", help="Remove from library"):
                st.session_state._confirm_remove[fh] = True
                st.rerun()

    st.markdown('<hr class="thin">', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  UPLOAD / MAIN SCREEN
# ═════════════════════════════════════════════════════════════════════════════

def render_upload_screen():
    st.markdown('<div class="main-header">🎯 Resume Scoring Agent</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Upload new resumes or pick from your saved library — '
        'then describe your ideal candidate and score everyone at once.</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1], gap="large")

    with left:
        _render_resume_sources()

    with right:
        st.markdown("#### 💬 Job Requirement")
        requirement = st.text_area(
            "Job requirement",
            height=230,
            placeholder=(
                "Example:\n"
                "Senior Python backend engineer with FastAPI and PostgreSQL, "
                "5+ years of experience. Kubernetes is a plus. "
                "Should have worked in a SaaS product environment."
            ),
            label_visibility="collapsed",
        )
        ch = len(requirement)
        if ch > 0:
            st.caption(f"{ch} characters")
        if 0 < ch < 20:
            st.markdown(
                '<div class="warning-box">⚠️ Requirement is short — '
                'more detail gives better scoring accuracy.</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # Candidate preview
    all_file_dicts = get_selected_file_dicts(list(st.session_state.lib_selected))
    _render_candidate_preview(all_file_dicts)

    # Score button
    can_score = bool(all_file_dicts) and len(requirement.strip()) >= 10
    if not can_score:
        hints = []
        if not all_file_dicts:
            hints.append("upload new files or check resumes in the sidebar library")
        if len(requirement.strip()) < 10:
            hints.append("enter a job requirement (at least 10 characters)")
        st.info(f"To get started: {' and '.join(hints)}.")

    btn_col, _ = st.columns([1, 3])
    with btn_col:
        label = (
            f"🚀 Score {len(all_file_dicts)} Candidate{'s' if len(all_file_dicts) != 1 else ''}"
            if all_file_dicts else "🚀 Score Candidates"
        )
        if st.button(label, disabled=not can_score, use_container_width=True, type="primary"):
            _run_agent(all_file_dicts, requirement.strip())


def _render_resume_sources():
    """Two-tab panel: upload new files, or select from library."""
    st.markdown("#### 📄 Resumes")

    tab_new, tab_lib = st.tabs(["⬆️ Upload New", "📚 From Library"])

    # ── Tab 1: Upload new ─────────────────────────────────────────────────────
    with tab_new:
        uploaded = st.file_uploader(
            "Drop resume files here",
            type=["pdf","docx","doc","pptx","ppt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="main_uploader",
        )
        if uploaded:
            newly, dupes = [], []
            for uf in uploaded:
                raw   = uf.getvalue()
                entry = add_resume(raw, uf.name)
                fh    = entry["file_hash"]
                st.session_state.lib_selected.add(fh)
                # Was it truly new? Check if this was already in the registry
                # before this upload (add_resume returns existing entry for dupes)
                if entry.get("_just_added", True):  # all adds are "new" from user's perspective
                    newly.append(entry)

            if newly:
                st.success(
                    f"✅ {len(newly)} file(s) saved to your library and selected for scoring."
                )
            for uf in uploaded:
                size_kb = len(uf.getvalue()) / 1024
                ext = Path(uf.name).suffix.upper().lstrip(".")
                st.caption(f"📎 {uf.name} — {size_kb:.1f} KB — {ext}")
        else:
            st.caption("Files you upload are automatically saved to your library for future use.")
            total = get_resume_count()
            if total:
                st.info(
                    f"📚 You have {total} resume(s) in your library. "
                    "Switch to the **From Library** tab to select them.",
                    icon="💡",
                )

    # ── Tab 2: Select from library ────────────────────────────────────────────
    with tab_lib:
        resumes = get_all_resumes()
        if not resumes:
            st.markdown(
                '<div class="lib-empty">Your library is empty.<br>'
                'Upload files in the Upload New tab first.</div>',
                unsafe_allow_html=True,
            )
        else:
            # Filter + sort bar
            fc, sc = st.columns([3, 2])
            with fc:
                q = st.text_input(
                    "Filter", placeholder="Filter by name…",
                    label_visibility="collapsed", key="lib_tab_filter"
                )
            with sc:
                sort_opt = st.selectbox(
                    "Sort", ["Newest first","Oldest first","A → Z"],
                    label_visibility="collapsed", key="lib_tab_sort"
                )

            filtered = resumes
            if q.strip():
                ql = q.strip().lower()
                filtered = [
                    r for r in resumes
                    if ql in (r.get("candidate_name") or "").lower()
                    or ql in r["file_name"].lower()
                ]
            if sort_opt == "Oldest first":
                filtered = list(reversed(filtered))
            elif sort_opt == "A → Z":
                filtered = sorted(
                    filtered,
                    key=lambda r: (r.get("candidate_name") or r["file_name"]).lower()
                )

            if not filtered:
                st.caption("No resumes match your filter.")
            else:
                # Column header
                hc1, hc2, hc3, hc4 = st.columns([0.07, 0.55, 0.22, 0.16])
                hc1.markdown("<small>**✓**</small>", unsafe_allow_html=True)
                hc2.markdown("<small>**Candidate**</small>", unsafe_allow_html=True)
                hc3.markdown("<small>**File**</small>", unsafe_allow_html=True)
                hc4.markdown("<small>**Added**</small>", unsafe_allow_html=True)

                for r in filtered:
                    fh   = r["file_hash"]
                    name = r.get("candidate_name") or r["file_name"]
                    ext  = r.get("file_ext", "")
                    age  = format_age(r.get("added_at",""))
                    fname_short = (
                        r["file_name"][:22] + "…"
                        if len(r["file_name"]) > 24 else r["file_name"]
                    )

                    rc1, rc2, rc3, rc4 = st.columns([0.07, 0.55, 0.22, 0.16])
                    with rc1:
                        checked = st.checkbox(
                            "", value=fh in st.session_state.lib_selected,
                            key=f"tab_chk_{fh}", label_visibility="collapsed"
                        )
                        if checked != (fh in st.session_state.lib_selected):
                            if checked:
                                st.session_state.lib_selected.add(fh)
                            else:
                                st.session_state.lib_selected.discard(fh)
                            st.rerun()
                    with rc2:
                        st.markdown(
                            f'{_badge(ext)} <span class="resume-name">{name}</span>',
                            unsafe_allow_html=True,
                        )
                    with rc3:
                        st.caption(fname_short)
                    with rc4:
                        st.caption(age)

            # Bulk helpers
            ba, bb = st.columns(2)
            with ba:
                if st.button("Select All", key="tab_all", use_container_width=True):
                    for r in resumes:
                        st.session_state.lib_selected.add(r["file_hash"])
                    st.rerun()
            with bb:
                if st.button("Clear Selection", key="tab_none", use_container_width=True):
                    st.session_state.lib_selected = set()
                    st.rerun()

            n_sel = len(st.session_state.lib_selected)
            if n_sel:
                st.success(f"✓ {n_sel} candidate(s) selected")


def _render_candidate_preview(file_dicts: list[dict]):
    if not file_dicts:
        return
    registry_map = {r["file_name"]: r for r in get_all_resumes()}
    st.markdown(f"**{len(file_dicts)} candidate(s) queued for scoring:**")
    cols = st.columns(min(len(file_dicts), 4))
    for i, fd in enumerate(file_dicts):
        reg  = registry_map.get(fd["file_name"], {})
        name = reg.get("candidate_name") or fd["file_name"]
        ext  = reg.get("file_ext", Path(fd["file_name"]).suffix.lstrip("."))
        with cols[i % 4]:
            st.markdown(f"{_badge(ext)} {name}", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  AGENT TRIGGER
# ═════════════════════════════════════════════════════════════════════════════

def _run_agent(file_dicts: list[dict], requirement: str):
    st.session_state.is_running           = True
    st.session_state.agent_result         = None
    st.session_state._pending_files       = file_dicts
    st.session_state._pending_requirement = requirement
    st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
#  RUNNING SCREEN
# ═════════════════════════════════════════════════════════════════════════════

STEPS = [
    ("📄 Parsing files",       "Extracting text from resumes…"),
    ("🔢 Indexing",            "Chunking and embedding into vector store…"),
    ("🧠 Parsing requirement", "Converting your description into a scoring rubric…"),
    ("🔍 Retrieving",          "Finding relevant sections per candidate…"),
    ("⚖️ Scoring",             "GPT-4o evaluating each candidate…"),
    ("🏆 Ranking",             "Sorting and generating summary…"),
]


def render_running_screen(file_dicts: list[dict], requirement: str):
    st.markdown("## ⏳ Evaluating Candidates…")
    st.markdown(f"Scoring **{len(file_dicts)}** candidate(s).")

    bar    = st.progress(0)
    status = st.empty()

    for i, (step, desc) in enumerate(STEPS):
        bar.progress((i + 1) / len(STEPS))
        status.markdown(f"**{step}** — {desc}")

    with st.spinner("Running agent pipeline…"):
        try:
            result = run_agent(
                uploaded_files=file_dicts,
                user_requirement=requirement,
                session_id=st.session_state.session_id,
            )
            st.session_state.agent_result = result
            st.session_state.run_count   += 1
        except Exception as e:
            st.session_state.agent_result = {"fatal_error": str(e), "status": "error"}

    bar.progress(1.0)
    status.markdown("**✅ Complete!**")
    st.session_state.is_running           = False
    st.session_state._pending_files       = None
    st.session_state._pending_requirement = None
    st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
#  RESULTS SCREEN
# ═════════════════════════════════════════════════════════════════════════════

MATCH_CSS = {
    MatchLevel.STRONG:   "match-strong",
    MatchLevel.GOOD:     "match-good",
    MatchLevel.PARTIAL:  "match-partial",
    MatchLevel.WEAK:     "match-weak",
    MatchLevel.NO_MATCH: "match-none",
}


def render_results_screen(state: dict[str, Any]):
    ranked         = state.get("ranked_result")
    parse_errors   = state.get("parse_errors", {})
    scoring_errors = state.get("scoring_errors", {})
    rubric_error   = state.get("rubric_error")
    fatal_error    = state.get("fatal_error")

    if fatal_error:
        st.error(f"❌ Pipeline error: {fatal_error}")
        if st.button("← Start Over"):
            st.session_state.agent_result = None
            st.rerun()
        return

    st.markdown("## 🏆 Scoring Results")

    if rubric_error:
        st.markdown(f'<div class="warning-box">⚠️ {rubric_error}</div>', unsafe_allow_html=True)
        st.markdown("")
    if parse_errors:
        with st.expander(f"⚠️ {len(parse_errors)} file(s) had parsing issues"):
            for f, e in parse_errors.items():
                st.markdown(f'<div class="error-box">📎 **{f}**: {e}</div>', unsafe_allow_html=True)
    if scoring_errors:
        with st.expander(f"⚠️ {len(scoring_errors)} candidate(s) had scoring errors"):
            for f, e in scoring_errors.items():
                st.markdown(f'<div class="error-box">📎 **{f}**: {e}</div>', unsafe_allow_html=True)

    if ranked is None or not ranked.candidates:
        st.warning("No candidates were successfully evaluated.")
        if st.button("← Start Over"):
            st.session_state.agent_result = None
            st.rerun()
        return

    if ranked.rubric_used:
        st.markdown(
            f'<div class="info-box">📋 <strong>Rubric:</strong> {ranked.rubric_used}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Evaluated",      ranked.total_processed)
    m2.metric("Strong (≥75)",   sum(1 for c in ranked.candidates if c.overall_score >= 75))
    m3.metric("Good (≥55)",     sum(1 for c in ranked.candidates if 55 <= c.overall_score < 75))
    m4.metric("No match (<30)", ranked.no_match_count)

    st.divider()

    if ranked.comparative_summary:
        st.markdown("### 💡 Analysis")
        st.info(ranked.comparative_summary)

    _render_score_chart(ranked.candidates)
    st.divider()
    st.markdown("### 📊 Candidate Rankings")
    for rank, c in enumerate(ranked.candidates, 1):
        _render_candidate_card(rank, c)
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("← Score New Batch", use_container_width=True):
            st.session_state.agent_result = None
            st.rerun()
    with col2:
        if st.button("📚 Back to Library", use_container_width=True):
            st.session_state.agent_result = None
            st.rerun()
    with col3:
        df  = _results_df(ranked.candidates)
        csv = df.to_csv(index=False)
        st.download_button(
            "⬇️ Export CSV", data=csv,
            file_name="resume_scores.csv", mime="text/csv",
            use_container_width=True,
        )


def _render_score_chart(candidates: list[CandidateScore]):
    if not candidates:
        return
    colors = [
        "#1a9e5c" if c.overall_score >= 75 else
        "#2563eb" if c.overall_score >= 55 else
        "#d97706" if c.overall_score >= 35 else "#dc2626"
        for c in candidates
    ]
    fig = go.Figure(go.Bar(
        x=[c.overall_score for c in candidates],
        y=[c.candidate_name for c in candidates],
        orientation="h",
        marker_color=colors,
        text=[f"{c.overall_score:.1f}" for c in candidates],
        textposition="outside",
        cliponaxis=False,
    ))
    fig.update_layout(
        xaxis=dict(range=[0,110], title="Overall Score (0–100)"),
        yaxis=dict(autorange="reversed"),
        height=max(200, len(candidates) * 50 + 80),
        margin=dict(l=10, r=60, t=20, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13), showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_candidate_card(rank: int, c: CandidateScore):
    badge = " 🔍 Manual review recommended" if c.needs_manual_review else ""
    with st.expander(
        f"{c.match_level_emoji} #{rank} — {c.candidate_name} — "
        f"**{c.overall_score:.1f}/100** ({c.match_level.value}){badge}",
        expanded=(rank == 1),
    ):
        if c.error:
            st.error(f"Scoring error: {c.error}")

        lft, rgt = st.columns(2, gap="large")
        with lft:
            st.markdown("**Score breakdown**")
            for label, score in [("Skills", c.skills_score), ("Experience", c.experience_score), ("Domain", c.domain_score)]:
                st.markdown(f"<small>{label}: **{score:.1f}**</small>", unsafe_allow_html=True)
                st.progress(int(score) / 100)
            st.markdown("")
            st.caption(f"Confidence: {c.confidence * 100:.0f}%")
            if c.experience_years_found is not None:
                st.caption(f"Experience found: ~{c.experience_years_found} years")
            if c.seniority_detected:
                st.caption(f"Seniority detected: {c.seniority_detected}")
        with rgt:
            if c.matched_skills:
                st.markdown("**✅ Matched required**")
                st.markdown(" ".join(_pill(s,"green") for s in c.matched_skills), unsafe_allow_html=True)
                st.markdown("")
            if c.matched_preferred_skills:
                st.markdown("**🔵 Matched preferred**")
                st.markdown(" ".join(_pill(s,"blue") for s in c.matched_preferred_skills), unsafe_allow_html=True)
                st.markdown("")
            if c.missing_skills:
                st.markdown("**❌ Missing required**")
                st.markdown(" ".join(_pill(s,"red") for s in c.missing_skills), unsafe_allow_html=True)

        st.markdown("")
        if c.justification:
            st.markdown(f"**📝 Assessment:** {c.justification}")

        gl, gr = st.columns(2)
        with gl:
            if c.key_strengths:
                st.markdown("**💪 Strengths**")
                for s in c.key_strengths: st.markdown(f"- {s}")
        with gr:
            if c.key_gaps:
                st.markdown("**⚠️ Gaps**")
                for g in c.key_gaps: st.markdown(f"- {g}")

        if c.retrieved_chunks:
            with st.expander("🔎 Evidence used for scoring"):
                for i, chunk in enumerate(c.retrieved_chunks[:5], 1):
                    st.markdown(f"**Excerpt {i}:**")
                    st.text(chunk[:600] + ("…" if len(chunk) > 600 else ""))
                    st.markdown("")


def _results_df(candidates: list[CandidateScore]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Rank":                   i,
            "Candidate":              c.candidate_name,
            "File":                   c.file_name,
            "Overall Score":          round(c.overall_score, 2),
            "Match Level":            c.match_level.value,
            "Skills Score":           round(c.skills_score, 2),
            "Experience Score":       round(c.experience_score, 2),
            "Domain Score":           round(c.domain_score, 2),
            "Confidence":             round(c.confidence, 2),
            "Experience Years Found": c.experience_years_found,
            "Matched Skills":         ", ".join(c.matched_skills),
            "Missing Skills":         ", ".join(c.missing_skills),
            "Preferred Matched":      ", ".join(c.matched_preferred_skills),
            "Key Strengths":          " | ".join(c.key_strengths),
            "Key Gaps":               " | ".join(c.key_gaps),
            "Justification":          c.justification,
            "Needs Manual Review":    c.needs_manual_review,
        }
        for i, c in enumerate(candidates, 1)
    ])


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═════════════════════════════════════════════════════════════════════════════

def main():
    render_sidebar()

    pending_files = st.session_state.get("_pending_files")
    pending_req   = st.session_state.get("_pending_requirement")

    if st.session_state.is_running and pending_files and pending_req:
        render_running_screen(pending_files, pending_req)
    elif st.session_state.agent_result is not None:
        render_results_screen(st.session_state.agent_result)
    else:
        render_upload_screen()


if __name__ == "__main__":
    main()