"""
app.py
───────
Streamlit UI for the Resume Scoring Agent.

Screens:
  1. Sidebar  — configuration, session management
  2. Upload   — drag-and-drop file upload + requirement input
  3. Running  — live progress indicator while agent runs
  4. Results  — ranked leaderboard, score cards, evidence viewer

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from loguru import logger

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Resume Scoring Agent",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load env before importing agent (config reads env at import time) ─────────
from dotenv import load_dotenv
load_dotenv()

from agent.graph import run_agent
from config import settings
from models.score import CandidateScore, MatchLevel, RankedResult
from vector_store.chroma_client import delete_collection


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .score-card {
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        background: #fafafa;
    }
    .score-badge {
        font-size: 2rem;
        font-weight: 700;
    }
    .match-strong { color: #1a9e5c; }
    .match-good   { color: #2563eb; }
    .match-partial{ color: #d97706; }
    .match-weak   { color: #ea580c; }
    .match-none   { color: #dc2626; }
    .skill-pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 99px;
        font-size: 0.82rem;
        margin: 2px;
    }
    .pill-green { background: #dcfce7; color: #166534; }
    .pill-red   { background: #fee2e2; color: #991b1b; }
    .pill-blue  { background: #dbeafe; color: #1e40af; }
    .pill-gray  { background: #f3f4f6; color: #374151; }
    .warning-box {
        background: #fffbeb;
        border: 1px solid #fbbf24;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 0.9rem;
    }
    .error-box {
        background: #fef2f2;
        border: 1px solid #fca5a5;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 0.9rem;
    }
    .info-box {
        background: #eff6ff;
        border: 1px solid #93c5fd;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 0.9rem;
    }
    div[data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
    }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ──────────────────────────────────────────────

def init_session():
    defaults = {
        "session_id": uuid.uuid4().hex[:12],
        "agent_result": None,
        "is_running": False,
        "run_count": 0,
        "weight_skills": 0.5,
        "weight_experience": 0.3,
        "weight_domain": 0.2,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        st.markdown("### Scoring Weights")
        st.caption("Adjust how much each dimension contributes to the overall score.")

        w_skills = st.slider("Skills weight", 0.0, 1.0, st.session_state.weight_skills, 0.05, key="w_skills")
        w_exp    = st.slider("Experience weight", 0.0, 1.0, st.session_state.weight_experience, 0.05, key="w_exp")
        w_domain = st.slider("Domain weight", 0.0, 1.0, st.session_state.weight_domain, 0.05, key="w_domain")

        total = round(w_skills + w_exp + w_domain, 4)
        if abs(total - 1.0) > 0.01:
            st.warning(f"Weights sum to {total:.2f} (should be 1.0). They will be auto-normalised.")
        else:
            st.success(f"✓ Weights sum to {total:.2f}")

        st.session_state.weight_skills = w_skills
        st.session_state.weight_experience = w_exp
        st.session_state.weight_domain = w_domain

        st.divider()

        st.markdown("### Model Info")
        st.code(f"LLM: {settings.llm_model}\nEmbedding: {settings.embedding_model}\nChunk size: {settings.chunk_size} tokens\nTop-K retrieval: {settings.top_k_retrieval}")

        st.divider()

        st.markdown("### Session")
        st.caption(f"Session ID: `{st.session_state.session_id}`")
        st.caption(f"Runs this session: {st.session_state.run_count}")

        if st.button("🗑️ Reset Session", width='stretch'):
            # Clean up ChromaDB collection
            try:
                delete_collection(f"resumes_{st.session_state.session_id}")
            except Exception:
                pass
            # Reset session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        st.divider()
        st.caption("Resume Scoring Agent v1.0 — POC")
        st.caption("Powered by LangGraph + LlamaIndex + OpenAI + ChromaDB")


# ── Upload screen ─────────────────────────────────────────────────────────────

def render_upload_screen():
    st.markdown('<div class="main-header">🎯 Resume Scoring Agent</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Upload resumes and describe your ideal candidate — '
        'the agent will rank and score each one.</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("#### 📄 Upload Resumes")
        uploaded_files = st.file_uploader(
            "Drop resume files here",
            type=["pdf", "docx", "doc", "pptx", "ppt"],
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX, DOC, PPTX, PPT",
            label_visibility="collapsed",
        )

        if uploaded_files:
            st.success(f"✓ {len(uploaded_files)} file(s) ready")
            for f in uploaded_files:
                size_kb = len(f.getvalue()) / 1024
                ext = Path(f.name).suffix.upper().lstrip(".")
                st.caption(f"📎 {f.name} — {size_kb:.1f} KB — {ext}")

    with col2:
        st.markdown("#### 💬 Job Requirement")
        requirement = st.text_area(
            "Describe the ideal candidate",
            height=180,
            placeholder=(
                "Example:\n"
                "I need a senior Python backend engineer with FastAPI and PostgreSQL, "
                "at least 5 years of experience. Kubernetes is a plus. "
                "They should have worked in a SaaS product company."
            ),
            label_visibility="collapsed",
        )

        char_count = len(requirement)
        if char_count > 0:
            st.caption(f"{char_count} characters")

        if char_count < 20 and char_count > 0:
            st.markdown(
                '<div class="warning-box">⚠️ Requirement seems too short. '
                'More detail leads to better scoring accuracy.</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # Validate before allowing run
    can_run = bool(uploaded_files) and len(requirement.strip()) >= 10

    if not can_run:
        reasons = []
        if not uploaded_files:
            reasons.append("upload at least one resume")
        if len(requirement.strip()) < 10:
            reasons.append("enter a job requirement (min 10 characters)")
        st.info(f"To get started: {' and '.join(reasons)}.")

    run_col, _ = st.columns([1, 3])
    with run_col:
        if st.button(
            "🚀 Score Candidates",
            disabled=not can_run,
            width='stretch',
            type="primary",
        ):
            # Prepare file dicts
            file_dicts = [
                {"file_bytes": f.getvalue(), "file_name": f.name}
                for f in uploaded_files
            ]
            _run_agent(file_dicts, requirement.strip())


def _run_agent(file_dicts: list[dict], requirement: str):
    """Trigger the agent and store result in session state."""
    st.session_state.is_running = True
    st.session_state.agent_result = None
    # Store inputs BEFORE rerun so they're available on the next render cycle
    st.session_state._pending_files = file_dicts
    st.session_state._pending_requirement = requirement
    st.rerun()


# ── Running screen ────────────────────────────────────────────────────────────

PIPELINE_STEPS = [
    ("📄 Parsing files", "Extracting text from your uploaded resumes..."),
    ("🔢 Indexing", "Chunking and embedding resumes into vector store..."),
    ("🧠 Parsing requirement", "Converting your job description into a scoring rubric..."),
    ("🔍 Retrieving", "Finding relevant sections from each resume..."),
    ("⚖️ Scoring", "GPT-4o evaluating each candidate against the rubric..."),
    ("🏆 Ranking", "Sorting candidates and generating summary..."),
]


def render_running_screen(file_dicts: list[dict], requirement: str):
    st.markdown("## ⏳ Evaluating Candidates...")
    st.markdown(f"Processing **{len(file_dicts)}** resume(s) against your requirement.")

    progress_bar = st.progress(0)
    status_text = st.empty()
    step_container = st.container()

    with step_container:
        for i, (step_name, step_desc) in enumerate(PIPELINE_STEPS):
            progress_bar.progress((i + 1) / len(PIPELINE_STEPS))
            status_text.markdown(f"**{step_name}** — {step_desc}")

    # Run the agent
    with st.spinner("Running agent pipeline..."):
        try:
            result = run_agent(
                uploaded_files=file_dicts,
                user_requirement=requirement,
                session_id=st.session_state.session_id,
            )
            st.session_state.agent_result = result
            st.session_state.run_count += 1
        except Exception as e:
            st.session_state.agent_result = {"fatal_error": str(e), "status": "error"}

    progress_bar.progress(1.0)
    status_text.markdown("**✅ Complete!**")
    st.session_state.is_running = False
    st.session_state._pending_files = None
    st.session_state._pending_requirement = None
    st.rerun()


# ── Results screen ────────────────────────────────────────────────────────────

MATCH_CSS = {
    MatchLevel.STRONG: "match-strong",
    MatchLevel.GOOD: "match-good",
    MatchLevel.PARTIAL: "match-partial",
    MatchLevel.WEAK: "match-weak",
    MatchLevel.NO_MATCH: "match-none",
}


def _skill_pill(skill: str, kind: str = "green") -> str:
    css = {"green": "pill-green", "red": "pill-red", "blue": "pill-blue", "gray": "pill-gray"}[kind]
    return f'<span class="skill-pill {css}">{skill}</span>'


def render_results_screen(state: dict[str, Any]):
    ranked: RankedResult | None = state.get("ranked_result")
    parse_errors: dict = state.get("parse_errors", {})
    scoring_errors: dict = state.get("scoring_errors", {})
    rubric_error: str | None = state.get("rubric_error")
    fatal_error: str | None = state.get("fatal_error")

    # Fatal error
    if fatal_error:
        st.error(f"❌ Pipeline error: {fatal_error}")
        if st.button("← Start Over"):
            st.session_state.agent_result = None
            st.rerun()
        return

    st.markdown("## 🏆 Scoring Results")

    # ── Error/warning banners ──
    if rubric_error:
        st.markdown(f'<div class="warning-box">⚠️ {rubric_error}</div>', unsafe_allow_html=True)
        st.markdown("")

    if parse_errors:
        with st.expander(f"⚠️ {len(parse_errors)} file(s) had parsing issues"):
            for fname, err in parse_errors.items():
                st.markdown(f'<div class="error-box">📎 **{fname}**: {err}</div>', unsafe_allow_html=True)

    if scoring_errors:
        with st.expander(f"⚠️ {len(scoring_errors)} candidate(s) had scoring errors"):
            for fname, err in scoring_errors.items():
                st.markdown(f'<div class="error-box">📎 **{fname}**: {err}</div>', unsafe_allow_html=True)

    if ranked is None or not ranked.candidates:
        st.warning("No candidates were successfully evaluated. Check the errors above.")
        if st.button("← Start Over"):
            st.session_state.agent_result = None
            st.rerun()
        return

    # ── Rubric summary ──
    if ranked.rubric_used:
        st.markdown(
            f'<div class="info-box">📋 <strong>Rubric:</strong> {ranked.rubric_used}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")

    # ── Key metrics ──
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Candidates evaluated", ranked.total_processed)
    m2.metric("Strong matches (≥75)", sum(1 for c in ranked.candidates if c.overall_score >= 75))
    m3.metric("Good matches (≥55)", sum(1 for c in ranked.candidates if 55 <= c.overall_score < 75))
    m4.metric("No matches (<30)", ranked.no_match_count)

    st.divider()

    # ── Comparative summary ──
    if ranked.comparative_summary:
        st.markdown("### 💡 Analysis")
        st.info(ranked.comparative_summary)

    # ── Score overview chart ──
    _render_score_chart(ranked.candidates)

    st.divider()

    # ── Ranked candidate cards ──
    st.markdown("### 📊 Candidate Rankings")

    for rank, candidate in enumerate(ranked.candidates, 1):
        _render_candidate_card(rank, candidate)

    st.divider()

    # ── Action buttons ──
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Score New Batch", width='stretch'):
            st.session_state.agent_result = None
            st.rerun()
    with col2:
        # Export results as CSV
        df = _build_results_df(ranked.candidates)
        csv = df.to_csv(index=False)
        st.download_button(
            "⬇️ Export Results CSV",
            data=csv,
            file_name="resume_scores.csv",
            mime="text/csv",
            use_container_width=True,
        )


def _render_score_chart(candidates: list[CandidateScore]):
    """Render a horizontal bar chart of overall scores."""
    if not candidates:
        return

    names = [c.candidate_name for c in candidates]
    scores = [c.overall_score for c in candidates]
    colors = []
    for c in candidates:
        if c.overall_score >= 75:
            colors.append("#1a9e5c")
        elif c.overall_score >= 55:
            colors.append("#2563eb")
        elif c.overall_score >= 35:
            colors.append("#d97706")
        else:
            colors.append("#dc2626")

    fig = go.Figure(go.Bar(
        x=scores,
        y=names,
        orientation="h",
        marker_color=colors,
        text=[f"{s:.1f}" for s in scores],
        textposition="outside",
        cliponaxis=False,
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 110], title="Overall Score (0–100)"),
        yaxis=dict(autorange="reversed"),
        height=max(200, len(candidates) * 50 + 80),
        margin=dict(l=10, r=60, t=20, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_candidate_card(rank: int, c: CandidateScore):
    """Render a detailed score card for a single candidate."""
    match_css = MATCH_CSS.get(c.match_level, "match-none")
    review_badge = " 🔍 Manual review recommended" if c.needs_manual_review else ""

    with st.expander(
        f"{c.match_level_emoji} #{rank} — {c.candidate_name} — "
        f"**{c.overall_score:.1f}/100** ({c.match_level.value}){review_badge}",
        expanded=(rank == 1),
    ):
        if c.error:
            st.error(f"Scoring error: {c.error}")

        left, right = st.columns([1, 1], gap="large")

        with left:
            # Dimension scores
            st.markdown("**Score breakdown**")
            _score_bar("Skills", c.skills_score)
            _score_bar("Experience", c.experience_score)
            _score_bar("Domain", c.domain_score)

            st.markdown("")
            st.caption(f"Confidence: {c.confidence * 100:.0f}%")
            if c.experience_years_found is not None:
                st.caption(f"Experience found: ~{c.experience_years_found} years")
            if c.seniority_detected:
                st.caption(f"Seniority detected: {c.seniority_detected}")

        with right:
            # Skills pills
            if c.matched_skills:
                st.markdown("**✅ Matched required skills**")
                pills = " ".join(_skill_pill(s, "green") for s in c.matched_skills)
                st.markdown(pills, unsafe_allow_html=True)
                st.markdown("")

            if c.matched_preferred_skills:
                st.markdown("**🔵 Matched preferred skills**")
                pills = " ".join(_skill_pill(s, "blue") for s in c.matched_preferred_skills)
                st.markdown(pills, unsafe_allow_html=True)
                st.markdown("")

            if c.missing_skills:
                st.markdown("**❌ Missing required skills**")
                pills = " ".join(_skill_pill(s, "red") for s in c.missing_skills)
                st.markdown(pills, unsafe_allow_html=True)

        st.markdown("")

        # Justification
        if c.justification:
            st.markdown(f"**📝 Assessment:** {c.justification}")

        # Strengths & gaps
        sg_left, sg_right = st.columns(2)
        with sg_left:
            if c.key_strengths:
                st.markdown("**💪 Strengths**")
                for s in c.key_strengths:
                    st.markdown(f"- {s}")
        with sg_right:
            if c.key_gaps:
                st.markdown("**⚠️ Gaps**")
                for g in c.key_gaps:
                    st.markdown(f"- {g}")

        # Evidence chunks (collapsed)
        if c.retrieved_chunks:
            with st.expander("🔎 View resume evidence used for scoring"):
                for i, chunk in enumerate(c.retrieved_chunks[:5], 1):
                    st.markdown(f"**Excerpt {i}:**")
                    st.text(chunk[:600] + ("..." if len(chunk) > 600 else ""))
                    st.markdown("")


def _score_bar(label: str, score: float):
    """Render a labelled progress bar for a score dimension."""
    color = (
        "normal" if score >= 60
        else "off"
    )
    st.markdown(f"<small>{label}: **{score:.1f}**</small>", unsafe_allow_html=True)
    st.progress(int(score) / 100)


def _build_results_df(candidates: list[CandidateScore]) -> pd.DataFrame:
    """Build a flat DataFrame of results for CSV export."""
    rows = []
    for i, c in enumerate(candidates, 1):
        rows.append({
            "Rank": i,
            "Candidate": c.candidate_name,
            "File": c.file_name,
            "Overall Score": round(c.overall_score, 2),
            "Match Level": c.match_level.value,
            "Skills Score": round(c.skills_score, 2),
            "Experience Score": round(c.experience_score, 2),
            "Domain Score": round(c.domain_score, 2),
            "Confidence": round(c.confidence, 2),
            "Experience Years Found": c.experience_years_found,
            "Matched Skills": ", ".join(c.matched_skills),
            "Missing Skills": ", ".join(c.missing_skills),
            "Preferred Skills Matched": ", ".join(c.matched_preferred_skills),
            "Key Strengths": " | ".join(c.key_strengths),
            "Key Gaps": " | ".join(c.key_gaps),
            "Justification": c.justification,
            "Needs Manual Review": c.needs_manual_review,
        })
    return pd.DataFrame(rows)


# ── Main render loop ──────────────────────────────────────────────────────────

def main():
    render_sidebar()

    # Check if we have pending files to run
    pending_files = st.session_state.get("_pending_files")
    pending_req = st.session_state.get("_pending_requirement")

    if st.session_state.is_running and pending_files and pending_req:
        render_running_screen(pending_files, pending_req)
    elif st.session_state.agent_result is not None:
        render_results_screen(st.session_state.agent_result)
    else:
        render_upload_screen()


if __name__ == "__main__":
    main()