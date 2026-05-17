from __future__ import annotations

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from loguru import logger

from config import settings
from models.score import CandidateScore


def is_smtp_configured() -> bool:
    return bool(settings.smtp_user and settings.smtp_password)


def send_email(to_address: str, subject: str, body: str) -> None:
    if not is_smtp_configured():
        raise ValueError(
            "SMTP credentials not configured. "
            "Add SMTP_USER and SMTP_PASSWORD to your .env file."
        )

    msg = MIMEMultipart()
    msg["From"]    = f"{settings.smtp_from_name} <{settings.smtp_user}>"
    msg["To"]      = to_address
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
        server.ehlo()
        server.starttls()
        server.login(settings.smtp_user, settings.smtp_password)
        server.sendmail(settings.smtp_user, to_address, msg.as_string())

    logger.info(f"Email sent → {to_address} | subject='{subject}'")


def draft_outreach_email(
    candidate: CandidateScore,
    rubric_text: str,
) -> tuple[str, str]:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0.7,
        api_key=settings.openai_api_key,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a professional recruiter writing a concise, warm outreach email "
            "to a candidate who was evaluated for a role. "
            "Be specific — reference their matched skills and overall score. "
            "Keep the body under 150 words. Professional but friendly tone. "
            "Do not add best reagrd or anything as such in the end becuase it will added by while sending."
            "Respond in EXACTLY this format, no extra text:\n"
            "SUBJECT: <subject line>\n"
            "BODY:\n<email body>"
        )),
        ("human", (
            "Candidate name: {name}\n"
            "Overall score: {score}/100 ({match_level})\n"
            "Matched skills: {matched}\n"
            "Missing skills: {missing}\n"
            "Job requirement: {rubric}\n\n"
            "Write the outreach email."
        )),
    ])

    chain = prompt | llm
    response = chain.invoke({
        "name":        candidate.candidate_name,
        "score":       f"{candidate.overall_score:.1f}",
        "match_level": candidate.match_level.value,
        "matched":     ", ".join(candidate.matched_skills[:6]) or "none",
        "missing":     ", ".join(candidate.missing_skills[:4]) or "none",
        "rubric":      rubric_text or "Not specified",
    })

    raw = response.content.strip()

    subject = f"Exciting opportunity for {candidate.candidate_name}"
    body    = raw

    if "SUBJECT:" in raw and "BODY:" in raw:
        lines   = raw.splitlines()
        subj_ln = next((l for l in lines if l.startswith("SUBJECT:")), "")
        subject = subj_ln.replace("SUBJECT:", "").strip()
        body    = raw[raw.find("BODY:") + 5:].strip()

    return subject, body
