"""
Streamlit frontend for Document Chatbot.
Calls FastAPI backend for RAG, dataset exploration, and evaluation features.
"""

import io
import json
import os
import textwrap
import unicodedata
from pathlib import Path

import requests
import streamlit as st
from fpdf import FPDF
from fpdf.enums import XPos, YPos

# FastAPI endpoint
API_URL = os.getenv("API_URL", "http://localhost:8000")


_UNICODE_REPLACEMENTS = str.maketrans({
    "\u2014": "-",   # em dash
    "\u2013": "-",   # en dash
    "\u2018": "'",   # left single quotation mark
    "\u2019": "'",   # right single quotation mark / apostrophe
    "\u201c": '"',   # left double quotation mark
    "\u201d": '"',   # right double quotation mark
    "\u2026": "...", # ellipsis
    "\u00a0": " ",   # non-breaking space
    "\u2022": "-",   # bullet
    "\u00b7": "-",   # middle dot
})


def _ascii_safe(s: str) -> str:
    """Convert text to ASCII-safe output for built-in PDF fonts."""
    normalized = unicodedata.normalize("NFKD", s.translate(_UNICODE_REPLACEMENTS))
    return normalized.encode("ascii", "ignore").decode("ascii")


def _extract_name_and_phone(resume_text: str) -> tuple[str, str]:
    """Extract candidate name and phone number from the top of the resume."""
    import re
    lines = [l.strip() for l in resume_text.splitlines() if l.strip()][:15]

    # Phone: look for common formats
    phone = ""
    phone_pattern = re.compile(r"(\+?1?\s*[\-.]?\s*\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4})")
    for line in lines:
        m = phone_pattern.search(line)
        if m:
            phone = m.group(1).strip()
            break

    # Name: first non-email, non-phone, non-URL line with 2-4 words and title case
    name = ""
    for line in lines[:6]:
        if re.search(r"@|http|linkedin|github|\d{5}|\+?\d[\d\s\-().]{7,}", line, re.IGNORECASE):
            continue
        words = line.split()
        if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w.isalpha()):
            name = line
            break

    return name, phone


def _cover_letter_to_pdf(text: str, resume_text: str = "") -> bytes:
    """Render plain-text cover letter as a clean A4 PDF."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(left=20, top=20, right=20)
    pdf.set_auto_page_break(auto=True, margin=20)

    usable_width = pdf.w - pdf.l_margin - pdf.r_margin

    # Header: candidate name + phone extracted from resume
    candidate_name, phone = _extract_name_and_phone(resume_text) if resume_text else ("", "")

    if candidate_name:
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 9, _ascii_safe(candidate_name), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    if phone:
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 6, _ascii_safe(phone), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(0, 0, 0)

    if candidate_name or phone:
        pdf.ln(4)
        # Thin separator line
        pdf.set_draw_color(200, 200, 200)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
        pdf.ln(6)

    # Body
    pdf.set_font("Helvetica", "", 11)
    for raw_line in text.splitlines():
        if raw_line.strip() == "":
            pdf.ln(5)
        else:
            safe_line = _ascii_safe(raw_line)
            wrapped = textwrap.wrap(safe_line, width=90) or [safe_line]
            for wline in wrapped:
                pdf.cell(usable_width, 6, wline, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    return bytes(pdf.output())


def _skill_gap_analysis_to_pdf(export_data: dict) -> bytes:
    """Render skill gap analysis data as a clean A4 PDF."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(left=20, top=20, right=20)
    pdf.set_auto_page_break(auto=True, margin=20)

    usable_width = pdf.w - pdf.l_margin - pdf.r_margin

    def write_line(text: str, *, bold: bool = False, spacing: int = 6) -> None:
        pdf.set_font("Helvetica", "B" if bold else "", 11)
        safe_text = _ascii_safe(text)
        wrapped_lines = textwrap.wrap(safe_text, width=90) or [safe_text]
        for line in wrapped_lines:
            pdf.cell(usable_width, spacing, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def write_section(title: str, items: list[str]) -> None:
        pdf.ln(2)
        write_line(title, bold=True)
        if not items:
            write_line("None")
            return
        for item in items:
            write_line(f"- {item}")

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Skill Gap Analysis Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    write_line(f"Resume source: {export_data.get('resume_source', 'Unknown')}")
    write_line(f"Match rate: {export_data.get('match_percentage', 0)}%")
    write_line(f"Your skills detected: {len(export_data.get('resume_skills', []))}")
    write_line(f"Job requirements detected: {len(export_data.get('job_skills', []))}")
    write_line(f"Matched skills: {len(export_data.get('matched_skills', []))}")
    write_line(f"Missing skills: {len(export_data.get('missing_skills', []))}")

    write_section("Matched skills", export_data.get("matched_skills", []))
    write_section("Missing skills", export_data.get("missing_skills", []))
    write_section("Recommendations", export_data.get("recommendations", []))

    return bytes(pdf.output())

st.set_page_config(
    page_title="Document chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top, #162238 0%, #0b1020 45%, #060816 100%);
        color: #f7f9fc;
    }
    .block-container {
        padding-top: 4rem;
        padding-bottom: 2rem;
    }
    .hero {
        padding: 0.8rem 1.4rem 0.9rem 1.4rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(18, 30, 56, 0.95), rgba(16, 16, 35, 0.88));
        box-shadow: 0 20px 60px rgba(0,0,0,0.25);
    }
    .card {
        padding: 1rem 1rem;
        border-radius: 18px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
    }
    .muted {
        color: rgba(255,255,255,0.7);
        font-size: 0.95rem;
    }
    .section-title {
        margin-top: 0.35rem;
        margin-bottom: 0.25rem;
        font-size: 1.3rem;
        font-weight: 700;
    }
    .small-label {
        display: inline-block;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-size: 0.7rem;
        font-weight: 600;
        color: #a78bfa;
        background: rgba(139,92,246,0.12);
        border: 1px solid rgba(139,92,246,0.38);
        border-radius: 999px;
        padding: 0.18rem 0.65rem;
        margin-bottom: 0.55rem;
    }
    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: rgba(8, 12, 28, 0.98) !important;
        border-right: 1px solid rgba(255,255,255,0.05) !important;
    }
    section[data-testid="stSidebar"] .stRadio > div {
        gap: 2px !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        padding: 0.42rem 0.65rem !important;
        border-radius: 9px !important;
        cursor: pointer !important;
        color: rgba(255,255,255,0.68) !important;
        font-size: 0.93rem !important;
        transition: background 0.15s, color 0.15s !important;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(139,92,246,0.10) !important;
        color: #ddd6fe !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.07) !important;
        margin: 0.4rem 0 !important;
    }
    /* Logout button — CSS-pinned to sidebar base */
    section[data-testid="stSidebar"] .stButton button {
        position: fixed !important;
        bottom: 1.25rem !important;
        left: 1.25rem !important;
        width: calc(21rem - 2.5rem) !important;
        background: linear-gradient(135deg, #e11d48, #9d174d) !important;
        border: 1px solid rgba(225,29,72,0.3) !important;
        color: #fff !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        letter-spacing: 0.03em !important;
        padding: 0.55rem 0 !important;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        background: linear-gradient(135deg, #f43f5e, #be185d) !important;
        box-shadow: 0 4px 16px rgba(225,29,72,0.4) !important;
        border-color: rgba(244,63,94,0.5) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "authenticated_user" not in st.session_state:
    st.session_state.authenticated_user = None
if "resume_text" not in st.session_state:
    st.session_state.resume_text = None
if "resume_source" not in st.session_state:
    st.session_state.resume_source = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "generated_cover_letter" not in st.session_state:
    st.session_state.generated_cover_letter = None
if "generated_cover_letter_meta" not in st.session_state:
    st.session_state.generated_cover_letter_meta = {}

# ============================================================================
# Login Page
# ============================================================================

if not st.session_state.authenticated:
    _cred_err = st.session_state.get("_login_cred_err", False)

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ═══════════════════════════════════════════════
       1. HIDE ALL STREAMLIT CHROME
    ═══════════════════════════════════════════════ */
    #MainMenu, footer, header            { display: none !important; }
    .stDeployButton                      { display: none !important; }
    div[data-testid="stToolbar"]         { display: none !important; }
    div[data-testid="stDecoration"]      { display: none !important; }
    div[data-testid="InputInstructions"] { display: none !important; }
    p[data-testid="InputInstructions"]   { display: none !important; }
    /* Hide browser autocomplete dropdown */
    input:-webkit-autofill,
    input:-webkit-autofill:hover,
    input:-webkit-autofill:focus {
        -webkit-box-shadow: 0 0 0px 1000px #0d0d18 inset !important;
        -webkit-text-fill-color: #ffffff !important;
        caret-color: #818cf8 !important;
    }
    /* Anchor links injected by Streamlit on headings */
    .login-heading a { display: none !important; }

    /* ═══════════════════════════════════════════════
       2. GLOBAL FONT + BOX SIZING
    ═══════════════════════════════════════════════ */
    *, *::before, *::after {
        box-sizing: border-box;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* ═══════════════════════════════════════════════
       3. CLEAN DARK BACKGROUND — no grid, no noise
    ═══════════════════════════════════════════════ */
    html, body,
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"],
    .main > div {
        background: #0a0a0f !important;
        background-image: none !important;
        background-color: #0a0a0f !important;
    }

    /* ═══════════════════════════════════════════════
       4. CENTER LAYOUT
    ═══════════════════════════════════════════════ */
    .block-container {
        max-width: 420px !important;
        margin: 0 auto !important;
        padding: 6vh 1rem 3rem !important;
    }

    /* ═══════════════════════════════════════════════
       5. ANIMATIONS
    ═══════════════════════════════════════════════ */
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(16px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes cardGlow {
        0%,100% { box-shadow: 0 0 30px rgba(99,102,241,0.15),
                              0 0 60px rgba(99,102,241,0.08),
                              0 24px 60px rgba(0,0,0,0.6); }
        50%     { box-shadow: 0 0 45px rgba(99,102,241,0.28),
                              0 0 90px rgba(99,102,241,0.14),
                              0 0 120px rgba(79,70,229,0.08),
                              0 24px 60px rgba(0,0,0,0.6); }
    }

    /* ═══════════════════════════════════════════════
       6. BRAND ICON
    ═══════════════════════════════════════════════ */
    .doc-icon-wrap {
        width: 72px; height: 72px;
        background: rgba(99,102,241,0.10);
        border: 1px solid rgba(99,102,241,0.18);
        border-radius: 20px;
        display: flex; align-items: center; justify-content: center;
        margin: 0 auto 1.2rem;
        box-shadow: 0 0 36px rgba(99,102,241,0.16), 0 8px 24px rgba(0,0,0,0.5);
        animation: fadeUp 0.5s ease both;
    }

    /* ═══════════════════════════════════════════════
       7. "DOCUMENT CHATBOT" PILL — no hard border
    ═══════════════════════════════════════════════ */
    .login-chip {
        display: inline-block;
        background: rgba(99,102,241,0.12);
        color: #c4b5fd;
        font-size: 12px; font-weight: 700;
        letter-spacing: 4px; text-transform: uppercase;
        border-radius: 999px; padding: 5px 18px;
        margin-bottom: 14px;
        border: 1px solid rgba(255,255,255,0.05);
        animation: fadeUp 0.5s 0.06s ease both;
    }

    /* ═══════════════════════════════════════════════
       8. HEADING + SUBTITLE
    ═══════════════════════════════════════════════ */
    .login-heading {
        font-size: 36px; font-weight: 700; color: #ffffff;
        letter-spacing: -0.03em; margin: 0 0 6px; line-height: 1.15;
        animation: fadeUp 0.5s 0.10s ease both;
    }
    .login-sub {
        font-size: 13.5px; color: #64748b !important; line-height: 1.55;
        margin: 0 auto; max-width: 100%;
        word-break: break-word;
        opacity: 0.75;
        animation: fadeUp 0.5s 0.14s ease both;
    }

    /* ═══════════════════════════════════════════════
       9. SINGLE GLASS CARD — wraps everything
    ═══════════════════════════════════════════════ */
    div[data-testid="stForm"] {
        background: rgba(255,255,255,0.02) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 20px !important;
        padding: 36px 32px 32px !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        animation: cardGlow 5s ease-in-out infinite,
                   fadeUp 0.5s 0.18s ease both !important;
    }

    /* ═══════════════════════════════════════════════
       10. CUSTOM FIELD LABELS
    ═══════════════════════════════════════════════ */
    .field-label {
        color: #94a3b8;
        font-size: 11px; font-weight: 500;
        letter-spacing: 2px; text-transform: uppercase;
        margin-bottom: 8px; display: block;
    }
    div[data-testid="stTextInput"] label { display: none !important; }

    /* ═══════════════════════════════════════════════
       11. INPUT FIELDS — dark background, purple focus
       Strategy: target the BaseWeb containers directly.
       The card bg is ~#0c0c13; inputs use rgba(255,255,255,0.05)
       so they're barely distinguishable — just a faint edge.
    ═══════════════════════════════════════════════ */

    /* Strip BaseWeb's outer shell */
    div[data-baseweb="input"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* The actual visible box — carved into the dark surface */
    div[data-testid="stTextInput"] div[data-baseweb="base-input"] {
        background-color: rgba(255,255,255,0.02) !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
        border-radius: 12px !important;
        box-shadow: none !important;
        transition: border-color 0.2s, box-shadow 0.2s, background-color 0.2s !important;
    }

    /* PURPLE focus — on the wrapper, not the raw input (fix 2) */
    div[data-testid="stTextInput"] div[data-baseweb="base-input"]:focus-within {
        background-color: rgba(99,102,241,0.06) !important;
        border-color: rgba(99,102,241,0.5) !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.10),
                    0 0 15px rgba(99,102,241,0.12) !important;
    }

    /* Raw <input> — transparent, white text, no outline of its own */
    div[data-testid="stTextInput"] input {
        background: transparent !important;
        background-color: transparent !important;
        color: #ffffff !important;
        padding: 14px 16px !important;
        font-size: 15px !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        caret-color: #818cf8 !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    div[data-testid="stTextInput"] input::placeholder {
        color: #3a3f52 !important;
        -webkit-text-fill-color: #3a3f52 !important;
        opacity: 1 !important;
    }
    /* Nuke every possible native focus ring — yellow, blue, red, all */
    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stTextInput"] input:focus-visible,
    div[data-testid="stTextInput"] input:focus-within {
        outline: none !important;
        outline-offset: 0 !important;
        box-shadow: none !important;
        border: none !important;
        background: transparent !important;
    }

    /* ═══════════════════════════════════════════════
       12. EYE ICON — subtle, no background
    ═══════════════════════════════════════════════ */
    div[data-baseweb="input-enhancer"] {
        background: transparent !important;
        border: none !important;
        padding-right: 4px !important;
    }
    div[data-baseweb="input-enhancer"] button {
        background: transparent !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        padding: 0 8px !important;
        cursor: pointer !important;
        color: #3a3f52 !important;
    }
    div[data-baseweb="input-enhancer"] button svg {
        fill: #3a3f52 !important;
        color: #3a3f52 !important;
        width: 16px !important;
        height: 16px !important;
    }
    div[data-baseweb="input-enhancer"] button:hover svg {
        fill: #64748b !important;
        color: #64748b !important;
    }

    /* ═══════════════════════════════════════════════
       13. SIGN IN BUTTON — gradient, no border
       NOTE: Do NOT use "div[data-testid="stForm"] button"
       here — that's too broad and would also catch the
       now-hidden eye-icon button if it reappears.
    ═══════════════════════════════════════════════ */
    button[kind="primaryFormSubmit"],
    button[data-testid="baseButton-primaryFormSubmit"],
    div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 50%, #3b82f6 100%) !important;
        background-image: linear-gradient(135deg, #6366f1 0%, #4f46e5 50%, #3b82f6 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-color: transparent !important;
        outline: none !important;
        outline-offset: 0 !important;
        -webkit-appearance: none !important;
        border-radius: 12px !important;
        padding: 14px 24px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        width: 100% !important;
        margin-top: 10px !important;
        box-shadow: 0 4px 20px rgba(99,102,241,0.35) !important;
        transition: all 0.25s ease !important;
        cursor: pointer !important;
    }
    button[kind="primaryFormSubmit"]:hover,
    button[data-testid="baseButton-primaryFormSubmit"]:hover,
    div[data-testid="stFormSubmitButton"] button:hover {
        filter: brightness(1.15) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 28px rgba(99,102,241,0.5) !important;
        border: none !important;
        outline: none !important;
    }
    button[kind="primaryFormSubmit"]:focus,
    button[kind="primaryFormSubmit"]:focus-visible,
    button[data-testid="baseButton-primaryFormSubmit"]:focus,
    button[data-testid="baseButton-primaryFormSubmit"]:focus-visible,
    div[data-testid="stFormSubmitButton"] button:focus,
    div[data-testid="stFormSubmitButton"] button:focus-visible {
        outline: none !important;
        box-shadow: 0 4px 20px rgba(99,102,241,0.35) !important;
        border: none !important;
    }

    /* ═══════════════════════════════════════════════
       14. WRONG-CREDENTIALS ERROR
    ═══════════════════════════════════════════════ */
    .cred-err {
        display: flex; align-items: center; gap: 8px;
        background: rgba(239,68,68,0.07);
        border: 1px solid rgba(239,68,68,0.16);
        border-radius: 10px; padding: 12px 14px;
        color: #f87171; font-size: 13px; margin-top: 10px;
    }

    /* ═══════════════════════════════════════════════
       15. DEMO BOX + BADGE (fix 6)
    ═══════════════════════════════════════════════ */
    .demo-box {
        display: flex; align-items: center; gap: 10px;
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 10px; padding: 12px 14px;
        margin-top: 14px; font-size: 13px; color: #64748b;
        animation: fadeUp 0.5s 0.32s ease both;
    }
    .demo-badge {
        background: #4c1d95;
        color: #c4b5fd;
        border-radius: 6px; padding: 3px 10px;
        font-size: 11px; font-weight: 700;
        letter-spacing: 1px; text-transform: uppercase;
        flex-shrink: 0; border: none;
    }
    .demo-cred { color: #ffffff; font-weight: 700; }

    /* ═══════════════════════════════════════════════
       16. FEATURE CHIPS
    ═══════════════════════════════════════════════ */
    .chips-row {
        display: flex; justify-content: center; gap: 8px;
        margin-top: 18px; flex-wrap: wrap;
        animation: fadeUp 0.5s 0.38s ease both;
    }
    .chip {
        display: inline-flex; align-items: center; gap: 5px;
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 999px; padding: 5px 14px;
        font-size: 11px; color: #475569; white-space: nowrap;
    }
    </style>

    """, unsafe_allow_html=True)

    # ── SINGLE GLASS CARD: header + form ──
    with st.form("login_form", clear_on_submit=False):
        # Header inside the card
        st.markdown("""
        <div style="text-align:center; padding-bottom:28px; border-bottom:1px solid rgba(255,255,255,0.05); margin-bottom:24px;">
          <div class="doc-icon-wrap">
            <svg width="38" height="38" viewBox="0 0 38 38" fill="none" xmlns="http://www.w3.org/2000/svg">
              <rect x="5" y="2" width="20" height="26" rx="3" fill="url(#lg1)" opacity="0.9"/>
              <rect x="9"  y="8"  width="12" height="2" rx="1" fill="#e0e7ff" opacity="0.6"/>
              <rect x="9"  y="13" width="12" height="2" rx="1" fill="#e0e7ff" opacity="0.6"/>
              <rect x="9"  y="18" width="8"  height="2" rx="1" fill="#e0e7ff" opacity="0.6"/>
              <circle cx="27" cy="27" r="9"   fill="#0a0a0f"/>
              <circle cx="27" cy="27" r="7.5" fill="url(#lg2)"/>
              <circle cx="27" cy="27" r="7.5" fill="none" stroke="rgba(96,165,250,0.3)" stroke-width="1"/>
              <path d="M23.5 27.5 L26.3 30.3 L30.8 24.8"
                    stroke="white" stroke-width="1.8"
                    stroke-linecap="round" stroke-linejoin="round"/>
              <defs>
                <linearGradient id="lg1" x1="5" y1="2" x2="25" y2="28" gradientUnits="userSpaceOnUse">
                  <stop stop-color="#818cf8"/>
                  <stop offset="1" stop-color="#6366f1"/>
                </linearGradient>
                <linearGradient id="lg2" x1="20" y1="20" x2="34" y2="34" gradientUnits="userSpaceOnUse">
                  <stop stop-color="#60a5fa"/>
                  <stop offset="1" stop-color="#3b82f6"/>
                </linearGradient>
              </defs>
            </svg>
          </div>
          <div class="login-chip">Document Chatbot</div>
          <div class="login-heading">Welcome Back</div>
          <p class="login-sub">Sign in to chat with your documents using AI-powered analysis</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<span class="field-label">Username</span>', unsafe_allow_html=True)
        username = st.text_input(
            "doc_chatbot_user_v1",
            placeholder="Enter your username",
            label_visibility="collapsed",
            key="login_username",
        )

        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

        st.markdown('<span class="field-label">Password</span>', unsafe_allow_html=True)
        password = st.text_input(
            "doc_chatbot_pass_v1",
            type="password",
            placeholder="Enter your password",
            label_visibility="collapsed",
            key="login_password",
        )

        if _cred_err:
            st.markdown(
                '<div class="cred-err">&#10007;&nbsp; Invalid credentials — please try again</div>',
                unsafe_allow_html=True,
            )

        submitted = st.form_submit_button("Sign In →", use_container_width=True)

    # ── AUTH LOGIC ──
    if submitted:
        if username.strip() == "apurva" and password == "resume123":
            st.session_state.authenticated      = True
            st.session_state.authenticated_user = "Apurva"
            st.session_state._login_cred_err    = False
            st.rerun()
        else:
            st.session_state._login_cred_err = True
            st.rerun()

    # ── DEMO HINT + FEATURE CHIPS (below card) ──
    st.markdown("""
    <div class="demo-box">
      <span class="demo-badge">Demo</span>
      <span>Use <span class="demo-cred">apurva</span> / <span class="demo-cred">resume123</span> to sign in</span>
    </div>
    <div class="chips-row">
      <span class="chip">Upload &amp; Chat</span>
      <span class="chip">RAG Pipeline</span>
      <span class="chip">Evaluation</span>
    </div>
    """, unsafe_allow_html=True)

    st.stop()

# ============================================================================
# Main App (After Login)
# ============================================================================

# Sidebar
with st.sidebar:
    # DocBot branding
    st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;padding:0.4rem 0 1.2rem 0;">
            <div style="background:linear-gradient(135deg,#7c3aed,#4f46e5);border-radius:10px;
                        width:38px;height:38px;display:flex;align-items:center;justify-content:center;
                        box-shadow:0 4px 12px rgba(124,58,237,0.4);flex-shrink:0;">
                <span style="font-size:1.25rem;line-height:1;">📄</span>
            </div>
            <span style="font-size:1.4rem;font-weight:800;
                         background:linear-gradient(135deg,#a78bfa,#818cf8);
                         -webkit-background-clip:text;-webkit-text-fill-color:transparent;">DocBot</span>
        </div>
    """, unsafe_allow_html=True)

    # User card
    st.markdown(f"""
        <div style="background:rgba(139,92,246,0.08);border:1px solid rgba(139,92,246,0.18);
                    border-radius:12px;padding:0.6rem 0.85rem;display:flex;align-items:center;
                    gap:10px;margin-bottom:0.5rem;">
            <div style="background:rgba(139,92,246,0.18);border-radius:50%;width:32px;height:32px;
                        display:flex;align-items:center;justify-content:center;
                        font-size:0.95rem;flex-shrink:0;">👤</div>
            <span style="font-weight:600;color:#ddd6fe;font-size:0.94rem;">
                {st.session_state.authenticated_user}
            </span>
        </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Navigation
    page = st.radio(
        "Navigation",
        ["Upload & Chat", "Skill Gap Analyzer", "Cover Letter Generator",
         "Dataset Explorer", "Evaluation Results", "Artifacts"],
        label_visibility="collapsed",
    )

    # Logout — CSS-pinned to sidebar bottom
    if st.button("→  Logout", use_container_width=True, key="sidebar_logout"):
        st.session_state.authenticated = False
        st.session_state.authenticated_user = None
        st.session_state.resume_text = None
        st.session_state.resume_source = None
        st.session_state.chat_messages = []
        st.session_state.generated_cover_letter = None
        st.session_state.generated_cover_letter_meta = {}
        st.rerun()

# ============================================================================
# Page: Upload & Chat (Main Page)
# ============================================================================

if page == "Upload & Chat":
    st.markdown(
        """
        <div class="hero">
            <div class="small-label">Resume Chatbot</div>
            <h1 style="margin:0; font-size:2.2rem;">Chat with your resume or document</h1>
            <p class="muted" style="margin-top:0.5rem; max-width: 920px;">
                Upload a PDF, TXT, or Markdown file. The backend will extract and index the content.
                Then ask questions and get grounded answers from your document.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📄 Step 1: Upload Your Resume")
    
    uploaded_file = st.file_uploader(
        "Upload a resume (PDF, TXT, or MD)",
        type=["pdf", "txt", "md"],
        key="resume_uploader",
    )

    if uploaded_file:
        st.info(f"📌 Uploaded: **{uploaded_file.name}**")
        
        # Parse the uploaded file via API
        with st.spinner("Parsing your resume..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                response = requests.post(f"{API_URL}/upload-resume", files=files, timeout=10)
                response.raise_for_status()
                result = response.json()
                
                if result.get("error"):
                    st.error(f"❌ {result['error']}")
                else:
                    st.session_state.resume_text = result["text"]
                    st.session_state.resume_source = f"Uploaded: {result['filename']}"
                    st.success("✅ Resume loaded successfully!")
            except Exception as exc:
                st.error(f"❌ Failed to upload: {exc}")

    # Show loaded resume info
    if st.session_state.resume_text:
        st.divider()
        with st.expander("📋 View loaded resume", expanded=False):
            st.code(st.session_state.resume_text[:3000])
        
        st.markdown("### 💬 Step 2: Ask Questions")
        st.info("Your resume is ready. Ask questions about education, skills, experience, projects, etc.")

        # Chat interface
        if "chat_messages" not in st.session_state or not st.session_state.chat_messages:
            st.session_state.chat_messages = [
                {
                    "role": "assistant",
                    "content": "I've loaded your resume. Ask me anything about your education, skills, experience, or projects!",
                }
            ]

        # Display chat history
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Chat input
        user_question = st.chat_input("Ask about your resume...")
        if user_question:
            st.session_state.chat_messages.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.write(user_question)

            # Call FastAPI backend
            with st.spinner("Searching your resume..."):
                try:
                    payload = {
                        "question": user_question,
                    }
                    response = requests.post(f"{API_URL}/answer", json=payload, timeout=30)
                    response.raise_for_status()
                    result = response.json()
                    
                    st.session_state.chat_messages.append({"role": "assistant", "content": result["answer"]})
                    with st.chat_message("assistant"):
                        st.write(result["answer"])
                    
                    # Only show retrieved chunks if answer was found
                    if result.get("show_chunks", False) and result.get("retrieved_chunks"):
                        with st.expander("📑 Retrieved resume sections"):
                            for idx, chunk in enumerate(result["retrieved_chunks"], 1):
                                st.markdown(f"**Section {idx}:**")
                                st.write(chunk)
                                st.divider()
                except Exception as exc:
                    st.error(f"❌ Failed to get answer: {exc}")
    else:
        st.warning("⬆️ Please upload a resume to start chatting.")

# ============================================================================
# Page: Skill Gap Analyzer
# ============================================================================

elif page == "Skill Gap Analyzer":
    st.markdown(
        """
        <div class="hero">
            <div class="small-label">Career Development</div>
            <h1 style="margin:0; font-size:2.2rem;">🚀 Skill Gap Analyzer</h1>
            <p class="muted" style="margin-top:0.5rem; max-width: 920px;">
                Compare your uploaded resume with job requirements. Identify missing skills and get 
                personalized recommendations to strengthen your qualifications.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Check if resume is uploaded
    if not st.session_state.resume_text:
        st.warning("📄 **No resume uploaded yet!**")
        st.info(
            "Please go to the **Upload & Chat** section and upload your resume first. "
            "Then come back here to analyze skill gaps."
        )
    else:
        st.success(f"✅ Resume loaded: **{st.session_state.resume_source}**")
        with st.expander("📋 View loaded resume", expanded=False):
            st.code(st.session_state.resume_text[:2000] + "..." if len(st.session_state.resume_text) > 2000 else st.session_state.resume_text)

        st.divider()

        st.markdown("### 💼 Job Description")
        st.caption("Paste the job description you want to compare against")
        job_description = st.text_area(
            "Job Description",
            height=400,
            placeholder="Paste the job description here...",
            key="gap_job_desc",
            label_visibility="collapsed",
        )

        st.divider()

        analyze_col1, analyze_col2, analyze_col3 = st.columns([1, 1, 2])
        with analyze_col2:
            run_analysis = st.button("🔍 Analyze Skills", type="primary", use_container_width=True)

        if run_analysis:
            if not job_description.strip():
                st.error("❌ Please paste a job description to analyze")
            else:
                with st.spinner("Analyzing skills..."):
                    try:
                        payload = {
                            "resume_text": st.session_state.resume_text,
                            "job_description": job_description,
                        }
                        response = requests.post(
                            f"{API_URL}/skill-gap-analyzer",
                            json=payload,
                            timeout=30,
                        )
                        response.raise_for_status()
                        result = response.json()

                        # Display results
                        st.markdown("---")
                        st.markdown("## 📊 Analysis Results")

                        # Metrics
                        metric_cols = st.columns(4)
                        with metric_cols[0]:
                            st.metric("Your Skills", len(result["resume_skills"]))
                        with metric_cols[1]:
                            st.metric("Job Requirements", len(result["job_skills"]))
                        with metric_cols[2]:
                            st.metric("Matched Skills", len(result["matched_skills"]))
                        with metric_cols[3]:
                            match_percentage = round(
                                (len(result["matched_skills"]) / max(len(result["job_skills"]), 1)) * 100, 1
                            )
                            st.metric("Match Rate", f"{match_percentage}%")

                        st.divider()

                        # Low match score warning
                        if match_percentage < 50:
                            st.warning(
                                "⚠️ **Low Match Score Detected**\n\n"
                                "Your resume does not align well with this job description. "
                                "Consider the following:\n\n"
                                "- **Update your resume** to highlight relevant skills, "
                                "experience, or projects that match the job requirements.\n"
                                "- **Review the job description** to confirm it matches the "
                                "role you are targeting — it may contain requirements outside "
                                "your current career track."
                            )

                        # Tabs for detailed breakdown
                        tab_matched, tab_missing, tab_all_skills = st.tabs(
                            ["✅ Matched Skills", "❌ Missing Skills", "All Skills"]
                        )

                        with tab_matched:
                            if result["matched_skills"]:
                                st.markdown("### Skills You Already Have:")
                                # Display as badges
                                matched_text = " ".join([f"🟢 `{skill}`" for skill in result["matched_skills"][:20]])
                                st.markdown(matched_text)
                                if len(result["matched_skills"]) > 20:
                                    st.caption(f"... and {len(result['matched_skills']) - 20} more")
                            else:
                                st.info("No matching skills found. This is an opportunity to learn new technologies!")

                        with tab_missing:
                            if result["missing_skills"]:
                                st.markdown("### Skills You Need to Develop:")
                                col1, col2 = st.columns([2, 1])
                                with col1:
                                    missing_text = " ".join([f"🔴 `{skill}`" for skill in result["missing_skills"][:15]])
                                    st.markdown(missing_text)
                                    if len(result["missing_skills"]) > 15:
                                        st.caption(f"... and {len(result['missing_skills']) - 15} more skills")
                                with col2:
                                    st.markdown(f"**Total Missing:** {len(result['missing_skills'])}")
                            else:
                                st.success("🎉 Perfect match! You have all required skills!")

                        with tab_all_skills:
                            c1, c2 = st.columns(2)
                            with c1:
                                st.markdown("**Your Skills:**")
                                st.code(", ".join(result["resume_skills"]) or "No skills detected")
                            with c2:
                                st.markdown("**Job Requirements:**")
                                st.code(", ".join(result["job_skills"]) or "No skills detected")

                        # Recommendations
                        st.divider()
                        st.markdown("## 💡 Personalized Recommendations")
                        for idx, rec in enumerate(result["recommendations"], 1):
                            st.markdown(f"**{idx}. {rec}**")

                        # Export results
                        st.divider()
                        st.markdown("### 📥 Export Analysis")
                        export_data = {
                            "resume_source": st.session_state.resume_source,
                            "resume_skills": result["resume_skills"],
                            "job_skills": result["job_skills"],
                            "matched_skills": result["matched_skills"],
                            "missing_skills": result["missing_skills"],
                            "recommendations": result["recommendations"],
                            "match_percentage": match_percentage,
                        }
                        export_pdf = _skill_gap_analysis_to_pdf(export_data)
                        st.download_button(
                            "📥 Download Analysis (.pdf)",
                            export_pdf,
                            "skill_gap_analysis.pdf",
                            "application/pdf",
                        )

                    except Exception as exc:
                        st.error(f"❌ Error analyzing skills: {exc}")

# ============================================================================
# Page: Cover Letter Generator
# ============================================================================

elif page == "Cover Letter Generator":
    st.markdown(
        """
        <div class="hero">
            <div class="small-label">Job Applications</div>
            <h1 style="margin:0; font-size:2.2rem;">Cover Letter Generator</h1>
            <p class="muted" style="margin-top:0.5rem; max-width: 920px;">
                Generate a tailored cover letter from your uploaded resume and a target job description.
                The draft stays grounded in your resume instead of inventing experience.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.resume_text:
        st.warning("📄 Upload a resume first in Upload & Chat before generating a cover letter.")
    else:
        st.success(f"✅ Resume loaded: **{st.session_state.resume_source}**")

        with st.expander("📋 View loaded resume", expanded=False):
            preview_text = st.session_state.resume_text[:2500]
            if len(st.session_state.resume_text) > 2500:
                preview_text += "..."
            st.code(preview_text)

        st.divider()

        meta_col1, meta_col2, meta_col3 = st.columns([1.1, 1.1, 0.8])
        with meta_col1:
            company_name = st.text_input("Company name", placeholder="Optional")
        with meta_col2:
            role_title = st.text_input("Role title", placeholder="Optional")
        with meta_col3:
            tone = st.selectbox("Tone", ["professional", "confident", "concise", "enthusiastic"], index=0)

        job_description = st.text_area(
            "Job Description",
            height=320,
            placeholder="Paste the job description here...",
        )

        generate_letter = st.button("Generate Cover Letter", type="primary", use_container_width=True)

        if generate_letter:
            if not job_description.strip():
                st.error("❌ Please paste a job description.")
            else:
                with st.spinner("Generating cover letter..."):
                    try:
                        payload = {
                            "resume_text": st.session_state.resume_text,
                            "job_description": job_description,
                            "tone": tone,
                        }
                        response = requests.post(f"{API_URL}/generate-cover-letter", json=payload, timeout=60)
                        response.raise_for_status()
                        result = response.json()
                        st.session_state.generated_cover_letter = result["cover_letter"]
                        st.session_state.generated_cover_letter_meta = {
                            "company_name": result["company_name"],
                            "role_title": result["role_title"],
                            "source": result["source"],
                            "tone": result["tone"],
                            "coverage_score": result.get("coverage_score", 0),
                            "job_skills": result.get("job_skills", []),
                            "matched_job_skills": result.get("matched_job_skills", []),
                            "uncovered_job_skills": result.get("uncovered_job_skills", []),
                            "covered_responsibilities": result.get("covered_responsibilities", []),
                            "uncovered_responsibilities": result.get("uncovered_responsibilities", []),
                        }
                    except Exception as exc:
                        st.error(f"❌ Failed to generate cover letter: {exc}")

        if st.session_state.generated_cover_letter:
            letter_meta = st.session_state.generated_cover_letter_meta
            safe_company = str(letter_meta.get("company_name", "company")).strip().lower().replace(" ", "_")
            safe_role = str(letter_meta.get("role_title", "role")).strip().lower().replace(" ", "_")

            st.divider()
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            with summary_col1:
                st.metric("Company", letter_meta.get("company_name", "-"))
            with summary_col2:
                st.metric("Role", letter_meta.get("role_title", "-"))
            with summary_col3:
                st.metric("Coverage", f"{letter_meta.get('coverage_score', 0)}%")

            # Resume-JD alignment warning
            job_skills_list = letter_meta.get("job_skills", [])
            matched_list = letter_meta.get("matched_job_skills", [])
            if job_skills_list:
                resume_match_rate = len(matched_list) / len(job_skills_list) * 100
                if resume_match_rate < 50:
                    st.warning(
                        "⚠️ **Your resume does not align well with this job description.**\n\n"
                        "The cover letter has been generated, but your resume covers fewer than "
                        f"**{int(resume_match_rate)}%** of the required skills in this JD. "
                        "You may want to:\n\n"
                        "- **Update your resume** to highlight skills and experience that match "
                        "the job requirements.\n"
                        "- **Double-check the job description** — it may not match the role you "
                        "are targeting."
                    )

            if job_skills_list:
                st.markdown("### Job Alignment Summary")
                align_tab1, align_tab2, align_tab3 = st.tabs([
                    "Supported Skills",
                    "Still Uncovered",
                    "Responsibilities",
                ])

                with align_tab1:
                    supported = letter_meta.get("matched_job_skills", [])
                    if supported:
                        st.write(", ".join(supported))
                    else:
                        st.caption("No JD skills were identified as directly supported by the uploaded resume.")

                with align_tab2:
                    uncovered = letter_meta.get("uncovered_job_skills", [])
                    if uncovered:
                        st.warning("These JD skills were not explicitly covered in the final letter:")
                        st.write(", ".join(uncovered))
                    else:
                        st.success("All extracted JD skills were explicitly covered in the final letter.")

                with align_tab3:
                    covered_resp = letter_meta.get("covered_responsibilities", [])
                    uncovered_resp = letter_meta.get("uncovered_responsibilities", [])
                    if covered_resp:
                        st.markdown("**Covered:**")
                        for item in covered_resp:
                            st.write(f"- {item}")
                    if uncovered_resp:
                        st.markdown("**Still not explicit enough:**")
                        for item in uncovered_resp:
                            st.write(f"- {item}")

            st.markdown("### Generated Cover Letter")
            st.text_area(
                "Generated Cover Letter",
                value=st.session_state.generated_cover_letter,
                height=420,
                key="generated_cover_letter_preview",
            )

            download_name = f"cover_letter_{safe_company}.pdf"
            pdf_bytes = _cover_letter_to_pdf(
                st.session_state.generated_cover_letter,
                st.session_state.resume_text or "",
            )

            action_col1, action_col2 = st.columns([1, 1.4])
            with action_col1:
                st.download_button(
                    "Download Cover Letter (.pdf)",
                    pdf_bytes,
                    download_name,
                    "application/pdf",
                    use_container_width=True,
                )
            with action_col2:
                st.caption(f"Saved filename: {download_name}")

# ============================================================================
# Page: Dataset Explorer
# ============================================================================

elif page == "Dataset Explorer":
    st.markdown("## 📊 Dataset Explorer")
    st.caption("Browse the 45-item evaluation dataset (30 typical + 10 edge + 5 adversarial)")

    with st.spinner("Loading dataset..."):
        try:
            response = requests.get(f"{API_URL}/dataset", timeout=10)
            response.raise_for_status()
            dataset = response.json()
            
            if not dataset.get("items"):
                st.warning("No dataset found.")
            else:
                items = dataset.get("items", [])
                st.metric("Total Items", len(items))
                
                # Filter by category
                categories = list(set(item.get("category", "unknown") for item in items))
                selected_category = st.selectbox("Filter by category", ["All"] + categories)
                
                filtered_items = items
                if selected_category != "All":
                    filtered_items = [item for item in items if item.get("category") == selected_category]
                
                st.markdown(f"### Showing {len(filtered_items)} items")
                for idx, item in enumerate(filtered_items, 1):
                    with st.expander(f"{idx}. **Q:** {item.get('query', '')[:80]}..."):
                        st.markdown(f"**Question:** {item.get('query', '')}")
                        st.markdown(f"**Ground Truth:** {item.get('ground_truth', '')}")
                        st.markdown(f"**Category:** `{item.get('category', 'unknown')}`")
        except Exception as exc:
            st.error(f"❌ Failed to load dataset: {exc}")

# ============================================================================
# Page: Evaluation Results
# ============================================================================

elif page == "Evaluation Results":
    st.markdown("## 📈 Evaluation Results")

    tab_live, tab_step3, tab_step4 = st.tabs(["Live on Uploaded Resume", "Step 3: RAG Pipeline", "Step 4: Meta-Prompting"])

    with tab_live:
        st.markdown("### Live Evaluation on Uploaded Resume")
        st.caption("Runs RAG + evaluation using the resume you uploaded in Upload & Chat.")

        col1, col2, col3 = st.columns([1, 1, 1.2])
        with col1:
            selected_split = st.selectbox("Dataset split", ["test", "dev", "train"], index=0)
        with col2:
            max_items = st.number_input("Max items (0 = all)", min_value=0, max_value=100, value=8, step=1)
        with col3:
            run_live_eval = st.button("Run evaluation on uploaded resume", type="primary")

        if run_live_eval:
            payload = {"split": selected_split}
            if max_items > 0:
                payload["max_items"] = int(max_items)

            with st.spinner("Running live RAG evaluation..."):
                try:
                    response = requests.post(f"{API_URL}/evaluate-uploaded", json=payload, timeout=120)
                    response.raise_for_status()
                    live = response.json()

                    if live.get("error"):
                        st.error(f"❌ {live['error']}")
                    else:
                        summary = live.get("summary", {})
                        d1, d2, d3, d4 = st.columns(4)
                        d1.metric("Accuracy", f"{summary.get('accuracy', 0)}%")
                        d2.metric("Correct", f"{summary.get('correct', 0)}/{summary.get('total', 0)}")
                        d3.metric("Abstention rate", f"{summary.get('abstention_rate', 0)}%")
                        d4.metric("Source", summary.get("source", "uploaded_resume"))

                        st.markdown("#### Detailed results")
                        for row in live.get("details", []):
                            status = "✅" if row.get("correct") else "❌"
                            with st.expander(f"{status} {row.get('id')} • {row.get('query', '')[:90]}"):
                                st.markdown(f"**Category:** `{row.get('category', 'unknown')}`")
                                st.markdown(f"**Question:** {row.get('query', '')}")
                                st.markdown(f"**Ground Truth:** {row.get('ground_truth', '')}")
                                st.markdown(f"**Answer:** {row.get('answer', '')}")
                                if row.get("show_chunks") and row.get("retrieved_chunks"):
                                    st.markdown("**Retrieved sections:**")
                                    for idx, chunk in enumerate(row.get("retrieved_chunks", []), 1):
                                        st.markdown(f"- Section {idx}: {chunk[:220]}...")
                except Exception as exc:
                    st.error(f"❌ Failed to run live evaluation: {exc}")

    with tab_step3:
        st.markdown("### Step 3: RAG Pipeline Results")
        st.caption("Query rewriting, retrieval, filtering, grounding — full pipeline evaluation")
        
        with st.spinner("Loading Step 3 results..."):
            try:
                response = requests.get(f"{API_URL}/step3-results", timeout=10)
                response.raise_for_status()
                results = response.json()
                
                if not results:
                    st.info("No Step 3 results found.")
                else:
                    st.json(results)
            except Exception as exc:
                st.error(f"❌ Failed to load results: {exc}")

    with tab_step4:
        st.markdown("### Step 4: Meta-Prompting Results")
        st.caption("Prompt critique and improvement — GPT-based meta-prompting evaluation")
        
        with st.spinner("Loading Step 4 results..."):
            try:
                response = requests.get(f"{API_URL}/step4-results", timeout=10)
                response.raise_for_status()
                results = response.json()
                
                if not results:
                    st.info("No Step 4 results found.")
                else:
                    st.json(results)
            except Exception as exc:
                st.error(f"❌ Failed to load results: {exc}")

# ============================================================================
# Page: Artifacts
# ============================================================================

elif page == "Artifacts":
    st.markdown("## 🎯 Fine-Tuning Artifacts")
    st.caption("Download training and evaluation data for fine-tuning")

    tab_train, tab_dev = st.tabs(["Training Data", "Dev Data"])

    with tab_train:
        st.markdown("### Training JSONL")
        with st.spinner("Loading training data preview..."):
            try:
                response = requests.get(f"{API_URL}/finetune-preview?file_type=train&max_lines=5", timeout=10)
                response.raise_for_status()
                data = response.json()
                
                st.metric("Total training lines", data.get("total", 0))
                st.markdown("**Preview (first 5 lines):**")
                for line in data.get("lines", []):
                    st.json(line)
            except Exception as exc:
                st.error(f"❌ Failed to load: {exc}")

    with tab_dev:
        st.markdown("### Dev JSONL")
        with st.spinner("Loading dev data preview..."):
            try:
                response = requests.get(f"{API_URL}/finetune-preview?file_type=dev&max_lines=5", timeout=10)
                response.raise_for_status()
                data = response.json()
                
                st.metric("Total dev lines", data.get("total", 0))
                st.markdown("**Preview (first 5 lines):**")
                for line in data.get("lines", []):
                    st.json(line)
            except Exception as exc:
                st.error(f"❌ Failed to load: {exc}")



# hi testing code
