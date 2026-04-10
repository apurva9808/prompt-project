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


def _cover_letter_to_pdf(text: str, company: str, role: str) -> bytes:
    """Render plain-text cover letter as a clean A4 PDF."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(left=20, top=20, right=20)
    pdf.set_auto_page_break(auto=True, margin=20)

    # Title line (ASCII-safe so Helvetica can render it)
    pdf.set_font("Helvetica", "B", 14)
    title = f"Cover Letter - {role} at {company}" if role and company else "Cover Letter"
    pdf.cell(0, 10, _ascii_safe(title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # Body
    pdf.set_font("Helvetica", "", 11)
    usable_width = pdf.w - pdf.l_margin - pdf.r_margin
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
        padding-top: 1.25rem;
        padding-bottom: 2rem;
    }
    .hero {
        padding: 1.2rem 1.4rem;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
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
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.74rem;
        color: rgba(255,255,255,0.6);
        margin-bottom: 0.35rem;
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
    st.markdown(
        """
        <div class="hero" style="max-width: 620px; margin: 7vh auto 0 auto; text-align: center;">
            <div class="small-label">Document chatbot</div>
            <h1 style="margin:0; font-size:2.6rem;">Login first</h1>
            <p class="muted" style="margin-top:0.5rem;">
                Sign in to access the resume chatbot and upload your documents.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    login_cols = st.columns([0.18, 0.64, 0.18])
    with login_cols[1]:
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="apurva")
            password = st.text_input("Password", type="password", placeholder="resume123")
            submitted = st.form_submit_button("Login", use_container_width=True)

        if submitted:
            # Simple demo authentication
            if username == "apurva" and password == "resume123":
                st.session_state.authenticated = True
                st.session_state.authenticated_user = username.capitalize()
                st.rerun()
            else:
                st.error("Invalid credentials. Try: apurva / resume123")

    st.caption('Demo: username: apurva, password: resume123')
    st.stop()

# ============================================================================
# Main App (After Login)
# ============================================================================

# Sidebar: Logout button
with st.sidebar:
    st.markdown(f"### 👤 {st.session_state.authenticated_user}")
    if st.button("Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.authenticated_user = None
        st.session_state.resume_text = None
        st.session_state.resume_source = None
        st.session_state.chat_messages = []
        st.session_state.generated_cover_letter = None
        st.session_state.generated_cover_letter_meta = {}
        st.rerun()
    st.divider()

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["Upload & Chat", "Skill Gap Analyzer", "Cover Letter Generator", "Dataset Explorer", "Evaluation Results", "Artifacts"],
    label_visibility="collapsed",
)

# ============================================================================
# Page: Upload & Chat (Main Page)
# ============================================================================

if page == "Upload & Chat":
    st.markdown(
        """
        <div class="hero">
            <div class="small-label">Resume Chatbot</div>
            <h1 style="margin:0; font-size:2.8rem;">Chat with your resume or document</h1>
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
            <h1 style="margin:0; font-size:2.8rem;">🚀 Skill Gap Analyzer</h1>
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
            <h1 style="margin:0; font-size:2.8rem;">Cover Letter Generator</h1>
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
                            "company_name": company_name,
                            "role_title": role_title,
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
            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
            with summary_col1:
                st.metric("Company", letter_meta.get("company_name", "-"))
            with summary_col2:
                st.metric("Role", letter_meta.get("role_title", "-"))
            with summary_col3:
                st.metric("Source", letter_meta.get("source", "-"))
            with summary_col4:
                st.metric("Coverage", f"{letter_meta.get('coverage_score', 0)}%")

            if letter_meta.get("job_skills"):
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

            download_name = f"cover_letter_{safe_company}_{safe_role}.pdf"
            pdf_bytes = _cover_letter_to_pdf(
                st.session_state.generated_cover_letter,
                letter_meta.get("company_name", ""),
                letter_meta.get("role_title", ""),
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
