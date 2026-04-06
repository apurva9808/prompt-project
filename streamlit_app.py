"""
Streamlit frontend for Document Chatbot.
Calls FastAPI backend for RAG, dataset exploration, and evaluation features.
"""

import json
import os
from pathlib import Path

import requests
import streamlit as st

# FastAPI endpoint
API_URL = os.getenv("API_URL", "http://localhost:8000")

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
        st.rerun()
    st.divider()

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["Upload & Chat", "Dataset Explorer", "Evaluation Results", "Artifacts"],
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
