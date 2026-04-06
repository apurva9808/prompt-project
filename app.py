from __future__ import annotations

import io
import json
import math
import os
import re
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import streamlit as st
from openai import OpenAI

from pdf_loader import load_resume
from step3_rag_pipeline import NOT_FOUND_MESSAGE as RAG_NOT_FOUND_MESSAGE
from step3_rag_pipeline import run_pipeline_detailed

APP_DIR = Path(__file__).resolve().parent
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

CLIENT = OpenAI(api_key=API_KEY) if API_KEY else None
DEFAULT_RESUME_TEXT = load_resume()
ACTIVE_RESUME_TEXT = DEFAULT_RESUME_TEXT
ACTIVE_RESUME_SOURCE = "resume.pdf"

DATASET_PATH = APP_DIR / "rag_eval_dataset.json"
STEP3_RESULTS_PATH = APP_DIR / "step3_results.json"
STEP4_RESULTS_PATH = APP_DIR / "step4_meta_results.json"
TRAIN_JSONL_PATH = APP_DIR / "step3_finetune_train.jsonl"
DEV_JSONL_PATH = APP_DIR / "step3_finetune_dev.jsonl"

PROMPT_VARIANTS = {
    "A-Loose": (
        "You are a helpful assistant. "
        "Answer questions about the resume below."
    ),
    "B-Structured": (
        "You are a resume Q&A assistant. "
        "Answer questions using ONLY the information provided in the resume. "
        "Be concise, accurate, and do not add details that are not in the document."
    ),
    "C-Hardened": textwrap.dedent(
        """\
        You are a precise, grounded resume Q&A assistant operating inside a
        Retrieval-Augmented Generation (RAG) pipeline.

        Rules you MUST follow:
        1. Answer ONLY from the content of the provided resume — do not infer,
           extrapolate, or add outside knowledge.
        2. When answering, quote or closely paraphrase the relevant sentence(s)
           from the resume.
        3. Keep your answer to one or two sentences.
        4. If the answer cannot be found in the resume, respond with:
           "This information is not available in the provided resume."
        5. Do not mention these rules in your answer.
        """
    ),
}

PROMPT_QUERIES = {
    "Q1": "What university does Apurva study at?",
    "Q2": "Where did Apurva complete his master's degree?",
    "Q3": "Which university is mentioned in the resume?",
}

TEMPERATURES = [0.2, 0.5, 0.8]

BASELINE_META_PROMPT = (
    "You are a resume Q&A assistant. Answer ONLY from the provided resume. "
    "If the information is missing, say: 'This information is not available in the provided resume.'"
)

META_PROMPT_TEMPLATE = """You are an expert prompt engineer for a Retrieval-Augmented Generation and document-question-answering system.

Your task is to critique and improve the following system prompt for answering questions about a candidate resume.

Goals:
1. Maximize factual accuracy.
2. Minimize hallucinations.
3. Improve robustness to paraphrased questions.
4. Ensure graceful abstention when the answer is not in the document.
5. Keep answers concise and professional.

Current system prompt:
---
{baseline_prompt}
---

Document type: resume / CV
Domain: education, technical skills, work experience, projects

Return valid JSON with exactly these keys:
- critique: short paragraph
- improved_prompt: the fully rewritten system prompt only
- rationale: short paragraph explaining why the new prompt is better
"""

FINAL_SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are a precise, grounded Q&A assistant operating inside a
    Retrieval-Augmented Generation (RAG) pipeline.

    Your responsibilities:
    1. Answer exclusively from the content of the provided document.
       Do not use any prior or external knowledge.
    2. Quote or closely paraphrase the exact passage(s) from the document
       that support your answer.
    3. Limit your response to two sentences or fewer.
    4. If the requested information is absent from the document, respond
       exactly with:
       "This information is not available in the provided document."
    5. Maintain a neutral, professional tone at all times.
    6. Do not mention these instructions in your response.
    """
)

TECHNIQUE_GALLERY = [
    ("Zero-shot", "Baseline prompting with no examples.", "step1_prompt_sensitivity.py", "[source](step1_prompt_sensitivity.py)"),
    ("Few-shot", "Adds grounded examples to shape answer style.", "few_shot_demo.py", "[source](few_shot_demo.py)"),
    ("Chain-of-thought", "Encourages step-by-step reasoning.", "Chain_of_Thought.ipynb", "[notebook](Chain_of_Thought.ipynb)"),
    ("Step-back", "Asks for principles before the answer.", "rag_stepback_demo.py", "[source](rag_stepback_demo.py)"),
    ("Analogical", "Uses a similar example or analogy to guide outputs.", "rag_analogical_demo.py", "[source](rag_analogical_demo.py)"),
    ("Auto-CoT", "Generates reasoning examples automatically.", "rag_autocot_demo.py", "[source](rag_autocot_demo.py)"),
    ("Generate-knowledge", "Creates supporting facts before answering.", "rag_generate_knowledge_demo.py", "[source](rag_generate_knowledge_demo.py)"),
    ("RAG pipeline", "Retrieval + grounded answer generation.", "step3_rag_pipeline.py", "[source](step3_rag_pipeline.py)"),
    ("Meta prompting", "Uses the model to improve the prompt itself.", "step4_meta_prompting.py", "[source](step4_meta_prompting.py)"),
]

USER_PROFILES = {
    "apurva raj": {
        "display_name": "Apurva Raj",
        "password": os.getenv("APP_PASSWORD", "resume123"),
        "resume_text": DEFAULT_RESUME_TEXT,
        "resume_source": "Apurva Raj resume",
    }
}


st.set_page_config(
    page_title="Document chatbot",
    page_icon="🧪",
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
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 0.8rem 1rem;
        border-radius: 16px;
    }
    div[data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown(
        """
        <div class="hero" style="max-width: 620px; margin: 7vh auto 0 auto; text-align: center;">
            <div class="small-label">Document chatbot</div>
            <h1 style="margin:0; font-size:2.6rem;">Login first</h1>
            <p class="muted" style="margin-top:0.5rem;">
                Sign in to open the landing page and chat with the selected resume profile.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    login_cols = st.columns([0.18, 0.64, 0.18])
    with login_cols[1]:
        with st.form("login_form", clear_on_submit=False):
            profile_key = st.selectbox(
                "Select profile",
                list(USER_PROFILES.keys()),
                format_func=lambda key: USER_PROFILES[key]["display_name"],
            )
            password = st.text_input("Password", type="password", placeholder="resume123")
            submitted = st.form_submit_button("Login", use_container_width=True)

        if submitted:
            profile = USER_PROFILES[profile_key]
            if password == profile["password"]:
                st.session_state.authenticated = True
                st.session_state.authenticated_user = profile["display_name"]
                st.session_state.active_resume_text = profile["resume_text"]
                st.session_state.active_resume_source = profile["resume_source"]
                st.rerun()
            else:
                st.error("Invalid password.")

    st.caption('Demo profile: Apurva Raj / password: resume123')
    st.stop()


@st.cache_data(show_spinner=False)
def load_json_file(path: str) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data(show_spinner=False)
def load_resume_sections(resume_text: str | None = None) -> list[dict[str, str]]:
    if resume_text is None:
        resume_text = DEFAULT_RESUME_TEXT

    section_headers = {"summary", "technical skills", "experience", "projects", "education"}
    sections: list[dict[str, str]] = []
    current_section = "Overview"
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines, current_section
        body = "\n".join(line for line in current_lines).strip()
        if body:
            sections.append({"section": current_section, "content": body})
        current_lines = []

    for line in resume_text.splitlines():
        normalized = line.strip().lower()
        if normalized in section_headers:
            flush()
            current_section = line.strip()
            continue
        if line.strip() or current_lines:
            current_lines.append(line)

    flush()
    return sections or [{"section": "Resume", "content": resume_text.strip()}]


def _read_uploaded_resume(uploaded_file) -> tuple[str, str]:
    filename = getattr(uploaded_file, "name", "resume")
    raw_bytes = uploaded_file.getvalue()

    if filename.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(raw_bytes))
            text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
            return text or "[ERROR] Could not extract any text from the uploaded PDF.", filename
        except Exception as exc:
            return f"[ERROR] Failed to read uploaded PDF: {exc}", filename

    try:
        return raw_bytes.decode("utf-8", errors="ignore").strip(), filename
    except Exception as exc:
        return f"[ERROR] Failed to read uploaded file: {exc}", filename


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _split_resume_chunks(resume_text: str) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", resume_text) if part.strip()]
    if paragraphs:
        return paragraphs
    lines = [line.strip() for line in resume_text.splitlines() if line.strip()]
    return lines or [resume_text.strip()]


def _extract_answer_from_context(question: str, context: str) -> str:
    if not context.strip():
        return RAG_NOT_FOUND_MESSAGE

    question_tokens = {token for token in _tokenize(question) if len(token) > 2}
    sentences = re.split(r"(?<=[.!?])\s+|\n+", context)
    scored: list[tuple[int, int, str]] = []

    for sentence in sentences:
        candidate = sentence.strip()
        if not candidate:
            continue
        tokens = set(_tokenize(candidate))
        overlap = len(question_tokens & tokens)
        if overlap:
            scored.append((overlap, -len(candidate), candidate))

    if scored:
        scored.sort(reverse=True)
        best = scored[0][2]
        if scored[0][0] < 2:
            return RAG_NOT_FOUND_MESSAGE
        return best

    first_sentence = next((sentence.strip() for sentence in sentences if sentence.strip()), "")
    return first_sentence if first_sentence else RAG_NOT_FOUND_MESSAGE


def _retrieve_resume_chunks(question: str, resume_text: str) -> list[str]:
    chunks = _split_resume_chunks(resume_text)
    question_tokens = {token for token in _tokenize(question) if len(token) > 2}

    def score(chunk: str) -> tuple[int, int]:
        chunk_tokens = set(_tokenize(chunk))
        overlap = len(question_tokens & chunk_tokens)
        return overlap, -len(chunk)

    ranked = sorted(chunks, key=score, reverse=True)
    return ranked[:4]


def answer_resume_question(question: str, resume_text: str, source: str = "resume") -> dict[str, Any]:
    retrieved_chunks = _retrieve_resume_chunks(question, resume_text)
    context = "\n\n".join(retrieved_chunks)

    if CLIENT is not None:
        response = CLIENT.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.2,
            max_tokens=180,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a resume chatbot. Answer ONLY from the provided resume context. "
                        "If the answer is missing, say: 'This information is not available in the provided resume.' "
                        "Keep the answer concise and grounded."
                    ),
                },
                {"role": "user", "content": f"Resume context:\n{context}\n\nQuestion: {question}"},
            ],
        )
        answer = response.choices[0].message.content.strip()
    else:
        answer = _extract_answer_from_context(question, context)

    if not answer:
        answer = RAG_NOT_FOUND_MESSAGE

    return {
        "question": question,
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "source": source,
        "grounded": answer != RAG_NOT_FOUND_MESSAGE,
    }


@st.cache_data(show_spinner=False)
def load_dataset() -> dict[str, Any]:
    return load_json_file(str(DATASET_PATH))


@st.cache_data(show_spinner=False)
def load_step3_results() -> dict[str, Any]:
    return load_json_file(str(STEP3_RESULTS_PATH))


@st.cache_data(show_spinner=False)
def load_step4_results() -> dict[str, Any]:
    return load_json_file(str(STEP4_RESULTS_PATH))


@st.cache_data(show_spinner=False)
def load_finetune_preview(path: str, max_lines: int = 3) -> list[str]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    lines: list[str] = []
    with open(file_path, "r", encoding="utf-8") as file:
        for index, line in enumerate(file):
            if index >= max_lines:
                break
            lines.append(line.rstrip())
    return lines


def metric_card(label: str, value: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="card">
            <div class="small-label">{label}</div>
            <div style="font-size: 1.55rem; font-weight: 800; line-height: 1.15;">{value}</div>
            <div class="muted" style="margin-top: 0.25rem;">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def is_not_in_document(answer: str) -> bool:
    return "not available" in answer.lower() or answer.strip() == RAG_NOT_FOUND_MESSAGE


def build_resume_messages(system_prompt: str, query: str, resume_text: str | None = None) -> list[dict[str, str]]:
    if resume_text is None:
        resume_text = ACTIVE_RESUME_TEXT

    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Resume:\n{resume_text}\n\nQuestion: {query}",
        },
    ]


def call_resume_model(system_prompt: str, query: str, temperature: float, resume_text: str | None = None) -> str:
    if CLIENT is None:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    response = CLIENT.chat.completions.create(
        model=OPENAI_MODEL,
        messages=build_resume_messages(system_prompt, query, resume_text=resume_text),
        temperature=temperature,
        max_tokens=150,
    )
    return response.choices[0].message.content.strip()


def run_prompt_sensitivity_experiment(resume_text: str | None = None) -> dict[str, Any]:
    if resume_text is None:
        resume_text = ACTIVE_RESUME_TEXT

    results: dict[str, Any] = {}
    total = len(PROMPT_VARIANTS) * len(TEMPERATURES) * len(PROMPT_QUERIES)
    progress = st.progress(0)
    status = st.empty()
    counter = 0

    for prompt_name, prompt_text in PROMPT_VARIANTS.items():
        results[prompt_name] = {}
        for temp in TEMPERATURES:
            results[prompt_name][temp] = {}
            for query_name, query_text in PROMPT_QUERIES.items():
                counter += 1
                status.info(f"Running {prompt_name} | T={temp} | {query_name} ({counter}/{total})")
                reply = call_resume_model(prompt_text, query_text, temp, resume_text=resume_text)
                results[prompt_name][temp][query_name] = {
                    "response": reply,
                    "correct": "northeastern university" in reply.lower(),
                    "tokens": len(reply.split()),
                }
                progress.progress(counter / total)

    status.success("Prompt sensitivity benchmark complete.")
    return results


def score_consistency(flags: list[bool]) -> str:
    if all(flags):
        return "FULL"
    if any(flags):
        return "PARTIAL"
    return "NONE"


def flatten_prompt_results(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prompt_name, temp_map in results.items():
        for temp, query_map in temp_map.items():
            flags = []
            tokens = []
            for query_name, record in query_map.items():
                flags.append(bool(record["correct"]))
                tokens.append(int(record["tokens"]))
                rows.append(
                    {
                        "prompt": prompt_name,
                        "temperature": temp,
                        "query": query_name,
                        "correct": record["correct"],
                        "tokens": record["tokens"],
                        "response": record["response"],
                    }
                )
            rows.append(
                {
                    "prompt": prompt_name,
                    "temperature": temp,
                    "query": "summary",
                    "correct": score_consistency(flags),
                    "tokens": round(sum(tokens) / len(tokens), 1),
                    "response": f"Average tokens: {round(sum(tokens) / len(tokens), 1)}",
                }
            )
    return rows


def estimate_perplexity_from_logprobs(response) -> float | None:
    try:
        items = response.choices[0].logprobs.content or []
        token_logprobs = [item.logprob for item in items if hasattr(item, "logprob")]
        if not token_logprobs:
            return None
        return math.exp(-sum(token_logprobs) / len(token_logprobs))
    except Exception:
        return None


def judge_answer_live(item: dict[str, Any], system_name: str, answer: str) -> dict[str, Any]:
    if CLIENT is None:
        return {"correct": False, "hallucinated": True, "rationale": "No API key configured."}

    judge_prompt = (
        "You are evaluating answers from a resume QA system. "
        "Return strict JSON with keys: correct (true/false), hallucinated (true/false), rationale (string). "
        "For adversarial or NOT_IN_DOCUMENT items, the answer is correct only if it clearly abstains. "
        "Mark hallucinated=true if the answer includes unsupported facts, fabricated details, or answers an unrelated question."
    )
    payload = {
        "query": item["query"],
        "ground_truth": item["ground_truth"],
        "category": item.get("category"),
        "system": system_name,
        "answer": answer,
    }
    response = CLIENT.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        max_tokens=150,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": judge_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    try:
        return json.loads(response.choices[0].message.content.strip())
    except Exception:
        return {"correct": False, "hallucinated": True, "rationale": "Judge parsing failed."}


def answer_with_prompt_live(system_prompt: str, query: str, resume_text: str | None = None) -> tuple[str, float | None]:
    if CLIENT is None:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    if resume_text is None:
        resume_text = ACTIVE_RESUME_TEXT

    response = CLIENT.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.2,
        max_tokens=180,
        logprobs=True,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Resume:\n{resume_text}\n\nQuestion: {query}"},
        ],
    )
    message = response.choices[0].message.content.strip()
    return message, estimate_perplexity_from_logprobs(response)


def generate_meta_improved_prompt_live() -> dict[str, str]:
    if CLIENT is None:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    meta_prompt = META_PROMPT_TEMPLATE.format(baseline_prompt=BASELINE_META_PROMPT)
    response = CLIENT.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": meta_prompt}],
    )
    result = json.loads(response.choices[0].message.content.strip())
    return {
        "critique": result["critique"],
        "improved_prompt": result["improved_prompt"],
        "rationale": result["rationale"],
    }


def evaluate_prompt_live(prompt_name: str, system_prompt: str, test_items: list[dict[str, Any]], resume_text: str | None = None) -> dict[str, Any]:
    if resume_text is None:
        resume_text = ACTIVE_RESUME_TEXT

    rows = []
    correct_count = 0
    hallucinated_count = 0
    perplexities: list[float] = []

    for item in test_items:
        answer, perplexity = answer_with_prompt_live(system_prompt, item["query"], resume_text=resume_text)
        judgment = judge_answer_live(item, prompt_name, answer)
        correct_count += int(bool(judgment.get("correct")))
        hallucinated_count += int(bool(judgment.get("hallucinated")))
        if perplexity is not None:
            perplexities.append(perplexity)
        rows.append(
            {
                "id": item["id"],
                "query": item["query"],
                "ground_truth": item["ground_truth"],
                "answer": answer,
                "perplexity": perplexity,
                "judgment": judgment,
            }
        )

    total = len(test_items)
    return {
        "summary": {
            "accuracy": round(100 * correct_count / total, 1),
            "hallucination_rate": round(100 * hallucinated_count / total, 1),
            "consistency": round(100 * correct_count / total, 1),
            "average_perplexity": round(mean(perplexities), 3) if perplexities else None,
            "correct_count": correct_count,
            "hallucinated_count": hallucinated_count,
            "total": total,
        },
        "details": rows,
    }


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
project_dataset = load_dataset()
step3_results = load_step3_results()
step4_results = load_step4_results()

# -----------------------------------------------------------------------------
# Sidebar navigation
# -----------------------------------------------------------------------------
st.sidebar.title("Document chatbot")
st.sidebar.caption("Resume RAG • Prompt Engineering • Evaluation")

uploaded_resume = st.sidebar.file_uploader(
    "Upload a resume",
    type=["pdf", "txt", "md"],
    help="Upload a resume to chat over a different document without changing the rest of the app.",
)

ACTIVE_RESUME_TEXT = st.session_state.get("active_resume_text", DEFAULT_RESUME_TEXT)
ACTIVE_RESUME_SOURCE = st.session_state.get("active_resume_source", "resume.pdf")

if uploaded_resume is not None:
    uploaded_text, uploaded_name = _read_uploaded_resume(uploaded_resume)
    ACTIVE_RESUME_TEXT = uploaded_text
    ACTIVE_RESUME_SOURCE = uploaded_name
    st.session_state.active_resume_text = uploaded_text
    st.session_state.active_resume_source = uploaded_name

resume_sections = load_resume_sections(ACTIVE_RESUME_TEXT)

page = st.sidebar.radio(
    "Navigate",
    [
        "Landing",
        "Resume Chat",
        "Prompt Lab",
        "RAG Playground",
        "Dataset Explorer",
        "Evaluation Dashboard",
        "Artifacts",
    ],
)

if API_KEY:
    st.sidebar.success("OpenAI key detected")
else:
    st.sidebar.warning("No OpenAI key found")

if st.sidebar.button("Logout"):
    st.session_state.authenticated = False
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**Current model**")
st.sidebar.write(OPENAI_MODEL)
st.sidebar.markdown("**Resume source**")
active_source = st.session_state.get("active_resume_source", ACTIVE_RESUME_SOURCE)
st.sidebar.write(active_source)
st.sidebar.markdown("**Upload status**")
st.sidebar.write("Using uploaded resume" if uploaded_resume is not None else "Using selected profile resume")

# -----------------------------------------------------------------------------
# Overview
# -----------------------------------------------------------------------------
if page == "Landing":
    st.markdown(
        """
        <div class="hero">
            <div class="small-label">Document chatbot</div>
            <h1 style="margin:0; font-size:2.8rem;">Chat with your resume or document</h1>
            <p class="muted" style="margin-top:0.5rem; max-width: 920px;">
                Upload a resume or use the built-in project resume. The app uses the Python files in this workspace as backend features:
                resume extraction, grounded answering, RAG retrieval, prompt sensitivity, and meta-prompting.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Chat UI", "Resume first", "Simple landing page and chat interface")
    with c2:
        metric_card("Backend", "Python", "All functionality comes from the project scripts")
    with c3:
        metric_card("Mode", "Upload ready", "Use your own resume or the default one")

    st.markdown("### What is included")
    st.markdown(
        """
        - Resume chatbot with upload support
        - RAG pipeline backed by Python code
        - Prompt sensitivity experiments
        - Meta prompting evaluation
        - Dataset explorer and artifact downloads
        """
    )

    st.markdown("### Start here")
    st.info("Use the sidebar to upload a resume, then open Resume Chat to ask questions.")

elif page == "Resume Chat":
    st.markdown(
        f"""
        <div class="hero">
            <div class="small-label">Resume Chatbot</div>
            <h1 style="margin:0; font-size:2.4rem;">Ask questions about the resume</h1>
            <p class="muted" style="margin-top:0.5rem; max-width: 920px;">
                Upload a resume in the sidebar or use the default project resume. The assistant will answer only
                from the provided document and will abstain when the answer is not present.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": "Upload a resume, then ask me anything about education, skills, experience, or projects.",
            }
        ]

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_question = st.chat_input("Ask about the resume...")
    if user_question:
        st.session_state.chat_messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.write(user_question)

        with st.spinner("Reading the resume..."):
            active_resume = st.session_state.get("active_resume_text", ACTIVE_RESUME_TEXT)
            active_source = st.session_state.get("active_resume_source", ACTIVE_RESUME_SOURCE)
            result = answer_resume_question(user_question, active_resume, active_source)

        st.session_state.chat_messages.append({"role": "assistant", "content": result["answer"]})
        with st.chat_message("assistant"):
            st.write(result["answer"])

        with st.expander("Retrieved resume chunks"):
            for idx, chunk in enumerate(result["retrieved_chunks"], start=1):
                st.markdown(f"**Chunk {idx}**")
                st.write(chunk)

    st.markdown("### Current document")
    active_source = st.session_state.get("active_resume_source", ACTIVE_RESUME_SOURCE)
    active_resume = st.session_state.get("active_resume_text", ACTIVE_RESUME_TEXT)
    st.info(active_source)
    st.code(active_resume[:2500])

elif page == "Overview":
    st.markdown(
        """
        <div class="hero">
            <div class="small-label">Prompt Engineering System Design</div>
            <h1 style="margin:0; font-size:2.4rem;">Document chatbot</h1>
            <p class="muted" style="margin-top:0.5rem; max-width: 920px;">
                A unified UI for the resume chatbot project: chat over a resume, inspect the RAG pipeline,
                explore the evaluation dataset, and review prompt-engineering results.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Technique gallery", "9", "Prompting patterns already implemented")
    with c2:
        metric_card("Evaluation items", "45", "Curated RAG dataset entries")
    with c3:
        metric_card("Held-out test", "8", "Used for the published metrics")
    with c4:
        metric_card("Live mode", "Ready", "Resume upload + grounded chat")

    st.markdown("### Project snapshot")
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown(
            """
            <div class="card">
                <div class="section-title">What this app does</div>
                <ul>
                    <li>Lets you upload a resume and chat with it directly.</li>
                    <li>Lets you ask resume questions in a grounded RAG playground.</li>
                    <li>Shows prompt sensitivity results across temperature and paraphrases.</li>
                    <li>Explores the curated evaluation dataset and split structure.</li>
                    <li>Displays Step 3 and Step 4 results with downloadable artifacts.</li>
                    <li>Can rerun prompt sensitivity and meta-prompting if an API key is configured.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="card">
                <div class="section-title">Best current outcome</div>
                <p class="muted">The RAG pipeline outperforms the prompt-only baseline on the held-out test split.</p>
                <p><strong>Prompt-only:</strong> 62.5% accuracy</p>
                <p><strong>RAG:</strong> 75.0% accuracy</p>
                <p><strong>Meta prompt:</strong> improved theory, but worse empirical results on this run.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Prompting techniques in the project")
    for technique, description, file_name, link_text in TECHNIQUE_GALLERY:
        st.markdown(
            f"- **{technique}** — {description} {link_text} ({file_name})",
            unsafe_allow_html=True,
        )

# -----------------------------------------------------------------------------
# Prompt Lab
# -----------------------------------------------------------------------------
elif page == "Prompt Lab":
    st.markdown("## Prompt Lab")
    st.caption("Run prompt sensitivity tests on the resume QA task.")

    tab_quick, tab_benchmark = st.tabs(["Quick compare", "Full benchmark"])

    with tab_quick:
        left, right = st.columns([1.1, 0.9])
        with left:
            selected_prompt = st.selectbox("System prompt", list(PROMPT_VARIANTS.keys()))
            selected_query = st.selectbox("Query", list(PROMPT_QUERIES.keys()))
            selected_temp = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
            run_quick = st.button("Run quick comparison", type="primary")
        with right:
            st.markdown(
                """
                <div class="card">
                    <div class="section-title">What this tests</div>
                    <p class="muted">The same resume question is phrased in three ways while the prompt and temperature vary.</p>
                    <p class="muted">This shows how stricter instructions reduce paraphrase sensitivity.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if run_quick:
            if CLIENT is None:
                st.error("OPENAI_API_KEY is not set, so live prompt testing is unavailable.")
            else:
                with st.spinner("Calling the model..."):
                    answer = call_resume_model(
                        PROMPT_VARIANTS[selected_prompt],
                        PROMPT_QUERIES[selected_query],
                        selected_temp,
                        resume_text=ACTIVE_RESUME_TEXT,
                    )
                st.success("Response generated")
                st.markdown("#### Answer")
                st.write(answer)

    with tab_benchmark:
        st.info("Full benchmark runs 27 API calls. Use it only if you want to reproduce the prompt sensitivity table.")
        if st.button("Run full benchmark", type="primary"):
            if CLIENT is None:
                st.error("OPENAI_API_KEY is not set.")
            else:
                with st.spinner("Running benchmark..."):
                    benchmark_results = run_prompt_sensitivity_experiment(ACTIVE_RESUME_TEXT)
                st.session_state["benchmark_results"] = benchmark_results

        benchmark_results = st.session_state.get("benchmark_results")
        if benchmark_results:
            st.markdown("#### Results")
            rows = flatten_prompt_results(benchmark_results)
            st.dataframe(rows, use_container_width=True)

            summary_rows = []
            for prompt_name, temp_map in benchmark_results.items():
                all_flags: list[bool] = []
                all_tokens: list[int] = []
                for temp, query_map in temp_map.items():
                    flags = [bool(record["correct"]) for record in query_map.values()]
                    tokens = [int(record["tokens"]) for record in query_map.values()]
                    all_flags.extend(flags)
                    all_tokens.extend(tokens)
                    summary_rows.append(
                        {
                            "prompt": prompt_name,
                            "temperature": temp,
                            "accuracy": round(100 * sum(flags) / len(flags), 1),
                            "consistency": score_consistency(flags),
                            "avg_tokens": round(sum(tokens) / len(tokens), 1),
                        }
                    )
                st.caption(f"Overall accuracy for {prompt_name}: {round(100 * sum(all_flags) / len(all_flags), 1)}%")

            st.markdown("#### Prompt-level summaries")
            st.dataframe(summary_rows, use_container_width=True)
        else:
            st.write("No benchmark has been run in this session yet.")

# -----------------------------------------------------------------------------
# RAG Playground
# -----------------------------------------------------------------------------
elif page == "RAG Playground":
    st.markdown("## RAG Playground")
    st.caption("Ask questions about the active resume and inspect retrieval details.")

    question = st.text_input(
        "Ask a question about Apurva's resume",
        placeholder="e.g. What cloud technologies does Apurva use?",
    )
    run_rag = st.button("Run grounded RAG answer", type="primary")

    if run_rag and question.strip():
        with st.spinner("Retrieving and generating..."):
            result = answer_resume_question(question.strip(), ACTIVE_RESUME_TEXT)
        st.session_state["rag_result"] = result

    result = st.session_state.get("rag_result")
    if result:
        left, right = st.columns([1.1, 0.9])
        with left:
            st.markdown("#### Answer")
            st.success(result["answer"])
            st.markdown("#### Diagnostics")
            st.write({
                "source": result.get("source"),
                "grounded": result.get("grounded"),
                "strategy": "upload_aware_chatbot",
            })
        with right:
            st.markdown("#### Retrieved chunks")
            for index, chunk in enumerate(result.get("retrieved_chunks", []), start=1):
                with st.expander(f"Chunk {index}"):
                    st.write(chunk)

        with st.expander("Filtered context"):
            st.write("\n\n".join(result.get("retrieved_chunks", [])))
    else:
        st.info("Run a question to see the retriever, filter, and answer steps.")

# -----------------------------------------------------------------------------
# Dataset Explorer
# -----------------------------------------------------------------------------
elif page == "Dataset Explorer":
    st.markdown("## Dataset Explorer")
    metadata = project_dataset.get("metadata", {})
    splits = project_dataset.get("splits", {})

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total items", str(metadata.get("total_items", "—")), "All curated prompts")
    with c2:
        metric_card("Typical", str(metadata.get("breakdown", {}).get("typical", "—")), "Direct factual lookups")
    with c3:
        metric_card("Edge cases", str(metadata.get("breakdown", {}).get("edge_cases", "—")), "Synthesis and reasoning")
    with c4:
        metric_card("Adversarial", str(metadata.get("breakdown", {}).get("adversarial", "—")), "Hallucination traps")

    st.markdown("### Metadata")
    st.json(metadata)

    split_choice = st.selectbox("Split", ["train", "dev", "test", "all"])
    category_choice = st.selectbox("Category", ["all", "typical", "edge_case", "adversarial"])
    query_filter = st.text_input("Search queries", placeholder="e.g. Kubernetes, salary, Admins")

    items: list[dict[str, Any]] = []
    if split_choice == "all":
        for split_name in ["train", "dev", "test"]:
            items.extend(splits.get(split_name, []))
    else:
        items = list(splits.get(split_choice, []))

    if category_choice != "all":
        items = [item for item in items if item.get("category") == category_choice]
    if query_filter.strip():
        needle = query_filter.lower()
        items = [item for item in items if needle in item.get("query", "").lower() or needle in item.get("ground_truth", "").lower()]

    st.markdown(f"### Matching items ({len(items)})")
    st.dataframe(
        [
            {
                "id": item.get("id"),
                "category": item.get("category"),
                "sub_category": item.get("sub_category"),
                "query": item.get("query"),
                "ground_truth": item.get("ground_truth"),
            }
            for item in items
        ],
        use_container_width=True,
    )

    if items:
        selection = st.selectbox("Inspect an item", [f"{item['id']} — {item['query']}" for item in items])
        selected_index = [f"{item['id']} — {item['query']}" for item in items].index(selection)
        st.json(items[selected_index])

# -----------------------------------------------------------------------------
# Evaluation Dashboard
# -----------------------------------------------------------------------------
elif page == "Evaluation Dashboard":
    st.markdown("## Evaluation Dashboard")

    step3_summary = step3_results.get("summary", {})
    step4_before = step4_results.get("before", {}).get("summary", {})
    step4_after = step4_results.get("after", {}).get("summary", {})

    st.markdown("### Step 3 results")
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Prompt-only accuracy", f"{step3_summary.get('prompt_only', {}).get('accuracy', '—')}%", "Baseline without retrieval")
    with c2:
        metric_card("RAG accuracy", f"{step3_summary.get('rag', {}).get('accuracy', '—')}%", "Retrieval-grounded answer quality")
    with c3:
        metric_card("Fine-tune status", "Prepared", "Train/dev JSONL files generated")

    st.dataframe(
        [
            {"system": "prompt_only", **step3_summary.get("prompt_only", {})},
            {"system": "rag", **step3_summary.get("rag", {})},
            {"system": "fine_tuned_model", **step3_summary.get("fine_tuned_model", {})},
        ],
        use_container_width=True,
    )

    st.markdown("### Step 4 meta prompting results")
    c4, c5, c6 = st.columns(3)
    with c4:
        metric_card("Before accuracy", f"{step4_before.get('accuracy', '—')}%", "Baseline prompt")
    with c5:
        metric_card("After accuracy", f"{step4_after.get('accuracy', '—')}%", "Meta-improved prompt")
    with c6:
        metric_card("Perplexity change", f"{step4_before.get('average_perplexity', '—')} → {step4_after.get('average_perplexity', '—')}", "Lower is better")

    st.dataframe(
        [
            {"prompt": "before", **step4_before},
            {"prompt": "after", **step4_after},
        ],
        use_container_width=True,
    )

    if st.button("Rerun meta prompting", type="primary"):
        if CLIENT is None:
            st.error("OPENAI_API_KEY is not set.")
        else:
            test_items = project_dataset.get("splits", {}).get("test", [])
            with st.spinner("Generating improved prompt and evaluating..."):
                meta = generate_meta_improved_prompt_live()
                before_eval = evaluate_prompt_live("baseline_prompt", BASELINE_META_PROMPT, test_items)
                after_eval = evaluate_prompt_live("meta_improved_prompt", meta["improved_prompt"], test_items)
            st.session_state["live_meta_results"] = {
                "meta": meta,
                "before": before_eval,
                "after": after_eval,
            }

    live_meta_results = st.session_state.get("live_meta_results")
    if live_meta_results:
        st.markdown("#### Latest live rerun")
        st.json(live_meta_results)

# -----------------------------------------------------------------------------
# Artifacts
# -----------------------------------------------------------------------------
elif page == "Artifacts":
    st.markdown("## Artifacts")

    tab1, tab2, tab3, tab4 = st.tabs(["Resume", "Prompts", "Fine-tune files", "Downloads"])

    with tab1:
        for section in resume_sections:
            with st.expander(section["section"], expanded=section["section"].lower() in {"summary", "technical skills"}):
                st.write(section["content"])

    with tab2:
        st.markdown("### Prompt sensitivity prompts")
        for name, prompt in PROMPT_VARIANTS.items():
            with st.expander(name):
                st.code(prompt)

        st.markdown("### Baseline meta prompt")
        st.code(BASELINE_META_PROMPT)

        st.markdown("### Meta prompt template")
        st.code(META_PROMPT_TEMPLATE.format(baseline_prompt=BASELINE_META_PROMPT))

        st.markdown("### Final hardened prompt")
        st.code(FINAL_SYSTEM_PROMPT)

    with tab3:
        st.markdown("### Fine-tuning preparation")
        st.write("The training and validation files are prepared but no live fine-tuning job was submitted.")
        c1, c2 = st.columns(2)
        with c1:
            st.write("Train preview")
            st.code("\n".join(load_finetune_preview(str(TRAIN_JSONL_PATH))))
        with c2:
            st.write("Dev preview")
            st.code("\n".join(load_finetune_preview(str(DEV_JSONL_PATH))))

    with tab4:
        st.markdown("### Download the saved JSON outputs")
        st.download_button(
            "Download step3_results.json",
            data=json.dumps(step3_results, indent=2, ensure_ascii=False),
            file_name="step3_results.json",
            mime="application/json",
        )
        st.download_button(
            "Download step4_meta_results.json",
            data=json.dumps(step4_results, indent=2, ensure_ascii=False),
            file_name="step4_meta_results.json",
            mime="application/json",
        )
        st.download_button(
            "Download the evaluation dataset",
            data=json.dumps(project_dataset, indent=2, ensure_ascii=False),
            file_name="rag_eval_dataset.json",
            mime="application/json",
        )

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption("Document chatbot • built from the Python backend already in the workspace.")
