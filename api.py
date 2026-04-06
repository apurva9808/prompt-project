"""
FastAPI backend server for Document Chatbot.
Exposes RAG pipeline, dataset exploration, and evaluation endpoints.
"""

import io
import json
import os
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

from step3_rag_pipeline import NOT_FOUND_MESSAGE as RAG_NOT_FOUND_MESSAGE
from pdf_loader import load_resume

NOT_RELATED_MESSAGE = "Question is not related to the resume."
UPLOAD_REQUIRED_MESSAGE = "Please upload a resume before asking questions."

app = FastAPI(title="Document Chatbot API", version="1.0.0")

# Enable CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
CLIENT = OpenAI(api_key=API_KEY) if API_KEY else None

APP_DIR = Path(__file__).resolve().parent
DATASET_PATH = APP_DIR / "rag_eval_dataset.json"
STEP3_RESULTS_PATH = APP_DIR / "step3_results.json"
STEP4_RESULTS_PATH = APP_DIR / "step4_meta_results.json"
TRAIN_JSONL_PATH = APP_DIR / "step3_finetune_train.jsonl"
DEV_JSONL_PATH = APP_DIR / "step3_finetune_dev.jsonl"

CURRENT_UPLOADED_RESUME_TEXT = ""
CURRENT_UPLOADED_RESUME_SOURCE = ""


# ============================================================================
# Request/Response Models
# ============================================================================

class QuestionRequest(BaseModel):
    question: str
    resume_text: str | None = None


class AnswerResponse(BaseModel):
    question: str
    answer: str
    retrieved_chunks: list[str]
    source: str
    grounded: bool
    show_chunks: bool = True


class UploadedResumeResponse(BaseModel):
    text: str
    filename: str
    error: str | None = None


class DatasetItem(BaseModel):
    query: str
    ground_truth: str
    category: str


class EvaluateUploadedRequest(BaseModel):
    split: str = "test"
    max_items: int | None = None


# ============================================================================
# Helper Functions
# ============================================================================

def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _split_resume_chunks(resume_text: str) -> list[str]:
    """Split resume into chunks (paragraphs or lines)."""
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", resume_text) if part.strip()]
    if paragraphs:
        return paragraphs
    lines = [line.strip() for line in resume_text.splitlines() if line.strip()]
    return lines or [resume_text.strip()]


def _extract_answer_from_context(question: str, context: str) -> str:
    """Extract answer from context using offline fallback (no API)."""
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
        if len(tokens) < 4:
            continue
        overlap = len(question_tokens & tokens)
        lower_candidate = candidate.lower()
        if "languages:" in lower_candidate and ("language" in question_tokens or "languages" in question_tokens):
            overlap += 3
        if "frameworks:" in lower_candidate and ("framework" in question_tokens or "frameworks" in question_tokens or "frontend" in question_tokens):
            overlap += 3
        if overlap:
            scored.append((overlap, -len(candidate), candidate))

    if scored:
        scored.sort(reverse=True)
        best = scored[0][2]
        if scored[0][0] < 1:
            return RAG_NOT_FOUND_MESSAGE
        return best

    first_sentence = next((sentence.strip() for sentence in sentences if sentence.strip()), "")
    return RAG_NOT_FOUND_MESSAGE


def _retrieve_resume_chunks(question: str, resume_text: str) -> list[str]:
    """Retrieve top chunks from resume based on question keywords (BM25-like)."""
    chunks = _split_resume_chunks(resume_text)
    question_tokens = {token for token in _tokenize(question) if len(token) > 2}

    def score(chunk: str) -> tuple[int, int]:
        chunk_tokens = set(_tokenize(chunk))
        overlap = len(question_tokens & chunk_tokens)
        return overlap, -len(chunk)

    ranked = sorted(chunks, key=score, reverse=True)
    return ranked[:4]


def _max_question_chunk_overlap(question: str, chunks: list[str]) -> int:
    question_tokens = {token for token in _tokenize(question) if len(token) > 2}
    if not question_tokens:
        return 0
    max_overlap = 0
    for chunk in chunks:
        chunk_tokens = set(_tokenize(chunk))
        overlap = len(question_tokens & chunk_tokens)
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap


def _read_uploaded_resume(file_content: bytes, filename: str) -> tuple[str, str | None]:
    """Parse uploaded resume file (PDF, TXT, MD)."""
    if filename.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(file_content))
            text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
            return text or "[ERROR] Could not extract any text from the uploaded PDF.", None
        except Exception as exc:
            return "", f"Failed to read PDF: {exc}"

    try:
        return file_content.decode("utf-8", errors="ignore").strip(), None
    except Exception as exc:
        return "", f"Failed to read file: {exc}"


def _answer_for_resume(question: str, resume_text: str, source: str) -> AnswerResponse:
    retrieved_chunks = _retrieve_resume_chunks(question, resume_text)
    context = "\n\n".join(retrieved_chunks)

    overlap_score = _max_question_chunk_overlap(question, retrieved_chunks)
    if overlap_score < 1:
        return AnswerResponse(
            question=question,
            answer=NOT_RELATED_MESSAGE,
            retrieved_chunks=[],
            source=source,
            grounded=False,
            show_chunks=False,
        )

    if CLIENT is not None:
        try:
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
        except Exception:
            answer = ""
    else:
        answer = _extract_answer_from_context(question, context)

    if not answer:
        answer = RAG_NOT_FOUND_MESSAGE

    is_grounded = answer != RAG_NOT_FOUND_MESSAGE
    return AnswerResponse(
        question=question,
        answer=answer,
        retrieved_chunks=retrieved_chunks if is_grounded else [],
        source=source,
        grounded=is_grounded,
        show_chunks=is_grounded,
    )


def _is_abstention(answer: str) -> bool:
    lower = answer.lower()
    return (
        "not related to the resume" in lower
        or "not available in the provided resume" in lower
        or "does not contain this information" in lower
        or "please upload a resume" in lower
    )


def _evaluate_correctness(item: dict[str, Any], answer: str) -> bool:
    ground_truth = str(item.get("ground_truth", "")).strip().lower()
    category = str(item.get("category", "")).strip().lower()
    answer_lower = answer.lower()

    if "not_in_document" in ground_truth or category == "adversarial":
        return _is_abstention(answer)

    if ground_truth and ground_truth in answer_lower:
        return True

    gt_tokens = {token for token in _tokenize(ground_truth) if len(token) > 2}
    ans_tokens = set(_tokenize(answer_lower))
    if not gt_tokens:
        return False
    overlap_ratio = len(gt_tokens & ans_tokens) / len(gt_tokens)
    return overlap_ratio >= 0.4


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "api_key_configured": API_KEY != "",
        "model": OPENAI_MODEL,
    }


@app.post("/answer", response_model=AnswerResponse)
async def answer_question(request: QuestionRequest) -> AnswerResponse:
    """
    Answer a question about the provided resume using RAG pipeline.
    
    - If OPENAI_API_KEY is set, uses ChatGPT
    - Otherwise uses offline extraction fallback
    """
    effective_resume_text = (CURRENT_UPLOADED_RESUME_TEXT or request.resume_text or "").strip()
    if not effective_resume_text:
        return AnswerResponse(
            question=request.question,
            answer=UPLOAD_REQUIRED_MESSAGE,
            retrieved_chunks=[],
            source="uploaded_resume",
            grounded=False,
            show_chunks=False,
        )

    return _answer_for_resume(
        question=request.question,
        resume_text=effective_resume_text,
        source=CURRENT_UPLOADED_RESUME_SOURCE or "uploaded_resume",
    )


@app.post("/evaluate-uploaded")
async def evaluate_uploaded_resume(request: EvaluateUploadedRequest) -> dict[str, Any]:
    """Run RAG + evaluation on the currently uploaded resume using dataset split items."""
    if not CURRENT_UPLOADED_RESUME_TEXT.strip():
        return {
            "error": UPLOAD_REQUIRED_MESSAGE,
            "summary": {},
            "details": [],
        }

    if not DATASET_PATH.exists():
        return {
            "error": "Evaluation dataset not found.",
            "summary": {},
            "details": [],
        }

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    split = request.split.lower().strip()
    split_items = dataset.get("splits", {}).get(split, [])
    if not split_items:
        return {
            "error": f"Split '{split}' has no items.",
            "summary": {},
            "details": [],
        }

    items = split_items[: request.max_items] if request.max_items else split_items
    details: list[dict[str, Any]] = []
    correct_count = 0
    abstained_count = 0

    for item in items:
        question = str(item.get("query", "")).strip()
        if not question:
            continue
        answer_result = _answer_for_resume(question, CURRENT_UPLOADED_RESUME_TEXT, CURRENT_UPLOADED_RESUME_SOURCE or "uploaded_resume")
        correct = _evaluate_correctness(item, answer_result.answer)
        abstained = _is_abstention(answer_result.answer)
        correct_count += int(correct)
        abstained_count += int(abstained)

        details.append(
            {
                "id": item.get("id"),
                "category": item.get("category"),
                "query": question,
                "ground_truth": item.get("ground_truth"),
                "answer": answer_result.answer,
                "correct": correct,
                "grounded": answer_result.grounded,
                "show_chunks": answer_result.show_chunks,
                "retrieved_chunks": answer_result.retrieved_chunks,
            }
        )

    total = len(details)
    accuracy = round((correct_count / total) * 100, 1) if total else 0.0
    abstention_rate = round((abstained_count / total) * 100, 1) if total else 0.0

    return {
        "error": None,
        "summary": {
            "split": split,
            "total": total,
            "correct": correct_count,
            "accuracy": accuracy,
            "abstained": abstained_count,
            "abstention_rate": abstention_rate,
            "source": CURRENT_UPLOADED_RESUME_SOURCE or "uploaded_resume",
            "pipeline": "uploaded_resume_rag_eval",
        },
        "details": details,
    }

@app.post("/upload-resume", response_model=UploadedResumeResponse)
async def upload_resume(file: UploadFile = File(...)) -> UploadedResumeResponse:
    """
    Upload and parse a resume file (PDF, TXT, MD).
    Returns the extracted text and filename.
    """
    try:
        global CURRENT_UPLOADED_RESUME_TEXT, CURRENT_UPLOADED_RESUME_SOURCE
        content = await file.read()
        text, error = _read_uploaded_resume(content, file.filename or "resume")
        if not error and text and not text.startswith("[ERROR]"):
            CURRENT_UPLOADED_RESUME_TEXT = text
            CURRENT_UPLOADED_RESUME_SOURCE = file.filename or "uploaded_resume"
        return UploadedResumeResponse(text=text, filename=file.filename or "resume", error=error)
    except Exception as exc:
        return UploadedResumeResponse(text="", filename=file.filename or "resume", error=str(exc))


@app.get("/default-resume")
async def get_default_resume() -> dict[str, str]:
    """Get the default project resume."""
    resume_text = load_resume()
    return {"text": resume_text, "source": "default_project_resume"}


@app.get("/dataset")
async def get_dataset() -> dict[str, Any]:
    """Get the evaluation dataset (45 items: 30 typical + 10 edge + 5 adversarial)."""
    file_path = DATASET_PATH
    if not file_path.exists():
        return {"items": [], "total": 0}
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@app.get("/step3-results")
async def get_step3_results() -> dict[str, Any]:
    """Get Step 3 RAG pipeline results and evaluation metrics."""
    file_path = STEP3_RESULTS_PATH
    if not file_path.exists():
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/step4-results")
async def get_step4_results() -> dict[str, Any]:
    """Get Step 4 meta-prompting results and prompt improvements."""
    file_path = STEP4_RESULTS_PATH
    if not file_path.exists():
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/finetune-preview")
async def get_finetune_preview(file_type: str = "train", max_lines: int = 5) -> dict[str, Any]:
    """Get preview of fine-tuning JSONL files."""
    if file_type == "train":
        path = TRAIN_JSONL_PATH
    elif file_type == "dev":
        path = DEV_JSONL_PATH
    else:
        raise HTTPException(status_code=400, detail="file_type must be 'train' or 'dev'")

    if not path.exists():
        return {"lines": [], "total": 0}

    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= max_lines:
                break
            lines.append(json.loads(line.strip()))

    # Count total lines
    total = sum(1 for _ in open(path))
    return {"lines": lines, "total": total}


@app.post("/retrieve-chunks")
async def retrieve_chunks(request: QuestionRequest) -> dict[str, list[str]]:
    """
    Retrieve top chunks from resume based on question keywords.
    Useful for debugging/inspecting the retrieval step.
    """
    chunks = _retrieve_resume_chunks(request.question, request.resume_text)
    return {"chunks": chunks}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
