# Document Chatbot - New Architecture

## Overview

The Document Chatbot is now built with a **client-server architecture**:

- **Backend (FastAPI):** `api.py` on port 8000
  - RAG pipeline for retrieving and answering questions
  - Resume upload handling
  - Dataset exploration
  - Fine-tuning artifact serving
  - Evaluation results retrieval

- **Frontend (Streamlit):** `streamlit_app.py` on port 8503
  - User authentication
  - Upload-first chat flow (upload mandatory before chatting)
  - Resume chat interface
  - Dataset explorer
  - Evaluation results visualization
  - Fine-tuning artifacts

## How It Works

### Upload-First Flow

1. **Login:** User authenticates with demo credentials (apurva / resume123)
2. **Upload:** User uploads a resume (PDF, TXT, or MD) via Streamlit UI
3. **Parse:** FastAPI endpoint `POST /upload-resume` parses the file
4. **Chat:** User asks questions; Streamlit calls `POST /answer` on FastAPI backend
5. **Retrieve:** FastAPI uses BM25-like keyword matching to retrieve relevant resume chunks
6. **Answer:** FastAPI calls OpenAI API (if key is set) or uses offline extraction fallback
7. **Display:** Streamlit shows answer and retrieved chunks

### Architecture Benefits

- **Separation of concerns:** Backend logic isolated from UI
- **Scalability:** FastAPI can be deployed separately, scaled horizontally
- **Reusability:** Backend endpoints can serve multiple frontends (web, mobile, CLI)
- **Testing:** API endpoints can be tested independently
- **Integration:** Existing Python backend files (step1-4 RAG pipeline) used directly

## Running the Application

### 1. Start FastAPI Backend

```bash
cd /Users/apurvaraj/Desktop/prompt-lab
source .venv-1/bin/activate
python api.py
```

FastAPI server will start on `http://localhost:8000`

### 2. Start Streamlit Frontend

In another terminal:

```bash
cd /Users/apurvaraj/Desktop/prompt-lab
source .venv-1/bin/activate
streamlit run streamlit_app.py --server.port 8503
```

Streamlit app will start on `http://localhost:8503`

### 3. Access the Application

Open browser to `http://localhost:8503`

Login with:
- **Username:** apurva
- **Password:** resume123

## API Endpoints

### POST `/answer`
Answer a question about the provided resume using RAG pipeline.

**Request:**
```json
{
  "question": "What are Apurva's main skills?",
  "resume_text": "..."
}
```

**Response:**
```json
{
  "question": "What are Apurva's main skills?",
  "answer": "...",
  "retrieved_chunks": ["...", "..."],
  "source": "uploaded_resume",
  "grounded": true
}
```

### POST `/upload-resume`
Upload and parse a resume file (PDF, TXT, MD).

**Request:** Multipart form data with `file` field

**Response:**
```json
{
  "text": "extracted resume text",
  "filename": "resume.pdf",
  "error": null
}
```

### GET `/default-resume`
Get the default project resume.

### GET `/dataset`
Get the 45-item evaluation dataset (30 typical + 10 edge + 5 adversarial).

### GET `/step3-results`
Get Step 3 RAG pipeline evaluation results.

### GET `/step4-results`
Get Step 4 meta-prompting evaluation results.

### GET `/finetune-preview?file_type=train&max_lines=5`
Get preview of fine-tuning JSONL files.

### POST `/retrieve-chunks`
Retrieve top chunks from resume based on question keywords (debugging endpoint).

## Configuration

Set environment variables:

```bash
export OPENAI_API_KEY="sk-..."              # For API-powered answers
export API_URL="http://localhost:8000"       # FastAPI endpoint (used by Streamlit)
export OPENAI_MODEL="gpt-4o-mini"           # OpenAI model to use
```

## Features

✅ **Upload Support:** PDF, TXT, Markdown  
✅ **RAG Pipeline:** BM25 keyword retrieval + LLM/offline extraction  
✅ **Offline Mode:** Works without API key (uses extraction fallback)  
✅ **Dataset Explorer:** Browse 45-item evaluation dataset  
✅ **Evaluation Dashboard:** View Step 3 & Step 4 results  
✅ **Fine-tuning Artifacts:** Download training/dev JSONL files  
✅ **Session-based Auth:** Simple login system  
✅ **Grounded Answers:** Answers validated against retrieved chunks  

## Backend Integration

The FastAPI backend integrates with existing Python files:

- `step3_rag_pipeline.py` - Core RAG pipeline
- `step2_dataset_generation.py` - Dataset curation
- `step1_prompt_sensitivity.py` - Prompt testing
- `step4_meta_prompting.py` - Meta-prompt evaluation
- `pdf_loader.py` - Resume parsing utilities

All evaluation results, datasets, and fine-tuning files are served via API endpoints.
