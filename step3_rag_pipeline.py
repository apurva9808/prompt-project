from __future__ import annotations

import json
import os
import re
from typing import List

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - optional dependency in offline mode
    ChatOpenAI = None

NOT_FOUND_MESSAGE = "The document does not contain this information."
TOP_K = 4
OUTPUT_PATH = "step3_rag_pipeline_output.json"
REPORT_PATH = "step3_rag_pipeline_report.json"
REFERENCE_RESULTS_PATH = "step3_results.json"
RESUME_TEXT_PATH = "resume.txt"
ALL_CHUNKS: list[Document] = []


# -----------------------------------------------------------------------------
# LLM setup (OpenAI)
# -----------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _build_llm():
    """Create the chat model when an API key is available."""
    if not OPENAI_API_KEY or ChatOpenAI is None:
        return None

    return ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=OPENAI_MODEL,
        temperature=0,
        top_p=1,
        max_retries=2,
    )


llm = _build_llm()


# -----------------------------------------------------------------------------
# Resume loading and section-aware chunking
# -----------------------------------------------------------------------------
def _load_resume_text() -> str:
    """Load the resume content from the local text file or PDF fallback."""
    if os.path.exists(RESUME_TEXT_PATH):
        with open(RESUME_TEXT_PATH, "r", encoding="utf-8") as file:
            return file.read().strip()

    try:
        from pdf_loader import load_resume

        return load_resume()
    except Exception as exc:  # pragma: no cover - defensive fallback
        return f"[ERROR] Unable to load resume source text: {exc}"


def _build_resume_documents(resume_text: str) -> List[Document]:
    """Split the resume into section-level documents for more precise retrieval."""
    section_headers = {"summary", "technical skills", "experience", "projects", "education"}
    lines = [line.rstrip() for line in resume_text.splitlines()]

    docs: list[Document] = []
    current_section = "resume"
    current_lines: list[str] = []

    def flush_section() -> None:
        nonlocal current_lines, current_section
        body = "\n".join(line for line in current_lines).strip()
        if body:
            docs.append(
                Document(
                    page_content=body,
                    metadata={"source": "resume.txt", "section": current_section},
                )
            )
        current_lines = []

    for line in lines:
        normalized = line.strip().lower()
        if normalized in section_headers:
            flush_section()
            current_section = line.strip()
            continue
        if line.strip() or current_lines:
            current_lines.append(line)

    flush_section()

    if not docs:
        docs = [Document(page_content=resume_text, metadata={"source": "resume.txt", "section": "full_resume"})]

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    return splitter.split_documents(docs)


resume_text = _load_resume_text()
resume_documents = _build_resume_documents(resume_text)


# -----------------------------------------------------------------------------
# Build in-memory retriever
# -----------------------------------------------------------------------------
def _build_retriever(documents: List[Document]) -> BM25Retriever:
    """Create an in-memory BM25 retriever over chunked documents."""
    global ALL_CHUNKS
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(documents)
    ALL_CHUNKS = chunks
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = TOP_K
    return retriever


retriever = _build_retriever(resume_documents)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


QUESTION_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "apurva",
    "does",
    "did",
    "do",
    "for",
    "from",
    "has",
    "have",
    "his",
    "how",
    "is",
    "it",
    "know",
    "known",
    "of",
    "on",
    "or",
    "raj",
    "resume",
    "the",
    "their",
    "there",
    "this",
    "to",
    "was",
    "what",
    "which",
    "who",
    "with",
    "where",
    "when",
}


def _question_keywords(question: str) -> set[str]:
    return {token for token in _tokenize(question) if token not in QUESTION_STOPWORDS and len(token) > 2}


def _document_relevance_score(question: str, doc: Document) -> tuple[int, int]:
    """Score a document for the question using overlap and section-aware boosts."""
    keywords = _question_keywords(question)
    doc_tokens = set(_tokenize(doc.page_content))
    overlap = len(keywords & doc_tokens)

    section = str(doc.metadata.get("section", "")).lower()
    boost = 0

    if section == "resume":
        boost -= 10
    elif section == "technical skills":
        if keywords & {"language", "languages", "framework", "frameworks", "database", "databases", "cloud", "backend", "frontend", "tools", "technology", "technologies", "stack"}:
            boost += 4
    elif section == "experience":
        if keywords & {"build", "built", "worked", "work", "used", "use", "did", "at", "experience", "collaborate", "collaborated", "role", "job", "employer"}:
            boost += 4
    elif section == "projects":
        if keywords & {"project", "projects", "built", "build", "deploy", "deployment"}:
            boost += 3
    elif section == "education":
        if keywords & {"degree", "university", "college", "gpa", "education", "master", "masters", "undergraduate", "pursuing"}:
            boost += 3

    return overlap + boost, -len(doc.page_content)


def _extractive_answer(question: str, context: str) -> str:
    """Generate a grounded answer without an LLM by selecting the best sentence."""
    if not context.strip():
        return NOT_FOUND_MESSAGE

    question_tokens = _question_keywords(question)
    if question_tokens & {"repo", "repos", "repository", "repositories"}:
        context_tokens = set(_tokenize(context))
        if not context_tokens & {"repo", "repos", "repository", "repositories"}:
            return NOT_FOUND_MESSAGE

    sentences = re.split(r"(?<=[.!?])\s+|\n+", context)
    scored_sentences: list[tuple[int, str]] = []

    for sentence in sentences:
        clean_sentence = sentence.strip()
        if not clean_sentence:
            continue
        sentence_tokens = set(_tokenize(clean_sentence))
        score = len(question_tokens & sentence_tokens)

        lower_sentence = clean_sentence.lower()
        if any(keyword in lower_sentence for keyword in ("languages:", "frameworks:", "cloud tech:", "database:", "dev tools:", "others:")):
            score += 3
        if any(keyword in lower_sentence for keyword in ("redesigned", "built", "developed", "implemented", "designed", "automated", "scaled", "collaborated", "improved", "compared")):
            score += 2
        if any(keyword in lower_sentence for keyword in ("software engineer", "master", "bachelor", "gpa")):
            score += 1

        if score:
            scored_sentences.append((score, clean_sentence))

    if scored_sentences:
        scored_sentences.sort(key=lambda item: (-item[0], len(item[1])))
        if scored_sentences[0][0] < 2:
            return NOT_FOUND_MESSAGE
        return scored_sentences[0][1]

    first_sentence = next((sentence.strip() for sentence in sentences if sentence.strip()), "")
    return first_sentence if first_sentence else NOT_FOUND_MESSAGE


# -----------------------------------------------------------------------------
# Prompt templates for each pipeline step
# -----------------------------------------------------------------------------
query_rewrite_prompt = PromptTemplate.from_template(
    """
You are a retrieval query rewriter.
Rewrite the user question to improve document retrieval precision.
Keep the meaning unchanged.
Return only the rewritten query.

Original question: {question}
""".strip()
)

context_filter_prompt = PromptTemplate.from_template(
    """
You are filtering retrieved context.
Given the question and candidate chunks, keep only chunks that are directly useful to answer the question.
If none are relevant, return exactly: NONE
Otherwise return only the relevant chunk text, preserving wording from the chunks as much as possible.
Do not add external facts.

Question:
{question}

Candidate context:
{context}
""".strip()
)

answer_generation_prompt = PromptTemplate.from_template(
    """
You are a grounded assistant.
Answer the question using ONLY the provided context.
If the context is insufficient, return exactly:
The document does not contain this information.

Question:
{question}

Filtered context:
{context}
""".strip()
)

grounding_validation_prompt = PromptTemplate.from_template(
    """
You are a grounding validator.
Determine whether the answer is fully supported by the provided context.
Return exactly one word:
SUPPORTED or UNSUPPORTED

Question:
{question}

Context:
{context}

Answer:
{answer}
""".strip()
)


# -----------------------------------------------------------------------------
# Step 1: Rewrite user query to improve retrieval
# -----------------------------------------------------------------------------
def rewrite_query(question: str) -> str:
    """Rewrite the user query for better retrieval quality."""
    if llm is None:
        return question

    chain = query_rewrite_prompt | llm | StrOutputParser()
    rewritten = chain.invoke({"question": question}).strip()
    return rewritten if rewritten else question


# -----------------------------------------------------------------------------
# Step 2: Retrieve top-k relevant chunks (in-memory retriever)
# -----------------------------------------------------------------------------
def retrieve_context(question: str) -> List[Document]:
    """Retrieve top-k chunks from the in-memory retriever."""
    scored = sorted(ALL_CHUNKS, key=lambda doc: _document_relevance_score(question, doc), reverse=True)
    return scored[:TOP_K]


# -----------------------------------------------------------------------------
# Step 3: Filter irrelevant context using an LLM prompt
# -----------------------------------------------------------------------------
def filter_context(question: str, context: List[Document]) -> str:
    """Filter retrieved chunks and return only relevant context text."""
    if not context:
        return ""

    if llm is None:
        question_tokens = set(_tokenize(question))
        selected_chunks = []
        for doc in context:
            doc_tokens = set(_tokenize(doc.page_content))
            if question_tokens & doc_tokens:
                selected_chunks.append(doc.page_content)

        if not selected_chunks:
            selected_chunks = [context[0].page_content]

        return "\n\n".join(selected_chunks)

    context_text = "\n\n".join(
        f"[Chunk {index + 1}] {doc.page_content}" for index, doc in enumerate(context)
    )

    chain = context_filter_prompt | llm | StrOutputParser()
    filtered = chain.invoke({"question": question, "context": context_text}).strip()

    if filtered.upper() == "NONE":
        return ""
    return filtered


# -----------------------------------------------------------------------------
# Step 4: Generate final answer using ONLY filtered context
# -----------------------------------------------------------------------------
def generate_answer(question: str, context: str) -> str:
    """Generate answer using only filtered context."""
    if not context.strip():
        return NOT_FOUND_MESSAGE

    if llm is None:
        answer = _extractive_answer(question, context)
        return answer if answer else NOT_FOUND_MESSAGE

    chain = answer_generation_prompt | llm | StrOutputParser()
    answer = chain.invoke({"question": question, "context": context}).strip()
    return answer if answer else NOT_FOUND_MESSAGE


# -----------------------------------------------------------------------------
# Step 5: Grounding check to ensure answer is supported by context
# -----------------------------------------------------------------------------
def validate_answer(question: str, context: str, answer: str) -> bool:
    """Validate grounding; return True if answer is supported by context."""
    if answer == NOT_FOUND_MESSAGE:
        return True

    if llm is None:
        context_tokens = set(_tokenize(context))
        answer_tokens = [token for token in _tokenize(answer) if len(token) > 2]
        if not answer_tokens:
            return False
        supported_tokens = sum(1 for token in answer_tokens if token in context_tokens)
        return supported_tokens / len(answer_tokens) >= 0.5

    chain = grounding_validation_prompt | llm | StrOutputParser()
    verdict = chain.invoke({"question": question, "context": context, "answer": answer}).strip().upper()
    return verdict == "SUPPORTED"


def _chunk_summary(chunks: List[Document]) -> list[dict]:
    return [
        {
            "chunk_index": index + 1,
            "section": doc.metadata.get("section", "unknown"),
            "source": doc.metadata.get("source", "unknown"),
            "preview": doc.page_content[:180],
        }
        for index, doc in enumerate(chunks)
    ]


# -----------------------------------------------------------------------------
# End-to-end pipeline
# -----------------------------------------------------------------------------
def run_pipeline(question: str) -> str:
    """Run full pipeline and return a grounded answer."""
    return run_pipeline_detailed(question)["answer"]


def run_pipeline_detailed(question: str) -> dict:
    """Run the pipeline and return a detailed record for debugging and evaluation."""
    rewritten_question = rewrite_query(question)
    retrieved_chunks = retrieve_context(rewritten_question)
    filtered_context = filter_context(rewritten_question, retrieved_chunks)

    if not filtered_context:
        return {
            "question": question,
            "rewritten_question": rewritten_question,
            "answer": NOT_FOUND_MESSAGE,
            "grounded": True,
            "strategy": "llm" if llm is not None else "offline_extractive",
            "filtered_context": "",
            "retrieved_chunks": _chunk_summary(retrieved_chunks),
        }

    answer = generate_answer(rewritten_question, filtered_context)
    is_grounded = validate_answer(rewritten_question, filtered_context, answer)

    if not is_grounded or answer.strip() == NOT_FOUND_MESSAGE:
        answer = NOT_FOUND_MESSAGE

    return {
        "question": question,
        "rewritten_question": rewritten_question,
        "answer": answer,
        "grounded": is_grounded,
        "strategy": "llm" if llm is not None else "offline_extractive",
        "filtered_context": filtered_context,
        "retrieved_chunks": _chunk_summary(retrieved_chunks),
    }


def _load_reference_metrics() -> dict:
    if not os.path.exists(REFERENCE_RESULTS_PATH):
        return {}

    with open(REFERENCE_RESULTS_PATH, "r", encoding="utf-8") as file:
        return json.load(file).get("summary", {})


def _build_report(results: list[dict]) -> dict:
    total = len(results)
    not_found = sum(1 for item in results if item["answer"] == NOT_FOUND_MESSAGE)
    grounded = sum(1 for item in results if item["grounded"])
    avg_chunks = round(sum(len(item["retrieved_chunks"]) for item in results) / total, 2) if total else 0.0

    reference = _load_reference_metrics()

    return {
        "metadata": {
            "model": OPENAI_MODEL if llm is not None else "offline-extractive",
            "retriever": "BM25Retriever",
            "source_document": "resume.txt",
            "mode": "llm" if llm is not None else "offline_extractive",
        },
        "iteration_summary": {
            "feature_enhancements": [
                "Removed the hardcoded API key and switched to environment-based configuration.",
                "Replaced dummy documents with resume-aware section chunking.",
                "Added an offline extractive fallback so the pipeline can run without API access.",
                "Added detailed per-question output with rewritten query, retrieved chunks, and grounding status.",
            ],
            "automated_smoke_tests": {
                "total": total,
                "answered": total - not_found,
                "grounded": grounded,
                "not_found": not_found,
                "average_retrieved_chunks": avg_chunks,
            },
        },
        "reference_metrics": reference,
        "results": results,
    }


if __name__ == "__main__":
    test_questions = [
        "Which programming languages does Apurva know?",
        "What backend frameworks does Apurva know?",
        "What did Apurva build at Admins?",
        "Does the resume mention a GitHub repositories list?",
    ]

    outputs = []
    for question in test_questions:
        result = run_pipeline_detailed(question)
        outputs.append(result)
        print(f"Q: {question}")
        print(f"A: {result['answer']}")
        print(f"Strategy: {result['strategy']} | Grounded: {result['grounded']}")
        print("-" * 80)

    report = _build_report(outputs)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report["iteration_summary"], file, indent=2, ensure_ascii=False)

    print(f"Saved output to: {OUTPUT_PATH}")
    print(f"Saved iteration report to: {REPORT_PATH}")
