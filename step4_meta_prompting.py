"""
=============================================================================
ASSIGNMENT: Prompt Engineering for RAG Systems
SECTION:    Step 4 — Apply Meta Prompting and Evaluate Using Perplexity
SYSTEM:     "Chat with Your Documents" (resume.pdf)
STUDENT:    Apurva Raj
=============================================================================

This script:
1. Uses a language model to critique and improve a baseline system prompt.
2. Evaluates baseline vs meta-improved prompt on the held-out test split.
3. Measures answer quality with accuracy and hallucination rate.
4. Estimates generation confidence using token-level perplexity.
5. Writes results to step4_meta_results.json.
"""

from __future__ import annotations

import json
import math
import os
from statistics import mean
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from pdf_loader import load_resume

load_dotenv()

MODEL = "gpt-4o-mini"
META_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o-mini"
RESULTS_PATH = "step4_meta_results.json"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
RESUME_TEXT = load_resume()

BASELINE_PROMPT = (
    "You are a resume Q&A assistant. Answer ONLY from the provided resume. "
    "If the information is missing, say: 'This information is not available in the provided resume.'"
)

META_PROMPT = """You are an expert prompt engineer for a Retrieval-Augmented Generation and document-question-answering system.

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

JUDGE_PROMPT = (
    "You are evaluating answers from a resume QA system. "
    "Return strict JSON with keys: correct (true/false), hallucinated (true/false), rationale (string). "
    "For adversarial or NOT_IN_DOCUMENT items, the answer is correct only if it clearly abstains. "
    "Mark hallucinated=true if the answer includes unsupported facts, fabricated details, or answers an unrelated question."
)


def load_dataset() -> dict[str, Any]:
    with open("rag_eval_dataset.json", "r", encoding="utf-8") as f:
        return json.load(f)



def generate_meta_improved_prompt() -> dict[str, str]:
    response = client.chat.completions.create(
        model=META_MODEL,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": META_PROMPT.format(baseline_prompt=BASELINE_PROMPT),
            }
        ],
    )
    content = response.choices[0].message.content.strip()
    result = json.loads(content)
    return {
        "critique": result["critique"],
        "improved_prompt": result["improved_prompt"],
        "rationale": result["rationale"],
    }



def answer_with_prompt(system_prompt: str, query: str) -> tuple[str, float | None]:
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.2,
        max_tokens=180,
        logprobs=True,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Resume:\n{RESUME_TEXT}\n\nQuestion: {query}"},
        ],
    )
    message = response.choices[0].message.content.strip()

    # Compute perplexity from returned token logprobs when available.
    perplexity = None
    try:
        content_items = response.choices[0].logprobs.content or []
        token_logprobs = [item.logprob for item in content_items if hasattr(item, "logprob")]
        if token_logprobs:
            perplexity = math.exp(-sum(token_logprobs) / len(token_logprobs))
    except Exception:
        perplexity = None

    return message, perplexity



def judge_answer(item: dict[str, Any], system_name: str, answer: str) -> dict[str, Any]:
    payload = {
        "query": item["query"],
        "ground_truth": item["ground_truth"],
        "category": item["category"],
        "system": system_name,
        "answer": answer,
    }
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0,
        max_tokens=150,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    try:
        return json.loads(response.choices[0].message.content.strip())
    except Exception:
        return {"correct": False, "hallucinated": True, "rationale": "Judge parsing failed."}



def evaluate_prompt(prompt_name: str, system_prompt: str, test_items: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    correct_count = 0
    hallucinated_count = 0
    perplexities: list[float] = []

    for item in test_items:
        answer, perplexity = answer_with_prompt(system_prompt, item["query"])
        judgment = judge_answer(item, prompt_name, answer)

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



def main() -> None:
    if RESUME_TEXT.startswith("[ERROR]"):
        raise SystemExit(RESUME_TEXT)
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set.")

    dataset = load_dataset()
    test_items = dataset["splits"]["test"]

    meta = generate_meta_improved_prompt()
    baseline_eval = evaluate_prompt("baseline_prompt", BASELINE_PROMPT, test_items)
    improved_eval = evaluate_prompt("meta_improved_prompt", meta["improved_prompt"], test_items)

    results = {
        "metadata": {
            "model": MODEL,
            "meta_model": META_MODEL,
            "judge_model": JUDGE_MODEL,
            "document": "resume.pdf",
            "test_items": len(test_items),
        },
        "baseline_prompt": BASELINE_PROMPT,
        "meta_prompt": META_PROMPT.format(baseline_prompt=BASELINE_PROMPT),
        "meta_output": meta,
        "before": baseline_eval,
        "after": improved_eval,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    print("STEP 4 RUN COMPLETE")
    print("=" * 72)
    print(f"Results written      : {RESULTS_PATH}")
    print("\nComparison:")
    print(
        "  Before meta prompting -> "
        f"perplexity={baseline_eval['summary']['average_perplexity']} | "
        f"accuracy={baseline_eval['summary']['accuracy']}% | "
        f"hallucination={baseline_eval['summary']['hallucination_rate']}%"
    )
    print(
        "  After meta prompting  -> "
        f"perplexity={improved_eval['summary']['average_perplexity']} | "
        f"accuracy={improved_eval['summary']['accuracy']}% | "
        f"hallucination={improved_eval['summary']['hallucination_rate']}%"
    )


if __name__ == "__main__":
    main()
