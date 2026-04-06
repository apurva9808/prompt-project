"""
RAG "Chat with Your Documents" - Auto-CoT Prompting Demo
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from pdf_loader import load_resume

load_dotenv()

client = OpenAI(api_key=os.getenv("WYWaxOADEWQ9kNOHfI7oxLAVpbNrrw7FfF7rnX4jxoOyFK2KuTPRPkZljJtcTW6PMjHxbNpSrKT3BlbkFJ5sOMKOsv1lInOyktFzwApyvUheBrN3MNOzMwARA8qYQrjEeC_CveYRSrn7ciEH77AcG_J7BjEA"))

SAMPLE_DOCUMENT = load_resume()
TEST_QUESTION = "What programming languages and technologies does the candidate know?"


def run_demo():
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.")
        return

    # Step 1: Auto-generate examples with brief reasoning
    generate_messages = [
        {
            "role": "system",
            "content": "You are a document Q&A assistant.",
        },
        {
            "role": "user",
            "content": (
                "Create 3 example Q&A pairs about a document, each with a short "
                "rationale grounded in the document."
            ),
        },
    ]

    examples = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=generate_messages,
        temperature=0.7,
        max_tokens=400,
    ).choices[0].message.content

    # Step 2: Use generated examples to answer the real question
    answer_messages = [
        {
            "role": "system",
            "content": (
                "You are a document Q&A assistant. Answer ONLY from the document. "
                "Use the examples as guidance for structure and grounding."
            ),
        },
        {
            "role": "user",
            "content": f"Examples:\n{examples}\n\nDocument:\n{SAMPLE_DOCUMENT}\n\nQuestion: {TEST_QUESTION}",
        },
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=answer_messages,
        temperature=0.7,
        max_tokens=300,
    )

    print("=" * 80)
    print("AUTO-COT PROMPTING DEMO")
    print("=" * 80)
    print("Generated Examples:")
    print(examples)
    print("\nQuestion:")
    print(TEST_QUESTION)
    print("\nResponse:")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    run_demo()
