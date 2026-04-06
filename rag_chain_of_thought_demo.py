"""
RAG "Chat with Your Documents" - Chain-of-Thought Prompting Demo
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from pdf_loader import load_resume

load_dotenv()

client = OpenAI(api_key=os.getenv("WYWaxOADEWQ9kNOHfI7oxLAVpbNrrw7FfF7rnX4jxoOyFK2KuTPRPkZljJtcTW6PMjHxbNpSrKT3BlbkFJ5sOMKOsv1lInOyktFzwApyvUheBrN3MNOzMwARA8qYQrjEeC_CveYRSrn7ciEH77AcG_J7BjEA"))

SAMPLE_DOCUMENT = load_resume()
TEST_QUESTION = "Walk me through the candidate's most recent work experience and key accomplishments."


def run_demo():
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.")
        return

    messages = [
        {
            "role": "system",
            "content": (
                "You are a document Q&A assistant. Answer ONLY from the document. "
                "Provide a short, structured rationale after the answer."
            ),
        },
        {
            "role": "user",
            "content": f"Document:\n{SAMPLE_DOCUMENT}\n\nQuestion: {TEST_QUESTION}",
        },
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
        max_tokens=300,
    )

    print("=" * 80)
    print("CHAIN-OF-THOUGHT PROMPTING DEMO (SHORT RATIONALE)")
    print("=" * 80)
    print(f"Question: {TEST_QUESTION}\n")
    print("Response:")
    print(response.choices[0].message.content)
    print("\nNote: Rationale is brief and grounded in document text.")


if __name__ == "__main__":
    run_demo()
