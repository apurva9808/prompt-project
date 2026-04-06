"""
RAG "Chat with Your Documents" - Step-Back Prompting Demo
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from pdf_loader import load_resume

load_dotenv()

client = OpenAI(api_key=os.getenv("WYWaxOADEWQ9kNOHfI7oxLAVpbNrrw7FfF7rnX4jxoOyFK2KuTPRPkZljJtcTW6PMjHxbNpSrKT3BlbkFJ5sOMKOsv1lInOyktFzwApyvUheBrN3MNOzMwARA8qYQrjEeC_CveYRSrn7ciEH77AcG_J7BjEA"))

SAMPLE_DOCUMENT = load_resume()
TEST_QUESTION = "What is the overall career trajectory and progression shown in this resume?"


def run_demo():
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.")
        return

    stepback_messages = [
        {
            "role": "system",
            "content": "You are a document Q&A assistant.",
        },
        {
            "role": "user",
            "content": (
                "Given the document, list 3 principles for answering questions "
                "without hallucinations."
            ),
        },
    ]

    principles = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=stepback_messages,
        temperature=0.3,
        max_tokens=200,
    ).choices[0].message.content

    answer_messages = [
        {
            "role": "system",
            "content": (
                "You are a document Q&A assistant. Follow these principles:\n"
                f"{principles}\nAnswer ONLY from the document."
            ),
        },
        {
            "role": "user",
            "content": f"Document:\n{SAMPLE_DOCUMENT}\n\nQuestion: {TEST_QUESTION}",
        },
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=answer_messages,
        temperature=0.5,
        max_tokens=300,
    )

    print("=" * 80)
    print("STEP-BACK PROMPTING DEMO")
    print("=" * 80)
    print("Principles:")
    print(principles)
    print("\nQuestion:")
    print(TEST_QUESTION)
    print("\nResponse:")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    run_demo()
