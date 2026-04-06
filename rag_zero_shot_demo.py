"""
RAG "Chat with Your Documents" - Zero-Shot Prompting Demo
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from pdf_loader import load_resume

load_dotenv()

client = OpenAI(api_key=os.getenv("WYWaxOADEWQ9kNOHfI7oxLAVpbNrrw7FfF7rnX4jxoOyFK2KuTPRPkZljJtcTW6PMjHxbNpSrKT3BlbkFJ5sOMKOsv1lInOyktFzwApyvUheBrN3MNOzMwARA8qYQrjEeC_CveYRSrn7ciEH77AcG_J7BjEA"))

SAMPLE_DOCUMENT = load_resume()
TEST_QUESTION = "What are the candidate's main technical skills and areas of expertise?"


def run_demo():
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.")
        return

    messages = [
        {
            "role": "system",
            "content": "You are a document Q&A assistant. Answer the user's question.",
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
    print("ZERO-SHOT PROMPTING DEMO")
    print("=" * 80)
    print(f"Question: {TEST_QUESTION}\n")
    print("Response:")
    print(response.choices[0].message.content)
    print("\nNote: Zero-shot may answer correctly but can be less consistent.")


if __name__ == "__main__":
    run_demo()
