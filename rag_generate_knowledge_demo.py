"""
RAG "Chat with Your Documents" - Generate-Knowledge Prompting Demo
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from pdf_loader import load_resume

load_dotenv()

client = OpenAI(api_key=os.getenv("WYWaxOADEWQ9kNOHfI7oxLAVpbNrrw7FfF7rnX4jxoOyFK2KuTPRPkZljJtcTW6PMjHxbNpSrKT3BlbkFJ5sOMKOsv1lInOyktFzwApyvUheBrN3MNOzMwARA8qYQrjEeC_CveYRSrn7ciEH77AcG_J7BjEA"))

SAMPLE_DOCUMENT = load_resume()
TEST_QUESTION = "What educational background and relevant certifications does the candidate have?"


def run_demo():
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.")
        return

    # Step 1: Generate relevant knowledge from the document
    knowledge_messages = [
        {
            "role": "system",
            "content": "You are a document Q&A assistant.",
        },
        {
            "role": "user",
            "content": (
                "From the document, extract only the facts needed to answer this question:\n"
                f"{TEST_QUESTION}\nDocument:\n{SAMPLE_DOCUMENT}"
            ),
        },
    ]

    knowledge = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=knowledge_messages,
        temperature=0.3,
        max_tokens=200,
    ).choices[0].message.content

    # Step 2: Answer using only extracted knowledge
    answer_messages = [
        {
            "role": "system",
            "content": (
                "You are a document Q&A assistant. Answer ONLY using the extracted facts."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Extracted facts:\n{knowledge}\n\nQuestion: {TEST_QUESTION}"
            ),
        },
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=answer_messages,
        temperature=0.5,
        max_tokens=300,
    )

    print("=" * 80)
    print("GENERATE-KNOWLEDGE PROMPTING DEMO")
    print("=" * 80)
    print("Extracted Facts:")
    print(knowledge)
    print("\nQuestion:")
    print(TEST_QUESTION)
    print("\nResponse:")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    run_demo()
