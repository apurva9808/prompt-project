"""
RAG-Based "Chat with Your Documents" - Few-Shot Prompting
============================================================
Project: Designing system prompts for a RAG application where users upload 
documents and ask questions. The AI must answer ONLY from document content.

Technique: Few-Shot Prompting
Goal: Guide the model to provide accurate, grounded responses and avoid hallucinations
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from pdf_loader import load_resume

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load the candidate's resume as the RAG document
RESUME_DOCUMENT = load_resume()


def baseline_without_fewshot(document_content, user_question):
    """Baseline: Without few-shot examples - may hallucinate"""
    print("\n" + "="*80)
    print("🔴 BASELINE: Without Few-Shot Examples")
    print("="*80)
    print(f"\nDocument Content:\n{document_content[:200]}...\n")
    print(f"User Question: {user_question}\n")
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system", 
                "content": "You are a helpful assistant for a document Q&A system."
            },
            {
                "role": "user", 
                "content": f"Document:\n{document_content}\n\nQuestion: {user_question}"
            }
        ],
        temperature=0.7,
        max_tokens=300
    )
    
    answer = response.choices[0].message.content
    print(f"AI Response:\n{answer}\n")
    print("⚠️  Risk: May add information not in the document or make assumptions")
    return answer


def fewshot_grounded_responses(document_content, user_question):
    """Few-Shot: With examples showing how to stay grounded in documents"""
    print("\n" + "="*80)
    print("✅ FEW-SHOT PROMPTING: With Grounded Response Examples")
    print("="*80)
    print(f"\nDocument Content:\n{document_content[:200]}...\n")
    print(f"User Question: {user_question}\n")
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system", 
                "content": """You are a document Q&A assistant. Answer questions using ONLY information from the provided document. 
If the answer is not in the document, say so clearly."""
            },
            # Example 1: Answering from document
            {
                "role": "user",
                "content": """Document: "Annual leave is 15 days per year. Sick leave is 10 days per year."
Question: How many days of annual leave do employees get?"""
            },
            {
                "role": "assistant",
                "content": "According to the document, employees get 15 days of annual leave per year."
            },
            # Example 2: Information not in document
            {
                "role": "user",
                "content": """Document: "Annual leave is 15 days per year. Sick leave is 10 days per year."
Question: What is the policy for bereavement leave?"""
            },
            {
                "role": "assistant",
                "content": "The document does not contain information about bereavement leave policy. I can only answer based on what's provided in the document."
            },
            # Example 3: Combining information from document
            {
                "role": "user",
                "content": """Document: "Remote work up to 2 days per week with manager approval. Full remote requires VP approval."
Question: Can I work from home?"""
            },
            {
                "role": "assistant",
                "content": "Yes, according to the document, you can work remotely up to 2 days per week with manager approval. If you need a full remote work arrangement, you would need VP approval."
            },
            # Actual question
            {
                "role": "user",
                "content": f"Document:\n{document_content}\n\nQuestion: {user_question}"
            }
        ],
        temperature=0.7,
        max_tokens=300
    )
    
    answer = response.choices[0].message.content
    print(f"AI Response:\n{answer}\n")
    print("✅ Benefits: Stays grounded, cites document, admits when info is missing")
    return answer


def fewshot_with_citations(document_content, user_question):
    """Few-Shot: Teaching the model to provide citations"""
    print("\n" + "="*80)
    print("📚 FEW-SHOT WITH CITATIONS: Teaching the model to quote sources")
    print("="*80)
    print(f"\nDocument Content:\n{document_content[:200]}...\n")
    print(f"User Question: {user_question}\n")
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system", 
                "content": """You are a document Q&A assistant. Answer questions using ONLY information from the provided document.
Always quote relevant parts of the document to support your answer."""
            },
            # Example 1: With direct quote
            {
                "role": "user",
                "content": """Document: "Employees receive 10 days of paid sick leave annually."
Question: How much sick leave do employees get?"""
            },
            {
                "role": "assistant",
                "content": """According to the document: "Employees receive 10 days of paid sick leave annually."

This means you get 10 paid sick days per year."""
            },
            # Example 2: Quoting with explanation
            {
                "role": "user",
                "content": """Document: "A medical certificate is required for absences exceeding 3 consecutive days."
Question: Do I need a doctor's note for sick leave?"""
            },
            {
                "role": "assistant",
                "content": """The document states: "A medical certificate is required for absences exceeding 3 consecutive days."

This means:
- No doctor's note needed for 1-3 days of sick leave
- Medical certificate required if you're sick for more than 3 days in a row"""
            },
            # Actual question
            {
                "role": "user",
                "content": f"Document:\n{document_content}\n\nQuestion: {user_question}"
            }
        ],
        temperature=0.7,
        max_tokens=300
    )
    
    answer = response.choices[0].message.content
    print(f"AI Response:\n{answer}\n")
    print("✅ Benefits: Provides evidence, increases trust, prevents hallucinations")
    return answer


def run_comparison_tests():
    """Run comparative tests showing few-shot prompting improvements"""
    
    print("\n")
    print("╔" + "═"*78 + "╗")
    print("║" + " "*15 + "RAG-BASED CHAT WITH YOUR DOCUMENTS" + " "*28 + "║")
    print("║" + " "*22 + "Few-Shot Prompting Demonstration" + " "*23 + "║")
    print("╚" + "═"*78 + "╝")
    print("\n📋 PROJECT GOAL: Answer user questions based ONLY on document content")
    print("🎯 CHALLENGE: Prevent hallucinations and ensure grounded responses")
    print("🔧 TECHNIQUE: Few-Shot Prompting with grounded examples\n")
    
    # Test Case 1: Company Policy
    print("\n" + "█"*80)
    print("TEST CASE 1: Company Policy Document")
    print("█"*80)
    
    question1 = "What are the candidate's main technical skills and tools?"
    
    baseline_without_fewshot(RESUME_DOCUMENT, question1)
    fewshot_grounded_responses(RESUME_DOCUMENT, question1)
    
    # Test Case 2: Information NOT in document
    print("\n" + "█"*80)
    print("TEST CASE 2: Handling Questions Beyond Document Scope")
    print("█"*80)
    
    question2 = "What is the candidate's expected salary range?"
    
    baseline_without_fewshot(RESUME_DOCUMENT, question2)
    fewshot_grounded_responses(RESUME_DOCUMENT, question2)
    
    # Test Case 3: Resume Details with Citations
    print("\n" + "█"*80)
    print("TEST CASE 3: Work Experience with Citations")
    print("█"*80)
    
    question3 = "What was the candidate's most recent role and what did they accomplish?"
    
    baseline_without_fewshot(RESUME_DOCUMENT, question3)
    fewshot_with_citations(RESUME_DOCUMENT, question3)
    
    # Test Case 4: Complex multi-part question
    print("\n" + "█"*80)
    print("TEST CASE 4: Complex Question Requiring Multiple Resume Sections")
    print("█"*80)
    
    question4 = "What education and certifications does the candidate have, and how do they relate to their work experience?"
    
    baseline_without_fewshot(RESUME_DOCUMENT, question4)
    fewshot_with_citations(RESUME_DOCUMENT, question4)


def print_summary():
    """Print summary of findings"""
    print("\n" + "="*80)
    print("📊 COMPARISON SUMMARY: Impact of Few-Shot Prompting on RAG Systems")
    print("="*80)
    print("""
┌──────────────────────┬─────────────────────┬────────────────────────┐
│ Aspect               │ Without Few-Shot    │ With Few-Shot Examples │
├──────────────────────┼─────────────────────┼────────────────────────┤
│ Grounding            │ May add extra info  │ Stays within document  │
│ Hallucinations       │ Higher risk         │ Significantly reduced  │
│ Missing Info         │ May guess/assume    │ Clearly states "not in │
│                      │                     │ document"              │
│ Citations            │ Rare                │ Provides quotes        │
│ User Trust           │ Lower               │ Higher                 │
│ Accuracy             │ Variable            │ More consistent        │
└──────────────────────┴─────────────────────┴────────────────────────┘
""")
    
    print("\n" + "="*80)
    print("🎓 KEY LEARNINGS FOR RAG SYSTEM PROMPTS")
    print("="*80)
    print("""
1. ✅ Few-shot examples teach the model to stay grounded
   - Show examples of answering FROM the document
   - Show examples of saying "not in document" when appropriate
   
2. ✅ Examples should demonstrate the desired behavior
   - Direct quotes from document
   - Clear attribution
   - Appropriate handling of missing information
   
3. ✅ Few-shot helps prevent common RAG problems
   - Hallucinations (making up information)
   - Assumptions beyond document content
   - Over-confident responses to unknown questions
   
4. ✅ Including citation examples improves transparency
   - Users can verify the response
   - Increases trust in the system
   - Makes it easier to trace answers back to source
   
5. 🎯 Best Practice Pattern for RAG Few-Shot:
   - Example 1: Simple fact from document
   - Example 2: Information NOT in document (show refusal)
   - Example 3: Complex answer requiring synthesis
   - All examples include direct quotes when possible
""")
    
    print("\n" + "="*80)
    print("💡 RECOMMENDED SYSTEM PROMPT STRUCTURE FOR RAG")
    print("="*80)
    print("""
System Prompt Components:
1. Role definition: "You are a document Q&A assistant"
2. Constraint: "Answer ONLY from provided document"
3. Missing info handling: "If not in document, say so clearly"
4. Citation requirement: "Quote relevant parts to support answers"
5. Few-shot examples: 3-5 examples showing desired behavior
""")
    
    print("\n" + "="*80)
    print("🚀 NEXT STEPS FOR YOUR PROJECT")
    print("="*80)
    print("""
1. Apply other techniques to the same RAG task:
   ✓ Zero-shot (completed - baseline)
   ✓ Few-shot (completed - this demo)
   ⏳ Chain-of-Thought (add reasoning to responses)
   ⏳ Step-back (identify principles for good RAG responses)
   ⏳ And 8 more foundational/advanced techniques
   
2. Document observations for each technique:
   - How does it affect grounding?
   - Does it reduce hallucinations?
   - How's the response quality?
   
3. Compare all techniques systematically
   
4. Design final system prompt incorporating best elements
""")
    
    print("\n" + "="*80 + "\n")


def main():
    """Run the complete demonstration"""
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY environment variable not set!")
        print("\nPlease set it using:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        return
    
    try:
        # Run all test cases
        run_comparison_tests()
        
        # Print summary
        print_summary()
        
        print("✅ Demo completed successfully!")
        print("\n📁 This demonstrates Few-Shot Prompting for your RAG project.")
        print("📝 Use these results in your assignment documentation.\n")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nMake sure:")
        print("1. OPENAI_API_KEY is set correctly")
        print("2. You have internet connection")
        print("3. Your OpenAI account has credits")


if __name__ == "__main__":
    main()
