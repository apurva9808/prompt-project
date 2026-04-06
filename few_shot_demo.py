"""
Few-Shot Learning Demonstration
This script demonstrates few-shot learning with automated test cases
"""

import os
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def test_even_odd_numbers():
    """Example 1: Teaching AI to recognize even and odd numbers with explanations"""
    print("="*80)
    print("EXAMPLE 1: Even/Odd Number Recognition with Few-Shot Learning")
    print("="*80)
    print("\nFew-shot examples provided to the model:")
    print("  User: 1 → Assistant: '1 is an odd number because it is not divisible by 2.'")
    print("  User: 2 → Assistant: '2 is an even number because it is divisible by 2.'")
    print("  User: 3 → Assistant: '3 is an odd number because it is not divisible by 2.'")
    print("\nNow testing with new numbers:\n")
    
    test_numbers = ["4", "7", "15", "100"]
    
    for number in test_numbers:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a smart and helpful assistant."},
                {"role": "user", "content": "1"},
                {"role": "assistant", "content": "1 is an odd number because it is not divisible by 2."},
                {"role": "user", "content": "2"},
                {"role": "assistant", "content": "2 is an even number because it is divisible by 2."},
                {"role": "user", "content": "3"},
                {"role": "assistant", "content": "3 is an odd number because it is not divisible by 2."},
                {"role": "user", "content": number}
            ],
            temperature=0.7,
            max_tokens=256
        )
        
        print(f"User: {number}")
        print(f"Felix: {response.choices[0].message.content}\n")


def test_animal_classification():
    """Example 2: Teaching AI to classify animals as mammals or not"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Animal Classification with Few-Shot Learning")
    print("="*80)
    print("\nFew-shot examples provided to the model:")
    print("  User: 'dog' → Assistant: 'A dog is a mammal because it gives birth to live young and has fur.'")
    print("  User: 'fish' → Assistant: 'A fish is not a mammal because it lays eggs and lives in water.'")
    print("\nNow testing with new animals:\n")
    
    test_animals = ["cat", "snake", "whale", "eagle"]
    
    for animal in test_animals:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a smart and helpful assistant."},
                {"role": "user", "content": "dog"},
                {"role": "assistant", "content": "A dog is a mammal because it gives birth to live young and has fur."},
                {"role": "user", "content": "fish"},
                {"role": "assistant", "content": "A fish is not a mammal because it lays eggs and lives in water."},
                {"role": "user", "content": animal}
            ],
            temperature=0.7,
            max_tokens=256
        )
        
        print(f"User: {animal}")
        print(f"Felix: {response.choices[0].message.content}\n")


def test_limitation_without_reasoning():
    """Example 3: Limitation - Few-shot without proper reasoning"""
    print("\n" + "="*80)
    print("EXAMPLE 3: LIMITATION - Few-Shot Without Reasoning")
    print("="*80)
    print("\nWe want AI to answer 'X' for odd numbers and 'Y' for even numbers.")
    print("\nFew-shot examples WITHOUT reasoning:")
    print("  User: '1' → Assistant: 'X'")
    print("  User: '2' → Assistant: 'Y'")
    print("\nProblem: The pattern is too abstract without explanation!")
    print("Let's test:\n")
    
    test_numbers = ["3", "4", "5"]
    
    for number in test_numbers:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a smart and helpful assistant."},
                {"role": "user", "content": "1"},
                {"role": "assistant", "content": "X"},
                {"role": "user", "content": "2"},
                {"role": "assistant", "content": "Y"},
                {"role": "user", "content": number}
            ],
            temperature=0.7,
            max_tokens=256
        )
        
        print(f"User: {number}")
        print(f"Felix: {response.choices[0].message.content}")
        print("  ⚠️  May not be consistent - pattern is too vague!\n")


def test_solution_with_reasoning():
    """Example 4: Solution - Using Chain of Thought reasoning in few-shot"""
    print("\n" + "="*80)
    print("EXAMPLE 4: SOLUTION - Few-Shot WITH Reasoning (Chain of Thought)")
    print("="*80)
    print("\nNow with reasoning in the examples:")
    print("  User: '1' → Assistant: '1 is an odd number so the answer is X.'")
    print("  User: '2' → Assistant: '2 is an even number so the answer is Y.'")
    print("  User: '3' → Assistant: '3 is an odd number so the answer is X.'")
    print("\nThis provides context! Let's test:\n")
    
    test_numbers = ["4", "7", "10", "13"]
    
    for number in test_numbers:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a smart and helpful assistant."},
                {"role": "user", "content": "1"},
                {"role": "assistant", "content": "1 is an odd number so the answer is X."},
                {"role": "user", "content": "2"},
                {"role": "assistant", "content": "2 is an even number so the answer is Y."},
                {"role": "user", "content": "3"},
                {"role": "assistant", "content": "3 is an odd number so the answer is X."},
                {"role": "user", "content": number}
            ],
            temperature=0.7,
            max_tokens=256
        )
        
        print(f"User: {number}")
        print(f"Felix: {response.choices[0].message.content}")
        print("  ✅ Much more consistent!\n")


def main():
    """Run all few-shot learning demonstrations"""
    print("\n")
    print("╔" + "═"*78 + "╗")
    print("║" + " "*20 + "FEW-SHOT LEARNING DEMONSTRATION" + " "*27 + "║")
    print("╚" + "═"*78 + "╝")
    print("\nThis demonstrates how providing examples (few-shot learning) helps the AI")
    print("understand patterns and respond in the desired format.\n")
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY environment variable not set!")
        print("\nPlease set it using:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        return
    
    try:
        # Run all examples
        test_even_odd_numbers()
        test_animal_classification()
        test_limitation_without_reasoning()
        test_solution_with_reasoning()
        
        print("\n" + "="*80)
        print("KEY TAKEAWAYS:")
        print("="*80)
        print("✅ Few-shot learning teaches AI through examples")
        print("✅ More examples = better pattern recognition")
        print("✅ Adding reasoning (Chain of Thought) improves reliability")
        print("✅ Without reasoning, complex patterns may fail")
        print("✅ This technique is foundational for prompt engineering")
        print("\n" + "="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nMake sure:")
        print("1. OPENAI_API_KEY is set correctly")
        print("2. You have internet connection")
        print("3. Your OpenAI account has credits")


if __name__ == "__main__":
    main()
