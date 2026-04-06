# AI App Concept Definition

## 📱 App Name
[Your app name here]

---

## 🎯 Purpose & Description
**What does this app do?**

[Describe your AI application in 2-3 sentences]

**Example:**
> "Smart Code Reviewer is an AI-powered assistant that analyzes code submissions, identifies potential bugs, suggests improvements, and explains best practices. It helps developers learn and improve code quality through constructive feedback."

---

## 👥 Target Users
**Who will use this app?**

- Primary users:
- Secondary users:
- Use cases:

**Example:**
- Primary: Junior to mid-level developers
- Secondary: Code reviewers, team leads
- Use cases: Pull request reviews, learning best practices, code quality checks

---

## 🧪 Core Test Task
**What specific task will you use to evaluate all prompting techniques?**

This should be:
- Representative of your app's main function
- Concrete and specific
- Repeatable across all experiments
- Neither too simple nor too complex

**Your Test Task:**
```
[Write your specific test task here]
```

**Example:**
```
Task: "Review the following Python function and provide feedback on code quality, potential bugs, and improvements:

def calculate_average(numbers):
    total = 0
    for num in numbers:
        total = total + num
    return total / len(numbers)
"
```

---

## ✅ Success Criteria
**How will you evaluate if responses are good?**

Rate each response on these dimensions (1-5 scale):

| Criterion | Definition | Weight |
|-----------|------------|--------|
| **Clarity** | Is the response easy to understand? | [Low/Medium/High] |
| **Accuracy** | Is the information correct? | [Low/Medium/High] |
| **Completeness** | Does it address all aspects? | [Low/Medium/High] |
| **Tone** | Is the tone appropriate for users? | [Low/Medium/High] |
| **Actionability** | Can users act on this information? | [Low/Medium/High] |
| **Format** | Is it well-structured? | [Low/Medium/High] |

Add your own criteria as needed!

---

## 🎨 Desired Tone & Style
- [ ] Professional
- [ ] Casual/Friendly
- [ ] Educational
- [ ] Empathetic
- [ ] Direct/Concise
- [ ] Detailed/Thorough
- [ ] Other: _____________

---

## 🚫 Constraints & Requirements
**What should the app NOT do?**

- Should not:
- Must avoid:
- Limitations:

**Example:**
- Should not be condescending or overly critical
- Must avoid suggesting changes without explanation
- Limitations: Focus on Python only, max 500 words per response

---

## 💡 Example Interactions

### Example 1: Ideal Interaction
**User Input:**
```
[Example user input]
```

**Desired Output:**
```
[What you want the AI to produce]
```

### Example 2: Edge Case
**User Input:**
```
[Example edge case]
```

**Desired Output:**
```
[How it should handle this]
```

---

## 📈 Metrics for Success
How will you measure if your final system prompt works?

- [ ] Response quality scores (average across criteria)
- [ ] Consistency across multiple runs
- [ ] User satisfaction (if testing with real users)
- [ ] Time to useful response
- [ ] Token efficiency
- [ ] Other: _____________

---

## 🔄 Iteration Notes
As you test different techniques, note insights here:

**Date: [Date]**
- Observation:
- Change made:
- Result:

---

## ✏️ Fill This Out First!
Before proceeding to implement any prompting techniques, complete this document. It will serve as your north star throughout the project.

**Status:** [ ] Not Started [ ] Draft [ ] Complete

---

*This document will be referenced in your final submission.*
