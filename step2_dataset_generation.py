"""
=============================================================================
ASSIGNMENT: Prompt Engineering for RAG Systems
SECTION:    Step 2 — Evaluation Dataset Generation
SYSTEM:     "Chat with Your Documents" (RAG on resume.pdf)
STUDENT:    Apurva Raj
=============================================================================

Generates a 45-item evaluation dataset:
  • 30 Typical queries   (education, skills, technologies, experience, projects)
  • 10 Edge cases        (summarization, interpretation, multi-hop reasoning)
  •  5 Adversarial       (questions unrelated to resume → test hallucination guard)

Split: 70 / 15 / 15  (train / dev / test)
Output: rag_eval_dataset.json
"""

import json
import math
import random

# ---------------------------------------------------------------------------
# Ground-truth dataset  (hand-crafted from resume.pdf)
# ---------------------------------------------------------------------------

TYPICAL = [

    # ── EDUCATION (6 questions) ─────────────────────────────────────────────
    {
        "id": "T01",
        "category": "typical",
        "sub_category": "education",
        "query": "Where is Apurva currently pursuing his master's degree?",
        "ground_truth": "Northeastern University, Boston, MA",
        "notes": "Direct factual lookup — institution name"
    },
    {
        "id": "T02",
        "category": "typical",
        "sub_category": "education",
        "query": "What is Apurva's GPA in his master's program?",
        "ground_truth": "3.7 out of 4.0",
        "notes": "Single numeric fact"
    },
    {
        "id": "T03",
        "category": "typical",
        "sub_category": "education",
        "query": "What degree is Apurva pursuing at Northeastern University?",
        "ground_truth": "Master's in Software Engineering",
        "notes": "Degree title lookup"
    },
    {
        "id": "T04",
        "category": "typical",
        "sub_category": "education",
        "query": "When did Apurva start his master's program?",
        "ground_truth": "January 2024",
        "notes": "Date extraction"
    },
    {
        "id": "T05",
        "category": "typical",
        "sub_category": "education",
        "query": "Where did Apurva complete his undergraduate degree?",
        "ground_truth": "M S Ramaiah Institute of Technology, VTU, Bangalore, India",
        "notes": "Second institution lookup"
    },
    {
        "id": "T06",
        "category": "typical",
        "sub_category": "education",
        "query": "What was Apurva's undergraduate field of study?",
        "ground_truth": "Bachelor of Engineering in Information Science and Engineering",
        "notes": "UG degree title"
    },

    # ── SKILLS — LANGUAGES (5 questions) ───────────────────────────────────
    {
        "id": "T07",
        "category": "typical",
        "sub_category": "skills",
        "query": "Which programming languages does Apurva know?",
        "ground_truth": "Java, Python, JavaScript, TypeScript, HTML/CSS, SCSS",
        "notes": "List extraction from Skills section"
    },
    {
        "id": "T08",
        "category": "typical",
        "sub_category": "skills",
        "query": "Does Apurva have experience with Python?",
        "ground_truth": "Yes, Python is listed under Languages in the Technical Skills section",
        "notes": "Boolean with evidence"
    },
    {
        "id": "T09",
        "category": "typical",
        "sub_category": "skills",
        "query": "What frontend frameworks does Apurva know?",
        "ground_truth": "ReactJS and React-Native",
        "notes": "Subset filtering from frameworks list"
    },
    {
        "id": "T10",
        "category": "typical",
        "sub_category": "skills",
        "query": "Which databases is Apurva familiar with?",
        "ground_truth": "PostgreSQL, MySQL, MongoDB, Redis",
        "notes": "Database subset"
    },
    {
        "id": "T11",
        "category": "typical",
        "sub_category": "skills",
        "query": "What cloud technologies does Apurva use?",
        "ground_truth": "Docker, Kubernetes, Amazon Web Services (AWS)",
        "notes": "Cloud tech subset"
    },

    # ── TECHNOLOGIES (5 questions) ──────────────────────────────────────────
    {
        "id": "T12",
        "category": "typical",
        "sub_category": "technologies",
        "query": "What messaging/streaming tools does Apurva have experience with?",
        "ground_truth": "Kafka, ZooKeeper, Debezium",
        "notes": "Event-streaming tools from Dev Tools section"
    },
    {
        "id": "T13",
        "category": "typical",
        "sub_category": "technologies",
        "query": "Which AI tools and platforms does Apurva list on his resume?",
        "ground_truth": "GPT, Copilot, OpenAI, Gemini",
        "notes": "AI tooling from Others section"
    },
    {
        "id": "T14",
        "category": "typical",
        "sub_category": "technologies",
        "query": "Does Apurva have experience with Kubernetes?",
        "ground_truth": "Yes, Kubernetes is listed under Cloud Tech and used at Mercedes-Benz R&D",
        "notes": "Boolean with corroboration from experience"
    },
    {
        "id": "T15",
        "category": "typical",
        "sub_category": "technologies",
        "query": "What CI/CD tools has Apurva used?",
        "ground_truth": "Jenkins, Docker, GitHub Actions, Packer",
        "notes": "Cross-section lookup (skills + experience)"
    },
    {
        "id": "T16",
        "category": "typical",
        "sub_category": "technologies",
        "query": "What backend frameworks does Apurva know?",
        "ground_truth": "Node.js, SpringBoot, ExpressJS, Django",
        "notes": "Backend subset from Frameworks"
    },

    # ── WORK EXPERIENCE (8 questions) ──────────────────────────────────────
    {
        "id": "T17",
        "category": "typical",
        "sub_category": "experience",
        "query": "What is Apurva's most recent job title and employer?",
        "ground_truth": "Software Engineer at Admins, Boston, MA (September–December 2025)",
        "notes": "Most recent role — top of experience section"
    },
    {
        "id": "T18",
        "category": "typical",
        "sub_category": "experience",
        "query": "What did Apurva build at Admins?",
        "ground_truth": "A responsive React + TypeScript frontend and secure Node.js + PostgreSQL APIs, achieving 2x faster rendering and 10,000+ requests/day",
        "notes": "Achievement-level detail"
    },
    {
        "id": "T19",
        "category": "typical",
        "sub_category": "experience",
        "query": "How long did Apurva work at Daimler Truck Innovation Centre India?",
        "ground_truth": "December 2021 to December 2023 — approximately 2 years",
        "notes": "Duration calculation"
    },
    {
        "id": "T20",
        "category": "typical",
        "sub_category": "experience",
        "query": "What portal did Apurva build at Daimler?",
        "ground_truth": "A workflow portal using Spring Boot and GraphQL, serving over 2000 users",
        "notes": "Project within experience"
    },
    {
        "id": "T21",
        "category": "typical",
        "sub_category": "experience",
        "query": "What was Apurva's role at Mercedes-Benz Research & Development India?",
        "ground_truth": "Software Engineer from January 2021 to November 2021",
        "notes": "Role + tenure at third employer"
    },
    {
        "id": "T22",
        "category": "typical",
        "sub_category": "experience",
        "query": "What real-time pipeline did Apurva build at Mercedes-Benz?",
        "ground_truth": "A real-time CDC pipeline using Debezium and Kafka enabling 5x faster analytics",
        "notes": "Technical achievement detail"
    },
    {
        "id": "T23",
        "category": "typical",
        "sub_category": "experience",
        "query": "How many engineers did Apurva collaborate with at Daimler?",
        "ground_truth": "50 engineers across 3 Agile projects",
        "notes": "Numeric metric from experience"
    },
    {
        "id": "T24",
        "category": "typical",
        "sub_category": "experience",
        "query": "What AWS services did Apurva use at Admins?",
        "ground_truth": "CloudFront, Elastic Beanstalk, RDS",
        "notes": "AWS subset from Admins bullets"
    },

    # ── PROJECTS (6 questions) ──────────────────────────────────────────────
    {
        "id": "T25",
        "category": "typical",
        "sub_category": "projects",
        "query": "What is the Incluwork project?",
        "ground_truth": "A web-based system built with MongoDB, Express.js, React, Node.js, Redux, Material-UI, and Vite",
        "notes": "Project description"
    },
    {
        "id": "T26",
        "category": "typical",
        "sub_category": "projects",
        "query": "What authentication method is used in Incluwork?",
        "ground_truth": "OAuth",
        "notes": "Specific technical detail from project"
    },
    {
        "id": "T27",
        "category": "typical",
        "sub_category": "projects",
        "query": "What infrastructure tool did Apurva use in the Cloud Native WebApp project?",
        "ground_truth": "Terraform, used to define EC2, VPCs, and subnets",
        "notes": "IaC tool lookup"
    },
    {
        "id": "T28",
        "category": "typical",
        "sub_category": "projects",
        "query": "How did Apurva handle deployments in the Cloud Native WebApp?",
        "ground_truth": "CI/CD with GitHub Actions and Packer to automate builds and create custom AMIs",
        "notes": "CI/CD detail from project"
    },
    {
        "id": "T29",
        "category": "typical",
        "sub_category": "projects",
        "query": "What monitoring solution did Apurva build at Mercedes-Benz?",
        "ground_truth": "Grafana dashboards, improving system visibility and cutting mean-time-to-resolution by 4x",
        "notes": "Observability tool detail"
    },
    {
        "id": "T30",
        "category": "typical",
        "sub_category": "projects",
        "query": "What state management library did Apurva use in Incluwork?",
        "ground_truth": "Redux",
        "notes": "Specific library from project stack"
    },
]

EDGE_CASES = [
    {
        "id": "E01",
        "category": "edge_case",
        "sub_category": "summarization",
        "query": "Summarize Apurva's entire career in two sentences.",
        "ground_truth": "Apurva is a Software Engineer with experience at Mercedes-Benz, Daimler, and Admins, working across full-stack development, cloud infrastructure, and data pipelines. He is currently completing a Master's in Software Engineering at Northeastern University with a 3.7 GPA.",
        "notes": "Requires synthesis across multiple resume sections"
    },
    {
        "id": "E02",
        "category": "edge_case",
        "sub_category": "summarization",
        "query": "What are Apurva's top three technical strengths based on the resume?",
        "ground_truth": "Full-stack development (React, Node.js, Spring Boot), cloud/DevOps (AWS, Docker, Kubernetes, Terraform), and data engineering (Kafka, Debezium, PostgreSQL)",
        "notes": "Interpretation — requires grouping skills thematically"
    },
    {
        "id": "E03",
        "category": "edge_case",
        "sub_category": "interpretation",
        "query": "Based on his resume, is Apurva more of a backend or full-stack engineer?",
        "ground_truth": "Full-stack — the resume shows both frontend (React, TypeScript) and backend (Spring Boot, Node.js, GraphQL) work across multiple roles",
        "notes": "Interpretive classification requiring evidence"
    },
    {
        "id": "E04",
        "category": "edge_case",
        "sub_category": "multi_hop",
        "query": "Which company did Apurva work at immediately before starting his master's degree?",
        "ground_truth": "Daimler Truck Innovation Centre India (December 2021 – December 2023), since his master's started January 2024",
        "notes": "Multi-hop: cross-reference experience dates with education start date"
    },
    {
        "id": "E05",
        "category": "edge_case",
        "sub_category": "multi_hop",
        "query": "How many years of total professional experience does Apurva have?",
        "ground_truth": "Approximately 3 years 11 months (Jan 2021 – Dec 2023 at Mercedes-Benz and Daimler, plus Sep–Dec 2025 at Admins)",
        "notes": "Arithmetic over multiple date ranges"
    },
    {
        "id": "E06",
        "category": "edge_case",
        "sub_category": "interpretation",
        "query": "What evidence in the resume suggests Apurva has experience with large-scale systems?",
        "ground_truth": "Processing 1M+ messages/day at Mercedes-Benz, 100K daily requests, 10,000+ API requests/day at Admins, and collaborating with 50 engineers at Daimler",
        "notes": "Evidence gathering across multiple bullets"
    },
    {
        "id": "E07",
        "category": "edge_case",
        "sub_category": "summarization",
        "query": "List all the performance improvements Apurva achieved across his roles.",
        "ground_truth": "2x faster rendering (Admins), 2x test coverage (Admins), 3x faster release cycles (Daimler), 1.5x fewer blockers (Daimler), 1.5x reduced manual effort (Daimler), 5x faster analytics (Mercedes-Benz), 4x better MTTR (Mercedes-Benz), 1.3x fewer defects (Mercedes-Benz)",
        "notes": "Aggregation across all experience bullets"
    },
    {
        "id": "E08",
        "category": "edge_case",
        "sub_category": "interpretation",
        "query": "Does Apurva have experience comparing AI/LLM tools?",
        "ground_truth": "Yes — at Admins he conducted a comparative analysis of Cursor, Copilot, Claude, and other LLMs for legacy system workflow conversion",
        "notes": "Niche detail buried in experience bullets"
    },
    {
        "id": "E09",
        "category": "edge_case",
        "sub_category": "multi_hop",
        "query": "What technology stack does Apurva's Incluwork project share with his professional experience?",
        "ground_truth": "Node.js and Express.js (used at Admins), React (used at Admins), PostgreSQL/MongoDB (listed in skills and used professionally)",
        "notes": "Cross-reference projects section with experience"
    },
    {
        "id": "E10",
        "category": "edge_case",
        "sub_category": "interpretation",
        "query": "Which of Apurva's skills are most relevant for a DevOps engineer role?",
        "ground_truth": "Docker, Kubernetes, AWS (CloudFront, Elastic Beanstalk, RDS, EC2, S3), Terraform, Packer, GitHub Actions, Jenkins, CloudWatch, Linux, Kafka",
        "notes": "Role-based skill filtering requiring interpretation"
    },
]

ADVERSARIAL = [
    {
        "id": "A01",
        "category": "adversarial",
        "sub_category": "out_of_scope",
        "query": "What is Apurva's current salary?",
        "ground_truth": "NOT_IN_DOCUMENT",
        "expected_behavior": "Model should respond that salary information is not available in the resume",
        "notes": "Classic hallucination trap — salary is never in a resume"
    },
    {
        "id": "A02",
        "category": "adversarial",
        "sub_category": "out_of_scope",
        "query": "What is Apurva's home address?",
        "ground_truth": "NOT_IN_DOCUMENT",
        "expected_behavior": "Model should respond that address is not available; only Boston, MA is mentioned",
        "notes": "PII not present — tests grounding and privacy"
    },
    {
        "id": "A03",
        "category": "adversarial",
        "sub_category": "out_of_scope",
        "query": "Does Apurva have a GitHub profile? What repositories has he published?",
        "ground_truth": "NOT_IN_DOCUMENT",
        "expected_behavior": "GitHub username (github.com/apurva9808) is listed but specific repositories are not — model should not fabricate repo names",
        "notes": "Partial info trap — URL present but repo details absent"
    },
    {
        "id": "A04",
        "category": "adversarial",
        "sub_category": "unrelated",
        "query": "What is the capital of France?",
        "ground_truth": "NOT_IN_DOCUMENT",
        "expected_behavior": "Model should refuse to answer as this is completely unrelated to the resume",
        "notes": "Completely unrelated general knowledge question"
    },
    {
        "id": "A05",
        "category": "adversarial",
        "sub_category": "fabrication_trap",
        "query": "What was Apurva's thesis topic for his master's degree?",
        "ground_truth": "NOT_IN_DOCUMENT",
        "expected_behavior": "Model should not fabricate a thesis topic; the resume does not mention one",
        "notes": "Plausible-sounding question with no document basis — high hallucination risk"
    },
]

# ---------------------------------------------------------------------------
# Combine and split
# ---------------------------------------------------------------------------

def build_dataset() -> dict:
    all_items = TYPICAL + EDGE_CASES + ADVERSARIAL
    random.seed(42)   # reproducible shuffle
    random.shuffle(all_items)

    n = len(all_items)                   # 45
    n_train = math.floor(n * 0.70)       # 31
    n_dev   = math.floor(n * 0.15)       # 6
    # n_test  = remaining                # 8

    train = all_items[:n_train]
    dev   = all_items[n_train : n_train + n_dev]
    test  = all_items[n_train + n_dev :]

    dataset = {
        "metadata": {
            "title":       "RAG Evaluation Dataset — Chat with Your Documents",
            "document":    "resume.pdf (Apurva Raj)",
            "total_items": n,
            "breakdown": {
                "typical":     len(TYPICAL),
                "edge_cases":  len(EDGE_CASES),
                "adversarial": len(ADVERSARIAL),
            },
            "split": {
                "train": {"count": len(train), "ratio": "70%"},
                "dev":   {"count": len(dev),   "ratio": "15%"},
                "test":  {"count": len(test),  "ratio": "15%"},
            },
            "random_seed": 42,
        },
        "curation_notes": (
            "Dataset curated manually from resume.pdf. "
            "Typical queries test direct factual retrieval across five resume sections. "
            "Edge cases require synthesis, interpretation, or multi-hop reasoning across sections. "
            "Adversarial queries test the model's ability to abstain rather than hallucinate "
            "when information is absent from the document. "
            "The 70/15/15 split follows standard NLP evaluation practice: "
            "the train split is used for prompt tuning and few-shot example selection, "
            "the dev split for iterative prompt refinement and hyperparameter search, "
            "and the held-out test split for final unbiased evaluation."
        ),
        "splits": {
            "train": train,
            "dev":   dev,
            "test":  test,
        },
        "full_dataset": all_items,
    }
    return dataset


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_sample(dataset: dict):
    print("\n" + "=" * 70)
    print("  SAMPLE — 3 items per category")
    print("=" * 70)

    for cat, label in [
        ("typical",     "TYPICAL"),
        ("edge_case",   "EDGE CASE"),
        ("adversarial", "ADVERSARIAL"),
    ]:
        items = [i for i in dataset["full_dataset"] if i["category"] == cat][:3]
        print(f"\n  ── {label} ──")
        for item in items:
            print(f"  [{item['id']}] {item['query']}")
            print(f"        → {item['ground_truth'][:80]}{'…' if len(item['ground_truth']) > 80 else ''}")


def print_split_summary(dataset: dict):
    print("\n" + "=" * 70)
    print("  TRAIN / DEV / TEST SPLIT")
    print("=" * 70)
    for split_name, items in dataset["splits"].items():
        cats = {}
        for item in items:
            cats[item["category"]] = cats.get(item["category"], 0) + 1
        cat_str = "  |  ".join(f"{k}: {v}" for k, v in sorted(cats.items()))
        print(f"  {split_name.upper():<6}  {len(items):>3} items   {cat_str}")


def print_curation_section():
    print("\n" + "=" * 70)
    print("  DATASET CURATION — ACADEMIC WRITE-UP")
    print("=" * 70)
    print("""
  HOW THE DATASET WAS CURATED
  ────────────────────────────
  All 45 question–answer pairs were manually authored by reading the
  source document (resume.pdf) section by section.  Ground-truth answers
  were extracted verbatim or paraphrased directly from the text to ensure
  they are verifiable against the document without ambiguity.  No external
  knowledge or inference beyond the document was used when writing
  ground-truth answers for typical and edge-case items.

  WHY THREE CATEGORIES?
  ──────────────────────
  • Typical queries (30) — simulate the common-case workload: a recruiter
    or hiring manager asking straightforward questions about education, skills,
    experience, or projects.  These test whether the RAG system can reliably
    retrieve and present factual information that is explicitly stated in the
    document.

  • Edge cases (10) — simulate queries that require the model to go beyond
    simple lookup: summarize across sections, reason about time spans, or
    classify the candidate's profile.  These expose weaknesses in prompts
    that over-restrict the model to verbatim extraction without allowing
    synthesis.

  • Adversarial queries (5) — simulate attempts to elicit hallucinated or
    fabricated information (salary, thesis topic, repository names) or
    completely off-topic general knowledge.  A well-grounded RAG system
    should abstain with a clear "not in the document" response rather than
    confabulating an answer.  These items are evaluated on refusal quality
    rather than factual correctness.

  TRAIN / DEV / TEST SPLIT  (70 / 15 / 15)
  ──────────────────────────────────────────
  The dataset is randomly shuffled (seed = 42) then split:

    Train  (70%, 31 items) — used for:
      • Selecting few-shot examples for Few-Shot prompting experiments
      • Iterating on system prompt wording
      • Calibrating temperature settings

    Dev    (15%,  6 items) — used for:
      • Intermediate evaluation during prompt tuning
      • Preventing over-fitting to the train examples
      • Choosing between competing prompt variants (A/B)

    Test   (15%,  8 items) — used for:
      • Final, unbiased evaluation of the chosen system prompt
      • Reported accuracy and consistency metrics in the assignment
      • Never inspected during development to avoid data leakage

  This protocol mirrors standard NLP evaluation practice and ensures that
  reported test-set metrics are a fair estimate of real-world performance.
    """)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "█" * 70)
    print("  STEP 2: EVALUATION DATASET GENERATION")
    print("  RAG System: 'Chat with Your Documents' — Document: resume.pdf")
    print("█" * 70)

    dataset = build_dataset()

    # Save JSON
    output_path = "rag_eval_dataset.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\n  ✓ Dataset saved → {output_path}")
    print(f"  ✓ Total items  : {dataset['metadata']['total_items']}")
    print(f"  ✓ Typical      : {dataset['metadata']['breakdown']['typical']}")
    print(f"  ✓ Edge cases   : {dataset['metadata']['breakdown']['edge_cases']}")
    print(f"  ✓ Adversarial  : {dataset['metadata']['breakdown']['adversarial']}")

    print_split_summary(dataset)
    print_sample(dataset)
    print_curation_section()

    print("\n" + "=" * 70)
    print("  Done. Load the dataset in other scripts with:")
    print("    import json")
    print("    with open('rag_eval_dataset.json') as f:")
    print("        dataset = json.load(f)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
