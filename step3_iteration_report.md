# Step 3 Iteration Report

## What was changed

- Removed the hardcoded OpenAI API key and switched to environment-based configuration.
- Replaced the dummy demo documents with resume-aware documents built from `resume.txt`.
- Added section-aware chunking so retrieval focuses on the correct resume sections.
- Added an offline extractive fallback so the pipeline still runs without an API key.
- Added a detailed output payload with:
  - rewritten query
  - retrieved chunks
  - filtered context
  - final answer
  - grounding status
- Added conservative abstention logic for unsupported repository-related questions.

## Test run results

Smoke tests were run on four questions:

1. Programming languages question → grounded answer from Technical Skills
2. Backend frameworks question → grounded answer from Technical Skills
3. Admins work question → grounded answer from Experience
4. GitHub repositories question → abstained correctly

## Output files

- [step3_rag_pipeline_output.json](step3_rag_pipeline_output.json)
- [step3_rag_pipeline_report.json](step3_rag_pipeline_report.json)

## Notes

- The pipeline now works in offline mode if no OpenAI API key is available.
- The JSON output contains the full run details and summary metrics.
- The report JSON contains the iteration summary used for documentation.
