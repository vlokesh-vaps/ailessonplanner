# VAPS Project Structure

This project is organized by concern so the experimental scripts, source modules, and generated data are separated cleanly.

## Layout

- `src/` — Python modules and app logic
  - `src/retrieval/` — Chroma/vector retrieval and indexing helpers
  - `src/generation/` — PDF generation and question-paper generation utilities
  - `src/lesson_planning/` — orchestration for lesson planning and RAG flows
  - `src/experiments/` — prototype and archived experiments
  - `src/analysis/` — analysis scripts
- `data/` — project data and generated artifacts
  - `data/inputs/` — source PDF inputs
  - `data/vector_db/` — persistent Chroma database
  - `data/output/` — generated lesson PDFs and other output artifacts
- `tests/` — validation and retrieval smoke tests
- `.env` — local environment configuration

## Compatibility note

Legacy imports such as `from pdfMaker import ...` or `from textbook_retriever import ...` are still supported through root-level shim files so older scripts continue to work while the project code lives under `src/`.
