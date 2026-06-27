# Legal Consequences RAG Handoff Instructions

## What Was Implemented

The Legal Consequences RAG feature now has a complete RAG-only backend path from legal source metadata through an integration-safe orchestrator. The implementation intentionally does not modify frontend/demo files and does not add UI, country selectors, legal panels, or final demo orchestration.

Implemented flow:

1. Legal source registry.
2. Safe input/output schemas.
3. HTML/PDF raw extraction.
4. Legal text cleaning and violence-related filtering.
5. Article/section chunking.
6. FAISS + SQLite metadata indexing.
7. Retrieval and deterministic ranking.
8. Grounded cautious summarization.
9. Guardrail validation.
10. Evaluation helpers.
11. Integration-safe orchestrator and mocked end-to-end tests.

## Modules Added

- `rag_service/legal_sources.py`
- `rag_service/schemas.py`
- `rag_service/legal_scraper.py`
- `rag_service/legal_cleaner.py`
- `rag_service/legal_chunker.py`
- `rag_service/legal_index.py`
- `rag_service/legal_retrieval.py`
- `rag_service/legal_summarizer.py`
- `rag_service/legal_guardrails.py`
- `rag_service/legal_evaluation.py`
- `rag_service/legal_orchestrator.py`

## Run Legal RAG Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_legal_source_registry.py tests\test_legal_scraper.py tests\test_legal_cleaner.py tests\test_legal_chunker.py tests\test_legal_index.py tests\test_legal_retrieval.py tests\test_legal_summarizer.py tests\test_legal_guardrails.py tests\test_legal_evaluation.py tests\test_legal_orchestrator.py tests\test_legal_rag_e2e.py
```

The tests use mocks where needed. They do not require network calls, legal scraping, real embedding downloads, or real LLM calls.

## Expected Input Schema

```json
{
  "country": "UAE",
  "verdict": "violence",
  "confidence": 0.94,
  "packet_summary": "...",
  "narrative": "...",
  "weapon_flag": true,
  "weapon_class": "bottle",
  "language": "en"
}
```

Do not pass or depend on `detected_actions`. The legal path intentionally avoids exact action labels such as punching, kicking, or slapping unless those words are explicitly present in the narrative or retrieved references.

## Expected Output Shape

The orchestrator returns a `LegalOrchestratorResult` with:

```json
{
  "legal_consequences": {
    "country": "UAE",
    "query_basis": {
      "verdict": "violence",
      "weapon_flag": true,
      "weapon_class": "bottle"
    },
    "retrieved_legal_references": [
      {
        "law_title": "...",
        "article_number": "...",
        "section_title": "...",
        "source_url": "...",
        "snippet": "...",
        "score": 0.87,
        "country": "UAE",
        "violence_category": "weapon_or_dangerous_object",
        "official_source": true
      }
    ],
    "summary": "According to the retrieved regulation, ... may be subject to ... depending on judicial determination.",
    "guardrail_status": "passed",
    "limitations_note": "This is not legal advice and does not determine guilt or predict court outcome."
  },
  "evaluation": {
    "retrieval_score": 0.0,
    "generation_score": 0.0,
    "overall_score": 0.0,
    "passed": false,
    "issues": [],
    "metric_breakdown": {}
  },
  "guardrails": {
    "status": "passed",
    "issues": [],
    "corrected_summary": null,
    "recommended_action": "pass"
  }
}
```

## Future Demo Integration

Future demo integration should call:

```python
from rag_service.legal_orchestrator import generate_legal_consequences

result = generate_legal_consequences(
    {
        "country": "UAE",
        "verdict": "violence",
        "confidence": 0.94,
        "packet_summary": packet_summary,
        "narrative": narrative,
        "weapon_flag": weapon_flag,
        "weapon_class": weapon_class,
        "language": language,
    },
    legal_index=loaded_legal_index,
    embedding_provider=embedding_provider,
)

legal_consequences = result.legal_consequences
```

For tests or offline wiring, inject `retriever`, `summarizer`, `guardrail_validator`, and `evaluator` instead of using real FAISS, embeddings, or model-backed summarization.

## Arabic Translation

The Legal RAG summarizer currently produces English internally. Arabic mode does not translate here. The final `legal_consequences` output should be translated later by the existing TranslateGemma wrapper, preserving the same schema.

## Safety Notes

- The legal output is not legal advice.
- It must not declare guilt.
- It must not predict court outcome.
- It must not guarantee punishment.
- It must remain grounded in retrieved legal references.
- No frontend or demo files were intentionally modified for this feature.

## TODO: Country-Specific KB Expansion

Current country legal content is split across:

- `FULL_RAG_Pipeline/data/legal_curated_docs/` for generated curated markdown documents.
- `FULL_RAG_Pipeline/rag_service/legal_sources.py` for source metadata and official/non-official URLs.
- `Backend_Ashour/BackEnd/services/curated_legal_service.py` for the runtime demo fallback records.

The current records are intentionally compact and demo-oriented. A later pass should expand the country-specific legal KB for Egypt, UAE, KSA, UK, Canada, and USA California with stronger article-level summaries, clearer source titles, and better referenced consequence language before rebuilding the legal index.
