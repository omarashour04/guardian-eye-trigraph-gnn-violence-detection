# Architecture Notes

## Existing Sara RAG

- `adapter.py`: converts classifier dictionaries into the shared RAG schema.
- `g1_evidence_packet.py`: deterministically summarizes classifier evidence.
- `g1_g2_integration.py`: joins evidence-packet creation with reference retrieval.
- `g2_reference_store.py`: embeds English queries and searches the reference FAISS index.
- `g3_incident_db.py`: persists and reads structured incidents from SQLite.
- `incident_vector_store.py`: builds and searches the historical-incident FAISS index.
- `g4_context_enrichment.py`: combines evidence, references, and similar incidents.
- `g4_historical_search.py`: exposes historical filters and the user semantic-query path.
- `main_pipeline.py`: orchestrates classifier conversion, enrichment, and explanation.
- `reference_corpus/`: English grounding documents used by reference retrieval.
- `data/`: persisted SQLite database, FAISS indexes, and index metadata.
- `docs/`: metadata contract documentation.
- `tests/`: focused unit tests for packets, persistence, and retrieval.

## Translation Addition

- `translation_service.py`: detects Arabic and manages request-scoped
  TranslateGemma 4B loading, translation, and unloading.
- `bilingual_rag.py`: translates Arabic input to English, calls an existing RAG
  function, and translates only user-facing output text back to Arabic.
- `g4_historical_search.semantic_search()`: integrated bilingual entry point.

The reference corpus and vector stores remain English. This avoids rebuilding
indexes and ensures Arabic requests use the same retrieval behavior as English
requests after translation.
