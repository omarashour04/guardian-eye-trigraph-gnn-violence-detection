# Tera Translator

This folder is an independent copy of Sara's Guardian Eye RAG with lazy
Arabic/English translation around user queries.

## Flow

1. `contains_arabic()` checks the user query without loading a model.
2. English queries go directly to the existing English RAG.
3. Arabic queries load `google/translategemma-4b-it`.
4. TranslateGemma translates the query to English.
5. The existing RAG runs with the English query.
6. User-facing result text is translated back to Arabic.
7. TranslateGemma is unloaded and GPU cache is released.

`g4_historical_search.semantic_search()` is already connected to this flow.
For another RAG entry point, wrap it with:

```python
from bilingual_rag import run_bilingual_rag

result = run_bilingual_rag(user_query, existing_rag_function, top_k=5)
```

## Configuration

- `TRANSLATEGEMMA_MODEL_ID`: local checkpoint path or Hugging Face model ID.
- `TRANSLATEGEMMA_DEVICE_MAP`: Transformers device map; defaults to `auto`.
- `TRANSLATEGEMMA_MAX_NEW_TOKENS`: generation limit; defaults to `512`.

The first Arabic request may need Hugging Face access to download the gated
model. For offline use, download the model in advance and set
`TRANSLATEGEMMA_MODEL_ID` to its local directory.
