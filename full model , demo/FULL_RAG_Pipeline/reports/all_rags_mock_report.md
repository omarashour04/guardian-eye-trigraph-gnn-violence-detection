# All RAGs Mock End-to-End Report

## Mock Input

```json
{
  "country": "UAE",
  "verdict": "violence",
  "confidence": 0.94,
  "packet_summary": "Two people are involved in a physical confrontation. One person appears to raise a bottle-like object during the incident.",
  "narrative": "The scene shows a confrontation between two individuals. A bottle-like object is visible. The classifier marked the incident as violent.",
  "weapon_flag": true,
  "weapon_class": "bottle",
  "language": "en"
}
```

## Explanation RAG Output

```json
{
  "status": "mocked",
  "verdict": "violence",
  "confidence": 0.94,
  "explanation": "The classifier marked the incident as violent because the packet describes a physical confrontation and a raised bottle-like object.",
  "evidence_basis": [
    "physical confrontation",
    "bottle-like object visible",
    "classifier confidence 0.94"
  ],
  "limitations": "This mock explanation is deterministic and does not use a real model."
}
```

## Incident Memory RAG Output

```json
{
  "status": "mocked",
  "query_basis": {
    "verdict": "violence",
    "weapon_flag": true,
    "weapon_class": "bottle"
  },
  "similar_incidents": [
    {
      "incident_id": "mock-memory-001",
      "summary": "Prior mock incident involving a confrontation where an object was visible but no legal conclusion was made.",
      "similarity": 0.82
    },
    {
      "incident_id": "mock-memory-002",
      "summary": "Prior mock violent-classified confrontation.",
      "similarity": 0.76
    }
  ],
  "memory_note": "This mock memory output does not query a real database or vector store."
}
```

## Legal Consequences RAG Output

```json
{
  "country": "UAE",
  "query_basis": {
    "verdict": "violence",
    "weapon_flag": true,
    "weapon_class": "bottle"
  },
  "retrieved_legal_references": [
    {
      "law_title": "UAE Crimes and Penalties Law",
      "article_number": "Article 1",
      "section_title": "Assault and Dangerous Object Context",
      "source_url": "https://uaelegislation.gov.ae/ar/legislations/1529",
      "snippet": "Retrieved mock legal text addresses assault, physical harm, dangerous objects, and possible penalties.",
      "score": 0.88,
      "country": "UAE",
      "violence_category": "weapon_or_dangerous_object",
      "official_source": true
    }
  ],
  "summary": "According to the retrieved regulation, including UAE Crimes and Penalties Law Article 1 source https://uaelegislation.gov.ae/ar/legislations/1529, the reported violence involving a potential weapon or dangerous object such as bottle may be subject to legal provisions reflected in the retrieved text. Possible legal consequences include exposure to statutory penalties, protective measures, or other court-determined outcomes depending on judicial determination and the facts established by competent authorities. Retrieved categories include weapon_or_dangerous_object.",
  "guardrail_status": "passed",
  "limitations_note": "This is not legal advice and does not determine guilt or predict court outcome."
}
```

## Legal RAG Scores

- Retrieval score: 0.91
- Generation score: 0.89
- Overall score: 0.9
- Passed: True

```json
{
  "retrieval": {
    "top_k_similarity": 0.88,
    "country_filter_correctness": 1.0,
    "keyword_overlap": 0.86,
    "source_officialness": 1.0,
    "article_level_match": 1.0
  },
  "generation": {
    "groundedness": 0.88,
    "citation_presence": 1.0,
    "guardrail_compliance": 1.0,
    "language_compatibility": 1.0,
    "unsupported_action_safety": 1.0
  }
}
```

## Guardrails and Limitations

- Guardrail status: passed
- Limitations note: This is not legal advice and does not determine guilt or predict court outcome.
- Overall status: mock_passed
