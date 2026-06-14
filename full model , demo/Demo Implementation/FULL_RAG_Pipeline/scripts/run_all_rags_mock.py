"""Run a mocked end-to-end example across all three Guardian Eye RAG paths."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag_service.legal_orchestrator import generate_legal_consequences
from rag_service.legal_retrieval import RankedLegalReference


FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "all_rags_mock_example.json"
REPORT_PATH = REPO_ROOT / "reports" / "all_rags_mock_report.md"


def load_mock_input(path: Path = FIXTURE_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_explanation_rag_mock(example_input: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "mocked",
        "verdict": example_input["verdict"],
        "confidence": example_input["confidence"],
        "explanation": (
            "The classifier marked the incident as violent because the packet "
            "describes a physical confrontation and a raised bottle-like object."
        ),
        "evidence_basis": [
            "physical confrontation",
            "bottle-like object visible",
            "classifier confidence 0.94",
        ],
        "limitations": (
            "This mock explanation is deterministic and does not use a real model."
        ),
    }


def run_incident_memory_rag_mock(example_input: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "mocked",
        "query_basis": {
            "verdict": example_input["verdict"],
            "weapon_flag": example_input["weapon_flag"],
            "weapon_class": example_input["weapon_class"],
        },
        "similar_incidents": [
            {
                "incident_id": "mock-memory-001",
                "summary": (
                    "Prior mock incident involving a confrontation where an "
                    "object was visible but no legal conclusion was made."
                ),
                "similarity": 0.82,
            },
            {
                "incident_id": "mock-memory-002",
                "summary": "Prior mock violent-classified confrontation.",
                "similarity": 0.76,
            },
        ],
        "memory_note": (
            "This mock memory output does not query a real database or vector store."
        ),
    }


class MockLegalRetriever:
    def retrieve(self, **kwargs: Any) -> list[RankedLegalReference]:
        return [
            RankedLegalReference(
                law_title="UAE Crimes and Penalties Law",
                article_number="Article 1",
                section_title="Assault and Dangerous Object Context",
                source_url="https://uaelegislation.gov.ae/ar/legislations/1529",
                snippet=(
                    "Retrieved mock legal text addresses assault, physical harm, "
                    "dangerous objects, and possible penalties."
                ),
                score=0.88,
                country="UAE",
                violence_category="weapon_or_dangerous_object",
                official_source=True,
            )
        ]


@dataclass(frozen=True)
class MockLegalEvaluation:
    retrieval_score: float = 0.91
    generation_score: float = 0.89
    overall_score: float = 0.9
    passed: bool = True
    issues: tuple[str, ...] = ()
    metric_breakdown: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieval_score": self.retrieval_score,
            "generation_score": self.generation_score,
            "overall_score": self.overall_score,
            "passed": self.passed,
            "issues": list(self.issues),
            "metric_breakdown": self.metric_breakdown
            or {
                "retrieval": {
                    "top_k_similarity": 0.88,
                    "country_filter_correctness": 1.0,
                    "keyword_overlap": 0.86,
                    "source_officialness": 1.0,
                    "article_level_match": 1.0,
                },
                "generation": {
                    "groundedness": 0.88,
                    "citation_presence": 1.0,
                    "guardrail_compliance": 1.0,
                    "language_compatibility": 1.0,
                    "unsupported_action_safety": 1.0,
                },
            },
        }


def mock_legal_evaluator(*args: Any, **kwargs: Any) -> MockLegalEvaluation:
    return MockLegalEvaluation()


def run_all_rags_mock(example_input: dict[str, Any] | None = None) -> dict[str, Any]:
    input_payload = dict(example_input or load_mock_input())
    input_payload.pop("detected_actions", None)

    explanation_output = run_explanation_rag_mock(input_payload)
    incident_memory_output = run_incident_memory_rag_mock(input_payload)
    legal_result = generate_legal_consequences(
        input_payload,
        retriever=MockLegalRetriever(),
        evaluator=mock_legal_evaluator,
    )

    combined = {
        "input": input_payload,
        "explanation_rag": explanation_output,
        "incident_memory_rag": incident_memory_output,
        "legal_consequences_rag": legal_result.legal_consequences,
        "legal_scores": legal_result.evaluation,
        "overall_status": "mock_passed",
    }
    write_report(combined)
    return combined


def write_report(combined: dict[str, Any], path: Path = REPORT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    legal_scores = combined["legal_scores"] or {}
    legal_output = combined["legal_consequences_rag"]
    report = f"""# All RAGs Mock End-to-End Report

## Mock Input

```json
{_json_block(combined["input"])}
```

## Explanation RAG Output

```json
{_json_block(combined["explanation_rag"])}
```

## Incident Memory RAG Output

```json
{_json_block(combined["incident_memory_rag"])}
```

## Legal Consequences RAG Output

```json
{_json_block(legal_output)}
```

## Legal RAG Scores

- Retrieval score: {legal_scores.get("retrieval_score")}
- Generation score: {legal_scores.get("generation_score")}
- Overall score: {legal_scores.get("overall_score")}
- Passed: {legal_scores.get("passed")}

```json
{_json_block(legal_scores.get("metric_breakdown", {}))}
```

## Guardrails and Limitations

- Guardrail status: {legal_output.get("guardrail_status")}
- Limitations note: {legal_output.get("limitations_note")}
- Overall status: {combined.get("overall_status")}
"""
    path.write_text(report, encoding="utf-8")


def _json_block(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def main() -> None:
    result = run_all_rags_mock()
    print(_json_block(result))


if __name__ == "__main__":
    main()
