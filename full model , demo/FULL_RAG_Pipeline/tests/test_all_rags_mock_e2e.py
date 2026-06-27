import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_all_rags_mock.py"
REPORT_PATH = REPO_ROOT / "reports" / "all_rags_mock_report.md"
FORBIDDEN_FIELD_NAME = "detected" + "_actions"


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("run_all_rags_mock", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_all_three_rags_mock_combined_run_and_report_generation(capsys):
    runner = _load_runner_module()

    result = runner.run_all_rags_mock()
    runner.main()
    printed = capsys.readouterr().out

    assert result["overall_status"] == "mock_passed"
    assert result["input"]
    assert result["explanation_rag"]
    assert result["incident_memory_rag"]
    assert result["legal_consequences_rag"]
    assert result["legal_scores"]

    legal_output = result["legal_consequences_rag"]
    assert legal_output["country"] == "UAE"
    assert legal_output["query_basis"]["weapon_flag"] is True
    assert legal_output["query_basis"]["weapon_class"] == "bottle"
    assert legal_output["retrieved_legal_references"]
    assert legal_output["summary"]
    assert legal_output["guardrail_status"] == "passed"
    assert "not legal advice" in legal_output["limitations_note"]

    legal_scores = result["legal_scores"]
    assert set(
        [
            "retrieval_score",
            "generation_score",
            "overall_score",
            "passed",
            "metric_breakdown",
        ]
    ).issubset(legal_scores.keys())
    assert legal_scores["retrieval_score"] == 0.91
    assert legal_scores["generation_score"] == 0.89
    assert legal_scores["overall_score"] == 0.9
    assert legal_scores["passed"] is True

    assert REPORT_PATH.exists()
    report = REPORT_PATH.read_text(encoding="utf-8")
    for heading in [
        "## Mock Input",
        "## Explanation RAG Output",
        "## Incident Memory RAG Output",
        "## Legal Consequences RAG Output",
        "## Legal RAG Scores",
        "## Guardrails and Limitations",
    ]:
        assert heading in report
    assert "Retrieval score: 0.91" in report
    assert "Generation score: 0.89" in report
    assert "Overall score: 0.9" in report

    combined_json = json.dumps(result, ensure_ascii=False)
    assert FORBIDDEN_FIELD_NAME not in combined_json
    assert FORBIDDEN_FIELD_NAME not in printed
    assert FORBIDDEN_FIELD_NAME not in report
