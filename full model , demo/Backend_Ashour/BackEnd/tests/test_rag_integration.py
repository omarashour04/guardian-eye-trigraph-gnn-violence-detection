from __future__ import annotations

import asyncio
import io
import os
import unittest
import uuid
from unittest.mock import patch

from fastapi import UploadFile

from database import get_db
from main import explain, legal_consequences, predict
from schemas import (
    ExplainRequest,
    GateWeights,
    GQSScores,
    LegalConsequencesRequest,
    PredictResponse,
    Telemetry,
    WeaponInfo,
)
from services.rag_adapter import _finalize_legal_output, _metadata_legal_output


os.environ.setdefault("GUARDIAN_LEGAL_AR_TRANSLATION", "fallback")
os.environ.setdefault("GUARDIAN_LEGAL_LLM_ENABLED", "0")


def _contains_arabic(text: str) -> bool:
    return any("\u0600" <= char <= "\u06ff" for char in text)


def _legal_payload() -> dict:
    return {
        "country": "UAE",
        "query_basis": {
            "verdict": "violence",
            "weapon_flag": True,
            "weapon_class": "bottle",
        },
        "retrieved_legal_references": [
            {
                "law_title": "UAE Demo Legal Source Fixture",
                "article_number": "Article 3",
                "section_title": "Weapon or dangerous object context",
                "source_url": "demo-fixture://legal-source/uae",
                "snippet": (
                    "Using a weapon, bottle, knife, or other dangerous object during "
                    "an alleged violent incident may be relevant to legal assessment."
                ),
                "score": 0.83,
                "country": "UAE",
                "violence_category": "weapon_or_dangerous_object",
                "official_source": False,
            }
        ],
        "summary": (
            "According to the retrieved regulation, possible legal consequences "
            "include exposure to statutory penalties depending on judicial determination."
        ),
        "guardrail_status": "passed",
        "limitations_note": (
            "This is not legal advice and does not determine guilt or predict court outcome. "
            "Arabic translation is not performed in this summarizer; the existing "
            "TranslateGemma wrapper can translate the final legal_consequences output later."
        ),
    }


def _legal_prediction(
    *,
    weapon_flag: bool = False,
    weapon_class: str | None = None,
) -> PredictResponse:
    return PredictResponse(
        verdict="violence",
        confidence=0.86,
        threshold=0.51,
        gate=GateWeights(skeleton=0.3, interaction=0.4, object=0.1, vit=0.2),
        gqs=GQSScores(q_skel=0.8, q_int=0.9, q_obj=0.45, q_po=0.5, valid_ratio=0.95),
        telemetry=Telemetry(
            people=3,
            peak_window=[6, 16],
            weapon=WeaponInfo(flag=weapon_flag, cls=weapon_class),
        ),
        clip_id="legal-rag-test.mp4",
    )


class RagIntegrationTests(unittest.TestCase):
    def _run(self, coro):
        return asyncio.run(coro)

    def _db(self):
        generator = get_db()
        db = next(generator)
        self.addCleanup(generator.close)
        return db

    def _predict(self):
        clip_id = f"test_{uuid.uuid4().hex}.mp4"
        upload = UploadFile(file=io.BytesIO(b"demo"), filename=clip_id)
        return self._run(predict(clip=upload, clip_id=clip_id))

    def test_predict_contract_stays_classifier_only(self) -> None:
        payload = self._predict().model_dump()

        self.assertIn("verdict", payload)
        self.assertIn("confidence", payload)
        self.assertIn("clip_id", payload)
        self.assertIn("active_modalities", payload)
        self.assertIn("inactive_modalities", payload)
        self.assertIn("gate_validity", payload)
        self.assertNotIn("legal_consequences_rag", payload)

    def test_explain_adds_rag_outputs_with_country(self) -> None:
        prediction = self._predict()
        response = self._run(
            explain(
                ExplainRequest(
                    clip_id=prediction.clip_id,
                    language="en",
                    country="UAE",
                ),
                db=self._db(),
            )
        )
        payload = response.model_dump()

        self.assertIn("narrative", payload)
        self.assertIn("incident_id", payload)
        self.assertEqual(payload["language"], "en")
        self.assertEqual(payload["legal_consequences_rag"]["country"], "UAE")
        self.assertIn("limitations_note", payload["legal_consequences_rag"])
        self.assertIn(
            payload["legal_consequences_rag"]["legal_rag_source"],
            {"mock", "real", "fallback"},
        )
        self.assertIn("rag_mode", payload["legal_consequences_rag"])
        self.assertIn("explanation_rag", payload)
        self.assertIn("incident_memory_rag", payload)

    def test_explain_returns_vlm_only_narration_mode(self) -> None:
        prediction = self._predict()
        narration_result = {
            "narrative": "Qwen-VL generated this grounded narration.",
            "vlm_summary": "Two people are visible in the incident.",
            "narration_mode": "vlm_only",
            "model_status": {"vlm": "ready", "llm": "skipped"},
            "reason_if_fallback": None,
        }

        with patch("main.generate_explanation", return_value=narration_result):
            response = self._run(
                explain(
                    ExplainRequest(clip_id=prediction.clip_id, language="en"),
                    db=self._db(),
                )
            )

        self.assertEqual(response.narration_mode, "vlm_only")
        self.assertEqual(response.model_status["llm"], "skipped")

    def test_explain_missing_country_keeps_flow_and_returns_fallback(self) -> None:
        prediction = self._predict()
        response = self._run(
            explain(
                ExplainRequest(clip_id=prediction.clip_id, language="en"),
                db=self._db(),
            )
        )
        payload = response.model_dump()

        self.assertIn("narrative", payload)
        legal_payload = payload["legal_consequences_rag"]
        if prediction.verdict == "non-violence":
            self.assertEqual(legal_payload["guardrail_status"], "passed")
            self.assertEqual(legal_payload["retrieved_legal_references"], [])
            self.assertIn("No legal consequences are suggested", legal_payload["summary"])
        else:
            self.assertEqual(legal_payload["guardrail_status"], "needs_review")
            self.assertEqual(legal_payload["warning"], "country_required")
        self.assertEqual(legal_payload["legal_rag_source"], "fallback")

    def test_legal_retry_route_uses_saved_incident(self) -> None:
        prediction = self._predict()
        explain_response = self._run(
            explain(
                ExplainRequest(
                    clip_id=prediction.clip_id,
                    language="en",
                    country="UAE",
                ),
                db=self._db(),
            )
        )

        response = self._run(
            legal_consequences(
                LegalConsequencesRequest(
                    incident_id=explain_response.incident_id,
                    language="en",
                    country="Canada",
                ),
                db=self._db(),
            )
        )

        self.assertEqual(response["legal_consequences_rag"]["country"], "Canada")
        self.assertIn(
            response["legal_consequences_rag"]["legal_rag_source"],
            {"mock", "real", "fallback"},
        )
        self.assertIn("legal_scores", response)

    def test_arabic_legal_output_translates_user_facing_text(self) -> None:
        payload = _finalize_legal_output(
            _legal_payload(),
            source="real",
            warning=None,
            language="ar",
        )

        reference = payload["retrieved_legal_references"][0]
        self.assertTrue(_contains_arabic(payload["summary"]))
        self.assertTrue(_contains_arabic(payload["limitations_note"]))
        self.assertTrue(_contains_arabic(reference["snippet"]))
        self.assertTrue(_contains_arabic(reference["section_title"]))
        self.assertNotIn("Arabic translation is not performed", payload["limitations_note"])

    def test_arabic_legal_output_preserves_non_text_reference_fields(self) -> None:
        payload = _finalize_legal_output(
            _legal_payload(),
            source="real",
            warning=None,
            language="ar",
        )
        reference = payload["retrieved_legal_references"][0]

        self.assertEqual(reference["source_url"], "demo-fixture://legal-source/uae")
        self.assertEqual(reference["article_number"], "Article 3")
        self.assertEqual(reference["score"], 0.83)
        self.assertEqual(reference["country"], "UAE")
        self.assertEqual(payload["legal_rag_source"], "real")

    def test_english_legal_output_uses_compact_top_summary(self) -> None:
        original = _legal_payload()
        payload = _finalize_legal_output(
            original,
            source="real",
            warning=None,
            language="en",
        )

        self.assertIn("Country: UAE", payload["summary"])
        self.assertIn("Relevant act:", payload["summary"])
        self.assertIn("Weapon/object context:", payload["summary"])
        self.assertIn("Possible consequence:", payload["summary"])
        self.assertIn("Limitation: This is legal context only, not legal advice.", payload["summary"])
        self.assertEqual(
            payload["retrieved_legal_references"][0]["snippet"],
            original["retrieved_legal_references"][0]["snippet"],
        )

    def test_legal_summary_removes_incomplete_confidence_fragment(self) -> None:
        original = _legal_payload()
        original["summary"] = (
            "Possible consequence: If responsibility is confirmed by authorities, "
            "in Canada, if an assault occurs with high confidence (97."
        )
        payload = _finalize_legal_output(
            original,
            source="real",
            warning=None,
            language="en",
            legal_mode="llm",
        )

        self.assertNotIn("(97.", payload["summary"])
        self.assertNotIn("confidence", payload["summary"].casefold())
        self.assertTrue(payload["summary"].rstrip().endswith("."))

    def test_legal_summary_does_not_return_unpunctuated_possible_consequence(self) -> None:
        original = _legal_payload()
        original["summary"] = "Possible consequence: Police review may lead to protective conditions"
        payload = _finalize_legal_output(
            original,
            source="real",
            warning=None,
            language="en",
            legal_mode="llm",
        )

        possible_line = next(
            line for line in payload["summary"].splitlines() if line.startswith("Possible consequence:")
        )
        self.assertTrue(possible_line.endswith("."))
        self.assertNotIn("protective conditions", possible_line)

    def test_metadata_legal_output_uses_llm_after_retrieval_when_enabled(self) -> None:
        import services.rag_adapter as adapter

        old_enabled = os.environ.get("GUARDIAN_LEGAL_LLM_ENABLED")
        original = adapter._run_metadata_legal_llm
        os.environ["GUARDIAN_LEGAL_LLM_ENABLED"] = "1"
        try:
            adapter._run_metadata_legal_llm = lambda **kwargs: (
                "Police review may lead to protective conditions."
            )
            output, scores = _metadata_legal_output(
                pred=_legal_prediction(),
                packet_summary="violent conduct involving a public confrontation",
                narrative="A confrontation appears visible.",
                country="Canada",
                language="en",
            )
        finally:
            adapter._run_metadata_legal_llm = original
            if old_enabled is None:
                os.environ.pop("GUARDIAN_LEGAL_LLM_ENABLED", None)
            else:
                os.environ["GUARDIAN_LEGAL_LLM_ENABLED"] = old_enabled

        self.assertEqual(output["legal_mode"], "llm")
        self.assertIsNone(output["reason_if_fallback"])
        self.assertIn("police review", output["summary"].casefold())
        self.assertEqual(scores["generation_score"], 0.9)

    def test_metadata_legal_output_falls_back_when_legal_llm_disabled(self) -> None:
        old_enabled = os.environ.get("GUARDIAN_LEGAL_LLM_ENABLED")
        os.environ["GUARDIAN_LEGAL_LLM_ENABLED"] = "0"
        try:
            output, scores = _metadata_legal_output(
                pred=_legal_prediction(),
                packet_summary="violent conduct involving a public confrontation",
                narrative="A confrontation appears visible.",
                country="Canada",
                language="en",
            )
        finally:
            if old_enabled is None:
                os.environ.pop("GUARDIAN_LEGAL_LLM_ENABLED", None)
            else:
                os.environ["GUARDIAN_LEGAL_LLM_ENABLED"] = old_enabled

        self.assertEqual(output["legal_mode"], "curated_fallback")
        self.assertEqual(output["reason_if_fallback"], "Legal LLM disabled")
        self.assertLess(scores["generation_score"], 0.9)


if __name__ == "__main__":
    unittest.main()
