from __future__ import annotations

import asyncio
import io
import os
import unittest
import uuid

from fastapi import UploadFile

from database import get_db
from main import explain, legal_consequences, predict
from schemas import ExplainRequest, LegalConsequencesRequest
from services.rag_adapter import _finalize_legal_output


os.environ.setdefault("GUARDIAN_LEGAL_AR_TRANSLATION", "fallback")


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
        self.assertEqual(
            payload["legal_consequences_rag"]["guardrail_status"],
            "needs_review",
        )
        self.assertEqual(payload["legal_consequences_rag"]["warning"], "country_required")
        self.assertEqual(payload["legal_consequences_rag"]["legal_rag_source"], "fallback")

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

    def test_english_legal_output_remains_unchanged(self) -> None:
        original = _legal_payload()
        payload = _finalize_legal_output(
            original,
            source="real",
            warning=None,
            language="en",
        )

        self.assertEqual(payload["summary"], original["summary"])
        self.assertEqual(
            payload["retrieved_legal_references"][0]["snippet"],
            original["retrieved_legal_references"][0]["snippet"],
        )


if __name__ == "__main__":
    unittest.main()
