from __future__ import annotations

import datetime
import asyncio
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ask_rag_service import AskContext, answer_question_rag_result, validate_history_answer
from database import Base, IncidentRecord
from history_qa_service import (
    ARABIC_EMPTY_HISTORY_ANSWER,
    ENGLISH_EMPTY_HISTORY_ANSWER,
    HistoryMatch,
    history_context_items,
)
from main import ask
from schemas import AskRequest


class GroundedHistoryAnswerTests(unittest.TestCase):
    def setUp(self) -> None:
        engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(engine)
        self.db = sessionmaker(bind=engine)()

    def tearDown(self) -> None:
        self.db.close()

    def _insert(
        self,
        incident_id: str,
        filename: str,
        verdict: str,
        confidence: float,
        *,
        people: int = 3,
        narrative: str = "Saved fixture summary with rapid interaction.",
        timestamp: datetime.datetime | None = None,
    ) -> IncidentRecord:
        record = IncidentRecord(
            incident_id=incident_id,
            clip_id=f"clip-{incident_id}.mp4",
            timestamp=timestamp or datetime.datetime.utcnow(),
            source=filename,
            verdict=verdict,
            confidence=confidence,
            threshold=0.51,
            gate_json=json.dumps([0.30, 0.40, 0.10, 0.20]),
            gqs_json=json.dumps([0.8, 0.9, 0.5, 0.4, 0.95]),
            people_count=people,
            peak_window_json=json.dumps([12, 20]),
            weapon_flag=False,
            packet_summary="Stored deterministic packet summary.",
            narrative=narrative,
        )
        self.db.add(record)
        self.db.commit()
        return record

    def _ask(self, question: str, *, incident_id: str | None = None, language: str = "en"):
        return answer_question_rag_result(
            question=question,
            language=language,
            db=self.db,
            clip_id=None,
            incident_id=incident_id,
            country=None,
            fallback=lambda: "THIS GENERIC FALLBACK MUST NOT BE USED",
        )

    def test_arabic_previous_violent_incidents_use_real_fixture_fields(self) -> None:
        self._insert("violent-1", "fight_camera_07.avi", "violence", 0.873)
        self._insert("calm-1", "quiet_hallway.avi", "non-violence", 0.04)

        result = self._ask("ما الحوادث العنيفة السابقة؟", language="en")

        self.assertEqual(result.ask_mode, "grounded")
        self.assertIn("الإجابة المختصرة", result.answer)
        self.assertIn("fight_camera_07.avi", result.answer)
        self.assertIn("87.3% (0.873)", result.answer)
        self.assertNotIn("quiet_hallway.avi", result.answer)
        self.assertNotIn("THIS GENERIC FALLBACK", result.answer)

    def test_arabic_similar_incidents_exclude_current_and_rank_saved_match(self) -> None:
        self._insert("current", "current_scene.avi", "violence", 0.82, people=4)
        self._insert("similar", "stored_similar_fight.avi", "violence", 0.79, people=4)
        self._insert("different", "stored_calm_scene.avi", "non-violence", 0.03, people=4)

        with patch(
            "ask_rag_service._run_ask_llm",
            return_value=(
                "الإجابة المختصرة: نعم، يوجد حادث مشابه في السجل. "
                "الملف stored_similar_fight.avi، والحكم violence، والثقة 79.0%."
            ),
        ):
            result = self._ask("هل توجد حوادث مشابهة للحادث الحالي؟", incident_id="current")

        self.assertIn("stored_similar_fight.avi", result.answer)
        self.assertEqual(result.ask_mode, "llm")
        self.assertEqual(result.grounding_label, "LLM + GROUNDED HISTORY")
        self.assertNotIn("current_scene.avi", result.answer)
        self.assertNotIn("stored_calm_scene.avi", result.answer)
        self.assertEqual(result.related_incident_ids, ("similar",))

    def test_empty_history_returns_exact_english_safety_answer(self) -> None:
        result = self._ask("Show previous violent incidents")

        self.assertEqual(result.answer, ENGLISH_EMPTY_HISTORY_ANSWER)
        self.assertEqual(result.related_incident_ids, ())

    def test_violent_query_with_non_violent_history_returns_arabic_safety_answer(self) -> None:
        self._insert("calm-only", "only_non_violent.avi", "non-violence", 0.02)

        result = self._ask("اعرض الحوادث العنيفة السابقة")

        self.assertEqual(result.answer, ARABIC_EMPTY_HISTORY_ANSWER)
        self.assertNotIn("only_non_violent.avi", result.answer)

    def test_answer_does_not_invent_records_people_or_dates(self) -> None:
        self._insert("known", "known_record.mp4", "violence", 0.66, people=2)

        result = self._ask("List previous incidents")

        self.assertIn("known_record.mp4", result.answer)
        self.assertIn("66.0% (0.66)", result.answer)
        self.assertNotIn("hockey", result.answer.lower())
        self.assertNotIn("victim", result.answer.lower())
        self.assertNotIn("attacker", result.answer.lower())
        self.assertEqual(result.related_incident_ids, ("known",))

    def test_arabic_count_uses_all_filtered_records_but_lists_four(self) -> None:
        for index in range(6):
            self._insert(
                f"violent-{index}",
                f"week_fight_{index}.avi",
                "violence",
                0.70 + index / 100,
            )

        result = self._ask("كم عدد الحوادث العنيفة التي تم اكتشافها هذا الأسبوع؟")

        self.assertIn("عدد الحوادث المطابقة في السجل: 6", result.answer)
        self.assertIn("أول 4 سجلات من أصل 6", result.answer)
        self.assertEqual(len(result.related_incident_ids), 6)

    def test_ask_endpoint_returns_only_retrieved_rows_and_detected_language(self) -> None:
        self._insert("endpoint-current", "endpoint_current.avi", "violence", 0.82)
        self._insert("endpoint-match", "endpoint_saved_match.avi", "violence", 0.81)

        response = asyncio.run(
            ask(
                AskRequest(
                    question="هل توجد حوادث مشابهة للحادث الحالي؟",
                    language="en",
                    incident_id="endpoint-current",
                ),
                self.db,
            )
        )

        self.assertEqual(response.language, "ar")
        self.assertEqual(response.ask_mode, "grounded")
        self.assertEqual([item.incident_id for item in response.incidents], ["endpoint-match"])
        self.assertIn("endpoint_saved_match.avi", response.answer)
        self.assertNotIn("endpoint_current.avi", response.answer)

    def test_highest_confidence_is_computed_before_llm_and_uses_real_filename(self) -> None:
        self._insert("low", "lower_confidence.avi", "violence", 0.61)
        self._insert("high", "highest_real_record.avi", "violence", 0.934)

        def grounded_llm(context: AskContext) -> str:
            computation = next(
                item for item in context.context_items
                if item.get("source") == "grounded_history_computation"
            )
            evidence = computation["result"]["highest_confidence_record"]
            self.assertEqual(evidence["filename"], "highest_real_record.avi")
            self.assertEqual(evidence["confidence"], 0.934)
            return (
                "أعلى حادث من حيث نسبة الثقة هو الملف highest_real_record.avi، "
                "بالحكم violence ونسبة ثقة 93.4%."
            )

        with patch("ask_rag_service._run_ask_llm", side_effect=grounded_llm):
            result = self._ask("ما هو أعلى حادث من حيث نسبة الثقة؟")

        self.assertEqual(result.ask_mode, "llm")
        self.assertEqual(result.related_incident_ids, ("high",))
        self.assertIn("highest_real_record.avi", result.answer)
        self.assertNotIn("حادث رقم", result.answer)

    def test_violent_count_is_computed_before_llm(self) -> None:
        self._insert("v1", "violence_one.avi", "violence", 0.71)
        self._insert("v2", "violence_two.avi", "violence", 0.72)
        self._insert("nv", "calm_only.avi", "non-violence", 0.03)

        def count_llm(context: AskContext) -> str:
            computation = next(
                item for item in context.context_items
                if item.get("source") == "grounded_history_computation"
            )
            self.assertEqual(computation["result"]["violent_incident_count"], 2)
            return "عدد الحوادث المصنفة كعنف هو 2: violence_one.avi وviolence_two.avi."

        with patch("ask_rag_service._run_ask_llm", side_effect=count_llm):
            result = self._ask("كم عدد الحوادث المصنفة كعنف؟")

        self.assertEqual(result.ask_mode, "llm")
        self.assertIn("violence_one.avi", result.answer)
        self.assertNotIn("calm_only.avi", result.answer)

    def test_invalid_incident_number_is_rejected_twice_then_falls_back(self) -> None:
        self._insert("real-id", "real_history.avi", "violence", 0.88)

        with patch(
            "ask_rag_service._run_ask_llm",
            return_value="الحادث رقم 10 هو الأعلى بنسبة ثقة 99%.",
        ) as llm:
            result = self._ask("ما هو أعلى حادث من حيث نسبة الثقة؟")

        self.assertEqual(llm.call_count, 2)
        self.assertEqual(result.ask_mode, "grounded")
        self.assertIn("real_history.avi", result.answer)
        self.assertNotIn("حادث رقم 10", result.answer)
        self.assertNotIn("99%", result.answer)

    def test_invalid_confidence_is_regenerated_with_supported_value(self) -> None:
        self._insert("supported", "supported_confidence.avi", "violence", 0.84)

        with patch(
            "ask_rag_service._run_ask_llm",
            side_effect=[
                "الملف supported_confidence.avi بنسبة ثقة 99%.",
                "الملف supported_confidence.avi هو الأعلى بنسبة ثقة 84.0%.",
            ],
        ) as llm:
            result = self._ask("ما هو أعلى حادث من حيث نسبة الثقة؟")

        self.assertEqual(llm.call_count, 2)
        self.assertEqual(result.ask_mode, "llm")
        self.assertIn("84.0%", result.answer)
        self.assertNotIn("99%", result.answer)

    def test_missing_confidence_must_be_reported_as_unavailable(self) -> None:
        record = SimpleNamespace(
            incident_id="missing-confidence",
            source="missing_confidence.avi",
            clip_id="missing_confidence.avi",
            verdict="violence",
            confidence=None,
            timestamp=None,
            narrative=None,
            packet_summary=None,
            peak_window=lambda: [],
        )
        items = history_context_items(
            [HistoryMatch(record, None, "saved match", "تطابق محفوظ")]
        )
        context = AskContext("history_memory", "ما نسبة الثقة؟", "ar", items, [])

        self.assertIsNone(items[0]["confidence"])
        self.assertEqual(
            validate_history_answer("الثقة غير متاحة للملف missing_confidence.avi.", context),
            [],
        )


if __name__ == "__main__":
    unittest.main()
