from __future__ import annotations

import datetime
import json
import os
import unittest
import uuid
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ask_rag_service import (
    answer_question_rag_result,
    build_ask_prompt,
    retrieve_ask_context,
    route_question,
)
from database import IncidentRecord, SessionLocal, create_tables
from explanation_service import answer_question
from incident_memory_service import MEMORY_STORE_PATH
from main import _ask_wants_history


class AskRagServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        create_tables()
        self.db = SessionLocal()
        self.clip_ids: list[str] = []
        self.old_local_only = os.environ.get("GUARDIAN_MODEL_LOCAL_ONLY")
        os.environ["GUARDIAN_MODEL_LOCAL_ONLY"] = "1"

    def tearDown(self) -> None:
        if self.clip_ids:
            self.db.query(IncidentRecord).filter(
                IncidentRecord.clip_id.in_(self.clip_ids)
            ).delete(synchronize_session=False)
            self.db.commit()
        self.db.close()
        if self.old_local_only is None:
            os.environ.pop("GUARDIAN_MODEL_LOCAL_ONLY", None)
        else:
            os.environ["GUARDIAN_MODEL_LOCAL_ONLY"] = self.old_local_only
        if MEMORY_STORE_PATH.exists():
            MEMORY_STORE_PATH.unlink()

    def _insert_incident(
        self,
        *,
        source: str = "ask-rag-test.mp4",
        verdict: str = "violence",
        confidence: float = 0.82,
        timestamp: datetime.datetime | None = None,
        narrative: str | None = None,
        vlm_summary: dict | None = None,
    ) -> IncidentRecord:
        clip_id = f"ask_rag_{uuid.uuid4().hex}.mp4"
        self.clip_ids.append(clip_id)
        record = IncidentRecord(
            incident_id=str(uuid.uuid4()),
            clip_id=clip_id,
            timestamp=timestamp or datetime.datetime.utcnow(),
            source=source,
            verdict=verdict,
            confidence=confidence,
            threshold=0.51,
            gate_json=json.dumps([0.25, 0.45, 0.12, 0.18]),
            gqs_json=json.dumps([0.9, 0.86, 0.5, 0.4, 0.95]),
            people_count=3,
            peak_window_json=json.dumps([8, 18]),
            weapon_flag=False,
            weapon_class=None,
            packet_summary="DECISION DRIVERS: interaction and skeleton were strongest.",
            narrative=narrative or "The saved narrative mentions rapid activity in the peak window.",
            vlm_summary_json=json.dumps(vlm_summary or {}),
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    def test_router_classifies_major_paths(self) -> None:
        self.assertEqual(route_question("Which stream contributed most?"), "current_incident")
        self.assertEqual(
            route_question("Which model stream contributed most in this current video?"),
            "current_incident",
        )
        self.assertEqual(
            route_question("Which gate contribution was highest in this uploaded video?"),
            "current_incident",
        )
        self.assertEqual(
            route_question("Compare interaction and skeleton for this video"),
            "current_incident",
        )
        self.assertEqual(route_question("Compare previous incidents"), "history_memory")
        self.assertEqual(route_question("Compare incidents from last week"), "history_memory")
        self.assertEqual(route_question("What are the legal consequences?"), "legal_rag")
        self.assertEqual(route_question("What should security do next?"), "reference_corpus")

    def test_ask_endpoint_history_hint_ignores_current_video_terms(self) -> None:
        self.assertFalse(
            _ask_wants_history("Which model stream contributed most in this current video?")
        )
        self.assertFalse(_ask_wants_history("Compare interaction and skeleton for this video"))
        self.assertTrue(_ask_wants_history("Compare previous incidents"))

    def test_current_incident_context_uses_saved_record_and_references(self) -> None:
        record = self._insert_incident(
            vlm_summary={
                "summary_source": "current_vlm",
                "people_count": 2,
                "violence_type": "physical altercation",
                "severity_estimate": "medium",
            }
        )

        context = retrieve_ask_context(
            route="current_incident",
            question="Which stream contributed most?",
            language="en",
            db=self.db,
            clip_id=record.clip_id,
            incident_id=None,
            country=None,
        )

        sources = {item["source"] for item in context.context_items}
        self.assertEqual(context.route, "current_incident")
        self.assertEqual(context.context_items[0]["source"], "current_vlm_summary")
        self.assertIn("current_incident", sources)
        self.assertIn("reference_corpus", sources)

    def test_answer_question_result_reports_vlm_summary_diagnostics(self) -> None:
        record = self._insert_incident(
            vlm_summary={
                "summary_source": "current_vlm",
                "people_count": 2,
                "violence_type": "physical altercation",
                "severity_estimate": "medium",
            }
        )

        result = answer_question_rag_result(
            question="What happened in this current video?",
            language="en",
            db=self.db,
            clip_id=record.clip_id,
            incident_id=None,
            country=None,
            fallback=lambda: "deterministic answer",
        )

        self.assertTrue(result.vlm_summary_used)
        self.assertEqual(result.summary_source, "current_vlm")
        self.assertEqual(result.vlm_people_count, 2)
        self.assertEqual(result.vlm_violence_type, "physical altercation")

    def test_stream_question_answers_from_gate_telemetry_before_vlm_scene(self) -> None:
        record = self._insert_incident(
            vlm_summary={
                "summary_source": "current_vlm",
                "people_count": 2,
                "violence_type": "hockey scene",
                "visual_summary": "Two people are in a hockey scene.",
            }
        )

        result = answer_question_rag_result(
            question="Which stream contributed most?",
            language="en",
            db=self.db,
            clip_id=record.clip_id,
            incident_id=None,
            country=None,
            fallback=lambda: "deterministic answer",
        )

        self.assertEqual(result.selected_route, "current_incident")
        self.assertIn("Interaction", result.answer)
        self.assertIn("45%", result.answer)
        self.assertIn("ViT/RGB", result.answer)
        self.assertNotIn("hockey", result.answer.lower())
        self.assertNotIn("VLM summary", result.answer)

    def test_history_context_retrieves_previous_incidents(self) -> None:
        self._insert_incident(
            source="camera-week-ago",
            timestamp=datetime.datetime.utcnow() - datetime.timedelta(days=7),
        )

        context = retrieve_ask_context(
            route="history_memory",
            question="Tell me about the fight from last week",
            language="en",
            db=self.db,
            clip_id=None,
            incident_id=None,
            country=None,
        )

        self.assertTrue(
            any(item.get("camera_or_source") == "camera-week-ago" for item in context.context_items)
        )

    def test_history_context_contains_only_grounded_saved_records(self) -> None:
        current = self._insert_incident(
            source="camera-current",
            narrative="Current incident involved rapid interaction near the entrance.",
        )
        self._insert_incident(
            source="camera-similar-pattern",
            narrative="A previous violent incident also involved rapid interaction near the entrance.",
        )

        context = retrieve_ask_context(
            route="history_memory",
            question="Find similar repeated patterns",
            language="en",
            db=self.db,
            clip_id=current.clip_id,
            incident_id=None,
            country=None,
        )

        history_items = [
            item for item in context.context_items
            if item.get("source") == "stored_history_record"
        ]
        self.assertTrue(history_items)
        self.assertEqual(history_items[0]["grounding_type"], "saved_incident_record")
        self.assertIn("filename", history_items[0])
        self.assertIn("confidence", history_items[0])
        self.assertFalse(
            any(item.get("source") == "reference_corpus" for item in context.context_items)
        )
        self.assertFalse(
            any(item.get("source") == "incident_memory_rag" for item in context.context_items)
        )

    def test_prompt_contains_retrieved_context_only_instruction(self) -> None:
        record = self._insert_incident()
        context = retrieve_ask_context(
            route="current_incident",
            question="Why was this violent?",
            language="en",
            db=self.db,
            clip_id=record.clip_id,
            incident_id=None,
            country=None,
        )

        prompt = build_ask_prompt(context)
        joined = "\n".join(message["content"] for message in prompt)

        self.assertIn("retrieved context only", joined.lower())
        self.assertIn(record.incident_id, joined)
        self.assertIn("Do not invent unavailable information", joined)

    def test_answer_question_falls_back_when_llm_unavailable(self) -> None:
        record = self._insert_incident()

        answer = answer_question(
            "How many people were involved?",
            language="en",
            db=self.db,
            clip_id=record.clip_id,
        )

        self.assertIn("Guardian Eye detected 3 people in the incident.", answer)

    def test_answer_question_result_returns_short_people_answer(self) -> None:
        record = self._insert_incident()

        result = answer_question_rag_result(
            question="How many people were involved?",
            language="en",
            db=self.db,
            clip_id=record.clip_id,
            incident_id=None,
            country=None,
            fallback=lambda: "deterministic answer",
        )

        self.assertEqual(result.ask_mode, "llm")
        self.assertEqual(result.selected_route, "current_incident")
        self.assertGreater(result.retrieved_context_count, 0)
        self.assertEqual(result.answer, "Guardian Eye detected 3 people in the incident.")
        self.assertIsNone(result.reason_if_fallback)


if __name__ == "__main__":
    unittest.main()
