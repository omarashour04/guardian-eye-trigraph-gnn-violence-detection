from __future__ import annotations

import datetime
import json
import sys
import unittest
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database import IncidentRecord, SessionLocal, create_tables
from explanation_service import answer_question


def _contains_arabic(text: str) -> bool:
    return any("\u0600" <= char <= "\u06ff" for char in text)


class AskCurrentContextTests(unittest.TestCase):
    def setUp(self) -> None:
        create_tables()
        self.db = SessionLocal()
        self.clip_ids: list[str] = []

    def tearDown(self) -> None:
        if self.clip_ids:
            self.db.query(IncidentRecord).filter(
                IncidentRecord.clip_id.in_(self.clip_ids)
            ).delete(synchronize_session=False)
            self.db.commit()
        self.db.close()

    def _insert_incident(
        self,
        *,
        verdict: str = "violence",
        confidence: float = 0.87,
        people_count: int = 3,
        peak_window: list[int] | None = None,
        weapon_flag: bool = False,
        weapon_class: str | None = None,
        timestamp: datetime.datetime | None = None,
        source: str = "ask-test.mp4",
        gate: list[float] | None = None,
        narrative: str | None = None,
        vlm_summary: dict | None = None,
    ) -> IncidentRecord:
        clip_id = f"ask_test_{uuid.uuid4().hex}.mp4"
        self.clip_ids.append(clip_id)
        record = IncidentRecord(
            incident_id=str(uuid.uuid4()),
            clip_id=clip_id,
            timestamp=timestamp or datetime.datetime.utcnow(),
            source=source,
            verdict=verdict,
            confidence=confidence,
            threshold=0.51,
            gate_json=json.dumps(gate or [0.31, 0.42, 0.08, 0.19]),
            gqs_json=json.dumps([0.9, 0.86, 0.5, 0.4, 0.95]),
            people_count=people_count,
            peak_window_json=json.dumps(peak_window or [8, 18]),
            weapon_flag=weapon_flag,
            weapon_class=weapon_class,
            packet_summary="DECISION DRIVERS: interaction and skeleton were strongest.",
            narrative=narrative or "Rapid closing distance was observed in the peak window.",
            vlm_summary_json=json.dumps(vlm_summary or {}),
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    def test_people_question_uses_current_clip(self) -> None:
        record = self._insert_incident(people_count=3)

        answer = answer_question(
            "How many people were involved?",
            db=self.db,
            clip_id=record.clip_id,
        )

        self.assertIn("Guardian Eye detected 3 people in the incident.", answer)
        self.assertNotIn("matching incident", answer)

    def test_stream_question_uses_gate_weights(self) -> None:
        record = self._insert_incident(gate=[0.21, 0.52, 0.07, 0.20])

        answer = answer_question(
            "Which stream contributed most?",
            db=self.db,
            clip_id=record.clip_id,
        )

        self.assertIn("Interaction", answer)
        self.assertIn("52%", answer)
        self.assertIn("Full gate weights", answer)

    def test_current_video_stream_question_does_not_use_history(self) -> None:
        current = self._insert_incident(
            source="current-video.mp4",
            gate=[0.11, 0.63, 0.04, 0.22],
        )
        self._insert_incident(
            source="previous-video.mp4",
            timestamp=datetime.datetime.utcnow() - datetime.timedelta(days=7),
            gate=[0.80, 0.10, 0.05, 0.05],
        )

        answer = answer_question(
            "Which model stream contributed most in this current video?",
            db=self.db,
            clip_id=current.clip_id,
        )

        self.assertIn("Interaction", answer)
        self.assertIn("63%", answer)
        self.assertIn("Full gate weights", answer)
        self.assertNotIn("matching incident", answer)
        self.assertNotIn("previous-video.mp4", answer)

    def test_people_question_prefers_vlm_summary_over_telemetry(self) -> None:
        record = self._insert_incident(
            people_count=5,
            vlm_summary={
                "summary_source": "current_vlm",
                "people_count": 2,
                "observed_actions": ["pushing", "physical confrontation"],
                "violence_type": "physical altercation",
                "severity_estimate": "medium",
                "visual_summary": "Two individuals engaged in a physical altercation.",
            },
        )

        answer = answer_question(
            "How many people were involved?",
            db=self.db,
            clip_id=record.clip_id,
        )

        self.assertIn("Guardian Eye detected 2 people in the incident.", answer)
        self.assertNotIn("5 tracked people", answer)

    def test_what_happened_question_prefers_vlm_summary(self) -> None:
        record = self._insert_incident(
            people_count=5,
            vlm_summary={
                "summary_source": "current_vlm",
                "people_count": 2,
                "observed_actions": ["pushing", "physical confrontation"],
                "possible_roles": ["one person appears to move toward another person"],
                "violence_type": "physical altercation",
                "severity_estimate": "medium",
                "visual_summary": "Two individuals engaged in a physical altercation.",
            },
        )

        answer = answer_question(
            "What happened in this video?",
            db=self.db,
            clip_id=record.clip_id,
        )

        self.assertIn("Guardian Eye's visual analysis", answer)
        self.assertIn("2 individuals", answer)
        self.assertIn("physical altercation", answer)
        self.assertNotIn("VLM summary", answer)

    def test_security_answer_keeps_disclaimer(self) -> None:
        record = self._insert_incident()

        answer = answer_question(
            "What should security do next?",
            db=self.db,
            clip_id=record.clip_id,
        )

        self.assertIn("decision-support alert", answer)
        self.assertIn("not treat the model output as proof of guilt", answer)

    def test_no_context_defaults_to_latest_incident(self) -> None:
        self._insert_incident(
            people_count=2,
            timestamp=datetime.datetime.utcnow() - datetime.timedelta(minutes=5),
        )
        latest = self._insert_incident(people_count=5)

        answer = answer_question("How many people were involved?", db=self.db)

        self.assertIn("Guardian Eye detected 5 people in the incident.", answer)
        self.assertNotIn(latest.clip_id, answer)

    def test_history_question_uses_date_search_not_current_clip(self) -> None:
        current = self._insert_incident(source="current-video.mp4")
        self._insert_incident(
            source="camera-week-ago",
            timestamp=datetime.datetime.utcnow() - datetime.timedelta(days=7),
            people_count=2,
        )

        answer = answer_question(
            "Tell me about the fight from last week",
            db=self.db,
            clip_id=current.clip_id,
        )

        self.assertIn("matching incident", answer)
        self.assertIn("camera-week-ago", answer)
        self.assertNotIn("current-video.mp4", answer)

    def test_arabic_current_people_answer(self) -> None:
        record = self._insert_incident(people_count=4)

        answer = answer_question(
            "كم عدد الأشخاص؟",
            language="ar",
            db=self.db,
            clip_id=record.clip_id,
        )

        self.assertIn("4", answer)
        self.assertTrue(_contains_arabic(answer))


if __name__ == "__main__":
    unittest.main()
