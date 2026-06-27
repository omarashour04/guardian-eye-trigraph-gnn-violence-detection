from __future__ import annotations

import datetime
import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database import IncidentRecord, SessionLocal, create_tables
from incident_memory_service import (
    search_incident_memory,
    sync_incident_memory,
)


class IncidentMemoryServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        create_tables()
        self.db = SessionLocal()
        self.clip_ids: list[str] = []
        self.old_embeddings = os.environ.get("GUARDIAN_MEMORY_EMBEDDINGS_ENABLED")

    def tearDown(self) -> None:
        if self.clip_ids:
            self.db.query(IncidentRecord).filter(
                IncidentRecord.clip_id.in_(self.clip_ids)
            ).delete(synchronize_session=False)
            self.db.commit()
        self.db.close()
        if self.old_embeddings is None:
            os.environ.pop("GUARDIAN_MEMORY_EMBEDDINGS_ENABLED", None)
        else:
            os.environ["GUARDIAN_MEMORY_EMBEDDINGS_ENABLED"] = self.old_embeddings

    def _insert_incident(
        self,
        *,
        source: str,
        verdict: str = "violence",
        confidence: float = 0.82,
        weapon_flag: bool = False,
        weapon_class: str | None = None,
        narrative: str = "Rapid interaction in the peak window.",
    ) -> IncidentRecord:
        clip_id = f"memory_{uuid.uuid4().hex}.mp4"
        self.clip_ids.append(clip_id)
        record = IncidentRecord(
            incident_id=str(uuid.uuid4()),
            clip_id=clip_id,
            timestamp=datetime.datetime.utcnow(),
            source=source,
            verdict=verdict,
            confidence=confidence,
            threshold=0.51,
            gate_json=json.dumps([0.25, 0.45, 0.12, 0.18]),
            gqs_json=json.dumps([0.9, 0.86, 0.5, 0.4, 0.95]),
            people_count=3,
            peak_window_json=json.dumps([8, 18]),
            weapon_flag=weapon_flag,
            weapon_class=weapon_class,
            packet_summary="DECISION DRIVERS: interaction and skeleton were strongest.",
            narrative=narrative,
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    def test_sync_writes_local_vector_store(self) -> None:
        self._insert_incident(source="camera-memory-store")

        with tempfile.TemporaryDirectory() as temp:
            store_path = Path(temp) / "incident_memory_store.json"
            documents = sync_incident_memory(self.db, store_path=store_path)

            self.assertTrue(store_path.exists())
            self.assertTrue(documents)
            self.assertIn("embedding", documents[0])
            self.assertEqual(len(documents[0]["embedding"]), 256)

    def test_semantic_search_retrieves_relevant_weapon_incident(self) -> None:
        self._insert_incident(
            source="camera-calm-lobby",
            verdict="non-violence",
            confidence=0.12,
            narrative="Routine lobby movement without conflict.",
        )
        weapon = self._insert_incident(
            source="camera-bottle-incident",
            weapon_flag=True,
            weapon_class="bottle",
            narrative="A bottle-like object appeared near the wrist during a violent interaction.",
        )

        with tempfile.TemporaryDirectory() as temp:
            result = search_incident_memory(
                self.db,
                "Find previous weapon or bottle incidents",
                top_k=2,
                store_path=Path(temp) / "memory.json",
            )

        self.assertEqual(result["retrieval_mode"], "local_vector")
        self.assertTrue(result["results"])
        self.assertEqual(result["results"][0].incident_id, weapon.incident_id)

    def test_keyword_fallback_when_embeddings_disabled(self) -> None:
        os.environ["GUARDIAN_MEMORY_EMBEDDINGS_ENABLED"] = "0"
        self._insert_incident(source="camera-keyword-fallback", weapon_flag=True, weapon_class="knife")

        with tempfile.TemporaryDirectory() as temp:
            result = search_incident_memory(
                self.db,
                "knife weapon",
                store_path=Path(temp) / "memory.json",
            )

        self.assertEqual(result["retrieval_mode"], "keyword_fallback")
        self.assertIn("embedding_search_unavailable", result["warnings"])
        self.assertTrue(result["results"])


if __name__ == "__main__":
    unittest.main()
