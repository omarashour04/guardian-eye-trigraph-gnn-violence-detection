import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from datetime import datetime
import uuid

import pytest

import g3_incident_db
from rag.schemas import (
    GQSScores,
    GateScores,
    IncidentRecord,
)


@pytest.fixture
def sample_incident():
    return IncidentRecord(
        incident_id=str(uuid.uuid4()),
        timestamp=datetime.fromisoformat(
            "2026-06-08T20:30:00"
        ),
        source="camera-01",
        verdict="violence",
        confidence=0.91,
        gate=GateScores(
            skeleton=0.87,
            interaction=0.83,
            object=0.42,
            vit=0.94,
        ),
        gqs=GQSScores(
            q_skel=0.88,
            q_int=0.81,
            q_obj=0.70,
            q_po=0.76,
            valid_ratio=0.93,
        ),
        people_count=2,
        peak_window=[24, 48],
        weapon_flag=True,
        weapon_class="bottle",
        packet_summary="Bottle fight detected.",
        narrative="Violent interaction detected.",
        frames_ref=["frame_024.jpg"],
    )


def test_insert_and_get_incident(sample_incident):
    g3_incident_db.create_tables()

    g3_incident_db.insert_incident(
        sample_incident
    )

    retrieved = g3_incident_db.get_incident(
        sample_incident.incident_id
    )

    assert retrieved is not None
    assert (
        retrieved.incident_id
        == sample_incident.incident_id
    )

    assert retrieved.verdict == "violence"
    assert retrieved.weapon_class == "bottle"


def test_count_incidents_increases(sample_incident):
    before = g3_incident_db.count_incidents()

    g3_incident_db.insert_incident(
        sample_incident
    )

    after = g3_incident_db.count_incidents()

    assert after == before + 1


def test_duplicate_insertion_raises(sample_incident):
    g3_incident_db.insert_incident(
        sample_incident
    )

    with pytest.raises(ValueError):
        g3_incident_db.insert_incident(
            sample_incident
        )


def test_delete_incident(sample_incident):
    g3_incident_db.insert_incident(
        sample_incident
    )

    g3_incident_db.delete_incident(
        sample_incident.incident_id
    )

    retrieved = g3_incident_db.get_incident(
        sample_incident.incident_id
    )

    assert retrieved is None


def test_list_incidents_returns_records(sample_incident):
    g3_incident_db.insert_incident(
        sample_incident
    )

    incidents = g3_incident_db.list_incidents()

    assert len(incidents) > 0

    assert any(
        incident.incident_id
        == sample_incident.incident_id
        for incident in incidents
    )