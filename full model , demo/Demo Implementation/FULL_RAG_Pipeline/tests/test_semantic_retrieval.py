import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pytest

import g4_historical_search as hs
from datetime import datetime

from rag.schemas import (
    GQSScores,
    GateScores,
    IncidentRecord,
)


@pytest.fixture
def mock_incidents():
    return [
        IncidentRecord(
            incident_id="incident-001",
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
            packet_summary="Bottle fight detected",
            narrative="Fight involving bottle",
            frames_ref=[],
        )
    ]


def test_semantic_search(
    monkeypatch,
):
    expected = [
        {
            "incident_id": "incident-001",
            "text": "bottle fight detected",
            "score": 0.92,
        }
    ]

    monkeypatch.setattr(
        hs,
        "semantic_incident_search",
        lambda query, top_k=5: expected,
    )

    results = hs.semantic_search(
        "bottle fight"
    )

    assert len(results) == 1
    assert (
        results[0]["incident_id"]
        == "incident-001"
    )
    assert results[0]["score"] == 0.92


def test_search_similar_incidents(
    monkeypatch,
    mock_incidents,
):
    monkeypatch.setattr(
        hs,
        "list_incidents",
        lambda limit=100: mock_incidents,
    )

    monkeypatch.setattr(
        hs,
        "semantic_incident_search",
        lambda query, top_k=5: [
            {
                "incident_id": "incident-001",
                "text": "same incident",
                "score": 0.99,
            },
            {
                "incident_id": "incident-002",
                "text": "similar incident",
                "score": 0.88,
            },
        ],
    )

    results = hs.search_similar_incidents(
        "incident-001"
    )

    assert len(results) == 1

    assert (
        results[0]["incident_id"]
        == "incident-002"
    )


def test_search_similar_incidents_not_found(
    monkeypatch,
):
    monkeypatch.setattr(
        hs,
        "list_incidents",
        lambda limit=100: [],
    )

    results = hs.search_similar_incidents(
        "missing-id"
    )

    assert results == []