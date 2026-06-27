import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from datetime import datetime

import pytest

import g4_historical_search as hs
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
            packet_summary="Bottle fight",
            narrative="Fight involving bottle",
            frames_ref=[],
        ),
        IncidentRecord(
            incident_id="incident-002",
            timestamp=datetime.fromisoformat(
                "2026-06-01T20:30:00"
            ),
            source="camera-02",
            verdict="non-violence",
            confidence=0.85,
            gate=GateScores(
                skeleton=0.50,
                interaction=0.40,
                object=0.30,
                vit=0.20,
            ),
            gqs=GQSScores(
                q_skel=0.50,
                q_int=0.50,
                q_obj=0.50,
                q_po=0.50,
                valid_ratio=0.90,
            ),
            people_count=1,
            peak_window=[10, 20],
            weapon_flag=False,
            weapon_class=None,
            packet_summary="Normal activity",
            narrative="People standing",
            frames_ref=[],
        ),
    ]


def test_search_by_weapon(
    monkeypatch,
    mock_incidents,
):
    monkeypatch.setattr(
        hs,
        "list_incidents",
        lambda limit=100: mock_incidents,
    )

    results = hs.search_by_weapon(
        "bottle"
    )

    assert len(results) == 1
    assert (
        results[0].weapon_class
        == "bottle"
    )


def test_search_by_verdict(
    monkeypatch,
    mock_incidents,
):
    monkeypatch.setattr(
        hs,
        "list_incidents",
        lambda limit=100: mock_incidents,
    )

    results = hs.search_by_verdict(
        "violence"
    )

    assert len(results) == 1
    assert (
        results[0].verdict
        == "violence"
    )


def test_search_by_confidence(
    monkeypatch,
    mock_incidents,
):
    monkeypatch.setattr(
        hs,
        "list_incidents",
        lambda limit=100: mock_incidents,
    )

    results = hs.search_by_confidence(
        0.90
    )

    assert len(results) == 1
    assert (
        results[0].confidence
        >= 0.90
    )


def test_search_by_date_range(
    monkeypatch,
    mock_incidents,
):
    monkeypatch.setattr(
        hs,
        "list_incidents",
        lambda limit=100: mock_incidents,
    )

    results = hs.search_by_date_range(
        datetime.fromisoformat(
            "2026-06-05T00:00:00"
        ),
        datetime.fromisoformat(
            "2026-06-10T00:00:00"
        ),
    )

    assert len(results) == 1
    assert (
        results[0].incident_id
        == "incident-001"
    )


def test_search_recent(
    monkeypatch,
    mock_incidents,
):
    monkeypatch.setattr(
        hs,
        "list_incidents",
        lambda limit=100: mock_incidents,
    )

    results = hs.search_recent()

    assert (
        results[0].timestamp
        >= results[1].timestamp
    )


def test_format_incident_summary(
    mock_incidents,
):
    summary = hs.format_incident_summary(
        mock_incidents[0]
    )

    assert "incident-001" in summary
    assert "violence" in summary
    assert "bottle" in summary

