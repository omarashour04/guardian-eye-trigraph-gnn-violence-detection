from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Union

import incident_vector_store
from bilingual_rag import run_bilingual_rag
from g3_incident_db import list_incidents
from incident_vector_store import semantic_incident_search
from rag.schemas import GQSScores, GateScores, IncidentRecord


SearchResult = Union[IncidentRecord, Dict[str, Any]]


def search_by_weapon(
    weapon_class: str,
) -> List[IncidentRecord]:
    """Return incidents whose weapon class matches case-insensitively."""

    target = weapon_class.lower()
    return [
        incident
        for incident in list_incidents()
        if incident.weapon_class is not None
        and incident.weapon_class.lower() == target
    ]


def search_by_verdict(
    verdict: str,
) -> List[IncidentRecord]:
    """Return incidents whose verdict matches case-insensitively."""

    target = verdict.lower()
    return [
        incident
        for incident in list_incidents()
        if incident.verdict.lower() == target
    ]


def search_by_confidence(
    minimum_confidence: float,
) -> List[IncidentRecord]:
    """Return incidents with confidence at or above the minimum value."""

    return [
        incident
        for incident in list_incidents()
        if incident.confidence >= minimum_confidence
    ]


def search_by_date_range(
    start_date: datetime,
    end_date: datetime,
) -> List[IncidentRecord]:
    """Return incidents whose timestamps fall within the date range."""

    return [
        incident
        for incident in list_incidents()
        if start_date <= incident.timestamp <= end_date
    ]


def search_recent(
    limit: int = 10,
) -> List[IncidentRecord]:
    """Return the latest incidents ordered by timestamp descending."""

    return sorted(
        list_incidents(limit=limit),
        key=lambda incident: incident.timestamp,
        reverse=True,
    )


def semantic_search(
    query: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Run semantic search, translating Arabic queries and results on demand."""

    return run_bilingual_rag(
        query,
        semantic_incident_search,
        top_k=top_k,
    )


def search_similar_incidents(
    incident_id: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Find historical incidents semantically similar to a given incident."""

    incidents = list_incidents()
    incident = next(
        (
            item
            for item in incidents
            if item.incident_id == incident_id
        ),
        None,
    )

    if incident is None:
        return []

    query = incident.packet_summary
    if not query:
        query = incident.narrative

    results = semantic_incident_search(
        query,
        top_k=top_k + 1,
    )

    filtered_results = [
        result
        for result in results
        if result["incident_id"] != incident_id
    ]
    return filtered_results[:top_k]


def format_incident_summary(
    incident: IncidentRecord,
) -> str:
    """Format an incident record as a human-readable summary."""

    weapon = incident.weapon_class if incident.weapon_class else "None"
    return "\n".join(
        [
            f"Incident ID: {incident.incident_id}",
            f"Timestamp: {incident.timestamp.isoformat()}",
            f"Verdict: {incident.verdict}",
            f"Confidence: {incident.confidence}",
            f"Weapon: {weapon}",
            f"Summary: {incident.packet_summary}",
        ]
    )


def print_search_results(
    results: List[SearchResult],
) -> None:
    """Pretty-print historical search results."""

    if not results:
        print("No results found.")
        return

    for result in results:
        if isinstance(result, IncidentRecord):
            print(format_incident_summary(result))
        else:
            print(f"Incident ID: {result['incident_id']}")
            print(f"Score: {result['score']:.4f}")
            print(f"Text: {result['text']}")
        print("---")


def _make_mock_incident(
    incident_id: str,
    verdict: str,
    confidence: float,
    weapon_class: str | None,
    packet_summary: str,
    narrative: str,
) -> IncidentRecord:
    """Create a mock incident for the smoke test."""

    return IncidentRecord(
        incident_id=incident_id,
        timestamp=datetime.fromisoformat("2026-06-08T20:30:00"),
        source="camera-01",
        verdict=verdict,
        confidence=confidence,
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
        weapon_flag=weapon_class is not None,
        weapon_class=weapon_class,
        packet_summary=packet_summary,
        narrative=narrative,
        frames_ref=["frames/mock/frame_024.jpg", "frames/mock/frame_048.jpg"],
    )


def main() -> None:
    """Run a smoke test for historical incident search."""

    mock_incidents = [
        _make_mock_incident(
            incident_id="mock-incident-001",
            verdict="violence",
            confidence=0.91,
            weapon_class="bottle",
            packet_summary="Violence detected with confidence 0.91. Bottle detected.",
            narrative="A likely fight involving a bottle was identified.",
        ),
        _make_mock_incident(
            incident_id="mock-incident-002",
            verdict="non-violence",
            confidence=0.86,
            weapon_class=None,
            packet_summary="Non-violence detected with confidence 0.86.",
            narrative="People were standing peacefully.",
        ),
        _make_mock_incident(
            incident_id="mock-incident-003",
            verdict="violence",
            confidence=0.94,
            weapon_class=None,
            packet_summary="Violence detected with confidence 0.94.",
            narrative="A physical altercation was detected.",
        ),
    ]

    def mock_list_incidents(limit: int = 100) -> List[IncidentRecord]:
        """Return mock incidents ordered by timestamp descending."""

        return mock_incidents[:limit]

    globals()["list_incidents"] = mock_list_incidents
    incident_vector_store.build_incident_vector_store(records=mock_incidents)

    print("Search By Weapon")
    print_search_results(search_by_weapon("bottle"))

    print("Search By Confidence")
    print_search_results(search_by_confidence(0.9))

    print("Semantic Search")
    print_search_results(semantic_search("bottle fight"))

    print("Similar Incidents")
    print_search_results(
        search_similar_incidents(
            "mock-incident-001"
        )
    )


if __name__ == "__main__":
    main()
