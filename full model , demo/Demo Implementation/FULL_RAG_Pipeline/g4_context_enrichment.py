from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from g1_evidence_packet import build_evidence_packet
from g2_reference_store import retrieve_reference
from incident_vector_store import semantic_incident_search
from rag.schemas import (
    ClassifierInput,
    GQSScores,
    GateScores,
    Telemetry,
    WeaponInfo,
)


def build_enriched_context(
    classifier_input: ClassifierInput,
    reference_top_k: int = 3,
    incident_top_k: int = 3,
) -> Dict[str, Any]:
    """Build enriched Guardian Eye context from evidence, references, and incident memory."""

    packet = build_evidence_packet(classifier_input)
    query = packet.packet_summary
    references = retrieve_reference(query, top_k=reference_top_k)
    incidents = semantic_incident_search(query, top_k=incident_top_k)

    return {
        "query": query,
        "evidence_packet": packet,
        "retrieved_references": references,
        "similar_incidents": incidents,
    }


def print_enriched_context(enriched_context: Dict[str, Any]) -> None:
    """Print enriched Guardian Eye context sections."""

    print("=================================")
    print("QUERY")
    print("=================================")
    print(enriched_context["query"])

    print("=================================")
    print("EVIDENCE PACKET")
    print("=================================")
    print(enriched_context["evidence_packet"])

    print("=================================")
    print("REFERENCE CORPUS RESULTS")
    print("=================================")
    references: List[Dict[str, Any]] = enriched_context["retrieved_references"]
    for reference in references:
        print(f"{reference['title']} ({reference['score']:.4f})")
        print(reference["snippet"])
        print("---")

    print("=================================")
    print("SIMILAR INCIDENTS")
    print("=================================")
    incidents: List[Dict[str, Any]] = enriched_context["similar_incidents"]
    for incident in incidents:
        print(f"{incident['incident_id']} ({incident['score']:.4f})")
        print(incident["text"])
        print("---")


def main() -> None:
    """Run a smoke test for enriched context construction."""

    classifier_input = ClassifierInput(
        clip_id="clip-001",
        incident_id="incident-001",
        timestamp=datetime.fromisoformat("2026-06-08T20:30:00"),
        source="camera-01",
        verdict="violence",
        confidence=0.91,
        threshold=0.75,
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
        telemetry=Telemetry(
            people=2,
            peak_window=[24, 48],
            weapon=WeaponInfo(
                flag=True,
                class_name="bottle",
            ),
        ),
        frames_ref=["frames/incident-001/frame_024.jpg", "frames/incident-001/frame_048.jpg"],
    )

    enriched_context = build_enriched_context(classifier_input)
    print_enriched_context(enriched_context)


if __name__ == "__main__":
    main()
