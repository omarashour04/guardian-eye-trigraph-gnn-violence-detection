from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from g1_evidence_packet import build_evidence_packet
from g2_reference_store import retrieve_reference
from rag.schemas import (
    ClassifierInput,
    GQSScores,
    GateScores,
    Telemetry,
    WeaponInfo,
)


def build_grounded_context(
    classifier_input: ClassifierInput,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Build an evidence packet and retrieve grounding references."""

    packet = build_evidence_packet(classifier_input)
    references = retrieve_reference(packet.packet_summary, top_k=top_k)

    return {
        "query": packet.packet_summary,
        "evidence_packet": packet,
        "retrieved_references": references,
    }


def print_grounded_context(grounded_context: Dict[str, Any]) -> None:
    """Print the evidence packet and retrieved references."""

    print("Evidence Packet")
    print(grounded_context["evidence_packet"])

    print("\nRetrieved References")
    references: List[Dict[str, Any]] = grounded_context["retrieved_references"]
    for reference in references:
        print(f"- {reference['title']} ({reference['score']:.4f})")
        print(reference["snippet"])


def main() -> None:
    """Run a smoke test for Guardian Eye G1/G2 integration."""

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

    grounded_context = build_grounded_context(classifier_input)
    print_grounded_context(grounded_context)


if __name__ == "__main__":
    main()
