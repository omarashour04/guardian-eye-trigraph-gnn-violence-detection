from __future__ import annotations

from datetime import datetime
from typing import Dict, List

from rag.schemas import (
    ClassifierInput,
    EvidencePacket,
    GQSScores,
    GateScores,
    Telemetry,
    WeaponInfo,
)


def get_stream_drivers(gate_scores: GateScores) -> List[Dict[str, float]]:
    """Return the two strongest gate streams sorted by descending score."""

    scores = gate_scores.model_dump()
    strongest = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:2]
    return [{name: score} for name, score in strongest]


def generate_quality_note(gqs: GQSScores) -> str:
    """Generate a deterministic quality note from GQS values."""

    if gqs.valid_ratio < 0.5:
        return "Low evidence coverage."

    strong_signals: List[str] = []

    if gqs.q_skel >= 0.8:
        strong_signals.append("skeleton")

    if gqs.q_int >= 0.8:
        strong_signals.append("interaction")

    if gqs.q_obj >= 0.8:
        strong_signals.append("object")

    if len(strong_signals) == 1:
        return f"Strong {strong_signals[0]} evidence."

    if len(strong_signals) == 2:
        return f"Strong {strong_signals[0]} and {strong_signals[1]} evidence."

    if len(strong_signals) == 3:
        return (
            f"Strong {strong_signals[0]}, "
            f"{strong_signals[1]}, and {strong_signals[2]} evidence."
        )

    return "Moderate evidence quality."


def _format_people_count(people: int) -> str:
    """Format people count for packet summaries."""

    if people == 1:
        return "One person detected."
    if people == 2:
        return "Two people detected."
    return f"{people} people detected."


def _format_peak_window(peak_window: List[int]) -> str:
    """Format peak interaction window for packet summaries."""

    if len(peak_window) >= 2:
        return f"Peak interaction window frames {peak_window[0]}-{peak_window[1]}."
    if len(peak_window) == 1:
        return f"Peak interaction window frame {peak_window[0]}."
    return "Peak interaction window unavailable."


def _format_weapon_status(weapon: WeaponInfo) -> str:
    """Format weapon status for packet summaries."""

    if not weapon.flag:
        return "No weapon detected."
    if weapon.class_name:
        return f"{weapon.class_name.capitalize()} detected."
    return "Weapon detected."


def generate_packet_summary(classifier_input: ClassifierInput) -> str:
    """Generate a deterministic summary from classifier metadata."""

    verdict = classifier_input.verdict.capitalize()
    confidence = f"{classifier_input.confidence:.2f}".rstrip("0").rstrip(".")
    telemetry = classifier_input.telemetry

    return " ".join(
        [
            f"{verdict} detected with confidence {confidence}.",
            _format_people_count(telemetry.people),
            _format_peak_window(telemetry.peak_window),
            _format_weapon_status(telemetry.weapon),
        ]
    )


def build_evidence_packet(classifier_input: ClassifierInput) -> EvidencePacket:
    """Build an EvidencePacket from classifier input metadata."""

    return EvidencePacket(
        verdict=classifier_input.verdict,
        confidence=classifier_input.confidence,
        people_count=classifier_input.telemetry.people,
        peak_window=classifier_input.telemetry.peak_window,
        weapon_flag=classifier_input.telemetry.weapon.flag,
        weapon_class=classifier_input.telemetry.weapon.class_name,
        stream_drivers=get_stream_drivers(classifier_input.gate),
        quality_note=generate_quality_note(classifier_input.gqs),
        packet_summary=generate_packet_summary(classifier_input),
    )


def main() -> None:
    """Run a minimal smoke test for evidence packet construction."""

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

    evidence_packet = build_evidence_packet(classifier_input)
    print(evidence_packet)


if __name__ == "__main__":
    main()
