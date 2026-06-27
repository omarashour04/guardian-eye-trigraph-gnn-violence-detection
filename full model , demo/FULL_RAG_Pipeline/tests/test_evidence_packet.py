import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))


from datetime import datetime

from g1_evidence_packet import (
    build_evidence_packet,
    generate_quality_note,
    get_stream_drivers,
)
from rag.schemas import (
    ClassifierInput,
    GQSScores,
    GateScores,
    Telemetry,
    WeaponInfo,
)


def make_classifier_input() -> ClassifierInput:
    return ClassifierInput(
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
        frames_ref=[
            "frame_024.jpg",
            "frame_048.jpg",
        ],
    )


def test_get_stream_drivers_returns_two_highest_scores():
    gate = GateScores(
        skeleton=0.87,
        interaction=0.83,
        object=0.42,
        vit=0.94,
    )

    drivers = get_stream_drivers(gate)

    assert len(drivers) == 2
    assert drivers[0] == {"vit": 0.94}
    assert drivers[1] == {"skeleton": 0.87}


def test_generate_quality_note_low_coverage():
    gqs = GQSScores(
        q_skel=0.90,
        q_int=0.90,
        q_obj=0.90,
        q_po=0.90,
        valid_ratio=0.40,
    )

    assert generate_quality_note(gqs) == "Low evidence coverage."


def test_generate_quality_note_strong_skeleton():
    gqs = GQSScores(
        q_skel=0.85,
        q_int=0.40,
        q_obj=0.30,
        q_po=0.50,
        valid_ratio=0.90,
    )

    assert generate_quality_note(gqs) == "Strong skeleton evidence."


def test_generate_quality_note_multiple_signals():
    gqs = GQSScores(
        q_skel=0.85,
        q_int=0.82,
        q_obj=0.30,
        q_po=0.50,
        valid_ratio=0.90,
    )

    assert (
        generate_quality_note(gqs)
        == "Strong skeleton and interaction evidence."
    )


def test_generate_quality_note_moderate():
    gqs = GQSScores(
        q_skel=0.60,
        q_int=0.50,
        q_obj=0.40,
        q_po=0.30,
        valid_ratio=0.90,
    )

    assert generate_quality_note(gqs) == "Moderate evidence quality."


def test_build_evidence_packet():
    classifier_input = make_classifier_input()

    packet = build_evidence_packet(classifier_input)

    assert packet.verdict == "violence"
    assert packet.confidence == 0.91

    assert packet.people_count == 2

    assert packet.weapon_flag is True
    assert packet.weapon_class == "bottle"

    assert len(packet.stream_drivers) == 2

    assert (
        packet.quality_note
        == "Strong skeleton and interaction evidence."
    )

    assert "Violence detected" in packet.packet_summary
    assert "Bottle detected" in packet.packet_summary