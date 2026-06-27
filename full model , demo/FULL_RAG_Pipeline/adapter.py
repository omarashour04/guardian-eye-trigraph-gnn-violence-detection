from datetime import datetime

from rag.schemas import (
    ClassifierInput,
    GateScores,
    GQSScores,
    Telemetry,
    WeaponInfo,
)


def convert_classifier_result(result):
    """
    Convert Tera classifier output into the RAG schema.
    """

    return ClassifierInput(
        clip_id=result["clip_id"],
        incident_id=result["incident_id"],
        timestamp=datetime.fromisoformat(result["timestamp"]),
        source=result["source"],
        verdict=result["verdict"],
        confidence=result["confidence"],
        threshold=result["threshold"],
        gate=GateScores(
            skeleton=result["gate"][0],
            interaction=result["gate"][1],
            object=result["gate"][2],
            vit=result["gate"][3],
        ),
        gqs=GQSScores(
            q_skel=result["gqs"][0],
            q_int=result["gqs"][1],
            q_obj=result["gqs"][2],
            q_po=result["gqs"][3],
            valid_ratio=result["gqs"][4],
        ),
        telemetry=Telemetry(
            people=result["telemetry"]["people"],
            peak_window=result["telemetry"]["peak_window"],
            weapon=WeaponInfo(
                flag=result["telemetry"]["weapon"]["flag"],
                class_name=result["telemetry"]["weapon"]["class_name"],
            ),
        ),
        frames_ref=result["frames_ref"],
    )