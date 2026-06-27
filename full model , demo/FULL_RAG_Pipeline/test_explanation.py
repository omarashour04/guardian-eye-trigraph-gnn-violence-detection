import sys
from pathlib import Path

ELWEEKA_DIR = (
    Path(__file__).resolve().parent
    / "RAG_Violence_Detection"
    / "Elweeka"
)

sys.path.insert(0, str(ELWEEKA_DIR))

from rag_service.explanation_service import (
    generate_clip_explanation,
)

result = generate_clip_explanation(
    verdict="violence",
    confidence=0.91,
    packet_summary="Violence detected with confidence 0.91. Bottle detected.",
    retrieved_references=[],
    frames_ref=[],
)

print(result)