from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from g3_incident_db import list_incidents
from rag.schemas import GQSScores, GateScores, IncidentRecord


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INDEX_PATH = DATA_DIR / "incident_vectors.index"
METADATA_PATH = DATA_DIR / "incident_vector_metadata.json"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Load the incident embedding model once."""

    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def build_incident_text(record: IncidentRecord) -> str:
    """Build a single searchable text string from an incident record."""

    parts = [
        record.verdict,
        f"confidence {record.confidence}",
    ]

    if record.weapon_class:
        parts.append(record.weapon_class)

    parts.extend(
        [
            record.packet_summary,
            record.narrative,
        ]
    )

    return " ".join(parts).lower()


def build_incident_vector_store(
    records: Optional[List[IncidentRecord]] = None,
) -> None:
    """Build and save the FAISS incident vector store from SQLite incidents."""

    if records is None:
        records = list_incidents()

    if not records:
        raise ValueError("No incidents available to build vector store.")

    metadata = [
        {
            "incident_id": record.incident_id,
            "text": build_incident_text(record),
        }
        for record in records
    ]

    model = get_embedding_model()
    texts = [item["text"] for item in metadata]
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    embeddings = np.asarray(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def rebuild_incident_vector_store() -> None:
    """Always rebuild the incident vector store from current SQLite incidents."""

    build_incident_vector_store()


def load_incident_vector_store() -> Tuple[faiss.IndexFlatIP, List[Dict[str, str]]]:
    """Load the saved FAISS incident index and metadata."""

    index = faiss.read_index(str(INDEX_PATH))
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return index, metadata


def semantic_incident_search(
    query: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Search incident memory semantically using FAISS."""

    if not query.strip():
        return []

    rebuild_if_missing()
    index, metadata = load_incident_vector_store()

    model = get_embedding_model()
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = index.search(
        query_embedding,
        min(top_k, len(metadata)),
    )

    results: List[Dict[str, Any]] = []
    for score, metadata_index in zip(scores[0], indices[0]):
        if metadata_index < 0:
            continue

        item = metadata[metadata_index]
        results.append(
            {
                "incident_id": item["incident_id"],
                "text": item["text"],
                "score": float(score),
            }
        )

    return results


def rebuild_if_missing() -> None:
    """Build the incident vector store when saved files are missing."""

    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        build_incident_vector_store()


def _make_mock_incident(
    verdict: str,
    confidence: float,
    weapon_class: str | None,
    packet_summary: str,
    narrative: str,
) -> IncidentRecord:
    """Create a mock incident for the smoke test."""

    return IncidentRecord(
        incident_id=str(uuid.uuid4()),
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
    """Run a smoke test for semantic incident retrieval."""

    mock_incidents = [
        _make_mock_incident(
            verdict="violence",
            confidence=0.91,
            weapon_class="bottle",
            packet_summary="Violence detected with confidence 0.91. Two people detected. Bottle detected.",
            narrative="A likely fight involving a bottle was identified in the peak window.",
        ),
        _make_mock_incident(
            verdict="non-violence",
            confidence=0.86,
            weapon_class=None,
            packet_summary="Non-violence detected with confidence 0.86. People standing peacefully.",
            narrative="The scene appears calm with no aggressive interaction.",
        ),
        _make_mock_incident(
            verdict="violence",
            confidence=0.88,
            weapon_class=None,
            packet_summary="Violence detected with confidence 0.88. Two people detected.",
            narrative="A physical altercation was detected without a visible weapon.",
        ),
    ]

    build_incident_vector_store(
        records=mock_incidents
    )
    results = semantic_incident_search(
        "bottle fight"
    )

    for result in results:
        print(result["incident_id"])
        print(f"{result['score']:.4f}")


if __name__ == "__main__":
    main()
