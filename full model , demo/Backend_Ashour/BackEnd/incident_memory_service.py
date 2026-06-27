from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from database import DB_DIR, IncidentRecord


MEMORY_STORE_PATH = DB_DIR / "incident_memory_store.json"
EMBEDDING_DIM = 256


@dataclass(frozen=True)
class IncidentMemoryResult:
    incident_id: str
    clip_id: str
    source: str
    timestamp: str | None
    verdict: str
    confidence: float
    similarity: float
    retrieval_mode: str
    summary: str
    narrative: str | None
    packet_summary: str | None
    people_count: int
    peak_window: list[int]
    weapon_flag: bool
    weapon_class: str | None


def search_incident_memory(
    db: Session,
    query: str,
    *,
    current_record: IncidentRecord | None = None,
    top_k: int = 4,
    store_path: Path = MEMORY_STORE_PATH,
) -> dict[str, Any]:
    """
    Retrieve semantically similar incidents from saved records.

    The vector store is a small JSON cache rebuilt from SQLite when records
    change. Embeddings are deterministic hashed lexical vectors by default, so
    search is CPU-only and safe for local demo hardware.
    """
    documents = sync_incident_memory(db, store_path=store_path)
    if not documents:
        return {
            "retrieval_mode": "empty",
            "results": [],
            "warnings": ["incident_memory_empty"],
        }

    query_text = _memory_query_text(query, current_record)
    exclude_id = current_record.incident_id if current_record is not None else None
    filtered = [
        document for document in documents
        if document.get("incident_id") != exclude_id
    ]
    if not filtered:
        filtered = documents

    try:
        if os.getenv("GUARDIAN_MEMORY_EMBEDDINGS_ENABLED", "1") == "0":
            raise RuntimeError("GUARDIAN_MEMORY_EMBEDDINGS_ENABLED=0")
        results = _vector_search(query_text, filtered, top_k=top_k)
        retrieval_mode = "local_vector"
        warnings: list[str] = []
    except Exception as exc:
        print(f"[incident-memory] embedding search failed; using keyword fallback: {exc}")
        results = _keyword_search(query_text, filtered, top_k=top_k)
        retrieval_mode = "keyword_fallback"
        warnings = ["embedding_search_unavailable"]

    return {
        "retrieval_mode": retrieval_mode,
        "results": [_result_from_document(document, retrieval_mode) for document in results],
        "warnings": warnings,
    }


def sync_incident_memory(
    db: Session,
    *,
    store_path: Path = MEMORY_STORE_PATH,
    limit: int = 500,
) -> list[dict[str, Any]]:
    records = (
        db.query(IncidentRecord)
        .order_by(IncidentRecord.timestamp.desc())
        .limit(limit)
        .all()
    )
    expected_signature = _records_signature(records)
    store = _load_store(store_path)
    if store.get("signature") == expected_signature and isinstance(store.get("documents"), list):
        return list(store["documents"])

    documents = [_document_from_record(record) for record in records]
    _save_store(
        store_path,
        {
            "version": 1,
            "embedding": {
                "type": "hashed_lexical",
                "dimension": EMBEDDING_DIM,
            },
            "generated_at": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "signature": expected_signature,
            "documents": documents,
        },
    )
    return documents


def rebuild_incident_memory(
    db: Session,
    *,
    store_path: Path = MEMORY_STORE_PATH,
) -> dict[str, Any]:
    if store_path.exists():
        store_path.unlink()
    documents = sync_incident_memory(db, store_path=store_path)
    return {
        "store_path": str(store_path),
        "documents": len(documents),
    }


def memory_context_item(
    db: Session,
    question: str,
    *,
    current_record: IncidentRecord | None,
    language: str,
    top_k: int = 4,
) -> dict[str, Any] | None:
    search = search_incident_memory(
        db,
        question,
        current_record=current_record,
        top_k=top_k,
    )
    results: list[IncidentMemoryResult] = search["results"]
    if not results:
        return None
    return {
        "source": "incident_memory_rag",
        "retrieval_mode": search["retrieval_mode"],
        "warnings": search["warnings"],
        "query": question,
        "language": language,
        "retrieved_incidents": [
            {
                "incident_id": result.incident_id,
                "clip_id": result.clip_id,
                "camera_or_source": result.source,
                "timestamp": result.timestamp,
                "verdict": result.verdict,
                "confidence": result.confidence,
                "similarity": result.similarity,
                "summary": result.summary,
                "narrative": result.narrative,
                "packet_summary": result.packet_summary,
                "people_count": result.people_count,
                "peak_window": result.peak_window,
                "weapon_flag": result.weapon_flag,
                "weapon_class": result.weapon_class,
            }
            for result in results
        ],
        "memory_note": (
            "Retrieved from local Incident Memory RAG. Similarity is computed "
            "from saved incident summaries, narratives, and metadata; it is not "
            "a new video classification."
        ),
        "grounding_type": "semantic_incident_memory",
    }


def _document_from_record(record: IncidentRecord) -> dict[str, Any]:
    text = _record_text(record)
    return {
        "incident_id": record.incident_id,
        "clip_id": record.clip_id,
        "source": record.source,
        "timestamp": record.timestamp.isoformat() if record.timestamp else None,
        "verdict": record.verdict,
        "confidence": float(record.confidence),
        "threshold": record.threshold,
        "gate": record.gate_dict(),
        "gqs": record.gqs_dict(),
        "people_count": int(record.people_count or 0),
        "peak_window": record.peak_window(),
        "weapon_flag": bool(record.weapon_flag),
        "weapon_class": record.weapon_class,
        "packet_summary": record.packet_summary,
        "narrative": record.narrative,
        "summary": _summary_from_record(record),
        "text": text,
        "embedding": embed_text(text),
        "keywords": sorted(set(_tokens(text)))[:120],
    }


def _record_text(record: IncidentRecord) -> str:
    peak = record.peak_window()
    weapon = (
        f"weapon object {record.weapon_class}"
        if record.weapon_flag
        else "no weapon no object"
    )
    return " ".join(
        str(part)
        for part in (
            record.source,
            record.clip_id,
            record.verdict,
            f"confidence {record.confidence:.2f}",
            f"people {record.people_count}",
            f"peak frames {peak[0]} {peak[1]}" if len(peak) >= 2 else "",
            weapon,
            _gate_text(record),
            record.packet_summary or "",
            record.narrative or "",
        )
        if part
    )


def _summary_from_record(record: IncidentRecord) -> str:
    peak = record.peak_window()
    weapon = (
        f"weapon/object flag for {record.weapon_class or 'object'}"
        if record.weapon_flag
        else "no weapon/object flag"
    )
    return (
        f"{record.source}: {record.verdict} at {record.confidence:.0%} confidence, "
        f"{record.people_count} tracked people, peak frames {peak[0]}-{peak[1]}, "
        f"{weapon}."
    )


def _gate_text(record: IncidentRecord) -> str:
    gate = record.gate_dict()
    if not gate:
        return ""
    ordered = sorted(gate.items(), key=lambda item: float(item[1] or 0.0), reverse=True)
    return " ".join(f"{name} gate {float(value or 0.0):.2f}" for name, value in ordered)


def _memory_query_text(query: str, current_record: IncidentRecord | None) -> str:
    if current_record is None:
        return query
    return f"{query} {_record_text(current_record)}"


def embed_text(text: str, *, dim: int = EMBEDDING_DIM) -> list[float]:
    tokens = _tokens(text)
    if not tokens:
        return [0.0] * dim
    vector = [0.0] * dim
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        weight = _token_weight(token)
        vector[index] += sign * weight
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1e-9:
        return vector
    return [round(value / norm, 8) for value in vector]


def _vector_search(
    query_text: str,
    documents: list[dict[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    query_vector = embed_text(query_text)
    scored = []
    for document in documents:
        similarity = _cosine(query_vector, document.get("embedding") or [])
        boosted = min(1.0, similarity + _metadata_boost(query_text, document))
        item = dict(document)
        item["similarity"] = round(boosted, 6)
        scored.append(item)
    scored.sort(key=lambda item: (-float(item["similarity"]), item.get("timestamp") or ""))
    return scored[:top_k]


def _keyword_search(
    query_text: str,
    documents: list[dict[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    query_tokens = set(_tokens(query_text))
    scored = []
    for document in documents:
        document_tokens = set(document.get("keywords") or _tokens(document.get("text") or ""))
        overlap = query_tokens.intersection(document_tokens)
        score = len(overlap) / max(len(query_tokens), 1)
        score += _metadata_boost(query_text, document)
        item = dict(document)
        item["similarity"] = round(min(1.0, score), 6)
        scored.append(item)
    scored.sort(key=lambda item: (-float(item["similarity"]), item.get("timestamp") or ""))
    return scored[:top_k]


def _metadata_boost(query_text: str, document: dict[str, Any]) -> float:
    q = query_text.casefold()
    boost = 0.0
    if any(term in q for term in ("violent", "violence", "fight", "attack", "assault")):
        boost += 0.12 if document.get("verdict") == "violence" else 0.0
    if any(term in q for term in ("non-violent", "nonviolent", "normal", "calm")):
        boost += 0.12 if document.get("verdict") == "non-violence" else 0.0
    if any(term in q for term in ("weapon", "object", "knife", "bottle")):
        boost += 0.12 if document.get("weapon_flag") else 0.0
    if any(term in q for term in ("similar", "compare", "pattern", "repeated", "previous")):
        boost += 0.05
    return boost


def _result_from_document(document: dict[str, Any], retrieval_mode: str) -> IncidentMemoryResult:
    return IncidentMemoryResult(
        incident_id=str(document.get("incident_id") or ""),
        clip_id=str(document.get("clip_id") or ""),
        source=str(document.get("source") or ""),
        timestamp=document.get("timestamp"),
        verdict=str(document.get("verdict") or ""),
        confidence=float(document.get("confidence") or 0.0),
        similarity=float(document.get("similarity") or 0.0),
        retrieval_mode=retrieval_mode,
        summary=str(document.get("summary") or ""),
        narrative=document.get("narrative"),
        packet_summary=document.get("packet_summary"),
        people_count=int(document.get("people_count") or 0),
        peak_window=list(document.get("peak_window") or [0, 0]),
        weapon_flag=bool(document.get("weapon_flag")),
        weapon_class=document.get("weapon_class"),
    )


def _tokens(text: str) -> list[str]:
    raw_tokens = re.findall(r"[\w\u0600-\u06FF]+", str(text).casefold())
    expanded: list[str] = []
    for token in raw_tokens:
        if len(token) < 2:
            continue
        expanded.append(token)
        expanded.extend(_synonyms(token))
    return expanded


def _synonyms(token: str) -> list[str]:
    groups = {
        "violence": {"violent", "violence", "fight", "fighting", "attack", "assault"},
        "calm": {"normal", "calm", "nonviolent", "nonviolence", "routine"},
        "weapon": {"weapon", "object", "knife", "bottle", "dangerous"},
        "history": {"history", "previous", "past", "similar", "compare", "pattern", "repeated"},
    }
    return [label for label, values in groups.items() if token in values]


def _token_weight(token: str) -> float:
    if token in {"violence", "weapon", "history", "calm"}:
        return 1.8
    if token in {"skeleton", "interaction", "object", "vit", "people", "peak"}:
        return 1.25
    return 1.0


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return max(0.0, sum(a * b for a, b in zip(left, right)))


def _records_signature(records: list[IncidentRecord]) -> str:
    payload = "|".join(
        f"{record.incident_id}:{record.timestamp.isoformat() if record.timestamp else ''}:"
        f"{record.verdict}:{record.confidence}:{len(record.narrative or '')}:"
        f"{len(record.packet_summary or '')}"
        for record in records
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_store(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_store(path: Path, store: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
