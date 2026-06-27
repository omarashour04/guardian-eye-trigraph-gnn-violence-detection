"""
Guardian Eye — Incident Service
Single place for all incident DB writes and reads.
main.py calls these functions; it never touches the ORM directly.

Public API
----------
save_incident(db, clip_id, pred, narrative, packet_summary,
              thumbnail_path, overlay_path)  -> IncidentRecord
get_incident(db, incident_id)               -> IncidentRecord | None
query_incidents(db, **filters)              -> list[IncidentRecord]
"""

from __future__ import annotations
import json, uuid, datetime
from typing import Optional
from sqlalchemy.orm import Session

from database import IncidentRecord
from schemas import PredictResponse


# ── Write ─────────────────────────────────────────────────────────────────────

def save_incident(
    db: Session,
    *,
    clip_id: str,
    pred: PredictResponse,
    narrative: str,
    packet_summary: str,
    thumbnail_path: Optional[str] = None,
    overlay_path: Optional[str]   = None,
) -> IncidentRecord:
    """
    Persist one full incident record from a completed predict + explain cycle.
    Called by POST /explain after the narrative has been generated.

    Parameters
    ----------
    clip_id          Stable handle returned by /predict (e.g. "cam3_clip_07.mp4").
    pred             Full PredictResponse from model_service.run_predict().
    narrative        VLM / mock narrative string.
    packet_summary   Deterministic evidence packet text (no hallucination).
    thumbnail_path   Relative URL to the JPEG thumbnail, e.g. "static/thumbnails/...jpg".
    overlay_path     Relative URL to the skeleton-overlay video, e.g. "static/overlays/...mp4".
                     Pass None until Phase 4 wires the overlay renderer.
    """
    record = IncidentRecord(
        incident_id      = str(uuid.uuid4()),
        clip_id          = clip_id,
        timestamp        = datetime.datetime.utcnow(),
        source           = clip_id,
        verdict          = pred.verdict,
        confidence       = pred.confidence,
        threshold        = pred.threshold,
        gate_json        = json.dumps([
            pred.gate.skeleton,
            pred.gate.interaction,
            pred.gate.object,
            pred.gate.vit,
        ]),
        gqs_json         = json.dumps([
            pred.gqs.q_skel,
            pred.gqs.q_int,
            pred.gqs.q_obj,
            pred.gqs.q_po,
            pred.gqs.valid_ratio,
        ]),
        people_count     = pred.telemetry.people,
        peak_window_json = json.dumps(pred.telemetry.peak_window),
        weapon_flag      = pred.telemetry.weapon.flag,
        weapon_class     = pred.telemetry.weapon.cls,
        thumbnail_path   = thumbnail_path,
        overlay_path     = overlay_path,
        packet_summary   = packet_summary,
        narrative        = narrative,
    )

    db.add(record)
    db.commit()
    db.refresh(record)
    return record


# ── Read ──────────────────────────────────────────────────────────────────────

def get_incident(db: Session, incident_id: str) -> Optional[IncidentRecord]:
    """Fetch a single incident by its UUID."""
    return db.query(IncidentRecord).filter(
        IncidentRecord.incident_id == incident_id
    ).first()


def get_by_clip(db: Session, clip_id: str) -> Optional[IncidentRecord]:
    """Return the most recent incident for a given clip_id."""
    return (
        db.query(IncidentRecord)
        .filter(IncidentRecord.clip_id == clip_id)
        .order_by(IncidentRecord.timestamp.desc())
        .first()
    )


def query_incidents(
    db: Session,
    *,
    verdict: Optional[str]           = None,
    weapon: Optional[bool]            = None,
    min_confidence: float             = 0.0,
    free_text: Optional[str]          = None,
    from_ts: Optional[datetime.datetime] = None,
    to_ts:   Optional[datetime.datetime] = None,
    limit: int  = 50,
    offset: int = 0,
) -> tuple[int, list[IncidentRecord]]:
    """
    Filtered query used by GET /history.
    Returns (total_count, page_of_records).
    """
    q = db.query(IncidentRecord)

    if verdict:
        q = q.filter(IncidentRecord.verdict == verdict)
    if weapon is not None:
        q = q.filter(IncidentRecord.weapon_flag == weapon)
    if min_confidence > 0.0:
        q = q.filter(IncidentRecord.confidence >= min_confidence)
    if from_ts:
        q = q.filter(IncidentRecord.timestamp >= from_ts)
    if to_ts:
        q = q.filter(IncidentRecord.timestamp <= to_ts)
    if free_text:
        q = q.filter(IncidentRecord.narrative.contains(free_text))

    total   = q.count()
    records = (
        q.order_by(IncidentRecord.timestamp.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return total, records


def recent_violence(db: Session, limit: int = 5) -> list[IncidentRecord]:
    """Most recent violent incidents — used by POST /ask."""
    return (
        db.query(IncidentRecord)
        .filter(IncidentRecord.verdict == "violence")
        .order_by(IncidentRecord.timestamp.desc())
        .limit(limit)
        .all()
    )


# ── Evidence packet builder ───────────────────────────────────────────────────

def build_packet_summary(pred: PredictResponse) -> str:
    """
    Deterministic text summary of classifier outputs + geometry telemetry.
    Matches the worked example in EXPLANATION_RAG_SYSTEM.md §5.
    No model inference — cannot hallucinate.
    """
    g  = pred.gate
    gq = pred.gqs
    t  = pred.telemetry

    dominant = max(
        [("skeleton", g.skeleton), ("interaction", g.interaction),
         ("object", g.object), ("vit", g.vit)],
        key=lambda x: x[1],
    )

    lines = [
        f"VERDICT: {pred.verdict.upper()}   confidence: {pred.confidence:.2f} (calibrated)",
        f"THRESHOLD: {pred.threshold:.2f}",
        f"PEOPLE: {t.people} tracked across the clip.",
        f"PEAK_WINDOW: frames {t.peak_window[0]}–{t.peak_window[1]} "
        f"(~{t.peak_window[0]/32*100:.0f}–{t.peak_window[1]/32*100:.0f}% of clip).",
    ]

    if t.weapon.flag:
        lines.append(
            f'OBJECT: detected near person wrist (class: "{t.weapon.cls}").'
        )

    lines += [
        f"DECISION DRIVERS (fusion gate): "
        f"interaction={g.interaction:.2f}, skeleton={g.skeleton:.2f}, "
        f"vit={g.vit:.2f}, object={g.object:.2f}  "
        f"[dominant: {dominant[0]}].",
        f"QUALITY: q_skel={gq.q_skel:.2f}, q_int={gq.q_int:.2f}, "
        f"q_obj={gq.q_obj:.2f}, q_po={gq.q_po:.2f}, "
        f"valid_ratio={gq.valid_ratio:.2f}.",
        "NOTE: classifier gives one clip-level score; "
        "person-level/timing claims are geometry-derived heuristics.",
    ]

    # Flag weak quality streams
    warnings = []
    if gq.q_skel < 0.5:
        warnings.append("pose evidence weak (q_skel < 0.5)")
    if gq.q_int < 0.5:
        warnings.append("interaction evidence weak (q_int < 0.5)")
    if warnings:
        lines.append("WARNINGS: " + "; ".join(warnings) + ".")

    return "\n".join(lines)
