"""
Guardian Eye — FastAPI Backend
Endpoints: POST /predict  POST /explain  GET /history  POST /ask

Run:
    uvicorn main:app --reload

Mock mode (default, for frontend development):
    GUARDIAN_MOCK=1 uvicorn main:app --reload

Real inference (Phase 4+):
    GUARDIAN_MOCK=0 uvicorn main:app --reload
"""

from __future__ import annotations
import os, uuid, shutil, datetime
from pathlib import Path
from typing import Optional

import numpy as np

from fastapi import FastAPI, UploadFile, File, Form, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from database import create_tables, get_db, IncidentRecord
from schemas import (
    PredictResponse, ExplainRequest, ExplainResponse,
    HistoryResponse, IncidentSummary, AskRequest, AskResponse,
    LegalConsequencesRequest, GateWeights, GQSScores, Telemetry, WeaponInfo,
)
from model_service import run_predict
from explanation_service import generate_explanation, answer_question
from incident_service import (
    save_incident, get_by_clip, query_incidents,
    recent_violence, build_packet_summary,
)
from overlay_service import render_overlay, render_mock_overlay, MOCK_MODE as _MOCK
from services.rag_adapter import build_full_rag_response, build_legal_response

# ── Directories ───────────────────────────────────────────────────────────────

for d in ("static/uploads", "static/thumbnails", "static/overlays", "db"):
    Path(d).mkdir(parents=True, exist_ok=True)

UPLOADS_DIR = Path("static/uploads")

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Guardian Eye",
    description="ST-HGT violence detection — demo backend",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
def on_startup():
    create_tables()
    mock = os.getenv("GUARDIAN_MOCK", "1") == "1"
    print(f"[Guardian Eye] started  mock={mock}")
    if not mock:
        from inference_classifier import load_v9_model
        load_v9_model()


# ── Shared helper ─────────────────────────────────────────────────────────────

def _to_summary(inc: IncidentRecord) -> IncidentSummary:
    pw = inc.peak_window()
    return IncidentSummary(
        incident_id      = inc.incident_id,
        timestamp        = inc.timestamp,
        source           = inc.source,
        verdict          = inc.verdict,
        confidence       = inc.confidence,
        thumbnail        = inc.thumbnail_path,
        overlay          = inc.overlay_path,
        people_count     = inc.people_count,
        weapon_flag      = inc.weapon_flag,
        weapon_class     = inc.weapon_class,
        peak_window      = pw,
        narrative_preview= (inc.narrative[:120] + "…") if inc.narrative else None,
    )


def _pred_from_record(record: IncidentRecord) -> PredictResponse:
    return PredictResponse(
        verdict=record.verdict,
        confidence=record.confidence,
        threshold=record.threshold or 0.0,
        gate=GateWeights(**record.gate_dict()),
        gqs=GQSScores(**record.gqs_dict()),
        telemetry=Telemetry(
            people=record.people_count,
            peak_window=record.peak_window(),
            weapon=WeaponInfo(
                flag=record.weapon_flag,
                cls=record.weapon_class,
            ),
        ),
        clip_id=record.clip_id,
    )


# ── POST /predict ─────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse, tags=["Inference"],
          summary="Upload a video clip and run the violence classifier")
async def predict(
    clip: UploadFile = File(..., description="Video file (.mp4 / .avi / .mov)"),
    clip_id: Optional[str] = Form(None, description="Stable ID; auto-generated if omitted"),
):
    """
    Returns **verdict, confidence, gate, GQS, telemetry, clip_id**.
    Pass the returned `clip_id` to `/explain` to generate a narrative and
    persist the full incident record.
    """
    if clip_id is None:
        suffix  = Path(clip.filename).suffix if clip.filename else ".mp4"
        clip_id = f"clip_{uuid.uuid4().hex[:8]}{suffix}"

    save_path = UPLOADS_DIR / clip_id
    with save_path.open("wb") as fh:
        shutil.copyfileobj(clip.file, fh)

    return run_predict(str(save_path), clip_id)


# ── POST /explain ─────────────────────────────────────────────────────────────

@app.post("/explain", response_model=ExplainResponse, tags=["Explanation"],
          summary="Generate narrative and persist the full incident record")
async def explain(body: ExplainRequest, db: Session = Depends(get_db)):
    """
    1. Re-runs prediction for the clip (Phase 4: pulled from in-memory cache).
    2. Builds the deterministic evidence packet.
    3. Generates a grounded narrative (mock → Qwen2.5-VL-7B in Phase 5).
    4. **Saves the full incident record** to SQLite via `incident_service`.
    5. Returns `narrative` + `incident_id`.

    Supports `language: "en"` (default) or `"ar"`.
    """
    pred   = run_predict(str(UPLOADS_DIR / body.clip_id), body.clip_id)
    packet = build_packet_summary(pred)

    # ── Overlay rendering ─────────────────────────────────────────────────────
    # Phase 3 (mock): animated placeholder MP4 + thumbnail, no NPZ needed.
    # Phase 4 (real): load NPZ, pass skeleton/int_nodes/obj_nodes arrays here.
    if _MOCK:
        ov = render_mock_overlay(
            clip_id    = body.clip_id,
            verdict    = pred.verdict,
            confidence = pred.confidence,
        )
    else:
        npz_path = Path("cache_npz") / f"{body.clip_id}.npz"
        if npz_path.exists():
            npz = np.load(str(npz_path), allow_pickle=False)
            ov = render_overlay(
                clip_id       = body.clip_id,
                frames_vit    = npz["frames_vit"],
                skeleton      = npz["skeleton"],
                int_nodes     = npz["int_nodes"],
                int_node_mask = npz["int_node_mask"],
                obj_nodes     = npz["obj_nodes"],
                obj_node_mask = npz["obj_node_mask"],
                verdict       = pred.verdict,
                confidence    = pred.confidence,
            )
        else:
            # NPZ missing for this clip_id (e.g. first explain without predict)
            ov = render_mock_overlay(body.clip_id, pred.verdict, pred.confidence)

    thumbnail_path = ov.thumbnail_path
    overlay_path   = ov.overlay_path

    narrative = generate_explanation(
        clip_id=body.clip_id,
        verdict=pred.verdict,
        language=body.language,
        pred=pred,
    )

    record = save_incident(
        db,
        clip_id        = body.clip_id,
        pred           = pred,
        narrative      = narrative,
        packet_summary = packet,
        thumbnail_path = thumbnail_path,
        overlay_path   = overlay_path,
    )

    rag = build_full_rag_response(
        pred=pred,
        packet_summary=packet,
        narrative=narrative,
        country=body.country,
        language=body.language,
    )

    return ExplainResponse(
        narrative   = narrative,
        incident_id = record.incident_id,
        language    = body.language,
        **rag,
    )


# ── POST /overlay ─────────────────────────────────────────────────────────────

@app.post("/legal-consequences", tags=["Legal"],
          summary="Generate possible legal consequences for an analysed incident")
async def legal_consequences(
    body: LegalConsequencesRequest,
    db: Session = Depends(get_db),
):
    from fastapi import HTTPException

    record = None
    if body.incident_id:
        from incident_service import get_incident as _get
        record = _get(db, body.incident_id)
    elif body.clip_id:
        record = get_by_clip(db, body.clip_id)

    if record is None:
        raise HTTPException(status_code=404, detail="Incident not found")

    pred = _pred_from_record(record)
    legal_output, legal_scores = build_legal_response(
        pred=pred,
        packet_summary=record.packet_summary or "",
        narrative=record.narrative or "",
        country=body.country,
        language=body.language,
    )
    return {
        "legal_consequences_rag": legal_output,
        "legal_scores": legal_scores,
        "language": body.language,
    }


@app.post("/overlay", tags=["Inference"],
          summary="Render (or re-render) the skeleton+box overlay for a clip")
async def overlay(
    clip_id: str = Form(..., description="clip_id returned by /predict"),
    db: Session  = Depends(get_db),
):
    """
    Renders the overlay MP4 + thumbnail for an already-uploaded clip and
    updates the incident record's paths in the DB (if a record exists).

    Returns: `{ overlay_path, thumbnail_path, clip_id }`
    """
    pred = run_predict(str(UPLOADS_DIR / clip_id), clip_id)

    if _MOCK:
        ov = render_mock_overlay(clip_id, pred.verdict, pred.confidence)
    else:
        ov = render_mock_overlay(clip_id, pred.verdict, pred.confidence)  # Phase 4: real NPZ

    # Patch existing incident record if one exists
    from incident_service import get_by_clip
    record = get_by_clip(db, clip_id)
    if record:
        record.thumbnail_path = ov.thumbnail_path
        record.overlay_path   = ov.overlay_path
        db.commit()

    return {
        "clip_id"        : clip_id,
        "overlay_path"   : ov.overlay_path,
        "thumbnail_path" : ov.thumbnail_path,
    }


# ── GET /history ──────────────────────────────────────────────────────────────

@app.get("/history", response_model=HistoryResponse, tags=["History"],
         summary="Query the incident history log with filters")
async def history(
    verdict:        Optional[str]              = Query(None, pattern="^(violence|non-violence)$"),
    weapon:         Optional[bool]             = Query(None),
    min_confidence: float                      = Query(0.0, ge=0.0, le=1.0),
    free_text:      Optional[str]              = Query(None),
    from_ts:        Optional[datetime.datetime]= Query(None, alias="from"),
    to_ts:          Optional[datetime.datetime]= Query(None, alias="to"),
    limit:          int                        = Query(50, ge=1, le=200),
    offset:         int                        = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    total, records = query_incidents(
        db,
        verdict        = verdict,
        weapon         = weapon,
        min_confidence = min_confidence,
        free_text      = free_text,
        from_ts        = from_ts,
        to_ts          = to_ts,
        limit          = limit,
        offset         = offset,
    )
    return HistoryResponse(total=total, incidents=[_to_summary(r) for r in records])


# ── POST /ask ─────────────────────────────────────────────────────────────────

@app.post("/ask", response_model=AskResponse, tags=["History"],
          summary='Natural-language query — "tell me about the fight from a week ago"')
async def ask(body: AskRequest, db: Session = Depends(get_db)):
    """
    Phase 3: static illustrative answer + recent violent incidents.
    Phase 5: NL → time/metadata filter + semantic retrieval → LLM summarise.
    """
    recent  = recent_violence(db, limit=5)
    answer  = answer_question(body.question, language=body.language)
    return AskResponse(
        answer    = answer,
        incidents = [_to_summary(r) for r in recent],
        language  = body.language,
    )


# ── GET /incident/{id} ────────────────────────────────────────────────────────

@app.get("/incident/{incident_id}", tags=["History"],
         summary="Fetch a single incident record by UUID")
async def get_incident(incident_id: str, db: Session = Depends(get_db)):
    from incident_service import get_incident as _get
    record = _get(db, incident_id)
    if record is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Incident not found")
    return {
        "incident_id"   : record.incident_id,
        "clip_id"       : record.clip_id,
        "timestamp"     : record.timestamp,
        "source"        : record.source,
        "verdict"       : record.verdict,
        "confidence"    : record.confidence,
        "threshold"     : record.threshold,
        "gate"          : record.gate_dict(),
        "gqs"           : record.gqs_dict(),
        "people_count"  : record.people_count,
        "peak_window"   : record.peak_window(),
        "weapon_flag"   : record.weapon_flag,
        "weapon_class"  : record.weapon_class,
        "thumbnail_path": record.thumbnail_path,
        "overlay_path"  : record.overlay_path,
        "packet_summary": record.packet_summary,
        "narrative"     : record.narrative,
    }


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Meta"])
def health():
    return {
        "status"  : "ok",
        "mock_mode": os.getenv("GUARDIAN_MOCK", "1") == "1",
        "phase"   : "3 — mock predictions, real DB persistence",
    }
