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
import os, uuid, shutil, datetime, re
from pathlib import Path
from typing import Optional

import numpy as np

from fastapi import FastAPI, UploadFile, File, Form, Depends, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from database import create_tables, get_db, IncidentRecord, SessionLocal
from schemas import (
    PredictResponse, ExplainRequest, ExplainResponse,
    HistoryResponse, IncidentSummary, AskRequest, AskResponse,
    LegalConsequencesRequest, GateWeights, GQSScores, Telemetry, WeaponInfo,
    ModelRoute,
)
from model_service import run_predict
from explanation_service import generate_explanation, answer_question
from incident_service import (
    save_incident, get_by_clip, get_incident as get_incident_record, query_incidents,
    recent_violence, build_packet_summary, seed_demo_incidents,
)
from overlay_service import (
    OverlayResult,
    render_overlay,
    render_mock_overlay,
    render_stream_overlays,
    overlay_status_from_paths,
    MOCK_MODE as _MOCK,
)
from services.rag_adapter import build_full_rag_response, build_legal_response
from arabic_translation_service import (
    get_translation_health,
    log_translation_health,
    log_translation_startup,
)

# ── Directories ───────────────────────────────────────────────────────────────

for d in ("static/uploads", "static/thumbnails", "static/overlays", "db"):
    Path(d).mkdir(parents=True, exist_ok=True)

log_translation_startup()

UPLOADS_DIR = Path("static/uploads")
_PLACEHOLDER_CLIP_IDS = {"", "string", "clip_id", "null", "none", "undefined"}
_UPLOAD_SOURCES: dict[str, str] = {}
_PREDICTION_CACHE: dict[str, PredictResponse] = {}

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
    if mock and os.getenv("GUARDIAN_SEED_DEMO", "1") != "0":
        db = SessionLocal()
        try:
            inserted = seed_demo_incidents(db)
            print(f"[Guardian Eye] demo seed inserted={inserted}")
        finally:
            db.close()
    if not mock:
        from inference_classifier import load_v9_model
        if os.getenv("GUARDIAN_MODEL_ROUTING_ENABLED", "0") == "1":
            from dataset_router import checkpoint_for_dataset
            startup_ckpt, _ = checkpoint_for_dataset("RLVS")
            load_v9_model(startup_ckpt)
        else:
            load_v9_model()


# ── Shared helper ─────────────────────────────────────────────────────────────

def _to_summary(inc: IncidentRecord) -> IncidentSummary:
    pw = inc.peak_window()
    route = inc.model_route()
    return IncidentSummary(
        incident_id      = inc.incident_id,
        clip_id          = inc.clip_id,
        timestamp        = inc.timestamp,
        source           = inc.source,
        verdict          = inc.verdict,
        confidence       = inc.confidence,
        thumbnail        = _existing_static_path(inc.thumbnail_path),
        overlay          = _existing_static_path(inc.overlay_path),
        people_count     = inc.people_count,
        weapon_flag      = inc.weapon_flag,
        weapon_class     = inc.weapon_class,
        peak_window      = pw,
        model_route      = ModelRoute(**route) if route else None,
        narrative_preview= (inc.narrative[:120] + "…") if inc.narrative else None,
    )


def _pred_from_record(record: IncidentRecord) -> PredictResponse:
    route = record.model_route()
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
        model_route=ModelRoute(**route) if route else None,
    )


def _existing_static_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    normalized = Path(path.replace("\\", "/"))
    if normalized.exists():
        return str(normalized)
    return None


def _upload_path_for_clip(clip_id: str) -> Optional[str]:
    path = UPLOADS_DIR / Path(clip_id).name
    if path.exists():
        return str(path)
    return None


def _stream_overlay_paths_for_clip(clip_id: str) -> dict[str, Optional[str]]:
    stem = Path(clip_id).stem
    streams = ("skeleton", "interaction", "object", "vit")
    overlays: dict[str, Optional[str]] = {}
    for stream in streams:
        path = Path("static/overlays") / f"{stem}_{stream}.mp4"
        overlays[stream] = str(path) if path.exists() else None
    return overlays


def _stream_overlay_status_for_clip(clip_id: str) -> dict[str, str]:
    return overlay_status_from_paths(_stream_overlay_paths_for_clip(clip_id))


def _cached_overlay_result(clip_id: str) -> Optional[OverlayResult]:
    stem = Path(clip_id).stem
    overlay_path = Path("static/overlays") / f"{stem}.mp4"
    thumbnail_path = Path("static/thumbnails") / f"{stem}.jpg"
    stream_paths = _stream_overlay_paths_for_clip(clip_id)
    stream_status = overlay_status_from_paths(stream_paths)
    if _MOCK and overlay_path.exists() and thumbnail_path.exists():
        placeholder_paths = {
            "skeleton": str(overlay_path),
            "interaction": str(overlay_path),
            "object": str(overlay_path),
            "vit": str(overlay_path),
        }
        print(f"[overlay] mock cache hit clip_id={clip_id!r}")
        return OverlayResult(
            overlay_path=str(overlay_path),
            thumbnail_path=str(thumbnail_path),
            overlays=placeholder_paths,
            overlay_status=overlay_status_from_paths(
                placeholder_paths,
                fallback_placeholder=True,
            ),
        )
    streams_ready = all(value == "available" for value in stream_status.values())
    if overlay_path.exists() and thumbnail_path.exists() and streams_ready:
        print(f"[overlay] cache hit clip_id={clip_id!r}")
        return OverlayResult(
            overlay_path=str(overlay_path),
            thumbnail_path=str(thumbnail_path),
            overlays=stream_paths,
            overlay_status=stream_status,
        )
    return None


def _render_or_reuse_clip_overlays(clip_id: str, pred) -> OverlayResult:
    cached = _cached_overlay_result(clip_id)
    if cached is not None:
        return cached
    return _render_clip_overlays(clip_id, pred)


def _predict_for_clip(
    clip_id: str,
    video_path: str,
    *,
    force_reprocess: bool = False,
    source: Optional[str] = None,
) -> PredictResponse:
    if not force_reprocess and clip_id in _PREDICTION_CACHE:
        print(f"[predict-cache] hit clip_id={clip_id!r}")
        return _PREDICTION_CACHE[clip_id]

    pred = run_predict(
        video_path,
        clip_id,
        force_reprocess=force_reprocess,
        source=source,
    )
    _PREDICTION_CACHE[clip_id] = pred
    return pred


def _safe_filename_part(name: Optional[str]) -> str:
    stem = Path(name or "upload").stem.strip()
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return safe[:48] or "upload"


def _safe_suffix(name: Optional[str]) -> str:
    suffix = Path(name or "").suffix.lower()
    if suffix and re.fullmatch(r"\.[A-Za-z0-9]{1,8}", suffix):
        return suffix
    return ".mp4"


def _generate_clip_id(filename: Optional[str]) -> str:
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    return f"{_safe_filename_part(filename)}_{timestamp}_{uuid.uuid4().hex[:8]}{_safe_suffix(filename)}"


def _normalize_requested_clip_id(
    requested_clip_id: Optional[str],
    filename: Optional[str],
) -> str:
    candidate = (requested_clip_id or "").strip()
    if candidate.lower() in _PLACEHOLDER_CLIP_IDS:
        return _generate_clip_id(filename)

    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(candidate).name).strip("._-")
    if (
        not safe
        or safe.lower() in _PLACEHOLDER_CLIP_IDS
        or Path(safe).stem.lower() in _PLACEHOLDER_CLIP_IDS
    ):
        return _generate_clip_id(filename)

    if not Path(safe).suffix:
        safe = f"{safe}{_safe_suffix(filename)}"
    if safe == candidate:
        # Avoid cache collisions even when a client sends a plausible explicit id.
        return _generate_clip_id(filename or safe)
    return f"{Path(safe).stem}_{uuid.uuid4().hex[:8]}{Path(safe).suffix}"


def _reject_placeholder_clip_id(clip_id: Optional[str]) -> None:
    value = (clip_id or "").strip()
    if (
        value.lower() in _PLACEHOLDER_CLIP_IDS
        or Path(value).stem.lower() in _PLACEHOLDER_CLIP_IDS
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid placeholder clip_id. Call /predict with a real upload first.",
        )


def _npz_array(npz: np.lib.npyio.NpzFile, key: str):
    if key not in npz.files:
        print(f"[overlay] NPZ key missing for stream rendering: {key}")
        return None
    return npz[key]


def _render_clip_overlays(clip_id: str, pred) -> object:
    if _MOCK:
        return render_mock_overlay(
            clip_id=clip_id,
            verdict=pred.verdict,
            confidence=pred.confidence,
        )

    npz_path = Path("cache_npz") / f"{clip_id}.npz"
    if not npz_path.exists():
        print(f"[overlay] NPZ missing for {clip_id}; falling back to mock overlay")
        return render_mock_overlay(clip_id, pred.verdict, pred.confidence)

    with np.load(str(npz_path), allow_pickle=False) as npz:
        frames_vit = _npz_array(npz, "frames_vit")
        if frames_vit is None:
            print(f"[overlay] frames_vit missing for {clip_id}; falling back to mock overlay")
            return render_mock_overlay(clip_id, pred.verdict, pred.confidence)

        skeleton = _npz_array(npz, "skeleton")
        int_nodes = _npz_array(npz, "int_nodes")
        int_edges = _npz_array(npz, "int_edges")
        int_node_mask = _npz_array(npz, "int_node_mask")
        int_edge_mask = _npz_array(npz, "int_edge_mask")
        obj_nodes = _npz_array(npz, "obj_nodes")
        obj_node_mask = _npz_array(npz, "obj_node_mask")
        po_edges = _npz_array(npz, "po_edges")
        po_edge_mask = _npz_array(npz, "po_edge_mask")

        ov = render_overlay(
            clip_id=clip_id,
            frames_vit=frames_vit,
            skeleton=skeleton,
            int_nodes=int_nodes,
            int_node_mask=int_node_mask,
            obj_nodes=obj_nodes,
            obj_node_mask=obj_node_mask,
            verdict=pred.verdict,
            confidence=pred.confidence,
        )
        ov.overlays = render_stream_overlays(
            clip_id=clip_id,
            frames_vit=frames_vit,
            skeleton=skeleton,
            int_nodes=int_nodes,
            int_edges=int_edges,
            int_node_mask=int_node_mask,
            int_edge_mask=int_edge_mask,
            obj_nodes=obj_nodes,
            obj_node_mask=obj_node_mask,
            po_edges=po_edges,
            po_edge_mask=po_edge_mask,
        )
        ov.overlay_status = overlay_status_from_paths(ov.overlays)
        return ov


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
    original_filename = clip.filename or "upload"
    requested_clip_id = clip_id
    clip_id = _normalize_requested_clip_id(requested_clip_id, original_filename)
    _UPLOAD_SOURCES[clip_id] = original_filename

    save_path = UPLOADS_DIR / clip_id
    print(
        "[predict] upload "
        f"filename={original_filename!r} "
        f"requested_clip_id={requested_clip_id!r} "
        f"clip_id={clip_id!r} "
        f"saved_path={save_path}"
    )
    with save_path.open("wb") as fh:
        shutil.copyfileobj(clip.file, fh)
    saved_size = save_path.stat().st_size if save_path.exists() else 0
    guardian_mock = os.getenv("GUARDIAN_MOCK", "1")
    real_mode_active = guardian_mock != "1"
    print(
        "[predict] saved "
        f"filename={original_filename!r} "
        f"requested_clip_id={requested_clip_id!r} "
        f"final_clip_id={clip_id!r} "
        f"saved_path={save_path} "
        f"saved_size_bytes={saved_size} "
        f"GUARDIAN_MOCK={guardian_mock!r} "
        f"real_mode_active={real_mode_active}"
    )

    return _predict_for_clip(
        clip_id,
        str(save_path),
        force_reprocess=real_mode_active,
        source=original_filename,
    )


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
    _reject_placeholder_clip_id(body.clip_id)
    pred   = _predict_for_clip(body.clip_id, str(UPLOADS_DIR / body.clip_id))
    packet = build_packet_summary(pred)

    # ── Overlay rendering ─────────────────────────────────────────────────────
    ov = _render_or_reuse_clip_overlays(body.clip_id, pred)

    thumbnail_path = ov.thumbnail_path
    overlay_path   = ov.overlay_path

    narrative_result = generate_explanation(
        clip_id=body.clip_id,
        verdict=pred.verdict,
        language=body.language,
        pred=pred,
        return_status=True,
    )
    narrative = narrative_result["narrative"]
    vlm_summary = narrative_result.get("vlm_summary")

    record = save_incident(
        db,
        clip_id        = body.clip_id,
        source         = _UPLOAD_SOURCES.get(body.clip_id, body.clip_id),
        pred           = pred,
        narrative      = narrative,
        packet_summary = packet,
        vlm_summary    = vlm_summary,
        thumbnail_path = thumbnail_path,
        overlay_path   = overlay_path,
    )

    rag = build_full_rag_response(
        pred=pred,
        packet_summary=packet,
        narrative=narrative,
        vlm_summary=vlm_summary,
        country=body.country,
        language=body.language,
    )

    return ExplainResponse(
        narrative   = narrative,
        incident_id = record.incident_id,
        language    = body.language,
        narration_mode=narrative_result["narration_mode"],
        model_status=narrative_result["model_status"],
        reason_if_fallback=narrative_result.get("reason_if_fallback"),
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
        vlm_summary=record.vlm_summary(),
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
    _reject_placeholder_clip_id(clip_id)
    pred = _predict_for_clip(clip_id, str(UPLOADS_DIR / clip_id))

    ov = _render_or_reuse_clip_overlays(clip_id, pred)

    # Patch existing incident record if one exists
    from incident_service import get_by_clip
    record = get_by_clip(db, clip_id)
    if record:
        record.thumbnail_path = ov.thumbnail_path
        record.overlay_path   = ov.overlay_path
        db.commit()

    return {
        "clip_id"        : clip_id,
        "overlay_url"    : ov.overlay_path,
        "overlay_path"   : ov.overlay_path,
        "thumbnail_path" : ov.thumbnail_path,
        "overlays"       : ov.overlays,
        "overlay_status" : ov.overlay_status,
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
    from history_qa_service import detect_question_language

    effective_language = detect_question_language(body.question, body.language)
    answer_result = answer_question(
        body.question,
        language=effective_language,
        db=db,
        clip_id=body.clip_id,
        incident_id=body.incident_id,
        country=body.country,
        return_status=True,
    )

    current = None
    wants_history = answer_result.get("selected_route") == "history_memory"
    if not wants_history:
        if body.incident_id:
            current = get_incident_record(db, body.incident_id)
        if current is None and body.clip_id:
            current = get_by_clip(db, body.clip_id)
    if current is not None:
        related = [current]
    elif wants_history:
        related = [
            record
            for incident_id in answer_result.get("related_incident_ids", [])
            if (record := get_incident_record(db, incident_id)) is not None
        ]
    else:
        _, latest = query_incidents(db, limit=1)
        related = latest
    return AskResponse(
        answer    = answer_result["answer"],
        incidents = [_to_summary(r) for r in related],
        language  = effective_language,
        ask_mode  = answer_result["ask_mode"],
        selected_route=answer_result["selected_route"],
        retrieved_context_count=answer_result["retrieved_context_count"],
        reason_if_fallback=answer_result.get("reason_if_fallback"),
        grounding_label=answer_result.get("grounding_label"),
        vlm_summary_used=bool(answer_result.get("vlm_summary_used")),
        summary_source=answer_result.get("summary_source"),
        vlm_people_count=answer_result.get("vlm_people_count"),
        vlm_violence_type=answer_result.get("vlm_violence_type"),
    )


def _ask_wants_history(question: str) -> bool:
    q = question.lower()
    if _ask_wants_current_incident(q):
        return False
    triggers = [
        "history",
        "previous incidents",
        "previous incident",
        "old incident",
        "old incidents",
        "history",
        "last week",
        "week ago",
        "yesterday",
        "compare incidents",
        "compare previous",
        "past events",
        "past event",
        "سجل",
        "السابقة",
        "سابق",
        "قديم",
        "القديمة",
        "قارن",
        "مقارنة",
        "كل الحوادث",
        "كل الفيديوهات",
        "الأسبوع الماضي",
        "اسبوع",
        "أمس",
    ]
    return any(trigger in q for trigger in triggers)


def _ask_wants_current_incident(q: str) -> bool:
    triggers = [
        "current video",
        "this video",
        "current incident",
        "this incident",
        "uploaded video",
        "model stream",
        "stream contributed",
        "streams contributed",
        "gate contribution",
        "gate contributions",
        "skeleton",
        "interaction",
        "object",
        "vit",
        "rgb",
    ]
    return any(trigger in q for trigger in triggers)


def _ask_history_records(question: str, db: Session) -> list[IncidentRecord]:
    q = question.lower()
    verdict = None
    weapon = None
    from_ts = None
    to_ts = None
    now = datetime.datetime.utcnow()

    if any(term in q for term in ["fight", "violen", "attack", "assault"]):
        verdict = "violence"
    elif any(term in q for term in ["peaceful", "normal", "calm", "non-viol"]):
        verdict = "non-violence"

    if "weapon" in q:
        weapon = True

    if "week ago" in q or "last week" in q:
        from_ts = now - datetime.timedelta(days=10)
        to_ts = now - datetime.timedelta(days=5)
    elif "yesterday" in q:
        from_ts = now - datetime.timedelta(days=2)
        to_ts = now - datetime.timedelta(days=1)

    _, records = query_incidents(
        db,
        verdict=verdict,
        weapon=weapon,
        from_ts=from_ts,
        to_ts=to_ts,
        limit=3,
    )
    return records


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
        "upload_path"   : _upload_path_for_clip(record.clip_id),
        "thumbnail_path": _existing_static_path(record.thumbnail_path),
        "overlay_path"  : _existing_static_path(record.overlay_path),
        "overlays"      : _stream_overlay_paths_for_clip(record.clip_id),
        "overlay_status": _stream_overlay_status_for_clip(record.clip_id),
        "packet_summary": record.packet_summary,
        "narrative"     : record.narrative,
        "model_route"   : record.model_route() or None,
    }


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Meta"])
def health():
    translation_health = get_translation_health()
    log_translation_health()
    return {
        "status"  : "ok",
        "mock_mode": os.getenv("GUARDIAN_MOCK", "1") == "1",
        "phase"   : "3 — mock predictions, real DB persistence",
        **translation_health,
    }
