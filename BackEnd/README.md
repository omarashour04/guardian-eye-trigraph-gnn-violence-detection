# Guardian Eye — FastAPI Backend

Violence detection demo backend
---

## What this is

A FastAPI backend that receives video clips, runs a violence classifier, generates natural-language explanations, and stores a searchable incident history. Built for the graduation defense demo.

The backend runs in **mock mode by default** — all endpoints return realistic data immediately without needing the real model, so the frontend can be built and tested in parallel.

---

## Project structure

```
FastAPI Backend/
├── main.py                  # FastAPI app — all endpoints wired here
├── schemas.py               # Pydantic request/response contracts (API shape Sara uses)
├── database.py              # SQLAlchemy model + SQLite engine
├── incident_service.py      # All DB reads/writes + evidence packet builder
├── model_service.py         # Mock predict + _real_predict() stub for Phase 4
├── explanation_service.py   # Mock narratives (EN/AR) + _real_explain() stub for Phase 5
├── overlay_service.py       # OpenCV skeleton/box renderer, MP4 export, thumbnail saver
├── cache_samples.py         # Seeds incident history with demo clips (run once)
├── requirements.txt
├── db/
│   └── guardian_eye.db      # SQLite database (auto-created on first run)
└── static/
    ├── uploads/             # Uploaded video clips
    ├── overlays/            # Rendered overlay MP4s
    └── thumbnails/          # JPEG thumbnails
```

---

## Setup

**1. Install dependencies (once)**
```bash
pip install -r requirements.txt
```

**2. Start the server**
```bash
uvicorn main:app --reload
```

Wait until you see:
```
INFO:     Application startup complete.
```

The `db/` folder and all `static/` subdirectories are created automatically — you do not need to create them manually.

**3. Seed the demo incident history (once, in a second terminal)**

Open a second terminal window while the server is still running in the first, then:
```bash
python cache_samples.py
```

This pre-loads 3 back-dated incidents so the *"tell me about the fight from a week ago"* demo query works on defense day. Only needs to be run once.

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/predict` | Upload a video clip → returns verdict, confidence, gate, GQS, telemetry |
| `POST` | `/explain` | Generate narrative + render overlay → writes full incident record to DB |
| `GET`  | `/history` | Query incident history with filters |
| `POST` | `/ask` | Natural-language query over incident history |
| `POST` | `/overlay` | Re-render overlay for an existing clip |
| `GET`  | `/incident/{id}` | Fetch a single full incident record by UUID |
| `GET`  | `/health` | Server status + mock mode flag |

Full interactive docs at **http://127.0.0.1:8000/docs** once the server is running.

---

## API contract (fixed — Sara builds against these shapes)

### `POST /predict`
**Request:** multipart form — `clip` (video file) + optional `clip_id` (string)

**Response:**
```json
{
  "clip_id": "clip_001",
  "verdict": "violence",
  "confidence": 0.94,
  "threshold": 0.51,
  "gate": {
    "skeleton": 0.34,
    "interaction": 0.41,
    "object": 0.07,
    "vit": 0.18
  },
  "gqs": {
    "q_skel": 0.91,
    "q_int": 0.88,
    "q_obj": 0.40,
    "q_po": 0.33,
    "valid_ratio": 0.97
  },
  "telemetry": {
    "people": 2,
    "peak_window": [14, 22],
    "weapon": {
      "flag": true,
      "cls": "bottle"
    }
  }
}
```

> ⚠️ Note: the weapon field uses `cls`, not `class` (Python reserved word).

---

### `POST /explain`
**Request:** `{ "clip_id": "clip_001", "language": "en" }`
Supports `language: "en"` or `"ar"`.

**Response:**
```json
{
  "incident_id": "4f2a...",
  "narrative": "The system detected a violent altercation...",
  "language": "en"
}
```

Side effect: renders overlay MP4 + thumbnail, writes full incident record to SQLite.

---

### `GET /history`
**Query params:** `verdict`, `weapon` (bool), `min_confidence`, `from`, `to`, `free_text`, `limit`, `offset`

**Response:**
```json
{
  "total": 3,
  "incidents": [
    {
      "incident_id": "4f2a...",
      "timestamp": "2026-06-06T15:30:00",
      "source": "clip_001.mp4",
      "verdict": "violence",
      "confidence": 0.94,
      "thumbnail": "static/thumbnails/clip_001.jpg",
      "overlay": "static/overlays/clip_001.mp4",
      "people_count": 2,
      "weapon_flag": true,
      "weapon_class": "bottle",
      "peak_window": [14, 22],
      "narrative_preview": "The system detected a violent altercation…"
    }
  ]
}
```

---

### `POST /ask`
**Request:** `{ "question": "tell me about the fight from a week ago", "language": "en" }`

**Response:**
```json
{
  "answer": "About a week ago, the system flagged a violent incident with 94% confidence...",
  "incidents": [ { "incident_id": "...", "thumbnail": "...", ... } ],
  "language": "en"
}
```

---

## Mock mode vs real mode

The server runs in **mock mode** by default (`GUARDIAN_MOCK=1`).  
Mock mode means all predictions and narratives are pre-written — no model is loaded.

| | Mock mode | Real mode |
|---|---|---|
| Predictions | Deterministic fake data | Real ST-HGT classifier |
| Narratives | Pre-written EN/AR strings | Qwen2.5-VL-7B-Instruct |
| Overlays | Animated stick-figure MP4 | Real skeleton from NPZ |
| DB writes | ✅ Real — always active | ✅ Real |

To switch to real mode (Phase 4+):
```bash
set GUARDIAN_MOCK=0        # Windows
uvicorn main:app --reload
```

---

## What each file does

| File | Role |
|------|------|
| `main.py` | FastAPI app. The only file uvicorn runs. Wires all services together. |
| `schemas.py` | Pydantic models. Defines every request/response field shape. |
| `database.py` | SQLAlchemy `IncidentRecord` table. Creates `db/` folder automatically. |
| `incident_service.py` | All DB logic — `save_incident()`, `query_incidents()`, `build_packet_summary()`. |
| `model_service.py` | Mock + stub for the real ST-HGT classifier (`_real_predict()`). |
| `explanation_service.py` | Mock narratives in EN/AR + stub for Qwen2.5-VL (`_real_explain()`). |
| `overlay_service.py` | OpenCV renderer. Mock animated overlay now; real NPZ overlay in Phase 4. |
| `cache_samples.py` | Seeds 3 back-dated demo incidents. Run once after server starts. |

---

## What's left for Phase 4 (wiring the real model)

1. In `model_service.py` — implement `_real_predict()` with the ST-HGT Stage G2 checkpoint.
2. In `main.py` `/explain` — uncomment the NPZ loading block and pass real arrays to `render_overlay()`.
3. In `explanation_service.py` — implement `_real_explain()` with Qwen2.5-VL-7B-Instruct 4-bit + ChromaDB RAG.

Everything else is already wired and working.

---

## Common issues

**`ConnectionRefusedError` in cache_samples.py** — the server wasn't running.  
Fix: start uvicorn in one terminal first, then run `cache_samples.py` in a second terminal.

**`404 Not Found` on `http://127.0.0.1:8000/`** — this is normal. No homepage is defined.  
Use `http://127.0.0.1:8000/docs` instead.
