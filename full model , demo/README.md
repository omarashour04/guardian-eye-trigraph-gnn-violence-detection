# Guardian Eye — Demo System

Violence detection demo built on the V9 ST-HGT classifier (E_full_qgf variant, RLVS dataset).

---

## What this is

A full-stack demo application:
- **FastAPI backend** — runs inference, stores incidents, serves overlays and narratives
- **FULL_RAG_Pipeline** — legal RAG, reference corpus retrieval, incident memory
- **React/Vite frontend** — verdict panel, skeleton overlay, gate bar, history, /ask

The system has two modes controlled by `GUARDIAN_MOCK`:
- `GUARDIAN_MOCK=1` (default) — deterministic mock predictions, no GPU needed, frontend can run immediately
- `GUARDIAN_MOCK=0` — real V9 inference + Qwen2.5-VL-7B explanation (requires checkpoints + GPU)

---

## Repository layout

```
demo/
├── Demo Implementation/
│   ├── BackEnd/                    FastAPI backend
│   │   ├── main.py                 All endpoints (/predict /explain /history /ask)
│   │   ├── v9_model.py             V9 model architecture (inference-only copy)
│   │   ├── inference_preprocess.py YOLO11x-pose + ByteTrack + VideoMAE → NPZ
│   │   ├── inference_classifier.py V9 forward pass + telemetry + VRAM management
│   │   ├── model_service.py        Mock/real routing (GUARDIAN_MOCK switch)
│   │   ├── explanation_service.py  Qwen2.5-VL-7B narrative + /ask answer
│   │   ├── overlay_service.py      Skeleton + bbox overlay renderer (OpenCV)
│   │   ├── incident_service.py     SQLite persistence + evidence packet builder
│   │   ├── database.py             SQLAlchemy ORM (incidents table)
│   │   ├── schemas.py              Pydantic request/response models
│   │   ├── services/rag_adapter.py Legal RAG bridge to FULL_RAG_Pipeline
│   │   ├── requirements.txt        Base deps (FastAPI, SQLAlchemy, OpenCV…)
│   │   └── requirements_real.txt   ML deps (torch, ultralytics, transformers…)
│   │
│   ├── FULL_RAG_Pipeline/          RAG stores + legal consequences
│   │   ├── reference_corpus/       4 .txt files (violence taxonomy, caveats, examples)
│   │   ├── data/                   Pre-built FAISS indexes + SQLite stores
│   │   ├── g2_reference_store.py   RAG store #1 — bge-small + FAISS retrieval
│   │   ├── g3_incident_db.py       RAG store #2 — incident SQLite
│   │   ├── incident_vector_store.py RAG store #2 — incident FAISS (semantic search)
│   │   ├── rag_service/            Legal RAG orchestrator
│   │   └── requirements.txt        RAG deps (sentence-transformers, faiss-cpu…)
│   │
│   └── FrontEnd/                   React + Vite + shadcn/ui
│       ├── src/                    Components, pages, API hooks
│       ├── package.json
│       └── vite.config.ts
```

---

## Where to put the model checkpoints

The backend expects these files on the machine running the server. Set the paths via environment variables.

### Required

| Env var | What it points to | Notes |
|---------|------------------|-------|
| `GUARDIAN_V9_CKPT` | `E_full_qgf/best.pt` | V9 classifier checkpoint from WS training output |

The checkpoint is saved by `trigraph_v9_rlvs_train.py` at:
```
C:\Violence detection\ashour\fresh\RLVS\train_output\runs_rlvs\E_full_qgf\best.pt
```
Transfer this file to the demo machine and point `GUARDIAN_V9_CKPT` at it.

### Optional but recommended

| Env var | What it points to | Fallback if missing |
|---------|------------------|-------------------|
| `GUARDIAN_VIDEOMAE_CKPT` | `videomae_best.pt` | vit_embedding = zeros (ViT stream contributes nothing) |
| `GUARDIAN_CAL_TEMP` | float from training log | 1.0 (no calibration) |
| `GUARDIAN_QWEN_CKPT` | local path to Qwen2.5-VL-7B-Instruct | Auto-downloads from HuggingFace (~15 GB) |

The VideoMAE checkpoint is saved by `trigraph_v9_rlvs_videomae.py` at:
```
C:\Violence detection\ashour\fresh\RLVS\finetune_output\videomae_best.pt
```

The calibration temperature is printed at the end of training in the RLVS training log
(`runs_rlvs/E_full_qgf/test_metrics.json` or stdout). Look for `cal_temp=`.

YOLO weights (`yolo11x-pose.pt`, `yolo11x.pt`) are **auto-downloaded by ultralytics**
on first inference call — no manual step needed if the machine has internet.

---

## How to run

### 1. Install dependencies

```bash
# Base stack (covers mock mode — no GPU needed)
cd "Demo Implementation/BackEnd"
pip install -r requirements.txt

# Real inference stack (needs CUDA GPU)
pip install -r requirements_real.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# RAG pipeline
cd ../FULL_RAG_Pipeline
pip install -r requirements.txt
```

### 2. Start in mock mode (no checkpoints needed)

```bash
cd "Demo Implementation/BackEnd"
GUARDIAN_MOCK=1 uvicorn main:app --reload --port 8000
```

### 3. Start in real mode (after transferring checkpoints)

```bash
cd "Demo Implementation/BackEnd"
set GUARDIAN_MOCK=0
set GUARDIAN_V9_CKPT=C:\path\to\E_full_qgf\best.pt
set GUARDIAN_VIDEOMAE_CKPT=C:\path\to\videomae_best.pt
set GUARDIAN_CAL_TEMP=1.15
uvicorn main:app --reload --port 8000
```

### 4. Start the frontend

```bash
cd "Demo Implementation/FrontEnd"
npm install
npm run dev
```

---

## Inference pipeline (real mode)

```
POST /predict  (video file upload)
  │
  ├─ inference_preprocess.preprocess_video()
  │    YOLO11x-pose + ByteTrack (32 frames) → skeleton, interaction graphs
  │    YOLO11x object detection (32 frames) → object nodes, person-object edges
  │    VideoMAE encoder → vit_embedding [768]
  │    GQS quality scores [5]
  │    → saved to cache_npz/<clip_id>.npz
  │
  └─ inference_classifier.classifier_forward()
       load NPZ → build torch batch
       V9 forward (CPU→GPU→CPU, empty_cache after)
       gate[4] extracted from QualityGatedFusion
       temperature calibration → confidence
       telemetry derived from NPZ geometry (peak_window, people, weapon)
       → verdict, confidence, gate, gqs, telemetry

POST /explain  (clip_id)
  │
  ├─ load cache_npz/<clip_id>.npz → frames_vit [16,224,224,3]
  ├─ render_overlay() → skeleton + bbox MP4 over frames_vit
  ├─ build_packet_summary() → deterministic evidence text (no model)
  ├─ g2_reference_store.retrieve_reference() → top-3 RAG snippets
  ├─ Qwen2.5-VL-7B-Instruct 4-bit (loaded on demand, ~7GB VRAM)
  │    prompt: system + evidence packet + RAG snippets + 16 frames
  │    guardrail: regenerate if narrative contradicts verdict
  └─ save_incident() → SQLite + returns narrative + incident_id
```

---

## V9 model constants (do not change)

These must match the training configuration exactly:

| Constant | Value |
|----------|-------|
| T (graph frames) | 32 |
| T_vit (VideoMAE frames) | 16 |
| M (max persons) | 6 |
| N (max objects) | 8 |
| V (COCO-17 joints) | 17 |
| embed_dim | 128 |
| vit_dim | 768 |
| fusion_mode | qgf (QualityGatedFusion) |
| Gate order | skeleton, interaction, object, vit |
| GQS order | q_skel, q_int, q_obj, q_po, valid_ratio |
| RLVS threshold | 0.28 (from checkpoint) |

---

## GPU lifecycle (RTX 3090 24 GB)

1. `/predict` — V9 on CPU → moved to GPU for forward → moved back to CPU → `empty_cache()`
2. `/explain` — Qwen2.5-VL loaded on demand (~7 GB VRAM), kept warm across requests
3. The two models never occupy VRAM simultaneously
