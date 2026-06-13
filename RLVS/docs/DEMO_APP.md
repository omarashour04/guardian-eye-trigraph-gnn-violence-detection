# Guardian Eye — Demo App (Design Spec)

> Status: design spec. No code yet. Companion to `EXPLANATION_RAG_SYSTEM.md`.
> Scope: **impressive graduation-defense showcase.** Stack: **FastAPI backend + web frontend**,
> all models **local** on one RTX 3090 (24 GB), fully offline.

---

## 1. Goal & experience

A demo that, on stage, takes a clip and within seconds shows:
1. a clear **VIOLENCE / NON-VIOLENCE** verdict with a confidence,
2. a **skeleton + bounding-box overlay** playing over the clip,
3. an animated **gate bar** revealing *which signal drove the decision*,
4. an **interaction timeline** marking the peak moment,
5. a **live natural-language explanation** of what is happening,
6. a **searchable incident history** you can ask in plain English
   ("tell me about the fight from a week ago").

The "wow" is the combination: a real model verdict, made *interpretable*, made *conversational*,
and *remembered over time* — all running locally.

---

## 2. How the model works (audience-level)

Guardian Eye fuses **four complementary streams**, then makes one decision:

| Stream | Sees | Catches |
|--------|------|---------|
| **Skeleton** (pose) | body joints over time | aggressive postures, strikes |
| **Interaction** (graph) | people relative to each other | sudden closing, shoving |
| **Object / person-object** | objects near hands | weapons, thrown items |
| **VideoMAE** (appearance) | raw RGB look | overall scene cues |

A **quality-gated fusion** combines them. The gate looks at how trustworthy each stream is on this
clip (the GQS quality scores) and weights them accordingly — then the model emits **one clip-level
probability** of violence. The demo surfaces that gate so the audience sees *why* this clip was
decided the way it was (e.g. "decided mostly by interaction + pose").

> One honest line for the audience: the model outputs a single score per clip. The *timeline* and
> *people* details in the explanation are reconstructed from geometry, not from the score itself.

---

## 3. How it connects to the RAG/VLM system

A 3-stage pipeline (full detail in `EXPLANATION_RAG_SYSTEM.md`):

```
STAGE 1  Inference          clip → preprocess → classifier
                            → verdict, confidence, gate[4], GQS[5], telemetry
STAGE 2  Evidence + RAG +   telemetry → evidence packet
         VLM                + retrieved reference snippets + 16 stored frames
                            → VLM narrative ; write incident record to memory
STAGE 3  Presentation       verdict + overlay + gate bar + timeline + narrative
                            + searchable incident history
```

**Authority vs narrator** (the credibility point for evaluators): the **classifier decides**; the
**VLM only explains**. The language model never changes the verdict or confidence.

---

## 4. Architecture & GPU lifecycle

```
┌────────────┐      HTTP/JSON      ┌──────────────────────────────────────────┐
│  Web        │ ◄─────────────────► │  FastAPI backend                          │
│  frontend   │                     │  /predict  /explain  /history  /ask       │
└────────────┘                     │                                            │
                                    │  GPU lifecycle on RTX 3090 (24 GB):        │
                                    │   1. load classifier → infer → FREE        │
                                    │   2. load Qwen2.5-VL-7B 4-bit (~7GB) on     │
                                    │      demand → explain → (keep warm / free)  │
                                    │  Stores: NPZ cache, vector DB, SQLite      │
                                    └──────────────────────────────────────────┘
```

**VRAM timeline (load-on-demand, the locked decision):**
1. Classifier + preprocessing models load, run inference, then their VRAM is released.
2. The 4-bit VLM (~7 GB) is loaded only when an explanation is requested.
3. Comfortable within 24 GB; brief one-time VLM load delay (mask it with a "analyzing…" state,
   or keep the VLM warm after first use during the demo session).

---

## 5. Endpoints / contract

### `POST /predict`
Request: `{ "clip": <upload | clip_id> }`
Response:
```json
{
  "verdict": "violence",
  "confidence": 0.94,
  "threshold": 0.51,
  "gate": {"skeleton":0.34,"interaction":0.41,"object":0.07,"vit":0.18},
  "gqs": {"q_skel":0.91,"q_int":0.88,"q_obj":0.40,"q_po":0.33,"valid_ratio":0.97},
  "telemetry": { "people":2, "peak_window":[14,22], "weapon":{"flag":true,"class":"bottle"} },
  "clip_id": "cam3_clip_07"
}
```

### `POST /explain`
Request: `{ "clip_id": "cam3_clip_07" }`
Response: `{ "narrative": "...", "incident_id": "4f2a..." }`
Side effect: writes an **incident record** (schema in `EXPLANATION_RAG_SYSTEM.md` §7.1) to the
vector + structured stores.

### `GET /history`
Query params: time range, verdict, weapon, min confidence, free-text.
Response: list of incident summaries (id, timestamp, source, verdict, confidence, thumbnail).

### `POST /ask`
Request: `{ "question": "tell me about the fight from a week ago" }`
Response: `{ "answer": "...", "incidents": [ {id, timestamp, thumbnail, ...} ] }`
(parses time/metadata filter + semantic query → retrieve → summarize; see §7.3 of the RAG spec.)

---

## 6. Frontend panels

1. **Verdict** — large VIOLENCE / NON-VIOLENCE badge + confidence dial.
2. **Overlay video** — the clip with COCO-17 skeleton + bounding boxes drawn (Section 7).
3. **Gate bar** — animated horizontal bars for the 4 streams; the dominant one highlighted
   ("decided mostly by interaction + pose").
4. **Interaction timeline** — a strip over the 32 frames marking the **peak-interaction window**
   (from `int_edges` distance/speed), with a weapon-proximity marker if present.
5. **Narrative** — the live VLM explanation, streamed in.
6. **Sample-clip gallery** — curated clips for one-click demoing (pre-cached, see §8/§9).
7. **Incident History** — searchable log of past analyses + a natural-language **ask box**
   ("tell me about the fight from a week ago"), backed by `/history` and `/ask`. Each result shows
   a thumbnail and can replay its recap frames.

---

## 7. Overlay rendering

Render from data already in the NPZ:
- Frames: `frames_vit [16, 224, 224, 3]` uint8 (16 uniformly sampled frames, 224×224).
- Skeleton: `skeleton [32, 6, 17, 3]` — (x, y, conf), **normalized by max(H, W)** at preprocessing.
  To draw on the 224×224 frame, **denormalize** (multiply by the stored max(H,W), then scale to the
  224 frame), and only draw joints with `conf > 0.25`. Connect joints using the **COCO-17** edge list
  (nose–eyes–ears, shoulders–elbows–wrists, hips–knees–ankles, plus the shoulder/hip torso links).
- Boxes: `int_nodes [32, 6, :4]` → (cx, cy, w, h), same denormalization; one color per person index
  `m` (tracks are stable via ByteTrack, so color = identity).
- Objects (optional): `obj_nodes [32, 8, :4]` boxes, labeled by `cls_norm`.

Note: skeleton is sampled at `T=32` and frames at `T_vit=16`; align by fractional index (nearest
frame) when overlaying, or render the overlay on the 32-frame grid and play that.

---

## 8. GPU / performance plan

- **Sequencing**: predict (classifier) → free → explain (VLM load-on-demand). Keep the VLM warm
  after first use within a demo session to avoid repeated load latency.
- **Expected latency**: preprocessing dominates; classifier inference is fast; VLM first-token after
  load is the main visible wait — cover with a loading state.
- **Fallback**: for the **sample gallery**, pre-compute and cache predictions, overlays, and
  narratives so those clips are instant and risk-free on stage. Live uploads run the full pipeline.

---

## 9. Demo-day robustness

- **Pre-cache the gallery** (predictions + overlays + narratives + incident records) so the core
  demo never depends on live timing.
- **Low-GQS handling**: if a clip has weak quality (e.g. `q_skel` low), the gate naturally
  downweights that stream and the narrative hedges — show this as a *feature* (graceful degradation),
  not a failure.
- **Calibrated confidence**: temperature scaling already exists in the diagnostics script — use the
  calibrated probability so the displayed confidence is trustworthy.
- **Seeded incident history**: pre-load the memory store with a handful of dated incidents so the
  "fight from a week ago" query has something compelling to retrieve during the defense.

---

## 10. Build checklist

1. FastAPI skeleton with `/predict`, `/explain`, `/history`, `/ask`.
2. Wire the classifier inference path (returns verdict + gate + GQS + telemetry; frees VRAM).
3. Implement load-on-demand VLM and the explanation call (from the RAG spec).
4. Implement incident-record write on every analysis (vector + structured stores).
5. Build the overlay renderer (skeleton + boxes over `frames_vit`).
6. Build the frontend panels (verdict, overlay, gate bar, timeline, narrative, gallery, history).
7. Pre-cache the sample gallery and seed the incident history.
8. Rehearse the GPU sequence end-to-end on the 3090 and measure the visible latencies.
```
