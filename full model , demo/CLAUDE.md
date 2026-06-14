# Guardian Eye — Claude Code Context (Demo & Full Model)

## Wiki
Project wiki: `c:/Users/lenovo/Desktop/school/graduation project/gp vault/gp/`
To log results or look up context: open vault, read `wiki/_hot.md` first.
- Update `_hot.md` and `log.md` only at checkpoints: end of a build phase, or when user
  says "checkpoint" / "save" / "wrap up"
- Do NOT update `_hot.md` or `log.md` for QUERY-only sessions

## Active task
Build the Guardian Eye demo system. Decision pending: which trained model(s) to load
for the demo. The full spec is split across two design docs (see below) — read both
before writing any code.

## Design spec docs (READ BEFORE CODING)
Full architecture, API contract, NPZ schema, evidence packet format, prompt template,
overlay rendering, GPU lifecycle, and build checklist are in:
- `c:/Users/lenovo/Desktop/school/graduation project/fresh/RLVS/docs/DEMO_APP.md`
- `c:/Users/lenovo/Desktop/school/graduation project/fresh/RLVS/docs/EXPLANATION_RAG_SYSTEM.md`

The same files are mirrored to:
- `c:/Users/lenovo/Desktop/school/graduation project/fresh/Paper and docs/docs/DEMO_APP.md`
- `c:/Users/lenovo/Desktop/school/graduation project/fresh/Paper and docs/docs/EXPLANATION_RAG_SYSTEM.md`

## Q1 portfolio — trained models (all COMPLETE ✅)
All four models are V9, E_full_qgf variant (deployed for interpretability — gate weights
are quality-conditioned). Checkpoints live on the WS under each dataset's folder.

| Dataset  | Macro-F1 | AUC    | Threshold | Notes                                      |
|----------|----------|--------|-----------|--------------------------------------------|
| RLVS     | 0.9900   | 0.9971 | 0.28      | Strongest — surveillance-style footage     |
| HF       | 0.9350   | 0.9780 | 0.63      | Hockey broadcast; object stream hurts (FPs)|
| RWF-2000 | 0.9175   | 0.9617 | 0.36      | In-the-wild footage                        |
| NTU      | 0.8744   | 0.9089 | 0.11      | Low-res CCTV/Mobile; VideoMAE carries most signal|

All checkpoints are the **E_full_qgf** variant. LW variant marginally outperforms on AUC
(≤0.003 gap) but QGF is used for interpretability (gate weights exposed to the demo UI).

## Demo model decision (user has NOT yet decided — confirm before building inference path)
Three options discussed:
1. **RLVS only** (recommended) — strongest model, surveillance-style clips. Demo value is
   in the explanation pipeline, not multi-dataset coverage.
2. **Dataset-aware routing** — user selects domain; system loads the right checkpoint.
3. **Other** — wait for user direction before implementing the inference path.

## System architecture (from spec — do not deviate without user confirmation)

```
FastAPI backend  (/predict  /explain  /history  /ask)
      │
      ├─ Stage 1: Inference
      │     clip → preprocess → V9 classifier
      │     → verdict, confidence, gate[4], GQS[5], telemetry
      │     → FREE classifier VRAM
      │
      ├─ Stage 2: Explanation
      │     telemetry → evidence packet (deterministic, no model)
      │     + RAG store #1 (reference corpus, local embeddings)
      │     + 16 stored frames_vit from NPZ
      │     → Qwen2.5-VL-7B 4-bit (loaded on demand, ~7GB VRAM)
      │     → 2-4 sentence grounded narrative
      │     → write incident record to RAG store #2 + SQLite
      │
      └─ Stage 3: Presentation (web frontend)
            verdict badge, skeleton+bbox overlay, gate bar,
            interaction timeline, narrative, gallery, incident history
```

**Authority rule (NEVER violate):** the classifier decides verdict and confidence. The VLM
is a narrator only — it cannot change or re-decide the verdict.

## V9 model invariants (inherited — do NOT change)
- Framework: PyTorch + HuggingFace Transformers, local WS, no Modal
- embed_dim = 768 (VideoMAE-Base CLS mean-pool)
- NPZ schema locked. frames_vit shape: [16, 224, 224, 3] uint8
- Graph constants: T=32, T_vit=16, M=6, N=8, V=17
- GATv2: mean-over-heads (NOT per-head einsum)
- gate shape from QualityGatedFusion: [B, 4], softmax, order = (skeleton, interaction, object, vit)
- GQS shape: [5] = [q_skel, q_int, q_obj, q_po, valid_ratio]
- Calibrated temperature scaling is available in diagnostics scripts; use calibrated p(violence)
  for displayed confidence. Exception: HF is overconfident (T*=0.567) — report raw there.

## NPZ keys used by the demo (from EXPLANATION_RAG_SYSTEM.md §4.4)
| Key            | Shape           | dtype | Use                                    |
|----------------|-----------------|-------|----------------------------------------|
| skeleton       | [32, 6, 17, 3]  | f32   | COCO-17 (x,y,conf), norm by max(H,W)  |
| int_nodes      | [32, 6, 7]      | f32   | cx,cy,w,h,conf,speed,continuity        |
| int_edges      | [32, 6, 6, 4]   | f32   | dist,iou,close,rel_spd (peak window)   |
| int_node_mask  | [32, 6]         | bool  | valid person                           |
| int_edge_mask  | [32, 6, 6]      | bool  | valid pair                             |
| obj_nodes      | [32, 8, 6]      | f32   | cx,cy,w,h,conf,cls_norm                |
| obj_node_mask  | [32, 8]         | bool  | valid object                           |
| po_edges       | [32, 6, 8, 5]   | f32   | wrist_d,body_d,iou,near_wrist,near_body|
| po_edge_mask   | [32, 6, 8]      | bool  | valid person-object edge               |
| gqs            | [5]             | f32   | quality scores                         |
| frames_vit     | [16, 224, 224, 3]| uint8| RGB frames for VLM + overlay           |

## Overlay rendering rules
- Denormalize skeleton: multiply (x,y) by stored max(H,W), then scale to the 224-px frame.
  Only draw joints with conf > 0.25. Use COCO-17 edge list.
- Denormalize boxes: int_nodes[:, :, :4] same denorm as skeleton.
- skeleton is T=32 frames; frames_vit is T_vit=16 — align by fractional index (nearest).
- Color per person index m (ByteTrack IDs are stable, so color = identity across frames).

## API contract (from DEMO_APP.md §5)
- POST /predict → verdict, confidence, threshold, gate[4], gqs[5], telemetry, clip_id
- POST /explain → narrative, incident_id  (side effect: writes incident record)
- GET  /history → list of incident summaries (time range, verdict, weapon, confidence filters)
- POST /ask     → answer + incidents  (NL → structured filter + semantic retrieval → LLM summary)

## GPU lifecycle (RTX 3090 24GB — LOCKED DECISION)
1. Load classifier + preprocessing models → run inference → FREE VRAM
2. Load Qwen2.5-VL-7B 4-bit (~7GB) on demand → explain → keep warm in session
Cover VLM load latency with an "Analyzing…" UI state on first call.

## Incident record schema (RAG store #2)
incident_id (uuid), timestamp (ISO-8601), source (clip/camera name), verdict, confidence,
gqs[5], gate[4], people_count, peak_window [int,int], weapon_flag (bool+class),
packet_summary (str), narrative (str), frames_ref (path to stored frames/thumbnail)

## Demo-day robustness rules
- Pre-cache the sample gallery (predictions + overlays + narratives + incident records)
  so the core demo never depends on live timing.
- Seed the incident history store with dated past incidents so "fight from a week ago" has
  something compelling to retrieve.
- Low-GQS clips: the gate naturally downweights the weak stream — present this as graceful
  degradation, not failure.
- Calibrated confidence: use temperature-scaled p_violence for displayed confidence.
  (Exception: HF raw — see invariants above.)

## Build checklist (from DEMO_APP.md §10 + EXPLANATION_RAG_SYSTEM.md §11)
1. FastAPI skeleton — /predict, /explain, /history, /ask
2. Inference path — loads checkpoint, returns verdict + gate + GQS + telemetry, frees VRAM
3. Evidence-packet builder — deterministic text from NPZ telemetry (no model, no hallucination)
4. RAG store #1 — reference corpus (violence taxonomy, non-violence cats, CUE-Net caveat,
   few-shot exemplars); local embedding model (bge-small or e5-small)
5. RAG store #2 + SQLite — incident record write on every analysis; vector + structured stores
6. VLM integration — Qwen2.5-VL-7B 4-bit, load-on-demand, prompt template + guardrails
7. Historical query flow — NL → filter + semantic → retrieve → LLM summary
8. Overlay renderer — skeleton + bbox over frames_vit, COCO-17 edges
9. Frontend panels — verdict, overlay, gate bar, interaction timeline, narrative, gallery, history
10. Pre-cache gallery + seed incident history
11. End-to-end rehearsal on 3090: measure visible latencies, confirm GPU sequence

## VLM guardrails (MUST implement — never relax)
1. Verdict and confidence are fixed facts passed in; regenerate if narrative flips them.
2. Person attribution and timing are heuristic — always hedge: "appears to", "around frames X–Y".
3. No objects/people may be named unless present in telemetry or visible in frames.

## Code style
- tqdm on every loop with unit label
- Inline comments on every non-obvious decision
- No emojis in print statements
- Return unified diff only unless told otherwise

## Key dataset-specific notes for the demo
- **RLVS**: motion-presence shortcut in NonViolence class is a known open limitation — hedge
  in the narrative for low-motion NV clips.
- **HF**: object/PO stream adds FPs (GQS_obj ≈ 0.111 mean, near-empty signal). Gate naturally
  downweights it. Calibration T*=0.567 (overconfident) — report raw confidence, not calibrated.
- **NTU**: no per-frame or per-person score from classifier; VideoMAE carries most signal.
  source-stratified: Mobile F1=0.879, CCTV F1=0.857.
- **RWF**: confusion matrix (E_full_qgf, thr=0.36): TN=186, FP=14, FN=19, TP=181.

## Citations
- Hockey Fights: Bermejo Nievas et al., "Violence Detection in Video using Computer Vision
  Techniques", CAIP 2011.
- RLVS: cite as per paper bib entry.
- RWF-2000: cite as per paper bib entry.
- NTU CCTV-Fights: Perez et al., ICASSP 2019.
- Qwen2.5-VL: cite as per HuggingFace model card.
