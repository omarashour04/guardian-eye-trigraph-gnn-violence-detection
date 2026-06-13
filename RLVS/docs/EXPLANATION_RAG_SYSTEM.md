# Guardian Eye — Explanation & RAG System (Design Spec)

> Status: design spec. No code yet. This document is the **contract** the later
> implementation must satisfy. Every shape and key cited here is verified against
> the V9 RLVS pipeline (`trigraph_v9_rlvs_preprocess.py`, `trigraph_v9_rlvs_train.py`).

---

## 1. Purpose & feasibility verdict

**Goal:** given a clip the classifier flagged, produce a natural-language explanation of
*what is happening in the fight* — and keep a queryable memory of past incidents so a user
can later ask "tell me about the fight from a week ago."

**Feasibility: yes.** With one honest caveat that must be stated up front:

> An LLM cannot *see* a fight. It can only describe what it is given. The quality of the
> explanation is bounded entirely by the signal we feed it. We feed it three things:
> (a) the classifier's structured telemetry, (b) the 16 RGB frames already stored per clip,
> and (c) retrieved reference snippets. With all three, grounded explanation is feasible
> **today** on a single local GPU, fully offline.

**The single most important rule:**

> **The classifier is the authority. The VLM is the narrator.**
> The violence/no-violence verdict and the confidence come from the V9 model and are *never*
> changed by the language model. The VLM only explains the verdict using grounded evidence.

---

## 2. Concepts: LLM vs VLM vs RAG

These three are routinely conflated. They are different:

| Term | What it is | In/out | Role here |
|------|-----------|--------|-----------|
| **LLM** | Large Language Model | text in → text out | Narrates from the *text* evidence packet (telemetry). Cannot see pixels. |
| **VLM** | Vision-Language Model = **an LLM with an attached vision encoder** | image(s) + text in → text out | Reads the 16 frames *plus* the evidence packet and describes the visible scene. |
| **RAG** | Retrieval-Augmented Generation — a **technique, not a model** | retrieves reference text → injects it into the prompt | Grounds vocabulary (taxonomy) and powers historical incident queries. |

Key clarifications (these were explicit user questions):

- **A VLM is not a different kind of model from an LLM.** It *is* an LLM, with eyes bolted on:
  same text "brain," plus a vision encoder that lets it accept images. Qwen2.5-VL is Qwen
  (an LLM) + a vision tower.
- **The VLM is not RAG, and not a classic NLP model.** It is a generative multimodal model.
- **RAG is orthogonal** and *wraps* the VLM (or LLM). It is the retrieve-then-generate pattern.
  We use RAG in two distinct places (Section 6 and Section 7).

---

## 3. Architecture (local, on-device)

```
                          ┌─────────────────────────────────────────┐
   clip ──► PREPROCESS ──►│ NPZ telemetry  +  frames_vit [16,224,224,3]│
            (pose/track/   └─────────────────────────────────────────┘
             objects/ViT)                 │
                                          ▼
                              CLASSIFIER (V9, authority)
                              ► p(violence), threshold
                              ► fusion gate [4]
                              ► GQS [5]
                                          │
                 ┌────────────────────────┼─────────────────────────┐
                 ▼                        ▼                          ▼
        EVIDENCE PACKET          16 STORED FRAMES          RAG RETRIEVAL
        (telemetry → text)        (pixels)                 (a) reference corpus
                 │                        │                 (b) incident memory
                 └───────────┬────────────┴──────────┬──────┘
                             ▼                        ▼
                      VLM  (Qwen2.5-VL-7B, 4-bit, loaded on demand)
                             │
                             ▼
                  grounded natural-language explanation
                  + new INCIDENT RECORD written to memory
```

Everything runs locally on the RTX 3090. No cloud, no data leaves the machine.

---

## 4. Required model outputs — the contract

The explanation layer consumes exactly these, all already produced by the pipeline.

### 4.1 Verdict (the authority)
| Field | Source | Notes |
|-------|--------|-------|
| `p_violence` | `torch.sigmoid(logits)` per clip | scalar in [0,1]; clip-level only |
| `threshold` | checkpoint `threshold` key | learned/calibrated decision threshold |
| `verdict` | `p_violence >= threshold` | binary: violence / non-violence |
| `confidence` | calibrated `p_violence` | temperature scaling already available in diagnostics |

> There is **no per-frame and no per-person score from the classifier.** Any "who started it"
> or "when it peaked" claim is *derived from geometry* (Section 5) and must be labeled heuristic.

### 4.2 Fusion gate — which stream drove the decision
- Obtained from the model forward with `return_gates=True` → `gates` shape **`[B, 4]`**, softmax,
  order **(skeleton, interaction, object, vit)**. Source: `QualityGatedFusion`
  (`trigraph_v9_rlvs_train.py`). Note the gate is computed from `gqs[:, :4]` (first four GQS).

### 4.3 GQS — graph quality scores `[5]`
Order: `[q_skel, q_int, q_obj, q_po, valid_ratio]` (`preprocess.py` ~L513–518).
- `q_skel` — fraction of frames with a valid skeleton (≥ `min_joints` joints conf > 0.25)
- `q_int` — fraction of frames with ≥ 2 tracked people
- `q_obj` — fraction of frames with ≥ 1 object
- `q_po` — fraction of frames with a person–object edge
- `valid_ratio` — fraction of frames with ≥ 1 person

### 4.4 NPZ telemetry (per clip) — verified keys & shapes
From `np.savez_compressed(...)` (`preprocess.py` ~L521–536):

| Key | Shape | dtype | Meaning |
|-----|-------|-------|---------|
| `skeleton` | `[32, 6, 17, 3]` | f32 | COCO-17 (x, y, conf), normalized by max(H,W) |
| `int_nodes` | `[32, 6, 7]` | f32 | per person: cx, cy, w, h, conf, speed, continuity |
| `int_edges` | `[32, 6, 6, 4]` | f32 | per pair: dist, iou, close, rel_spd |
| `int_node_mask` | `[32, 6]` | bool | valid person |
| `int_edge_mask` | `[32, 6, 6]` | bool | valid pair |
| `obj_nodes` | `[32, 8, 6]` | f32 | per object: cx, cy, w, h, conf, cls_norm |
| `obj_node_mask` | `[32, 8]` | bool | valid object |
| `po_edges` | `[32, 6, 8, 5]` | f32 | person↔object: wrist_d, body_d, iou, near_wrist, near_body |
| `po_edge_mask` | `[32, 6, 8]` | bool | valid person-object edge |
| `gqs` | `[5]` | f32 | see 4.3 |
| `frames_vit` | `[16, 224, 224, 3]` | uint8 | RGB frames for the VLM |
| `label`, `split` | scalar | — | ground truth / split |

`vit_embedding [768]` (f32) is added in Phase 2.

Constants: `T=32`, `T_vit=16`, `M=6` people, `N=8` objects, `V=17` joints. Tracks via ByteTrack
(`persist=True`), so index `m` is the same person across frames.

---

## 5. The evidence packet (telemetry → text)

A deterministic function turns raw NPZ telemetry into a compact text block the VLM reads.
It is **not** model-generated — it is computed, so it cannot hallucinate.

Computation notes:
- **Peak-interaction window**: scan `int_edges[t, i, j, :]` over valid pairs; flag the frame range
  where `dist` is minimal *and* `rel_spd` (or `close`) spikes. Report as a frame window
  (and as a fraction of the clip).
- **Weapon/object proximity**: scan `po_edges[..., near_wrist]` / `wrist_d`; if a wrist is near an
  object during the peak window, surface it (cls_norm → object class name via the detector's label map).
- **People count**: max over `int_node_mask[t].sum()`.
- **Stream emphasis**: argmax / top-2 of the gate `[4]`.
- **Quality**: GQS values, flag any low one (e.g. `q_skel < 0.5` → "pose evidence weak").

### Worked example packet
```
VERDICT: VIOLENCE   confidence: 0.94 (calibrated)
PEOPLE: 2 tracked across the clip.
INTERACTION: persons came within 0.08 (normalized) at frames 14–22 (~44–69% of clip),
             with a sharp relative-speed spike — consistent with rapid contact.
OBJECT: an object (class: "bottle") detected near person A's wrist during frames 16–20.
DECISION DRIVERS (fusion gate): interaction 0.41, skeleton 0.34, vit 0.18, object 0.07.
QUALITY: q_skel 0.91, q_int 0.88, q_obj 0.40, q_po 0.33, valid_ratio 0.97.
NOTE: classifier gives one clip-level score; person-level / timing claims are geometry-derived.
```

---

## 6. RAG store #1 — reference corpus (grounds vocabulary)

A small curated corpus retrieved into the prompt so the narrative uses correct, consistent
terms and does not invent categories.

Contents:
- **Violence taxonomy**: bare-hand fight vs weapon-involved.
- **Non-violence categories** (from RLVS): handshake, conversation, eating, walking, horse riding.
- **Evaluation caveats**: the CUE-Net (2024) note that some RLVS test labels are disputed — so the
  narrative can hedge appropriately on borderline cases.
- **A few annotated exemplars** (telemetry → ideal explanation) as few-shot anchors.

Mechanics: local embedding model (e.g. `bge-small` / `e5-small`, runs on CPU or a sliver of GPU);
top-k semantic retrieval; inject snippets into the prompt.

> This is intentionally a **light** retrieval layer over a tiny corpus. It is still legitimately
> RAG (retrieve-then-generate). Say so plainly in the thesis; do not overclaim a giant index.

---

## 7. RAG store #2 — incident memory (retrieval over history)

This is the **classic RAG pattern** and the feature behind *"tell me about the fight from a week
ago."* Every analyzed clip becomes a persistent, queryable **incident record**.

### 7.1 Incident record schema
| Field | Type | Source |
|-------|------|--------|
| `incident_id` | uuid | generated |
| `timestamp` | ISO-8601 absolute datetime | analysis time |
| `source` | str | clip / camera / file name |
| `verdict` | enum {violence, non-violence} | classifier |
| `confidence` | float | calibrated `p_violence` |
| `gqs` | float[5] | NPZ |
| `gate` | float[4] | model `return_gates=True` |
| `people_count` | int | telemetry |
| `peak_window` | [int, int] (frame range) | evidence packet |
| `weapon_flag` | bool + class | po_edges |
| `packet_summary` | str | the evidence packet |
| `narrative` | str | VLM output |
| `frames_ref` | path | stored 16 frames / a montage thumbnail |

### 7.2 Two stores
- **Vector store** (local, e.g. Chroma/FAISS): embed `narrative + packet_summary` for semantic
  search ("a fight with a bottle", "two men shoving").
- **Structured store** (SQLite or parquet): for time and metadata filters — `timestamp`,
  `verdict`, `confidence`, `weapon_flag`, `source`. Enables "last week", "weapon involved",
  "confidence > 0.9".

### 7.3 Query flow
```
NL question
  → parse into (time/metadata filter) + (semantic query)
  → structured store filters candidates (e.g. timestamp within last 7 days, verdict=violence)
  → vector store ranks by semantic similarity
  → retrieve top record(s)
  → LLM/VLM summarizes from the stored record(s)
     (optionally re-load frames_ref for a visual recap)
```

### 7.4 Worked example — "tell me about the fight from a week ago"
```
Parsed → time filter: [now-8d, now-6d], verdict=violence; semantic query: "fight"
Retrieved → incident 4f2a…, timestamp 2026-05-30 18:42, source "cam3_clip_07",
            confidence 0.94, weapon "bottle", peak frames 14–22, 2 people.
Answer (LLM) → "About a week ago, on May 30 at 6:42 PM (camera 3), the system flagged a
            violent incident with 94% confidence. Two people were involved; they closed
            distance rapidly around the middle of the clip, and an object resembling a
            bottle was detected near one person's hand. [show recap frames]"
```

---

## 8. VLM stage

- **Model**: Qwen2.5-VL-7B-Instruct, **4-bit** quantized (~7 GB VRAM), **loaded on demand**.
- **Inputs**: the 16 `frames_vit` + the evidence packet (Section 5) + retrieved reference snippets
  (Section 6).
- **Output**: 2–4 sentence grounded narrative.

### Prompt template (sketch)
```
SYSTEM: You explain a violence-detection result. You are a NARRATOR, not the judge.
        The VERDICT and CONFIDENCE are final and come from a separate classifier —
        never contradict or re-decide them. Describe only what the frames and the
        evidence support. If person-level or timing detail is geometry-derived, hedge.
        Use the reference vocabulary. Do not invent objects or people not in the evidence.

REFERENCE (retrieved): {taxonomy / caveats / exemplars}
EVIDENCE PACKET: {packet}
IMAGES: {16 frames}

USER: Explain what is happening in this clip, consistent with the verdict.
```

### Guardrails (must implement)
1. Verdict/confidence are passed as fixed facts; reject/regenerate if the narrative flips them.
2. Person attribution and exact timing are presented as heuristic ("appears to", "around
   frames X–Y"), because the classifier has no per-person/per-frame score.
3. No objects/people may be named unless present in telemetry or clearly visible.

---

## 9. Does it need the videos?

**No raw video files at inference.** The pipeline already stores `frames_vit [16,224,224,3]` per
clip — those 16 frames are what the VLM sees, and also what powers historical visual recaps. All
other evidence is read from the NPZ telemetry. This fits the local/offline/private setup.

---

## 10. Honest limitations (state these in the thesis)

- Classifier output is **clip-level only** — no per-frame, no per-person score.
- "Who" and "when" in the narrative are **geometry-derived heuristics**, not classifier outputs.
- The reference corpus is **small** — a light RAG layer, not a large knowledge base.
- The VLM can still err on visual detail; the verdict does not depend on it.

### Optional upgrade (small code change, big payoff)
Expose, from the model forward: (a) pre-pooling **per-frame scores** in the interaction stream, and
(b) **GATv2 attention** `[B, T, M, M, heads]`. This would convert the heuristic timing/person claims
into model-grounded ones (true temporal + person localization). Flagged as future work.

---

## 11. Build checklist

1. Add an inference entry point that returns `{verdict, confidence, gate[4], gqs[5], telemetry}`
   and frees classifier VRAM after use.
2. Implement the deterministic **evidence-packet** builder from NPZ telemetry (Section 5).
3. Stand up RAG store #1: curate the reference corpus, pick a local embedding model, build the index.
4. Stand up RAG store #2: define the incident-record schema, wire vector + structured stores,
   write a record on every analysis.
5. Integrate the VLM (Qwen2.5-VL-7B 4-bit, load-on-demand) with the prompt template + guardrails.
6. Implement the historical query flow (NL → filter + semantic → retrieve → summarize).
7. Validate guardrails: confirm the narrative never overrides the verdict; confirm hedging language
   on heuristic claims.
```
