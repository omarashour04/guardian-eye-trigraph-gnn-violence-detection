# Guardian Eye Runtime Status Validation

Date: 2026-06-15

This note documents the final demo runtime paths added after the architecture audit.

## Runtime Modes

- Prediction: V9 remains the authority for verdict, confidence, threshold, gates, GQS, and telemetry.
- Narration: `/explain` returns `narration_mode`.
  - `vlm_llm`: VLM visual pass and LLM final narration completed.
  - `fallback`: deterministic narration was used. `reason_if_fallback` gives a safe user-facing reason.
- Legal: legal responses return `legal_mode`.
  - `llm`: curated legal KB retrieval plus Qwen LLM generation completed.
  - `curated_fallback`: curated KB retrieval and deterministic legal summary were used.
- Ask Guardian Eye: `/ask` returns `ask_mode`, `selected_route`, `retrieved_context_count`, and `reason_if_fallback`.

## Overlay Streams

The backend now returns per-stream overlay status for:

- `skeleton`
- `interaction`
- `object`
- `vit`

Status values:

- `available`: stream MP4 exists and is returned.
- `missing`: stream MP4 is not present.
- `generating`: reserved for async generation.
- `fallback_placeholder`: demo placeholder stream is returned.

Existing overlays are reused when the combined overlay, thumbnail, and stream MP4s are present. The backend renders only when a required overlay artifact is genuinely missing.

## ViT / VideoMAE

VideoMAE does not need to run live for ViT to be valid. ViT contribution validity is based on the cached NPZ:

- `vit_embedding` exists and is non-zero: ViT stream active.
- `vit_embedding` missing or all zeros: ViT stream unavailable.

Utility:

```powershell
.\.venv\Scripts\python.exe scripts\inspect_npz_vit.py cache_npz\fight_1.avi.npz --pretty
```

Sample result observed for `fight_1.avi.npz`:

- `vit_embedding_exists`: true
- `shape`: `[768]`
- `mean`: `0.0`
- `std`: `0.0`
- `all_zero`: true
- `active`: false

## GPU Lifecycle

Prediction still completes before generation. `/explain`, legal generation, and Ask generation load LLM/VLM only when requested.

Preprocessing now logs CUDA memory at:

- `before_yolo`
- `after_yolo_load`
- `after_preprocessing`
- `after_preprocess_cleanup`

After NPZ creation, preprocessing models are moved/released and CUDA cache is cleared by default:

```python
gc.collect()
torch.cuda.empty_cache()
```

Set `GUARDIAN_RELEASE_PREPROCESS_MODELS=0` only for profiling repeated local predictions; leave it enabled for the 8GB demo path.

## Remaining Risks

- If Qwen models are not present locally and `GUARDIAN_MODEL_LOCAL_ONLY=1`, narration, legal, and Ask correctly run fallback modes.
- Releasing YOLO/VideoMAE after preprocessing protects VRAM but can make a second uncached prediction reload preprocessing models.
- In-process prediction cache prevents duplicate `/predict -> /overlay -> /explain` inference during the same server session, but it is cleared on backend restart.
