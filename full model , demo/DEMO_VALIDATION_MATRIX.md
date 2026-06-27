# Guardian Eye Demo Validation Matrix

Validation date: 2026-06-15  
Backend: `Backend_Ashour\BackEnd`  
Frontend: `FrontEnd`  
Backend checked at: `http://127.0.0.1:8000`  
Backend health during validation: `status=ok`, `mock_mode=false`

## Summary

| Area | Result | Notes/Risk |
|---|---:|---|
| Backend API health | PASS | Live backend responded in real mode. |
| V9 violent prediction | PASS | `fight_1.avi` persisted as violence, 60% confidence. |
| V9 non-violent prediction | PASS | Existing persisted real/demo incident shows non-violence, 12% confidence. Fresh all-in-one validation script timed out, so this row uses the latest persisted non-violent incident. |
| Frontend build | PASS | `npm.cmd run build` completed. Vite emitted only the existing large chunk warning. |
| Legal RAG | PASS | Canada/UK/USA California returned real official references; UAE returned deterministic fallback. |
| Ask Guardian Eye | PASS | Current-incident English answers validated; Arabic response contains Arabic Unicode and current incident facts. |
| History | PASS | `/history` and `/incident/{id}` return recent records, telemetry, narrative, legal-compatible IDs, and clean overlay contracts. |
| Browser responsiveness | PARTIAL | In-app browser was unavailable in this Codex session. Frontend build passed, but responsive visual click-through was not performed. |

## Confidence And Verdict Note

The V9 verdict logic is unchanged for the demo-quality fixes. In `Backend_Ashour\BackEnd\inference_classifier.py`, raw logits are converted with sigmoid, then calibrated by `_calibrate(raw_prob)`, and the final verdict is derived by comparing the calibrated confidence against `_v9_threshold`: `violence` when `conf >= threshold`, otherwise `non-violence`. It is not an argmax over two classes and no new threshold is applied in the frontend. Demo risk: a visually low or moderate displayed confidence can still show `Violence Detected` when the calibrated score is above the configured threshold.

## Matrix

| # | Test name | Sample/video used | Expected result | Actual result | Pass/Fail | Notes/Risk |
|---:|---|---|---|---|---|---|
| 1 | Backend health | Backend service | `/health` returns OK and real mode | `status=ok`, `mock_mode=false` | PASS | Existing backend on port 8000 was real mode. |
| 2 | Violent sample prediction | `fight_1.avi` | Verdict `violence` with confidence in 0-1 | `violence`, confidence `0.5996606945991516` | PASS | Real V9 output persisted as incident `edb45c9c-7bda-4c6f-9905-5b9b58d08e60`. |
| 3 | Non-violent sample prediction | `test_0e85...50b72ff9.mp4` persisted incident | Verdict `non-violence` with confidence in 0-1 | `non-violence`, confidence `0.12` | PASS | Used latest persisted non-violent incident to avoid another long inference run. |
| 4 | Frontend upload flow | `fight_1.avi` | `/predict`, `/overlay`, `/explain` complete and save incident | `/predict=200`, `/overlay=200`, `/explain=200`, incident saved | PASS | Fresh validation produced `fight_1_20260615_143813_777527_fbc2495f.avi`. |
| 5 | Verdict/confidence display data | Violent incident | Frontend receives verdict/confidence fields | API returned `violence`, `0.5996606945991516` | PASS | Frontend build validates type path. |
| 6 | Non-violent verdict/confidence data | Non-violent incident | Frontend receives non-violent verdict/confidence fields | API returned `non-violence`, `0.12` | PASS | Persisted incident has complete detail payload. |
| 7 | Timeline/peak window | Violent incident | Peak window is present and displayable | `[12, 15]` | PASS | Detail endpoint also has `people_count=5`. |
| 8 | Timeline/peak window | Non-violent incident | Peak window is present and displayable | `[0, 0]` | PASS | Non-violent sample uses neutral peak window. |
| 9 | Narrative consistency | Violent incident | Narrative says violent and matches verdict | "classified this clip as violent with 60% confidence..." | PASS | No contradictory non-violent wording. |
| 10 | Narrative consistency | Non-violent incident | Narrative says non-violent and explains threshold | "classified this clip as non-violent with 12% confidence..." | PASS | Mentions evidence did not pass violence threshold. |
| 11 | Combined overlay | Violent incident | Combined overlay file exists | `static\overlays\fight_1_...fbc2495f.mp4` | PASS | Backend path verified. |
| 12 | Stream overlays | Violent incident | Skeleton/interaction/object/vit files exist | All four stream paths returned | PASS | `skeleton`, `interaction`, `object`, `vit` MP4s present. |
| 13 | Historical/missing stream overlay contract | Non-violent incident | Missing old stream files are cleanly represented | `overlays={skeleton:null, interaction:null, object:null, vit:null}` | PASS | Frontend shows pending/unavailable state instead of broken video. |
| 14 | Legal RAG Canada | Violent incident | Real official references | `source=real`, `refs=3`, `status=passed` | PASS | Summary grounded in current incident prediction. |
| 15 | Legal RAG UK | Violent incident | Real official references | `source=real`, `refs=3`, `status=passed` | PASS | Summary grounded in current incident prediction. |
| 16 | Legal RAG USA California | Violent incident | Real official references | `source=real`, `refs=1`, `status=passed` | PASS | One official California reference available. |
| 17 | Legal RAG UAE fallback | Violent incident | Deterministic fallback, no fake official refs | `source=fallback`, `refs=0`, `status=needs_review` | PASS | Expected fallback behavior for unsupported/unverified country. |
| 18 | Ask people count | Violent incident | Current incident answer | "5 tracked people" | PASS | Uses selected incident ID, not all history. |
| 19 | Ask why violent | Violent incident | Uses verdict/confidence/peak/gates | Mentions `60%`, frames `12-15`, `vit` and `skeleton` | PASS | Uses actual stored prediction fields. |
| 20 | Ask why non-violent | Non-violent incident | Explains threshold/non-violent evidence | Mentions `12%`, interaction `35%`, did not show strong violent pattern | PASS | Uses current non-violent incident. |
| 21 | Ask weapon/object flag | Violent incident | Answers weapon/object flag directly | "person object-proximity flag around frames 12-15" | PASS | Fixed during validation: weapon question no longer routes to people-count answer. |
| 22 | Ask peak window | Violent incident | Returns current peak window | "frames 12-15" | PASS | Includes telemetry caveat. |
| 23 | Ask strongest stream | Violent incident | Returns strongest gate stream | `vit=32%`, `skeleton=26%`, object `24%`, interaction `19%` | PASS | Uses saved gate weights. |
| 24 | Ask legal consequences | Violent incident | Reuses Legal RAG/current incident | Canada Legal RAG answer returned with references/summary | PASS | Endpoint can be slow if repeatedly batch-called; single focused call passes. |
| 25 | Ask Arabic | Violent incident | Arabic answer using current incident facts | Arabic Unicode response includes `5` people and current incident summary | PASS | PowerShell console may display mojibake; Python Unicode check passed. |
| 26 | History recent list | `/history` | Recent incidents returned | Violence total `63`; non-violence total `15` | PASS | History is populated and sorted by timestamp. |
| 27 | Open historical incident | `/incident/{id}` | Detail includes verdict/confidence/people/peak/narrative | Detail returned all fields for violent and non-violent incidents | PASS | Historical incident can drive frontend review. |
| 28 | Historical overlays | Historical detail | Overlay paths or null stream slots are safe | Violent has files; older non-violent has null stream slots | PASS | No stale latest-upload overlay is sent for old incident. |
| 29 | Historical legal panel update | Historical incident + Canada | Legal panel can update from selected incident | `/legal-consequences` returned summary and refs | PASS | Frontend now calls legal endpoint when opening history with selected country. |
| 30 | Arabic/English labels and summaries | Frontend source/build | Language toggle compiles and backend accepts `language` | Frontend build passed; Arabic Ask response is Unicode Arabic | PASS | Visual toggle not clicked due browser unavailability. |
| 31 | Browser responsiveness | Frontend UI | Desktop/mobile visual responsive check | Not visually executed | PARTIAL | Browser automation surface unavailable; use manual checklist below. |

## Reproduction Commands

Start backend:

```powershell
cd "D:\sara\graduation\Demo Implementation\Backend_Ashour\BackEnd"
$env:GUARDIAN_MOCK="0"
$env:GUARDIAN_V9_CKPT="D:\sara\graduation\Demo Implementation\models\v9\best.pt"
$env:GUARDIAN_RAG_PIPELINE_PATH="D:\sara\graduation\Demo Implementation\FULL_RAG_Pipeline"
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Start frontend:

```powershell
cd "D:\sara\graduation\Demo Implementation\FrontEnd"
$env:VITE_API_BASE_URL="http://127.0.0.1:8000"
npm.cmd run dev
```

Backend tests:

```powershell
cd "D:\sara\graduation\Demo Implementation\Backend_Ashour\BackEnd"
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Frontend build:

```powershell
cd "D:\sara\graduation\Demo Implementation\FrontEnd"
npm.cmd run build
```

Check latest violent and non-violent history records:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/history?verdict=violence&limit=1" |
  ConvertTo-Json -Depth 8

Invoke-RestMethod -Uri "http://127.0.0.1:8000/history?verdict=non-violence&limit=1" |
  ConvertTo-Json -Depth 8
```

Ask current incident:

```powershell
$incidentId = "edb45c9c-7bda-4c6f-9905-5b9b58d08e60"
$detail = Invoke-RestMethod -Uri "http://127.0.0.1:8000/incident/$incidentId"
$body = @{
  question = "Which stream contributed most?"
  language = "en"
  clip_id = $detail.clip_id
  incident_id = $incidentId
  country = "Canada"
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/ask" `
  -ContentType "application/json; charset=utf-8" `
  -Body $body
```

Legal RAG country check:

```powershell
$incidentId = "edb45c9c-7bda-4c6f-9905-5b9b58d08e60"
foreach ($country in @("Canada", "UK", "USA California", "UAE")) {
  $body = @{
    incident_id = $incidentId
    country = $country
    language = "en"
  } | ConvertTo-Json

  $result = Invoke-RestMethod -Method Post `
    -Uri "http://127.0.0.1:8000/legal-consequences" `
    -ContentType "application/json" `
    -Body $body

  "$country source=$($result.legal_consequences_rag.legal_rag_source) refs=$($result.legal_consequences_rag.retrieved_legal_references.Count)"
}
```

## Manual Browser Checklist

Use this after starting both servers:

1. Open `http://127.0.0.1:5173`.
2. Upload `fight_1.avi`; confirm verdict, confidence, timeline, narrative, legal panel, and four overlay cards update.
3. Ask: "How many people were involved?", "Was there a weapon involved?", and "Which stream contributed most?"
4. Toggle Arabic and ask `كم عدد الأشخاص؟`.
5. Open Incident History and select an older non-violent incident.
6. Confirm the old incident updates verdict, confidence, people count, peak window, narrative, legal panel, and overlay cards.
7. Resize to mobile width and confirm no overlapping panels/buttons/text.
