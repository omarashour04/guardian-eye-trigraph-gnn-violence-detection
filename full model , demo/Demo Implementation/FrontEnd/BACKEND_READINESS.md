# Guardian Eye Backend Readiness Audit

## Summary

The frontend is mostly ready for future FastAPI integration because API access is centralized in `src/api/guardianApi.ts`, shared response types live in `src/types/guardian.ts`, and mock records are isolated in `src/mocks/mockData.ts`.

The main remaining risks are not architectural blockers. They are contract-shape cleanup items: a few components still contain fallback mock constants, some UI data is optional or duplicated locally, and the mock service function return shapes should be aligned more tightly with the future `/predict`, `/explain`, `/history`, and `/ask` responses before replacing mock calls with Axios.

## Endpoint Readiness

### `/predict`

Current frontend entry points:

- `predictVideo(file)` in `src/api/guardianApi.ts`
- `predictSample(clipId)` in `src/api/guardianApi.ts`

Readiness:

- `predictVideo(file)` already accepts a `File`, so it can later become a `FormData` upload to `POST /predict`.
- `DemoPage.tsx` consumes the returned `GuardianPrediction` object and updates verdict, gate chart, timeline, overlay, narrative, history, and Q&A context from that response.
- `predictSample(clipId)` is a mock-only helper for gallery-driven demo predictions. It should remain frontend-only or map to a future sample/demo endpoint, not the production `/predict` endpoint.

Risk:

- `predictVideo(file)` currently ignores `file` and always returns the first mock sample. This is fine for mock mode, but the Axios implementation should be the only place this changes.

### `/explain`

Current frontend entry point:

- `explainIncident(incidentId)` in `src/api/guardianApi.ts`

Readiness:

- The function exists and is centralized.
- `GuardianExplanation` exists in `src/types/guardian.ts`.

Risk:

- `explainIncident()` currently returns `Promise<string>`, while the type file defines `GuardianExplanation`. For backend readiness, this should likely return `Promise<GuardianExplanation>` so `/explain` can provide `narrative` and `dominant_signals` consistently.
- The dashboard currently reads `prediction.explanation.narrative` directly instead of calling `explainIncident()`.

### `/history`

Current frontend entry point:

- `getHistory()` in `src/api/guardianApi.ts`

Readiness:

- `GuardianIncident` is centralized.
- `HistoryPanel` accepts `incidents?: GuardianIncident[]`.
- `HistoryPanel` now supports empty and loading states.

Risk:

- `DemoPage.tsx` currently uses `prediction?.incidents` instead of calling `getHistory()`.
- `GuardianIncident.thumbnail_url` is required. If FastAPI may omit thumbnails, this should become optional or nullable.

### `/ask`

Current frontend entry point:

- `askGuardianEye(question, clipId?)` in `src/api/guardianApi.ts`

Readiness:

- `AskBox` accepts an async `onAsk` prop.
- `DemoPage.tsx` calls `askGuardianEye(question, selectedSample?.clip_id)`.
- `GuardianAskResponse` is centralized.

Risk:

- The function currently ignores `question` and returns a sample-linked answer. This is acceptable for mock mode.
- The future request body shape is not typed yet. A `GuardianAskRequest` type would make the Axios swap safer.

## Component Data-Driven Audit

### `DemoPage.tsx`

Status:

- Mostly API-driven through `getSamples()`, `predictSample()`, `predictVideo()`, and `askGuardianEye()`.
- Holds selected sample and prediction state cleanly.

Hardcoded assumptions:

- Demo/status notification text is hardcoded in the page.
- Sample loading assumes `loadedSamples[0]` exists.
- History is sourced from `prediction.incidents`, not `/history`.
- Narrative is sourced from `prediction.explanation.narrative`, not `/explain`.

Backend readiness:

- Good. The page should not need a redesign for backend integration.
- Add empty/error handling if `getSamples()` returns an empty array or if a backend request fails before initial data is available.

### `SampleGallery.tsx`

Status:

- Fully prop-driven through `samples`, `selectedClipId`, and `onSelectSample`.

Hardcoded assumptions:

- Requires `GuardianSample`, which extends `GuardianPrediction` with `title`, `description`, and `ask_response`.
- This is appropriate for demo samples, but production `/predict` responses should not need sample-gallery fields.

Backend readiness:

- Good for a demo/sample endpoint.
- Keep `GuardianSample` separate from `GuardianPrediction`, as it is now.

### `UploadPanel.tsx`

Status:

- API-agnostic and parent-driven through `onUpload(file)` and `isLoading`.

Hardcoded assumptions:

- None affecting backend contracts.

Backend readiness:

- Good.

### `VerdictPanel.tsx`

Status:

- Driven by `verdict` and `confidence`.

Hardcoded assumptions:

- Props duplicate the verdict union instead of importing `GuardianVerdict`.

Backend readiness:

- Good.
- Consider changing the prop type to `GuardianVerdict` later to reduce duplicated contract definitions.

### `GateBar.tsx`

Status:

- Driven by optional `gate?: GuardianGate`.

Hardcoded assumptions:

- Contains `defaultGate` fallback mock values.
- Gate labels and colors are UI constants, which is fine.

Backend readiness:

- Good.
- If backend gate keys change, only `GuardianGate` and this component need updates.

### `Timeline.tsx`

Status:

- Driven by optional `telemetry?: GuardianTelemetry`.

Hardcoded assumptions:

- Contains `defaultTelemetry` fallback mock values.
- Assumes `peak_window` and `total_frames` always exist when telemetry exists.

Backend readiness:

- Good if `/predict` always returns telemetry.
- If telemetry can be absent or partial, make fields optional and display `EmptyState`.

### `OverlayVideo.tsx`

Status:

- Driven by `media`.
- Supports future `overlay_video_url`.

Hardcoded assumptions:

- Defines a local `OverlayMedia` subset that overlaps with `GuardianMedia`.
- Contains fallback metadata values.
- `GuardianMedia` still includes legacy `overlay_url`, while the component uses `overlay_video_url`.

Backend readiness:

- Good directionally.
- Remove `overlay_url` after backend contract is finalized, or map backend media fields inside `guardianApi.ts`.

### `NarrativePanel.tsx`

Status:

- Accepts `narrative?: string`.

Hardcoded assumptions:

- Contains a default mock narrative fallback.

Backend readiness:

- Good for display.
- For `/explain`, consider accepting `GuardianExplanation` instead of only a string if dominant signals will be shown later.

### `HistoryPanel.tsx`

Status:

- Accepts `incidents?: GuardianIncident[]` and `isLoading?: boolean`.
- Uses `EmptyState` and `LoadingSkeleton`.

Hardcoded assumptions:

- Contains `defaultHistory`.

Backend readiness:

- Good.
- Consider removing the default history when `/history` becomes real, so empty state appears when no incidents exist.

### `AskBox.tsx`

Status:

- Accepts async `onAsk(question)` and displays returned answer.

Hardcoded assumptions:

- Contains a default mock answer fallback.

Backend readiness:

- Good.
- Add a typed request payload for `/ask` when backend contract is final.

## TypeScript Contract Audit

Central types currently exist in `src/types/guardian.ts`:

- `GuardianPrediction`
- `GuardianGate`
- `GuardianGqs`
- `GuardianTelemetry`
- `GuardianWeapon`
- `GuardianMedia`
- `GuardianExplanation`
- `GuardianIncident`
- `GuardianAskResponse`
- `GuardianSample`

Missing or useful future types:

- `GuardianPredictRequest` or equivalent upload metadata type.
- `GuardianExplainResponse`, if `/explain` returns more than `GuardianExplanation`.
- `GuardianHistoryResponse`, if `/history` returns pagination or metadata.
- `GuardianAskRequest`, likely `{ question: string; clip_id?: string; incident_id?: string }`.
- `GuardianApiError`, for consistent error display.

Potential type issues:

- `GuardianMedia` mixes `overlay_video_url` with legacy `overlay_url`.
- `GuardianIncident.thumbnail_url` is required, but the UI already treats thumbnails as placeholders. It may be safer as `thumbnail_url?: string`.
- `GuardianMedia.source`, `resolution`, and `mode` are required. If FastAPI does not always return them, either make them optional or normalize them in `guardianApi.ts`.
- `PredictionResponse` is an alias exported from `guardianApi.ts`. It is not harmful, but components could import `GuardianPrediction` directly to reduce indirection.

## Duplicate Interface Audit

No duplicate backend model interfaces were found outside `src/types/guardian.ts`.

Minor local overlaps:

- `VerdictPanel.tsx` duplicates the verdict union instead of importing `GuardianVerdict`.
- `OverlayVideo.tsx` defines `OverlayMedia`, overlapping with `GuardianMedia`.
- `DemoPage.tsx` defines `NotificationState`, which is UI-only and acceptable.

These are not blockers, but reducing the first two would make the backend contract easier to maintain.

## Values That Should Eventually Come From API Responses

Currently hardcoded fallback/demo values:

- `GateBar.tsx`: `defaultGate`
- `Timeline.tsx`: `defaultTelemetry`
- `OverlayVideo.tsx`: `defaultMedia`
- `NarrativePanel.tsx`: default narrative text
- `HistoryPanel.tsx`: `defaultHistory`
- `AskBox.tsx`: `defaultMockAnswer`
- `DemoPage.tsx`: notification copy, status copy, presentation banner copy

Recommendation:

- Keep UI text hardcoded.
- Remove or stop using fallback mock data once backend calls are reliable.
- Prefer `EmptyState` or `LoadingSkeleton` instead of silently showing demo defaults when real API data is missing.

## Axios Switch Readiness

`src/api/guardianApi.ts` is ready to become the backend boundary with minimal page/component changes.

Expected future implementation shape:

```ts
export async function predictVideo(file: File): Promise<GuardianPrediction> {
  const formData = new FormData()
  formData.append("file", file)

  const response = await guardianHttp.post<GuardianPrediction>("/predict", formData)
  return response.data
}
```

Other likely mappings:

- `explainIncident(incidentId)` -> `GET /explain/{incidentId}` or `POST /explain`
- `getHistory()` -> `GET /history`
- `askGuardianEye(question, clipId?)` -> `POST /ask`
- `predictSample(clipId)` and `getSamples()` should remain demo-only unless the backend adds sample/demo endpoints.

## Overall Readiness Rating

Backend readiness: **Good, with small contract cleanup recommended before integration.**

Highest-priority cleanup before FastAPI connection:

1. Make `explainIncident()` return `GuardianExplanation` instead of `string`.
2. Finalize `GuardianMedia`: choose `overlay_video_url` and remove or map `overlay_url`.
3. Add request/response wrapper types for `/ask`, `/history`, and `/explain` if backend returns metadata.
4. Replace component default mock fallbacks with loading/empty states when real API mode is enabled.
5. Handle empty `getSamples()` safely in `DemoPage.tsx`.

No UI redesign is needed for backend integration.
