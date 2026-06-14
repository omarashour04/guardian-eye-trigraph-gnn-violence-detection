import axios from "axios"

import { mockHistory, mockSamples } from "@/mocks/mockData"
import type {
  GuardianApiMode,
  GuardianAskResponse,
  GuardianBackendStatus,
  GuardianExplanation,
  GuardianIncident,
  GuardianLegalConsequencesRag,
  GuardianLegalScores,
  GuardianPrediction,
  GuardianRagEnrichment,
  GuardianSample,
} from "@/types/guardian"

const configuredApiMode = String(import.meta.env.VITE_API_MODE ?? "auto").toLowerCase()
const configuredBaseUrl =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000"
const isExplicitMockMode = configuredApiMode === "mock"

export const guardianApiMode: GuardianApiMode =
  isExplicitMockMode ? "mock" : "backend"

export const guardianHttp = axios.create({
  baseURL: configuredBaseUrl,
})

const logApiDebug = (message: string, details?: Record<string, unknown>) => {
  console.info(`[Guardian API] ${message}`, details ?? {})
}

logApiDebug("client initialized", {
  apiBaseUrl: configuredBaseUrl,
  configuredApiMode,
  serviceMode: guardianApiMode,
  mockOverride: isExplicitMockMode,
})

guardianHttp.interceptors.request.use((config) => {
  logApiDebug("real backend request", {
    apiBaseUrl: configuredBaseUrl,
    method: config.method?.toUpperCase(),
    target: `${configuredBaseUrl}${config.url ?? ""}`,
  })

  return config
})

export type PredictionResponse = GuardianPrediction

export type ExplanationResponse = GuardianExplanation & GuardianRagEnrichment

type BackendGqs = {
  q_skel: number
  q_int: number
  q_obj: number
  q_po: number
  valid_ratio: number
}

type BackendTelemetry = {
  people: number
  peak_window: number[]
  weapon: {
    flag: boolean
    cls: string | null
  }
}

type BackendPredictResponse = {
  verdict: GuardianPrediction["verdict"]
  confidence: number
  threshold: number
  gate: GuardianPrediction["gate"]
  gqs: BackendGqs
  telemetry: BackendTelemetry
  clip_id: string
}

type BackendIncidentSummary = {
  incident_id: string
  timestamp: string
  source: string
  verdict: GuardianIncident["verdict"]
  confidence: number
  thumbnail?: string | null
  overlay?: string | null
  people_count: number
  weapon_flag: boolean
  weapon_class?: string | null
  peak_window: number[]
  narrative_preview?: string | null
}

type BackendIncidentDetail = {
  incident_id: string
  clip_id: string
  timestamp: string
  source: string
  verdict: GuardianIncident["verdict"]
  confidence: number
  threshold?: number | null
  gate?: Partial<GuardianPrediction["gate"]> | null
  gqs?: Partial<BackendGqs> | null
  people_count?: number | null
  peak_window?: number[] | null
  weapon_flag?: boolean | null
  weapon_class?: string | null
  thumbnail_path?: string | null
  overlay_path?: string | null
  packet_summary?: string | null
  narrative?: string | null
}

type BackendHistoryResponse = {
  total: number
  incidents: BackendIncidentSummary[]
}

type BackendAskResponse = {
  answer: string
  incidents: BackendIncidentSummary[]
  language: string
}

type BackendExplainResponse = {
  narrative: string
  incident_id: string
  language: string
  explanation_rag?: GuardianRagEnrichment["explanation_rag"]
  incident_memory_rag?: GuardianRagEnrichment["incident_memory_rag"]
  legal_consequences_rag?: GuardianLegalConsequencesRag | null
  legal_scores?: GuardianLegalScores | null
}

type BackendOverlayResponse = {
  clip_id: string
  overlay_path: string
  thumbnail_path: string
}

const wait = (ms: number) =>
  new Promise((resolve) => {
    window.setTimeout(resolve, ms)
  })

const findSampleByClipId = (clipId: string) =>
  mockSamples.find((sample) => sample.clip_id === clipId) ?? mockSamples[0]

const warnAndFallback = (operation: string, error: unknown) => {
  console.warn(`Guardian Eye ${operation} failed. Falling back to mock data.`, error)
}

const warnAndContinue = (operation: string, error: unknown) => {
  console.warn(`Guardian Eye ${operation} failed. Continuing without it.`, error)
}

const clamp01 = (value: number) => Math.min(1, Math.max(0, value))

const normalizeBackendPath = (path?: string | null) =>
  path?.replace(/\\/g, "/")

const toBackendMediaUrl = (url?: string | null) => {
  if (!url) {
    return url
  }

  if (/^(https?:|blob:|data:)/i.test(url)) {
    return url
  }

  try {
    return new URL(url, configuredBaseUrl).toString()
  } catch {
    return url
  }
}

const toOptionalBackendMediaUrl = (url?: string | null) =>
  toBackendMediaUrl(url) ?? undefined

const toStaticMediaUrl = (url?: string | null) =>
  toOptionalBackendMediaUrl(normalizeBackendPath(url))

const toFrameWindow = (peakWindow?: number[]): [number, number] => {
  const start = Number(peakWindow?.[0] ?? 0)
  const end = Number(peakWindow?.[1] ?? 0)

  return [start, end]
}

const getDominantSignals = (gate: GuardianPrediction["gate"]) =>
  Object.entries(gate)
    .sort(([, left], [, right]) => right - left)
    .slice(0, 2)
    .map(([signal]) => signal)

const normalizeBackendGqs = (gqs: BackendGqs): GuardianPrediction["gqs"] => {
  const score = clamp01(
    ((gqs.q_skel + gqs.q_int + gqs.q_obj + gqs.q_po) / 4) *
      gqs.valid_ratio,
  )

  return {
    score,
    label: score >= 0.75 ? "high" : score >= 0.45 ? "medium" : "low",
    threshold_passed: score >= 0.5,
  }
}

const normalizeBackendTelemetry = (
  telemetry: BackendTelemetry,
  fallback: GuardianPrediction["telemetry"],
): GuardianPrediction["telemetry"] => {
  const peakWindow = toFrameWindow(telemetry.peak_window)

  return {
    peak_window: peakWindow,
    total_frames: Math.max(fallback.total_frames, peakWindow[1], 32),
    processing_ms: fallback.processing_ms,
    detected_people: telemetry.people,
  }
}

const normalizeBackendWeapons = (
  telemetry: BackendTelemetry,
  confidence: number,
): GuardianPrediction["weapons"] => [
  {
    label: telemetry.weapon.cls ?? "weapon",
    present: telemetry.weapon.flag,
    confidence: telemetry.weapon.flag ? confidence : 0,
  },
]

const normalizeIncidentDetailGate = (
  gate: BackendIncidentDetail["gate"],
  fallback: GuardianPrediction["gate"],
): GuardianPrediction["gate"] => ({
  skeleton: gate?.skeleton ?? fallback.skeleton,
  interaction: gate?.interaction ?? fallback.interaction,
  object: gate?.object ?? fallback.object,
  vit: gate?.vit ?? fallback.vit,
})

const normalizeIncidentDetailGqs = (
  gqs: BackendIncidentDetail["gqs"],
  fallback: GuardianPrediction["gqs"],
): GuardianPrediction["gqs"] => {
  if (
    gqs?.q_skel === undefined ||
    gqs.q_int === undefined ||
    gqs.q_obj === undefined ||
    gqs.q_po === undefined ||
    gqs.valid_ratio === undefined
  ) {
    return fallback
  }

  return normalizeBackendGqs(gqs as BackendGqs)
}

const getBackendPredictionMedia = (
  clipId: string,
  fallback: GuardianPrediction["media"],
  overlay?: BackendOverlayResponse | null,
): GuardianPrediction["media"] => ({
  original_video_url: toStaticMediaUrl(`static/uploads/${clipId}`),
  overlay_video_url: toStaticMediaUrl(overlay?.overlay_path),
  thumbnail_url: toStaticMediaUrl(overlay?.thumbnail_path),
  overlay_url: null,
  source: clipId,
  resolution: fallback.resolution,
  mode: "Live",
})

const getIncidentDetailMedia = (
  incident: BackendIncidentDetail,
  fallback: GuardianPrediction["media"],
): GuardianPrediction["media"] => {
  const source = incident.source || incident.clip_id

  return {
    original_video_url: toStaticMediaUrl(`static/uploads/${source}`),
    overlay_video_url: toStaticMediaUrl(incident.overlay_path),
    thumbnail_url: toStaticMediaUrl(incident.thumbnail_path),
    overlay_url: null,
    source,
    resolution: fallback.resolution,
    mode: "Live",
  }
}

const normalizeIncidentMedia = (
  incident: GuardianIncident,
): GuardianIncident => ({
  ...incident,
  thumbnail_url: toBackendMediaUrl(incident.thumbnail_url) ?? incident.thumbnail_url,
})

const normalizePredictionMedia = (
  prediction: GuardianPrediction,
): GuardianPrediction => ({
  ...prediction,
  media: {
    ...prediction.media,
    original_video_url: toOptionalBackendMediaUrl(
      prediction.media.original_video_url,
    ),
    overlay_video_url: toOptionalBackendMediaUrl(
      prediction.media.overlay_video_url,
    ),
    thumbnail_url: toOptionalBackendMediaUrl(prediction.media.thumbnail_url),
    overlay_url: toBackendMediaUrl(prediction.media.overlay_url),
  },
  incidents: prediction.incidents.map(normalizeIncidentMedia),
})

const withPredictionFallbacks = (
  response: BackendPredictResponse,
  fallback: GuardianSample = mockSamples[0],
  overlay?: BackendOverlayResponse | null,
  incidents: GuardianIncident[] = [],
): GuardianPrediction => {
  const dominantSignals = getDominantSignals(response.gate)

  return normalizePredictionMedia({
    ...fallback,
    clip_id: response.clip_id,
    verdict: response.verdict,
    confidence: response.confidence,
    threshold: response.threshold,
    gate: response.gate,
    gqs: normalizeBackendGqs(response.gqs),
    telemetry: normalizeBackendTelemetry(response.telemetry, fallback.telemetry),
    media: getBackendPredictionMedia(response.clip_id, fallback.media, overlay),
    weapons: normalizeBackendWeapons(response.telemetry, response.confidence),
    explanation: {
      narrative: `Backend prediction complete: ${response.verdict} (${Math.round(
        response.confidence * 100,
      )}% confidence). Request an explanation after prediction for the persisted incident narrative.`,
      dominant_signals: dominantSignals,
    },
    incidents,
  })
}

const renderBackendOverlay = async (
  clipId: string,
): Promise<BackendOverlayResponse | null> => {
  try {
    const formData = new FormData()
    formData.append("clip_id", clipId)

    const response = await guardianHttp.post<BackendOverlayResponse>(
      "/overlay",
      formData,
    )

    return response.data
  } catch (error) {
    warnAndContinue("overlay render", error)

    return null
  }
}

const normalizeBackendIncident = (
  incident: BackendIncidentSummary,
): GuardianIncident => ({
  incident_id: incident.incident_id,
  clip_id: incident.source,
  timestamp: incident.timestamp,
  verdict: incident.verdict,
  confidence: incident.confidence,
  thumbnail_url: toStaticMediaUrl(incident.thumbnail) ?? "",
  narrative: incident.narrative_preview ?? "",
})

const normalizeBackendIncidentDetail = (
  incident: BackendIncidentDetail,
  fallback: GuardianPrediction = mockSamples[0],
  incidents: GuardianIncident[] = [],
): GuardianPrediction => {
  const gate = normalizeIncidentDetailGate(incident.gate, fallback.gate)
  const peakWindow = toFrameWindow(incident.peak_window ?? undefined)
  const narrative =
    incident.narrative ??
    incident.packet_summary ??
    fallback.explanation.narrative

  return normalizePredictionMedia({
    ...fallback,
    clip_id: incident.clip_id || incident.source || incident.incident_id,
    verdict: incident.verdict,
    confidence: incident.confidence,
    threshold: incident.threshold ?? fallback.threshold,
    gate,
    gqs: normalizeIncidentDetailGqs(incident.gqs, fallback.gqs),
    telemetry: {
      peak_window: peakWindow,
      total_frames: Math.max(fallback.telemetry.total_frames, peakWindow[1], 32),
      processing_ms: fallback.telemetry.processing_ms,
      detected_people:
        incident.people_count ?? fallback.telemetry.detected_people,
    },
    media: getIncidentDetailMedia(incident, fallback.media),
    weapons: [
      {
        label: incident.weapon_class ?? "weapon",
        present: Boolean(incident.weapon_flag),
        confidence: incident.weapon_flag ? incident.confidence : 0,
      },
    ],
    explanation: {
      narrative,
      dominant_signals: getDominantSignals(gate),
    },
    incidents,
  })
}

const normalizeBackendAskResponse = (
  response: BackendAskResponse,
): GuardianAskResponse => {
  const incidents = response.incidents.map(normalizeBackendIncident)
  const confidence = incidents.reduce(
    (highest, incident) => Math.max(highest, incident.confidence),
    0,
  )

  return {
    answer: response.answer,
    related_incident_ids: incidents.map((incident) => incident.incident_id),
    confidence,
  }
}

export async function getSamples(): Promise<GuardianSample[]> {
  await wait(500)

  return mockSamples
}

export async function checkBackendStatus(): Promise<GuardianBackendStatus> {
  if (guardianApiMode === "mock") {
    logApiDebug("mock service selected for backend status", {
      reason: "VITE_API_MODE=mock",
    })
    return "mock"
  }

  try {
    const response = await guardianHttp.get("/health", {
      timeout: 1500,
      validateStatus: () => true,
    })

    return response.status >= 200 && response.status < 300
      ? "connected"
      : "unavailable"
  } catch {
    return "unavailable"
  }
}

export async function predictSample(
  clipId: string,
): Promise<GuardianPrediction> {
  logApiDebug("local sample prediction selected", {
    clipId,
    serviceMode: "mock-sample",
  })
  await wait(800)

  return findSampleByClipId(clipId)
}

export async function predictVideo(file: File): Promise<GuardianPrediction> {
  if (guardianApiMode === "mock") {
    logApiDebug("mock service selected for /predict", {
      reason: "VITE_API_MODE=mock",
      fileName: file.name,
    })
    await wait(1000)

    return mockSamples[0]
  }

  try {
    const formData = new FormData()
    formData.append("clip", file)
    formData.append("clip_id", file.name)

    const response = await guardianHttp.post<BackendPredictResponse>(
      "/predict",
      formData,
    )
    logApiDebug("/predict FastAPI response received", {
      clipId: response.data.clip_id,
      verdict: response.data.verdict,
      confidence: response.data.confidence,
    })
    const overlay = await renderBackendOverlay(response.data.clip_id)
    const incidents = await getHistory()

    return withPredictionFallbacks(
      response.data,
      mockSamples[0],
      overlay,
      incidents,
    )
  } catch (error) {
    warnAndFallback("video prediction", error)

    return mockSamples[0]
  }
}

export async function explainIncident(
  clipId: string,
  language: "en" | "ar" = "en",
  country?: string,
): Promise<ExplanationResponse> {
  if (guardianApiMode === "mock") {
    logApiDebug("mock service selected for /explain", {
      reason: "VITE_API_MODE=mock",
      clipId,
      country,
    })
    await wait(500)

    const incident = mockHistory.find(
      (historyItem) => historyItem.incident_id === clipId,
    )

    return {
      narrative: incident?.narrative ?? mockSamples[0].explanation.narrative,
      dominant_signals: mockSamples[0].explanation.dominant_signals,
      explanation_rag: mockSamples[0].explanation_rag,
      incident_memory_rag: mockSamples[0].incident_memory_rag,
      legal_consequences_rag: country
        ? {
            ...mockSamples[0].legal_consequences_rag!,
            country,
          }
        : {
            ...mockSamples[0].legal_consequences_rag!,
            country: "",
            retrieved_legal_references: [],
            summary: "Select a country to request possible legal consequences.",
            guardrail_status: "needs_review",
            rag_mode: "mock",
            legal_rag_source: "fallback",
            legal_rag_warning:
              "Country is required before Legal RAG can retrieve references.",
            warning: "country_required",
          },
      legal_scores: country ? mockSamples[0].legal_scores : null,
    }
  }

  try {
    const response = await guardianHttp.post<BackendExplainResponse>(
      "/explain",
      {
        clip_id: clipId,
        language,
        country,
      },
    )
    logApiDebug("/explain FastAPI response received", {
      clipId,
      country,
      ragMode: response.data.legal_consequences_rag?.rag_mode,
      legalRagSource: response.data.legal_consequences_rag?.legal_rag_source,
      legalRagWarning: response.data.legal_consequences_rag?.legal_rag_warning,
      references:
        response.data.legal_consequences_rag?.retrieved_legal_references.length ?? 0,
    })

    return {
      narrative: response.data.narrative,
      dominant_signals: mockSamples[0].explanation.dominant_signals,
      explanation_rag: response.data.explanation_rag,
      incident_memory_rag: response.data.incident_memory_rag,
      legal_consequences_rag: response.data.legal_consequences_rag,
      legal_scores: response.data.legal_scores,
    }
  } catch (error) {
    warnAndContinue("incident explanation", error)

    throw error
  }
}

export async function getHistory(): Promise<GuardianIncident[]> {
  if (guardianApiMode === "mock") {
    logApiDebug("mock service selected for /history", {
      reason: "VITE_API_MODE=mock",
    })
    await wait(500)

    return mockHistory
  }

  try {
    const response = await guardianHttp.get<BackendHistoryResponse>("/history")

    return response.data.incidents.map(normalizeBackendIncident)
  } catch (error) {
    warnAndFallback("history request", error)

    return mockHistory
  }
}

export async function getIncidentReview(
  incidentId: string,
): Promise<GuardianPrediction> {
  if (guardianApiMode === "mock") {
    logApiDebug("mock service selected for incident review", {
      reason: "VITE_API_MODE=mock",
      incidentId,
    })
    await wait(500)

    const incident = mockHistory.find(
      (historyItem) => historyItem.incident_id === incidentId,
    )
    const fallback = findSampleByClipId(incident?.clip_id ?? mockSamples[0].clip_id)

    return normalizePredictionMedia({
      ...fallback,
      explanation: {
        ...fallback.explanation,
        narrative: incident?.narrative ?? fallback.explanation.narrative,
      },
      incidents: mockHistory,
    })
  }

  try {
    const [detailResponse, incidents] = await Promise.all([
      guardianHttp.get<BackendIncidentDetail>(
        `/incident/${encodeURIComponent(incidentId)}`,
      ),
      getHistory(),
    ])

    return normalizeBackendIncidentDetail(
      detailResponse.data,
      mockSamples[0],
      incidents,
    )
  } catch (error) {
    warnAndFallback("incident review", error)

    return mockSamples[0]
  }
}

export async function askGuardianEye(
  question: string,
  clipId?: string,
  language: "en" | "ar" = "en",
): Promise<GuardianAskResponse> {
  if (guardianApiMode === "mock") {
    logApiDebug("mock service selected for /ask", {
      reason: "VITE_API_MODE=mock",
      clipId,
    })
    await wait(700)

    return findSampleByClipId(clipId ?? mockSamples[0].clip_id).ask_response
  }

  try {
    const request = {
      question,
      language,
    }

    const response = await guardianHttp.post<BackendAskResponse>(
      "/ask",
      request,
    )

    return normalizeBackendAskResponse(response.data)
  } catch (error) {
    warnAndFallback("ask request", error)

    return findSampleByClipId(clipId ?? mockSamples[0].clip_id).ask_response
  }
}
