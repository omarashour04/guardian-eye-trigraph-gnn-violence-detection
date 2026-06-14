export type GuardianVerdict = "violence" | "non-violence"

export type GuardianApiMode = "mock" | "backend"

export type GuardianBackendStatus = "mock" | "connected" | "unavailable"

export type GuardianGate = {
  skeleton: number
  interaction: number
  object: number
  vit: number
}

export type GuardianGqs = {
  score: number
  label: "low" | "medium" | "high"
  threshold_passed: boolean
}

export type GuardianTelemetry = {
  peak_window: [number, number]
  total_frames: number
  processing_ms: number
  detected_people: number
}

export type GuardianWeapon = {
  label: string
  present: boolean
  confidence: number
}

export type GuardianMedia = {
  original_video_url?: string
  overlay_video_url?: string
  thumbnail_url?: string
  overlay_url?: string | null
  source: string
  resolution: string
  mode: "Demo" | "Live"
}

export type GuardianExplanation = {
  narrative: string
  dominant_signals: string[]
}

export type GuardianExplanationRag = {
  status: string
  verdict: GuardianVerdict
  confidence: number
  explanation: string
  evidence_basis: string[]
  limitations: string
}

export type GuardianSimilarIncident = {
  incident_id: string
  summary: string
  similarity: number
}

export type GuardianIncidentMemoryRag = {
  status: string
  query_basis: {
    verdict: GuardianVerdict
    weapon_flag: boolean
    weapon_class?: string | null
  }
  similar_incidents: GuardianSimilarIncident[]
  memory_note: string
}

export type GuardianLegalReference = {
  law_title: string
  article_number?: string | null
  section_title?: string | null
  source_url: string
  snippet: string
  score: number
  country?: string | null
  violence_category?: string | null
  official_source?: boolean | null
}

export type GuardianLegalConsequencesRag = {
  country: string
  query_basis: {
    verdict: GuardianVerdict
    weapon_flag: boolean
    weapon_class?: string | null
  }
  retrieved_legal_references: GuardianLegalReference[]
  summary: string
  guardrail_status: "passed" | "blocked" | "needs_review"
  limitations_note: string
  rag_mode?: string
  legal_rag_source?: "mock" | "real" | "fallback"
  legal_rag_warning?: string | null
  warning?: string | null
}

export type GuardianLegalScores = {
  retrieval_score?: number | null
  generation_score?: number | null
  overall_score?: number | null
  passed?: boolean | null
}

export type GuardianRagEnrichment = {
  explanation_rag?: GuardianExplanationRag | null
  incident_memory_rag?: GuardianIncidentMemoryRag | null
  legal_consequences_rag?: GuardianLegalConsequencesRag | null
  legal_scores?: GuardianLegalScores | null
}

export type GuardianIncident = {
  incident_id: string
  clip_id: string
  timestamp: string
  verdict: GuardianVerdict
  confidence: number
  thumbnail_url: string
  narrative: string
}

export type GuardianAskResponse = {
  answer: string
  related_incident_ids: string[]
  confidence: number
}

export type GuardianPrediction = {
  clip_id: string
  verdict: GuardianVerdict
  confidence: number
  threshold: number
  gate: GuardianGate
  gqs: GuardianGqs
  telemetry: GuardianTelemetry
  media: GuardianMedia
  weapons: GuardianWeapon[]
  explanation: GuardianExplanation
  explanation_rag?: GuardianRagEnrichment["explanation_rag"]
  incident_memory_rag?: GuardianRagEnrichment["incident_memory_rag"]
  legal_consequences_rag?: GuardianRagEnrichment["legal_consequences_rag"]
  legal_scores?: GuardianRagEnrichment["legal_scores"]
  incidents: GuardianIncident[]
}

export type GuardianPredictResponse = {
  clip_id: string
  verdict: GuardianVerdict
  confidence: number
  threshold: number
  gate: GuardianGate
  gqs: GuardianGqs
  telemetry: GuardianTelemetry
  media: GuardianMedia
}

export type GuardianAskRequest = {
  question: string
  clip_id?: string
  incident_id?: string
}

export type GuardianSample = GuardianPrediction & {
  title: string
  description: string
  ask_response: GuardianAskResponse
}
