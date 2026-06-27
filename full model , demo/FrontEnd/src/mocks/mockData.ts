import type {
  GuardianAskResponse,
  GuardianExplanationRag,
  GuardianIncident,
  GuardianIncidentMemoryRag,
  GuardianLegalConsequencesRag,
  GuardianLegalScores,
  GuardianSample,
} from "@/types/guardian"

const mockExplanationRag: GuardianExplanationRag = {
  status: "mocked",
  verdict: "violence",
  confidence: 0.94,
  explanation:
    "The mock explanation RAG summarizes the classifier result without changing the original verdict or confidence.",
  evidence_basis: [
    "classifier confidence 0.94",
    "interaction and skeleton streams",
    "packet summary available",
  ],
  limitations: "This mock explanation is deterministic and does not use a heavy model.",
}

const mockIncidentMemoryRag: GuardianIncidentMemoryRag = {
  status: "mocked",
  query_basis: {
    verdict: "violence",
    weapon_flag: false,
    weapon_class: null,
  },
  similar_incidents: [
    {
      incident_id: "mock-memory-001",
      summary: "Prior mock confrontation with similar interaction intensity.",
      similarity: 0.82,
    },
    {
      incident_id: "mock-memory-002",
      summary: "Prior mock incident classified as violence for demo comparison.",
      similarity: 0.76,
    },
  ],
  memory_note: "This mock memory output does not query a real vector store.",
}

const mockLegalConsequencesRag: GuardianLegalConsequencesRag = {
  country: "UK",
  query_basis: {
    verdict: "violence",
    weapon_flag: false,
    weapon_class: null,
  },
  retrieved_legal_references: [
    {
      law_title: "MOCK legal placeholder",
      article_number: "MOCK",
      section_title: "Mock-only demo reference",
      source_url: "https://example.com/mock-legal-reference",
      snippet:
        "Mock-only placeholder reference. This was not retrieved from a real country-specific legal index.",
      score: 0,
      country: "UK",
      violence_category: "mock_only",
      official_source: false,
    },
  ],
  summary:
    "Mock-only Legal RAG output: no real country-specific legal index was queried. Use this only to verify the UI shape, not as grounded legal retrieval.",
  guardrail_status: "needs_review",
  limitations_note:
    "This is not legal advice, does not determine guilt, and does not predict court outcome.",
  rag_mode: "mock",
  legal_rag_source: "mock",
  legal_rag_warning: "Frontend mock mode is active; real Legal RAG was not used.",
  warning: "mock_only",
}

const mockLegalScores: GuardianLegalScores = {
  retrieval_score: null,
  generation_score: null,
  overall_score: null,
  passed: false,
}

export const mockSamples: GuardianSample[] = [
  {
    clip_id: "clip_violence_high_001",
    title: "High-Risk Corridor Incident",
    description: "Two-person confrontation with strong motion and contact cues.",
    verdict: "violence",
    confidence: 0.94,
    threshold: 0.7,
    gate: {
      skeleton: 0.34,
      interaction: 0.41,
      object: 0.07,
      vit: 0.18,
    },
    gqs: {
      score: 0.88,
      label: "high",
      threshold_passed: true,
    },
    telemetry: {
      peak_window: [14, 22],
      total_frames: 40,
      processing_ms: 1260,
      detected_people: 2,
    },
    media: {
      thumbnail_url: "/static/thumbnails/clip_001.jpg",
      overlay_url: null,
      source: "Mock overlay stream",
      resolution: "720p",
      mode: "Demo",
    },
    weapons: [
      {
        label: "weapon",
        present: false,
        confidence: 0.08,
      },
    ],
    explanation: {
      narrative:
        "The model detected a violent interaction primarily from interaction and skeleton signals. The highest activity occurred between frames 14 and 22. The confidence score indicates a strong likelihood of violent behavior.",
      dominant_signals: ["interaction", "skeleton"],
    },
    explanation_rag: mockExplanationRag,
    incident_memory_rag: mockIncidentMemoryRag,
    legal_consequences_rag: mockLegalConsequencesRag,
    legal_scores: mockLegalScores,
    incidents: [
      {
        incident_id: "incident_001",
        clip_id: "clip_violence_high_001",
        timestamp: "2026-06-06 15:30",
        verdict: "violence",
        confidence: 0.94,
        thumbnail_url: "/static/thumbnails/clip_001.jpg",
        narrative:
          "High-risk violent interaction detected with strong interaction and skeleton evidence.",
      },
    ],
    ask_response: {
      answer:
        "I found one violent incident. It involved two people with strong interaction and skeleton signals. The peak activity occurred between frames 14 and 22.",
      related_incident_ids: ["incident_001"],
      confidence: 0.91,
    },
  },
  {
    clip_id: "clip_non_violence_low_002",
    title: "Normal Lobby Movement",
    description: "Low-risk passage through monitored area with no conflict cues.",
    verdict: "non-violence",
    confidence: 0.18,
    threshold: 0.7,
    gate: {
      skeleton: 0.18,
      interaction: 0.12,
      object: 0.08,
      vit: 0.09,
    },
    gqs: {
      score: 0.22,
      label: "low",
      threshold_passed: false,
    },
    telemetry: {
      peak_window: [0, 0],
      total_frames: 40,
      processing_ms: 980,
      detected_people: 1,
    },
    media: {
      thumbnail_url: "/static/thumbnails/clip_002.jpg",
      overlay_url: null,
      source: "Mock overlay stream",
      resolution: "720p",
      mode: "Demo",
    },
    weapons: [
      {
        label: "weapon",
        present: false,
        confidence: 0.03,
      },
    ],
    explanation: {
      narrative:
        "The model classified this clip as non-violent. Motion patterns remained stable, interaction intensity stayed below the threshold, and no high-risk peak window was identified.",
      dominant_signals: ["vit"],
    },
    explanation_rag: {
      ...mockExplanationRag,
      verdict: "non-violence",
      confidence: 0.18,
      evidence_basis: ["classifier confidence 0.18", "calm movement pattern"],
    },
    incident_memory_rag: {
      ...mockIncidentMemoryRag,
      query_basis: {
        verdict: "non-violence",
        weapon_flag: false,
        weapon_class: null,
      },
      similar_incidents: [],
    },
    legal_consequences_rag: {
      ...mockLegalConsequencesRag,
      query_basis: {
        verdict: "non-violence",
        weapon_flag: false,
        weapon_class: null,
      },
      summary:
        "Possible legal consequences are not summarized confidently for this low-risk mock sample without stronger retrieved legal references.",
      guardrail_status: "needs_review",
      retrieved_legal_references: [],
    },
    legal_scores: null,
    incidents: [
      {
        incident_id: "incident_002",
        clip_id: "clip_non_violence_low_002",
        timestamp: "2026-06-06 15:44",
        verdict: "non-violence",
        confidence: 0.18,
        thumbnail_url: "/static/thumbnails/clip_002.jpg",
        narrative: "Normal movement detected with no sustained risk signals.",
      },
    ],
    ask_response: {
      answer:
        "I found one low-risk non-violent sample. It shows normal movement with no sustained interaction peak or strong skeleton aggression cues.",
      related_incident_ids: ["incident_002"],
      confidence: 0.84,
    },
  },
  {
    clip_id: "clip_violence_medium_003",
    title: "Medium-Risk Entrance Event",
    description: "Brief aggressive movement with moderate interaction evidence.",
    verdict: "violence",
    confidence: 0.76,
    threshold: 0.7,
    gate: {
      skeleton: 0.28,
      interaction: 0.31,
      object: 0.11,
      vit: 0.16,
    },
    gqs: {
      score: 0.71,
      label: "medium",
      threshold_passed: true,
    },
    telemetry: {
      peak_window: [18, 25],
      total_frames: 40,
      processing_ms: 1180,
      detected_people: 2,
    },
    media: {
      thumbnail_url: "/static/thumbnails/clip_003.jpg",
      overlay_url: null,
      source: "Mock overlay stream",
      resolution: "720p",
      mode: "Demo",
    },
    weapons: [
      {
        label: "weapon",
        present: false,
        confidence: 0.11,
      },
    ],
    explanation: {
      narrative:
        "The model detected a medium-confidence violent event. Interaction and skeleton gates crossed the decision threshold, with the strongest activity concentrated between frames 18 and 25.",
      dominant_signals: ["interaction", "skeleton", "vit"],
    },
    explanation_rag: {
      ...mockExplanationRag,
      confidence: 0.76,
      evidence_basis: ["classifier confidence 0.76", "interaction and skeleton streams"],
    },
    incident_memory_rag: mockIncidentMemoryRag,
    legal_consequences_rag: mockLegalConsequencesRag,
    legal_scores: mockLegalScores,
    incidents: [
      {
        incident_id: "incident_003",
        clip_id: "clip_violence_medium_003",
        timestamp: "2026-06-06 16:05",
        verdict: "violence",
        confidence: 0.76,
        thumbnail_url: "/static/thumbnails/clip_003.jpg",
        narrative:
          "Medium-confidence violent event detected near the entrance zone.",
      },
    ],
    ask_response: {
      answer:
        "I found a medium-confidence violent incident. The strongest evidence came from interaction and skeleton features, with peak activity between frames 18 and 25.",
      related_incident_ids: ["incident_003"],
      confidence: 0.82,
    },
  },
]

export const mockHistory: GuardianIncident[] = mockSamples.flatMap(
  (sample) => sample.incidents,
)

export const mockAskResponses: GuardianAskResponse[] = mockSamples.map(
  (sample) => sample.ask_response,
)
