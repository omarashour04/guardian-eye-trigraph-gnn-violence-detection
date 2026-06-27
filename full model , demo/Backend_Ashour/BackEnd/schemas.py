"""
Guardian Eye — Pydantic Schemas
All request/response contracts matching DEMO_APP.md §5 exactly.
"""

from __future__ import annotations
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field
import datetime


# ── Shared sub-schemas ────────────────────────────────────────────────────────

class GateWeights(BaseModel):
    skeleton: float = Field(..., ge=0.0, le=1.0)
    interaction: float = Field(..., ge=0.0, le=1.0)
    object: float = Field(..., ge=0.0, le=1.0)
    vit: float = Field(..., ge=0.0, le=1.0)


class GateValidity(BaseModel):
    status: Literal["valid", "partial", "unknown"] = "unknown"
    message: str = "Modality activity was not checked for this prediction."
    raw_gate_sum: float = Field(default=1.0, ge=0.0)
    active_gate_sum: float = Field(default=1.0, ge=0.0)
    inactive_gate_sum: float = Field(default=0.0, ge=0.0)
    unavailable_contributions: list[str] = []


class GQSScores(BaseModel):
    q_skel: float = Field(..., ge=0.0, le=1.0)
    q_int: float = Field(..., ge=0.0, le=1.0)
    q_obj: float = Field(..., ge=0.0, le=1.0)
    q_po: float = Field(..., ge=0.0, le=1.0)
    valid_ratio: float = Field(..., ge=0.0, le=1.0)


class WeaponInfo(BaseModel):
    flag: bool
    cls: Optional[str] = None


class Telemetry(BaseModel):
    """Geometry-derived heuristics — NOT classifier outputs."""
    people: int
    peak_window: list[int] = Field(..., min_length=2, max_length=2)
    weapon: WeaponInfo


class ModelRoute(BaseModel):
    selected_dataset: Literal["RLVS", "HF", "NTU"]
    routing_confidence: float = Field(..., ge=0.0, le=1.0)
    routing_reason: str
    dataset_similarity: dict[str, float] = Field(default_factory=dict)
    selected_v9_checkpoint: Optional[str] = None
    selected_videomae_checkpoint: Optional[str] = None
    route_source: str = "rules"
    fallback_used: bool = False
    requested_dataset: Optional[Literal["RLVS", "HF", "NTU"]] = None


# ── /predict ──────────────────────────────────────────────────────────────────

class PredictResponse(BaseModel):
    verdict: str
    confidence: float
    threshold: float
    gate: GateWeights
    active_modalities: list[str] = ["skeleton", "interaction", "object", "vit"]
    inactive_modalities: list[str] = []
    gate_validity: GateValidity = Field(default_factory=GateValidity)
    gqs: GQSScores
    telemetry: Telemetry
    clip_id: str
    source: Optional[str] = None
    model_route: Optional[ModelRoute] = None


class ExplanationRagOutput(BaseModel):
    status: str
    verdict: str
    confidence: float
    explanation: str
    evidence_basis: list[str] = []
    limitations: str


class SimilarIncident(BaseModel):
    incident_id: str
    summary: str
    similarity: float


class IncidentMemoryRagOutput(BaseModel):
    status: str
    query_basis: dict[str, Optional[str] | bool]
    similar_incidents: list[SimilarIncident] = []
    memory_note: str


class LegalQueryBasis(BaseModel):
    verdict: str
    weapon_flag: bool
    weapon_class: Optional[str] = None


class LegalReference(BaseModel):
    law_title: str
    article_number: Optional[str] = None
    section_title: Optional[str] = None
    source_url: str
    snippet: str
    score: float = Field(..., ge=0.0, le=1.0)
    country: Optional[str] = None
    violence_category: Optional[str] = None
    official_source: Optional[bool] = None


class LegalConsequencesRagOutput(BaseModel):
    country: str
    query_basis: LegalQueryBasis
    incident_context: Optional[dict[str, Any]] = None
    retrieved_legal_references: list[LegalReference] = []
    summary: str
    guardrail_status: Literal["passed", "blocked", "needs_review"]
    limitations_note: str
    rag_mode: str = "auto"
    legal_rag_source: Literal["mock", "real", "fallback"] = "fallback"
    legal_mode: Literal["llm", "curated_fallback"] = "curated_fallback"
    legal_rag_warning: Optional[str] = None
    reason_if_fallback: Optional[str] = None
    warning: Optional[str] = None
    vlm_summary_used: bool = False
    summary_source: Optional[str] = None
    vlm_people_count: Optional[int] = None
    vlm_violence_type: Optional[str] = None


class LegalScores(BaseModel):
    retrieval_score: Optional[float] = None
    generation_score: Optional[float] = None
    overall_score: Optional[float] = None
    passed: Optional[bool] = None


# ── /explain ──────────────────────────────────────────────────────────────────

class ExplainRequest(BaseModel):
    clip_id: str
    language: str = Field(default="en", pattern="^(en|ar)$")
    country: Optional[str] = None


class ExplainResponse(BaseModel):
    narrative: str
    incident_id: str
    language: str
    narration_mode: Literal["vlm_llm", "vlm_only", "fallback"] = "fallback"
    model_status: dict[str, str] = {}
    reason_if_fallback: Optional[str] = None
    explanation_rag: Optional[ExplanationRagOutput] = None
    incident_memory_rag: Optional[IncidentMemoryRagOutput] = None
    legal_consequences_rag: Optional[LegalConsequencesRagOutput] = None
    legal_scores: Optional[LegalScores] = None


class LegalConsequencesRequest(BaseModel):
    incident_id: Optional[str] = None
    clip_id: Optional[str] = None
    country: Optional[str] = None
    language: str = Field(default="en", pattern="^(en|ar)$")


# ── /history ──────────────────────────────────────────────────────────────────

class IncidentSummary(BaseModel):
    incident_id: str
    clip_id: str
    timestamp: datetime.datetime
    source: str
    verdict: str
    confidence: float
    thumbnail: Optional[str] = None
    overlay: Optional[str] = None          # NEW — skeleton-overlay video path
    people_count: int
    weapon_flag: bool
    weapon_class: Optional[str] = None
    peak_window: list[int]
    narrative_preview: Optional[str] = None
    model_route: Optional[ModelRoute] = None


class HistoryResponse(BaseModel):
    total: int
    incidents: list[IncidentSummary]


# ── /ask ──────────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    language: str = Field(default="en", pattern="^(en|ar)$")
    clip_id: Optional[str] = None
    incident_id: Optional[str] = None
    country: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    incidents: list[IncidentSummary]
    language: str
    ask_mode: Literal["llm", "fallback", "grounded"] = "fallback"
    selected_route: Optional[str] = None
    retrieved_context_count: int = 0
    reason_if_fallback: Optional[str] = None
    grounding_label: Optional[str] = None
    vlm_summary_used: bool = False
    summary_source: Optional[str] = None
    vlm_people_count: Optional[int] = None
    vlm_violence_type: Optional[str] = None
