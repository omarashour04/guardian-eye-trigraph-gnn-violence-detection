"""Safe schemas for the Legal Consequences RAG feature.

These schemas define the contract only. They do not perform retrieval,
summarization, indexing, scraping, or LLM calls.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LegalConsequencesInput(StrictSchema):
    country: str
    verdict: str
    confidence: float = Field(ge=0.0, le=1.0)
    packet_summary: str
    narrative: str
    weapon_flag: bool
    weapon_class: str | None = None
    language: str


class LegalQueryBasis(StrictSchema):
    verdict: str
    weapon_flag: bool
    weapon_class: str | None = None


class RetrievedLegalReference(StrictSchema):
    law_title: str
    article_number: str | None = None
    section_title: str | None = None
    source_url: str
    snippet: str
    score: float = Field(ge=0.0, le=1.0)
    country: str | None = None
    violence_category: str | None = None
    official_source: bool | None = None


class LegalConsequencesPayload(StrictSchema):
    country: str
    query_basis: LegalQueryBasis
    retrieved_legal_references: list[RetrievedLegalReference]
    summary: str
    guardrail_status: Literal["passed", "blocked", "needs_review"]
    limitations_note: str = (
        "This is not legal advice and does not determine guilt or predict court outcome."
    )


class LegalConsequencesOutput(StrictSchema):
    legal_consequences: LegalConsequencesPayload
