from __future__ import annotations

import sys
from pathlib import Path

# Make Elweeka's rag_service importable
ELWEEKA_DIR = (
    Path(__file__).resolve().parent
    / "RAG_Violence_Detection"
    / "Elweeka"
)

sys.path.insert(0, str(ELWEEKA_DIR))

from adapter import convert_classifier_result
from g1_evidence_packet import build_evidence_packet
from g4_context_enrichment import build_enriched_context
from g3_incident_db import create_tables

from rag_service.explanation_service import (
    generate_clip_explanation,
)


def run_pipeline(classifier_result):
    """
    End-to-end Guardian Eye pipeline.
    """

    # Step 1: Convert Tera output into internal schema
    classifier_input = convert_classifier_result(
        classifier_result
    )

    # Step 2: Build evidence packet
    evidence_packet = build_evidence_packet(
        classifier_input
    )

    # Step 3: Retrieve references and similar incidents
    enriched_context = build_enriched_context(
        classifier_input
    )

    # Step 4: Convert retrieved references into Elweeka schema
    formatted_references = []

    for i, ref in enumerate(
        enriched_context["retrieved_references"]
    ):
        formatted_references.append(
            {
                "reference_id": f"ref_{i + 1}",
                "source": "rag_store_1",
                "title": ref.get("title"),
                "snippet": ref.get("snippet"),
                "score": ref.get("score"),
                "metadata": {},
            }
        )

    # Step 5: Generate explanation
    explanation = generate_clip_explanation(
        verdict=classifier_input.verdict,
        confidence=classifier_input.confidence,
        packet_summary=evidence_packet.packet_summary,
        retrieved_references=formatted_references,
        frames_ref=classifier_input.frames_ref,
    )

    # Step 6: Ensure database exists
    create_tables()

    return {
        "classifier_input": classifier_input,
        "evidence_packet": evidence_packet,
        "context": enriched_context,
        "explanation": explanation,
    }