from __future__ import annotations

import contextlib
import io
import os
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from schemas import GateWeights, GQSScores, PredictResponse, Telemetry, WeaponInfo
from services.curated_legal_service import (
    NON_VIOLENT_LEGAL_MESSAGE,
    build_curated_legal_response,
    build_legal_prompt,
    infer_act_type,
    infer_severity,
    retrieve_curated_legal_context,
)


def _prediction(
    *,
    verdict: str = "violence",
    confidence: float = 0.86,
    weapon_flag: bool = False,
    weapon_class: str | None = None,
) -> PredictResponse:
    return PredictResponse(
        verdict=verdict,
        confidence=confidence,
        threshold=0.51,
        gate=GateWeights(skeleton=0.3, interaction=0.4, object=0.1, vit=0.2),
        gqs=GQSScores(q_skel=0.8, q_int=0.9, q_obj=0.45, q_po=0.5, valid_ratio=0.95),
        telemetry=Telemetry(
            people=3,
            peak_window=[6, 16],
            weapon=WeaponInfo(flag=weapon_flag, cls=weapon_class),
        ),
        clip_id="curated-legal-test.mp4",
    )


def _contains_arabic(text: str) -> bool:
    return any("\u0600" <= char <= "\u06ff" for char in text)


class CuratedLegalServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.old_llm_enabled = os.environ.get("GUARDIAN_LEGAL_LLM_ENABLED")
        os.environ["GUARDIAN_LEGAL_LLM_ENABLED"] = "0"

    def tearDown(self) -> None:
        if self.old_llm_enabled is None:
            os.environ.pop("GUARDIAN_LEGAL_LLM_ENABLED", None)
        else:
            os.environ["GUARDIAN_LEGAL_LLM_ENABLED"] = self.old_llm_enabled

    def test_non_violent_returns_fixed_safe_message_without_references(self) -> None:
        output, scores = build_curated_legal_response(
            pred=_prediction(verdict="non-violence", confidence=0.72),
            packet_summary="classifier packet",
            narrative="narrative",
            country="Canada",
            language="en",
        )

        self.assertIsNone(scores)
        self.assertEqual(output["summary"], NON_VIOLENT_LEGAL_MESSAGE)
        self.assertEqual(output["retrieved_legal_references"], [])
        self.assertEqual(output["guardrail_status"], "passed")

    def test_violent_canada_uses_curated_reference_and_cautious_fallback(self) -> None:
        output, scores = build_curated_legal_response(
            pred=_prediction(weapon_flag=True, weapon_class="bottle"),
            packet_summary="weapon/object stream contributed to the decision",
            narrative="The saved narration mentions only model evidence.",
            country="Canada",
            language="en",
        )

        self.assertIsNotNone(scores)
        self.assertEqual(output["country"], "Canada")
        self.assertEqual(output["legal_rag_source"], "real")
        self.assertEqual(output["legal_mode"], "curated_fallback")
        self.assertIsNotNone(output["reason_if_fallback"])
        self.assertTrue(output["retrieved_legal_references"])
        self.assertTrue(
            output["retrieved_legal_references"][0]["source_url"].startswith(
                "curated-legal-kb://"
            )
        )
        self.assertIn("If responsibility is confirmed by authorities", output["summary"])
        self.assertNotIn("the attacker is", output["summary"].lower())
        self.assertNotIn("the victim is", output["summary"].lower())

    def test_person_object_class_is_not_legal_weapon_context(self) -> None:
        pred = _prediction(
            confidence=0.7,
            weapon_flag=True,
            weapon_class="person",
        )

        act_type = infer_act_type(
            pred,
            "weapon/object stream detected person proximity",
            "A person appears near another person.",
        )
        severity = infer_severity(
            pred,
            "weapon/object stream detected person proximity",
            "A person appears near another person.",
        )
        output, scores = build_curated_legal_response(
            pred=pred,
            packet_summary="weapon/object stream detected person proximity",
            narrative="A person appears near another person.",
            country="Canada",
            language="en",
        )

        self.assertEqual(act_type, "violent_conduct")
        self.assertEqual(severity, "medium")
        self.assertIsNotNone(scores)
        self.assertFalse(output["query_basis"]["weapon_flag"])
        self.assertFalse(output["query_basis"]["dangerous_object_context"])
        self.assertEqual(output["incident_context"]["act_type"], "violent_conduct")
        self.assertIn(
            "Weapon/object context: No weapon or dangerous object was identified.",
            output["summary"],
        )
        self.assertTrue(output["retrieved_legal_references"])
        self.assertTrue(
            all(
                reference["violence_category"] != "weapon_or_dangerous_object"
                for reference in output["retrieved_legal_references"]
            )
        )

    def test_person_object_leak_is_removed_from_llm_summary_and_references(self) -> None:
        import services.curated_legal_service as service

        original = service._run_legal_llm
        os.environ["GUARDIAN_LEGAL_LLM_ENABLED"] = "1"
        try:
            service._run_legal_llm = lambda **kwargs: (
                "Country: UK\n"
                "Relevant act: Offensive weapon in a public place\n"
                "Weapon/object context: present (person)\n"
                "Possible consequence: Weapon-specific enforcement may apply."
            )
            output, _scores = build_curated_legal_response(
                pred=_prediction(weapon_flag=True, weapon_class="person"),
                packet_summary="object_label=person",
                narrative="A person appears near another person.",
                country="UK",
                language="en",
            )
        finally:
            service._run_legal_llm = original

        self.assertFalse(output["query_basis"]["weapon_flag"])
        self.assertFalse(output["incident_context"]["dangerous_object_context"])
        self.assertIn(
            "Weapon/object context: No weapon or dangerous object was identified.",
            output["summary"],
        )
        self.assertNotIn("present (person)", output["summary"])
        self.assertNotIn("Offensive weapon in a public place", output["summary"])
        self.assertTrue(
            all(
                reference["violence_category"] != "weapon_or_dangerous_object"
                for reference in output["retrieved_legal_references"]
            )
        )

    def test_vlm_person_object_label_is_not_dangerous_object_category(self) -> None:
        pred = _prediction(confidence=0.7, weapon_flag=False, weapon_class=None)

        act_type = infer_act_type(
            pred,
            "object_label=person",
            "A person appears near another person.",
            vlm_summary={"objects": ["person"], "violence_type": "physical altercation"},
        )

        self.assertEqual(act_type, "violent_conduct")

    def test_knife_and_bottle_remain_dangerous_object_context(self) -> None:
        for weapon_class in ("knife", "bottle"):
            with self.subTest(weapon_class=weapon_class):
                pred = _prediction(weapon_flag=True, weapon_class=weapon_class)
                act_type = infer_act_type(pred, f"{weapon_class} object", "")
                output, _scores = build_curated_legal_response(
                    pred=pred,
                    packet_summary=f"{weapon_class} object",
                    narrative="",
                    country="Canada",
                    language="en",
                )

                self.assertEqual(act_type, "weapon_or_dangerous_object")
                self.assertTrue(output["query_basis"]["weapon_flag"])
                self.assertTrue(output["query_basis"]["dangerous_object_context"])
                self.assertEqual(
                    output["retrieved_legal_references"][0]["violence_category"],
                    "weapon_or_dangerous_object",
                )

    def test_violent_canada_reports_llm_when_generation_succeeds(self) -> None:
        import services.curated_legal_service as service

        original = service._run_legal_llm
        os.environ["GUARDIAN_LEGAL_LLM_ENABLED"] = "1"
        try:
            service._run_legal_llm = lambda **kwargs: (
                "possible actions may include police review and protective conditions."
            )
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                output, scores = build_curated_legal_response(
                    pred=_prediction(weapon_flag=True, weapon_class="bottle"),
                    packet_summary="weapon/object stream contributed to the decision",
                    narrative="This narrative should not be repeated in legal output.",
                    country="Canada",
                    language="en",
                )
        finally:
            service._run_legal_llm = original

        self.assertIsNotNone(scores)
        self.assertEqual(output["legal_mode"], "llm")
        self.assertIsNone(output["reason_if_fallback"])
        self.assertIn("If responsibility is confirmed by authorities", output["summary"])
        self.assertIn("not legal advice", output["summary"])
        self.assertNotIn("This narrative should not be repeated", output["summary"])
        self.assertIn("legal_llm_load_start", stream.getvalue())
        self.assertIn("legal_llm_success", stream.getvalue())

    def test_curated_legal_summary_removes_incomplete_confidence_fragment(self) -> None:
        import services.curated_legal_service as service

        original = service._run_legal_llm
        os.environ["GUARDIAN_LEGAL_LLM_ENABLED"] = "1"
        try:
            service._run_legal_llm = lambda **kwargs: (
                "If responsibility is confirmed by authorities, in Canada, if an assault "
                "occurs with high confidence (81."
            )
            output, _scores = build_curated_legal_response(
                pred=_prediction(),
                packet_summary="classifier packet",
                narrative="narrative",
                country="Canada",
                language="en",
            )
        finally:
            service._run_legal_llm = original

        self.assertNotIn("(81.", output["summary"])
        self.assertNotIn("confidence", output["summary"].casefold())
        self.assertTrue(output["summary"].rstrip().endswith("."))

    def test_fallback_reason_is_logged_when_legal_llm_disabled(self) -> None:
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            output, _scores = build_curated_legal_response(
                pred=_prediction(),
                packet_summary="classifier packet",
                narrative="narrative",
                country="Canada",
                language="en",
            )

        logs = stream.getvalue()
        self.assertEqual(output["legal_mode"], "curated_fallback")
        self.assertIn("[legal] env_enabled=0", logs)
        self.assertIn("[legal] attempted_llm=false", logs)
        self.assertIn("[legal] mode=curated_fallback", logs)
        self.assertIn("[legal] fallback_reason='Legal LLM disabled'", logs)
        self.assertIn("[legal] legal_fallback_reason='Legal LLM disabled'", logs)

    def test_legal_mode_is_llm_only_when_llm_generation_is_used(self) -> None:
        import services.curated_legal_service as service

        output, _scores = build_curated_legal_response(
            pred=_prediction(),
            packet_summary="classifier packet",
            narrative="narrative",
            country="Canada",
            language="en",
        )
        self.assertEqual(output["legal_mode"], "curated_fallback")

        original = service._run_legal_llm
        os.environ["GUARDIAN_LEGAL_LLM_ENABLED"] = "1"
        try:
            service._run_legal_llm = lambda **kwargs: "Police review may lead to protective conditions."
            output, _scores = build_curated_legal_response(
                pred=_prediction(),
                packet_summary="classifier packet",
                narrative="narrative",
                country="Canada",
                language="en",
            )
        finally:
            service._run_legal_llm = original
        self.assertEqual(output["legal_mode"], "llm")

    def test_arabic_country_returns_arabic_curated_summary(self) -> None:
        output, scores = build_curated_legal_response(
            pred=_prediction(weapon_flag=True, weapon_class="knife"),
            packet_summary="object evidence available",
            narrative="",
            country="UAE",
            language="ar",
        )

        self.assertIsNotNone(scores)
        self.assertEqual(output["country"], "UAE")
        self.assertTrue(_contains_arabic(output["summary"]))
        self.assertTrue(_contains_arabic(output["limitations_note"]))
        self.assertTrue(_contains_arabic(output["retrieved_legal_references"][0]["snippet"]))

    def test_all_demo_countries_have_low_medium_high_curated_context(self) -> None:
        cases = [
            ("Canada", "en"),
            ("UK", "en"),
            ("USA California", "en"),
            ("Egypt", "ar"),
            ("UAE", "ar"),
            ("KSA", "ar"),
        ]
        for country, language in cases:
            for severity in ("low", "medium", "high"):
                with self.subTest(country=country, severity=severity):
                    records = retrieve_curated_legal_context(
                        country=country,
                        language=language,
                        act_type="violent_conduct",
                        severity=severity,
                    )
                    self.assertTrue(records)
                    self.assertEqual(records[0].country, country)
                    self.assertIn(records[0].language, {language, "en", "ar"})

    def test_english_countries_have_distinct_curated_consequences(self) -> None:
        summaries = {}
        for country in ("Canada", "UK", "USA California"):
            records = retrieve_curated_legal_context(
                country=country,
                language="en",
                act_type="violent_conduct",
                severity="medium",
            )
            summaries[country] = records[0].consequence

        self.assertEqual(len(set(summaries.values())), 3)

    def test_arabic_countries_have_country_specific_references(self) -> None:
        references = {}
        for country in ("Egypt", "UAE", "KSA"):
            records = retrieve_curated_legal_context(
                country=country,
                language="ar",
                act_type="violent_conduct",
                severity="low",
            )
            references[country] = records[0].reference
            self.assertTrue(_contains_arabic(records[0].consequence))

        self.assertEqual(len(set(references.values())), 3)

    def test_legal_response_uses_vlm_summary_diagnostics(self) -> None:
        output, scores = build_curated_legal_response(
            pred=_prediction(weapon_flag=False),
            packet_summary="telemetry summary",
            narrative="narrative",
            country="Canada",
            language="en",
            vlm_summary={
                "summary_source": "current_vlm",
                "people_count": 2,
                "observed_actions": ["pushing", "physical confrontation"],
                "objects": ["vehicle"],
                "violence_type": "physical altercation",
                "severity_estimate": "medium",
                "visual_summary": "Two individuals engaged in a physical altercation.",
            },
        )

        self.assertIsNotNone(scores)
        self.assertTrue(output["vlm_summary_used"])
        self.assertEqual(output["summary_source"], "current_vlm")
        self.assertEqual(output["vlm_people_count"], 2)
        self.assertEqual(output["vlm_violence_type"], "physical altercation")
        self.assertEqual(output["incident_context"]["act_type"], "violent_conduct")
        self.assertEqual(output["incident_context"]["severity"], "medium")

    def test_prompt_marks_curated_context_as_guidance_not_incident_fact(self) -> None:
        pred = _prediction(weapon_flag=True, weapon_class="knife")
        act_type = infer_act_type(pred, "knife object", "")
        severity = infer_severity(pred, "", "")
        records = retrieve_curated_legal_context(
            country="USA California",
            language="en",
            act_type=act_type,
            severity=severity,
        )
        messages = build_legal_prompt(
            pred=pred,
            packet_summary="packet",
            narrative="narrative",
            country="USA California",
            language="en",
            act_type=act_type,
            severity=severity,
            records=records,
        )
        joined = "\n".join(message["content"] for message in messages)

        self.assertIn("Curated legal context is the only legal source", joined)
        self.assertIn("vlm_incident_understanding", joined)
        self.assertIn("incident narrative is intentionally excluded", joined)
        self.assertNotIn('"narrative": "narrative"', joined)
        self.assertNotIn('"peak_window"', joined)
        self.assertNotIn('"people"', joined)
        self.assertIn("If responsibility is confirmed by authorities", joined)
        self.assertIn("Do not identify attacker, victim, guilt", joined)


if __name__ == "__main__":
    unittest.main()
