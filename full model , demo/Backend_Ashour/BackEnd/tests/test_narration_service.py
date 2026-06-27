from __future__ import annotations

import contextlib
import io
import os
import shutil
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from explanation_service import _deterministic_explain_from_pred
from narration_service import (
    _compact_summary_text,
    _deterministic_clean_narrative,
    _export_evidence_frames,
    _format_narrative_for_display,
    _guardrail_ok,
    _narrative_labels,
    _prepare_structured_translation_source,
    _restore_arabic_translation_structure,
    build_evidence_packet,
    build_vlm_summary,
    generate_grounded_narration,
    generate_grounded_narration_result,
    retrieve_reference_context,
)
from text_quality import polish_generated_text


def _fake_pred(clip_id: str):
    return SimpleNamespace(
        clip_id=clip_id,
        verdict="violence",
        confidence=0.6,
        threshold=0.28,
        gate=SimpleNamespace(
            skeleton=0.26,
            interaction=0.19,
            object=0.24,
            vit=0.32,
        ),
        gqs=SimpleNamespace(
            q_skel=0.88,
            q_int=0.77,
            q_obj=0.66,
            q_po=0.55,
            valid_ratio=0.91,
        ),
        telemetry=SimpleNamespace(
            people=5,
            peak_window=[12, 15],
            weapon=SimpleNamespace(flag=True, cls="bottle"),
        ),
    )


def _fake_nonviolent_pred(clip_id: str):
    pred = _fake_pred(clip_id)
    pred.verdict = "non-violence"
    pred.confidence = 0.12
    pred.telemetry.weapon.flag = False
    pred.telemetry.weapon.cls = None
    pred.telemetry.people = 2
    pred.telemetry.peak_window = [0, 0]
    return pred


class NarrationServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.old_enabled = os.environ.get("GUARDIAN_NARRATION_ENABLED")
        self.old_local = os.environ.get("GUARDIAN_MODEL_LOCAL_ONLY")
        self.old_vlm_enabled = os.environ.get("GUARDIAN_NARRATION_VLM_ENABLED")
        self.old_narration_mode = os.environ.get("GUARDIAN_NARRATION_MODE")
        self.old_unload_before_translation = os.environ.get(
            "GUARDIAN_NARRATION_UNLOAD_VLM_BEFORE_TRANSLATION"
        )
        os.environ["GUARDIAN_MODEL_LOCAL_ONLY"] = "1"
        self.clip_id = f"narration_test_{uuid.uuid4().hex}.mp4"

    def tearDown(self) -> None:
        if self.old_enabled is None:
            os.environ.pop("GUARDIAN_NARRATION_ENABLED", None)
        else:
            os.environ["GUARDIAN_NARRATION_ENABLED"] = self.old_enabled
        if self.old_local is None:
            os.environ.pop("GUARDIAN_MODEL_LOCAL_ONLY", None)
        else:
            os.environ["GUARDIAN_MODEL_LOCAL_ONLY"] = self.old_local
        for name, old_value in (
            ("GUARDIAN_NARRATION_VLM_ENABLED", self.old_vlm_enabled),
            ("GUARDIAN_NARRATION_MODE", self.old_narration_mode),
            (
                "GUARDIAN_NARRATION_UNLOAD_VLM_BEFORE_TRANSLATION",
                self.old_unload_before_translation,
            ),
        ):
            if old_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old_value

        npz_path = Path("cache_npz") / f"{self.clip_id}.npz"
        if npz_path.exists():
            npz_path.unlink()
        frame_dir = Path("static") / "evidence_frames" / Path(self.clip_id).stem
        if frame_dir.exists():
            shutil.rmtree(frame_dir)

    def test_disabled_narration_uses_fallback_without_models(self) -> None:
        os.environ["GUARDIAN_NARRATION_ENABLED"] = "0"

        result = generate_grounded_narration(
            clip_id=self.clip_id,
            pred=_fake_pred(self.clip_id),
            language="en",
            fallback=lambda: "deterministic fallback",
        )

        self.assertEqual(result, "deterministic fallback")

    def test_disabled_narration_reports_fallback_mode(self) -> None:
        os.environ["GUARDIAN_NARRATION_ENABLED"] = "0"

        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            result = generate_grounded_narration_result(
                clip_id=self.clip_id,
                pred=_fake_pred(self.clip_id),
                language="en",
                fallback=lambda: "deterministic fallback",
            )

        self.assertEqual(result.narration_mode, "fallback")
        self.assertEqual(result.narrative, "deterministic fallback")
        self.assertIn("disabled", result.reason_if_fallback.lower())
        self.assertEqual(result.model_status["vlm"], "disabled")
        self.assertIn("narration_fallback_reason='Narration AI disabled'", stream.getvalue())

    def test_missing_npz_uses_fallback_without_models(self) -> None:
        os.environ["GUARDIAN_NARRATION_ENABLED"] = "1"

        result = generate_grounded_narration(
            clip_id=self.clip_id,
            pred=_fake_pred(self.clip_id),
            language="en",
            fallback=lambda: "deterministic fallback",
        )

        self.assertEqual(result, "deterministic fallback")

    def test_valid_vlm_unloads_before_arabic_translation_and_skips_llm(self) -> None:
        os.environ["GUARDIAN_NARRATION_ENABLED"] = "1"
        os.environ["GUARDIAN_NARRATION_VLM_ENABLED"] = "1"
        os.environ["GUARDIAN_NARRATION_MODE"] = "vlm_llm"
        os.environ["GUARDIAN_NARRATION_UNLOAD_VLM_BEFORE_TRANSLATION"] = "1"
        events: list[str] = []
        evidence = SimpleNamespace(
            paths=["frame.jpg"],
            peak_vit_indices=(0,),
        )
        packet = {"verdict": "violence", "confidence": 0.6}
        english_vlm_narration = (
            "Scene: An indoor room. People observed: Two people are visible. "
            "Observed activity: One person appears to push another person. "
            "Guardian Eye assessment: The classifier verdict is violence."
        )

        def fake_vlm(*args, **kwargs):
            events.append("vlm_generated")
            from narration_service import _complete_vlm_unload

            _complete_vlm_unload()
            return english_vlm_narration

        def fake_unload():
            events.append("vlm_unloaded")

        def fake_translate(text):
            events.append("translation_started")
            return "سرد عربي صالح"

        with patch("narration_service._load_frames_vit", return_value=np.zeros((16, 1, 1, 3))), patch(
            "narration_service._export_evidence_frames", return_value=evidence
        ), patch("narration_service.build_evidence_packet", return_value=packet), patch(
            "narration_service._run_vlm", side_effect=fake_vlm
        ), patch("narration_service._complete_vlm_unload", side_effect=fake_unload) as unload, patch(
            "narration_service._run_llm"
        ) as llm, patch(
            "narration_service._translate_narration_to_arabic", side_effect=fake_translate
        ), patch(
            "narration_service._format_narrative_for_display", side_effect=lambda text, **kwargs: text
        ), patch(
            "narration_service.polish_generated_text", side_effect=lambda text, *args, **kwargs: text
        ), patch("narration_service._guardrail_ok", return_value=True), patch(
            "narration_service.build_vlm_summary", return_value={"summary_source": "current_vlm"}
        ), patch("arabic_translation_service.unload_translation_model"):
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                result = generate_grounded_narration_result(
                    clip_id=self.clip_id,
                    pred=_fake_pred(self.clip_id),
                    language="ar",
                    fallback=lambda: "fallback",
                )

        self.assertEqual(result.narration_mode, "vlm_only")
        self.assertEqual(result.model_status["llm"], "skipped")
        self.assertEqual(events, ["vlm_generated", "vlm_unloaded", "translation_started"])
        unload.assert_called_once()
        llm.assert_not_called()
        self.assertIn("narration_vlm_load_start", stream.getvalue())
        self.assertIn("narration_vlm_success", stream.getvalue())

    def test_exports_16_frames_and_structured_packet(self) -> None:
        pred = _fake_pred(self.clip_id)
        frames = np.zeros((16, 224, 224, 3), dtype=np.uint8)
        for idx in range(16):
            frames[idx, :, :, 0] = idx

        evidence_frames = _export_evidence_frames(self.clip_id, frames, pred)
        packet = build_evidence_packet(pred, evidence_frames)

        self.assertEqual(len(evidence_frames.paths), 16)
        self.assertEqual(len(packet["selected_frames"]), 16)
        self.assertEqual(packet["verdict"], "violence")
        self.assertEqual(packet["confidence"], 0.6)
        self.assertEqual(packet["peak_window"], [12, 15])
        self.assertEqual(packet["people_count"], 5)
        self.assertEqual(packet["dominant_contributors"][0]["stream"], "vit")
        self.assertTrue(all(Path(path).exists() for path in evidence_frames.paths))

    def test_reference_context_included_as_guidance_not_facts(self) -> None:
        pred = _fake_pred(self.clip_id)
        frames = np.zeros((16, 224, 224, 3), dtype=np.uint8)

        evidence_frames = _export_evidence_frames(self.clip_id, frames, pred)
        packet = build_evidence_packet(pred, evidence_frames)
        references = packet["reference_context"]

        self.assertGreaterEqual(len(references), 3)
        self.assertIn("severity", packet)
        self.assertEqual(packet["severity"], "high")
        self.assertTrue(
            any(reference["id"] == "object-proximity" for reference in references)
        )
        self.assertTrue(
            all(
                reference["grounding_type"] == "narration_guidance_not_incident_fact"
                for reference in references
            )
        )
        self.assertTrue(all(reference["guidance_en"] for reference in references))
        self.assertTrue(all(reference["guidance_ar"] for reference in references))
        self.assertIn(
            "Reference context is narration guidance",
            " ".join(packet["caveats"]),
        )
        caveats = " ".join(packet["caveats"])
        self.assertIn("appears to", caveats)
        self.assertIn("The frames do not clearly support identifying individual roles.", caveats)
        self.assertIn("لا تدعم الإطارات بوضوح تحديد أدوار الأفراد", caveats)

    def test_reference_retrieval_uses_nonviolent_threshold_context(self) -> None:
        references = retrieve_reference_context(_fake_nonviolent_pred(self.clip_id))
        reference_ids = {reference["id"] for reference in references}

        self.assertIn("non-violence-threshold", reference_ids)
        self.assertNotIn("object-proximity", reference_ids)

    def test_nonviolent_fallback_narration_is_one_concise_sentence(self) -> None:
        narrative = _deterministic_explain_from_pred(
            _fake_nonviolent_pred(self.clip_id),
            "en",
        )

        self.assertIn("non-violent", narrative)
        self.assertIn("88% confidence", narrative)
        self.assertEqual(narrative.count("."), 1)
        self.assertNotIn("12% confidence", narrative)
        self.assertNotIn("frames", narrative.casefold())
        self.assertNotIn("telemetry", narrative.casefold())
        self.assertNotIn("legal", narrative.casefold())
        self.assertNotIn("attacker", narrative.casefold())
        self.assertNotIn("victim", narrative.casefold())

    def test_display_formatter_preserves_nonviolent_one_sentence_rule(self) -> None:
        narrative = _format_narrative_for_display(
            "Guardian Eye classified the clip as non-violent. Extra caveat should not appear.",
            verdict="non-violence",
            language="en",
        )

        self.assertEqual(narrative.count("."), 1)

    def test_display_formatter_breaks_violent_observations_into_lines(self) -> None:
        narrative = _format_narrative_for_display(
            "Scene: The video appears to show an indoor area. Person 1: A person appears to stand near the entrance. "
            "Person 2: Another person seems to be on the floor. Action: The interaction may indicate an unsafe event. "
            "Guardian Eye assessment: The classifier identified violence with high confidence.",
            verdict="violence",
            language="en",
        )

        self.assertIn("Scene:\n", narrative)
        self.assertIn("\n\nPeople observed:\n", narrative)
        self.assertIn("\n\nObserved activity:\n", narrative)
        self.assertIn("\n\nGuardian Eye assessment:\n", narrative)
        self.assertNotIn("\n\nLimitations:\n", narrative)
        self.assertNotIn("**", narrative)

    def test_display_formatter_preserves_labeled_person_descriptions_under_people(self) -> None:
        narrative = _format_narrative_for_display(
            "Scene: The video appears to show an indoor area. "
            "People observed: Person 1: A person in a dark shirt appears to stand near the doorway. "
            "Person 2: A person in a blue shirt seems to be on the floor. "
            "Person 3: A third person may be moving close to the others. "
            "Observed activity: Person 1 appears to move toward Person 2. "
            "Person 3 appears to intervene nearby. "
            "Guardian Eye assessment: Guardian Eye classified the clip as violent with 76.5% confidence.",
            verdict="violence",
            language="en",
        )

        self.assertIn("People observed:\nPerson 1:\n", narrative)
        self.assertIn("dark shirt", narrative)
        self.assertIn("Person 2:\n", narrative)
        self.assertIn("blue shirt", narrative)
        self.assertIn("Person 3:\n", narrative)
        self.assertNotIn("Guardian Eye detected 3 visible people", narrative)

    def test_display_formatter_removes_forbidden_person_role_terms(self) -> None:
        narrative = _format_narrative_for_display(
            "Scene: The video appears to show an indoor area. "
            "People observed: Person 1: The attacker appears to wear a black shirt. "
            "Person 2: The victim seems to be on the floor. "
            "Person 3: A nearby person appears to stand close to the others. "
            "Observed activity: The exact sequence and intent cannot be determined. "
            "Guardian Eye assessment: Guardian Eye classified the clip as violent with 76.5% confidence.",
            verdict="violence",
            language="en",
        )

        self.assertNotIn("attacker", narrative.casefold())
        self.assertNotIn("victim", narrative.casefold())
        self.assertIn("Person 3:\n", narrative)
        self.assertIn("nearby person", narrative)

    def test_display_formatter_removes_caveat_and_legal_responsibility_wording(self) -> None:
        narrative = _format_narrative_for_display(
            "Guardian Eye classified the clip as violent with 76.5% confidence. "
            "Visible Scene Entities: Three individuals are visible in the selected evidence frames. "
            "Interaction: One person appears to be on the floor while another stands nearby. "
            "Caveats: Legal responsibility must remain human-reviewed. Frame-level timing is telemetry.",
            verdict="violence",
            language="en",
        )

        self.assertIn("People observed:", narrative)
        self.assertIn("Observed activity:", narrative)
        self.assertNotIn("Caveats:", narrative)
        self.assertNotIn("selected evidence frames", narrative)
        self.assertNotIn("legal responsibility", narrative.casefold())
        self.assertNotIn("telemetry", narrative.casefold())

    def test_display_formatter_removes_evidence_file_artifacts(self) -> None:
        narrative = _format_narrative_for_display(
            "Scene: evidence_00_vit_00.jpg shows an indoor area. "
            "[evidence_01_vit_01.jpg] People observed: Three individuals are visible. "
            "Observed activity: One person appears to be on the floor while another stands nearby. "
            "Guardian Eye assessment: Guardian Eye classified the clip as violent with 76.5% confidence.",
            verdict="violence",
            language="en",
            packet={"people_count": 3, "confidence": 0.765},
        )

        self.assertIn("Scene:", narrative)
        self.assertNotIn("evidence_00_vit_00.jpg", narrative)
        self.assertNotIn("evidence_01_vit_01.jpg", narrative)
        self.assertNotIn("[", narrative)

    def test_display_formatter_removes_gate_telemetry_text(self) -> None:
        narrative = _format_narrative_for_display(
            "Summary: People Count is 3. V9 classifier. evidence_02.png. "
            "Gate Contribution: Moderate (0.28). "
            "Interaction Contribution: Low (0.20). Scene: The video appears to show an indoor area. "
            "People observed: Three individuals are visible. "
            "Observed activity: One person appears to be on the floor at frame index 12 while another stands nearby. "
            "Peak frames show strongest model evidence. "
            "Guardian Eye assessment: Guardian Eye classified the clip as violent with 76.5% confidence.",
            verdict="violence",
            language="en",
            packet={"people_count": 3, "confidence": 0.765},
        )

        self.assertIn("People observed:", narrative)
        self.assertNotIn("Summary:", narrative)
        self.assertNotIn("Gate Contribution", narrative)
        self.assertNotIn("Interaction Contribution", narrative)
        self.assertNotIn("(0.28)", narrative)
        self.assertNotIn("telemetry", narrative.casefold())
        self.assertNotIn("V9", narrative)
        self.assertNotIn("evidence_02.png", narrative)
        self.assertNotIn("frame index", narrative.casefold())
        self.assertNotIn("peak", narrative.casefold())
        self.assertNotIn("strongest model evidence", narrative.casefold())

    def test_display_formatter_removes_limitations_and_dangling_fragments(self) -> None:
        narrative = _format_narrative_for_display(
            "Scene: The video appears to show an indoor area. "
            "People observed: Three individuals are visible. Wearing "
            "Observed activity: One person appears to be on the floor during the at frame index 11 and 12. "
            "Another person seems to stand nearby. "
            "Guardian Eye assessment: Guardian Eye classified the clip as violent with high confidence (81.",
            verdict="violence",
            language="en",
            packet={"people_count": 3, "confidence": 0.81},
        )

        self.assertNotIn("Limitations:", narrative)
        self.assertNotIn("Wearing", narrative)
        self.assertNotIn("frame index", narrative.casefold())
        self.assertNotIn("(81.", narrative)
        self.assertTrue(narrative.rstrip().endswith("."))

    def test_deterministic_clean_narrative_uses_vlm_person_descriptions(self) -> None:
        narrative = _deterministic_clean_narrative(
            packet={"people_count": 3, "confidence": 0.765},
            visual_observations=(
                "Scene: indoor area. "
                "Person 1: A person in a dark shirt appears to stand near the doorway. "
                "Person 2: A person in a blue shirt seems to be on the floor. "
                "Person 3: A third person may be standing nearby."
            ),
            language="en",
        )

        self.assertIn("People observed:\nPerson 1:\n", narrative)
        self.assertIn("dark shirt", narrative)
        self.assertIn("Person 2:\n", narrative)
        self.assertIn("blue shirt", narrative)
        self.assertNotIn("Guardian Eye detected 3 visible people", narrative)

    def test_display_formatter_uses_arabic_section_labels(self) -> None:
        labels = _narrative_labels("ar")
        narrative = _format_narrative_for_display(
            "Scene: The video appears to show an indoor area. "
            "People observed: Person 1: A person appears to stand near the doorway. "
            "Observed activity: The interaction may indicate unsafe close contact. "
            "Guardian Eye assessment: Guardian Eye classified the clip as violent with 76.5% confidence.",
            verdict="violence",
            language="ar",
        )

        self.assertIn(f"{labels['scene']}:\n", narrative)
        self.assertIn(f"{labels['people']}:\n", narrative)
        self.assertIn("الشخص 1:\n", narrative)
        self.assertNotIn("Scene:", narrative)
        self.assertNotIn("People observed:", narrative)
        self.assertNotIn("Observed activity:", narrative)
        self.assertNotIn("Guardian Eye assessment:", narrative)

    def test_display_formatter_preserves_arabic_violent_structure(self) -> None:
        narrative = _format_narrative_for_display(
            "المشهد: يبدو أن الفيديو يعرض منطقة داخلية. "
            "الأشخاص المرصودون: الشخص 1: يبدو أن شخصا يرتدي قميصا داكنا يقف قرب المدخل. "
            "الشخص 2: يبدو أن شخصا يرتدي قميصا أزرق موجود على الأرض. "
            "النشاط المرصود: يبدو أن الشخص 1 يتحرك باتجاه الشخص 2. "
            "تقييم Guardian Eye: صنف Guardian Eye المقطع على أنه عنيف بثقة 76.5%.",
            verdict="violence",
            language="ar",
        )

        self.assertIn("المشهد:\n", narrative)
        self.assertIn("الأشخاص المرصودون:\n", narrative)
        self.assertIn("الشخص 1:\n", narrative)
        self.assertIn("الشخص 2:\n", narrative)
        self.assertIn("النشاط المرصود:\n", narrative)
        self.assertIn("تقييم Guardian Eye:\n", narrative)
        self.assertNotIn("Scene:", narrative)
        self.assertNotIn("People observed:", narrative)
        self.assertNotIn("Observed activity:", narrative)
        self.assertNotIn("Guardian Eye assessment:", narrative)

    def test_translation_structure_markers_restore_complete_arabic_sections(self) -> None:
        source, required = _prepare_structured_translation_source(
            "Scene: A room. People observed: Person 1: A person is on the floor. "
            "Person 2: Another person stands nearby. Observed activity: Close contact. "
            "Guardian Eye assessment: Violence."
        )
        translated = (
            source.replace("A room.", "غرفة.")
            .replace("A person is on the floor.", "شخص على الأرض.")
            .replace("Another person stands nearby.", "شخص آخر يقف بالقرب منه.")
            .replace("Close contact.", "تلامس قريب.")
            .replace("Violence.", "عنف.")
        )

        restored = _restore_arabic_translation_structure(
            translated,
            required_markers=required,
        )

        self.assertIn("المشهد:", restored)
        self.assertIn("الأشخاص المرصودون:", restored)
        self.assertIn("الشخص 1:", restored)
        self.assertIn("الشخص 2:", restored)
        self.assertIn("النشاط المرصود:", restored)
        self.assertIn("تقييم Guardian Eye:", restored)

    def test_translation_structure_markers_reject_missing_sections(self) -> None:
        with self.assertRaisesRegex(ValueError, "did not preserve structure markers"):
            _restore_arabic_translation_structure(
                "[[GE_ASSESSMENT]] العنف.",
                required_markers=("[[GE_SCENE]]", "[[GE_ASSESSMENT]]"),
            )

    def test_arabic_formatter_does_not_reinsert_english_vlm_people(self) -> None:
        narrative = _format_narrative_for_display(
            "المشهد: يبدو أن الفيديو يعرض منطقة داخلية. "
            "الأشخاص المرصودون: يظهر شخصان في المكان. "
            "النشاط المرصود: تبدو هناك مواجهة جسدية محتملة. "
            "تقييم Guardian Eye: صنف النظام المقطع على أنه عنيف بثقة 76.5%.",
            verdict="violence",
            language="ar",
            visual_observations=(
                "Person 1: A person appears to be on the ground, wearing a blue shirt. "
                "Person 2: Another person seems to stand nearby."
            ),
        )

        self.assertNotIn("appears", narrative.casefold())
        self.assertNotIn("wearing", narrative.casefold())
        self.assertNotIn("Person 1", narrative)

    def test_display_formatter_removes_arabic_mode_internal_artifacts(self) -> None:
        narrative = _format_narrative_for_display(
            "المشهد: evidence_00_vit_00.jpg يبدو أن الفيديو يعرض منطقة داخلية. "
            "الأشخاص المرصودون: الشخص 1: يبدو أن شخصا يقف قرب المدخل. "
            "النشاط المرصود: Gate Contribution: Moderate (0.28). V9 classifier. "
            "يبدو أن النشاط قريب من الآخرين عند frame index 12. Peak frames show telemetry. "
            "تقييم Guardian Eye: صنف Guardian Eye المقطع على أنه عنيف بثقة 76.5%.",
            verdict="violence",
            language="ar",
            packet={"people_count": 1, "confidence": 0.765},
        )

        self.assertIn("المشهد:", narrative)
        self.assertNotIn("evidence_00_vit_00.jpg", narrative)
        self.assertNotIn("Gate Contribution", narrative)
        self.assertNotIn("V9", narrative)
        self.assertNotIn("telemetry", narrative.casefold())
        self.assertNotIn("frame index", narrative.casefold())
        self.assertNotIn("Peak", narrative)

    def test_display_formatter_preserves_arabic_nonviolent_one_sentence_rule(self) -> None:
        narrative = _format_narrative_for_display(
            "صنف Guardian Eye المقطع على أنه غير عنيف بثقة 88%. لا يظهر سلوك عنيف واضح. جملة إضافية.",
            verdict="non-violence",
            language="ar",
        )

        self.assertEqual(narrative.count("."), 1)
        self.assertIn("غير عنيف", narrative)
        self.assertNotIn("Scene:", narrative)

    def test_compact_summary_truncates_at_sentence_boundary(self) -> None:
        text = "First sentence is complete. Second sentence is also complete. Third sentence is cut midway"
        compact = _compact_summary_text(text, 55)

        self.assertEqual(compact, "First sentence is complete.")
        self.assertTrue(compact.endswith("."))

    def test_guardrail_rejects_forbidden_role_claims(self) -> None:
        self.assertTrue(
            _guardrail_ok(
                "The person in the dark shirt appears to move toward another person.",
                "violence",
            )
        )
        self.assertFalse(_guardrail_ok("The attacker is the person in black.", "violence"))
        self.assertFalse(_guardrail_ok("This person committed assault.", "violence"))
        self.assertFalse(_guardrail_ok("هذا الشخص مسؤول قانونيا.", "violence"))

        self.assertFalse(
            _guardrail_ok(
                "Guardian Eye classified this clip as non-violent with 88% confidence, with peak frames 1-4.",
                "non-violence",
            )
        )

    def test_builds_structured_vlm_summary_from_visual_text(self) -> None:
        packet = {
            "people_count": 3,
            "severity": "medium",
            "verdict": "violence",
            "weapon": {"flag": False, "class": None},
        }

        summary = build_vlm_summary(
            visual_observations=(
                "Two individuals are visible on a street at night. The person in a "
                "dark shirt appears to move toward another person, and one person "
                "appears to push or strike. A vehicle is nearby."
            ),
            narrative="Two individuals engaged in a physical altercation.",
            packet=packet,
            language="en",
        )

        self.assertEqual(summary["summary_source"], "current_vlm")
        self.assertEqual(summary["people_count"], 2)
        self.assertEqual(summary["environment"], "street at night")
        self.assertIn("physical altercation", summary["violence_type"])
        self.assertIn("vehicle", summary["objects"])
        self.assertTrue(summary["possible_roles"])

    def test_arabic_narration_preserves_english_vlm_scene_entities_without_raw_telemetry(self) -> None:
        english_narration = (
            "The Guardian Eye classifier classified the clip as violent with 60% confidence. "
            "Two people are visible on a street at night, and one person in a dark shirt "
            "appears to push another person near a vehicle."
        )
        arabic_translation = (
            "صنف مصنف Guardian Eye المقطع على أنه عنيف بثقة 60%. "
            "يظهر شخصان في شارع ليلا، ويبدو أن شخصا يرتدي قميصا داكنا "
            "يدفع شخصا آخر بالقرب من مركبة. frames 12-15 V9 classifier peak evidenced"
        )

        clean_arabic = polish_generated_text(arabic_translation, "ar", label="narration")

        for required in ("شخصان", "شارع", "قميص", "داكنا", "يدفع", "مركبة"):
            self.assertIn(required, clean_arabic)
        self.assertNotIn("frames", clean_arabic.casefold())
        self.assertNotIn("peak", clean_arabic.casefold())
        self.assertNotIn("evidenced", clean_arabic.casefold())
        self.assertNotIn("V9", clean_arabic)
        self.assertNotRegex(clean_arabic, r"\b\d+\s*[-–]\s*\d+\b")
        self.assertIn("Guardian Eye", english_narration)


if __name__ == "__main__":
    unittest.main()
