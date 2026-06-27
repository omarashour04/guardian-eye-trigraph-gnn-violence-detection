from __future__ import annotations

import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from bilingual_rag import run_bilingual_rag
from translation_service import contains_arabic


class FakeTranslator:
    def __init__(self) -> None:
        self.calls = []
        self.unloaded = False

    def translate(self, text, source_language, target_language):
        self.calls.append((text, source_language, target_language))
        if target_language == "English":
            return "bottle fight"
        return "شجار بزجاجة"

    def unload(self):
        self.unloaded = True


def test_contains_arabic():
    assert contains_arabic("هل حدث شجار؟")
    assert not contains_arabic("Was there a fight?")


def test_english_query_does_not_construct_translator():
    calls = []

    def rag(query, top_k):
        calls.append((query, top_k))
        return [{"text": "bottle fight", "score": 0.9}]

    result = run_bilingual_rag("bottle fight", rag, top_k=3)

    assert calls == [("bottle fight", 3)]
    assert result[0]["text"] == "bottle fight"


def test_arabic_query_translates_query_and_user_facing_result():
    translator = FakeTranslator()
    received_queries = []

    def rag(query):
        received_queries.append(query)
        return [
            {
                "incident_id": "incident-001",
                "text": "bottle fight",
                "score": 0.9,
            }
        ]

    result = run_bilingual_rag(
        "هل حدث شجار بزجاجة؟",
        rag,
        translator=translator,
    )

    assert received_queries == ["bottle fight"]
    assert result == [
        {
            "incident_id": "incident-001",
            "text": "شجار بزجاجة",
            "score": 0.9,
        }
    ]
    assert translator.unloaded is True
