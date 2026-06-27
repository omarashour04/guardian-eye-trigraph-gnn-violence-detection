from rag_service.legal_sources import LEGAL_SOURCE_REGISTRY
from rag_service.schemas import LegalConsequencesInput, LegalConsequencesOutput


REQUIRED_COUNTRIES = {"UK", "USA California", "Canada", "KSA", "UAE", "Egypt"}
REQUIRED_SOURCE_FIELDS = {
    "source_url",
    "law_title",
    "source_language",
    "official_source",
    "source_type",
}


def _schema_fields(model):
    if hasattr(model, "model_fields"):
        return model.model_fields
    return model.__fields__


def test_all_required_countries_exist():
    assert REQUIRED_COUNTRIES.issubset(LEGAL_SOURCE_REGISTRY.keys())


def test_each_country_has_at_least_one_source():
    for country in REQUIRED_COUNTRIES:
        assert LEGAL_SOURCE_REGISTRY[country]


def test_every_source_has_required_metadata():
    for sources in LEGAL_SOURCE_REGISTRY.values():
        for source in sources:
            assert REQUIRED_SOURCE_FIELDS.issubset(source.keys())
            assert source["source_type"] in {"html", "pdf"}
            assert isinstance(source["official_source"], bool)


def test_detected_actions_is_not_required_by_legal_input_schema():
    fields = _schema_fields(LegalConsequencesInput)
    assert "detected_actions" not in fields


def test_legal_output_schema_supports_references_and_limitations_note():
    output = LegalConsequencesOutput(
        legal_consequences={
            "country": "UAE",
            "query_basis": {
                "verdict": "violence",
                "weapon_flag": True,
                "weapon_class": "bottle",
            },
            "retrieved_legal_references": [
                {
                    "law_title": "UAE Crimes and Penalties Law",
                    "article_number": "example",
                    "source_url": "https://uaelegislation.gov.ae/ar/legislations/1529",
                    "snippet": "Example retrieved legal reference snippet.",
                    "score": 0.87,
                }
            ],
            "summary": (
                "According to the retrieved regulation, this type of violent "
                "conduct may be subject to legal consequences."
            ),
            "guardrail_status": "passed",
            "limitations_note": (
                "This is not legal advice and does not determine guilt or "
                "predict court outcome."
            ),
        }
    )

    legal_consequences = output.legal_consequences
    assert legal_consequences.retrieved_legal_references[0].score == 0.87
    assert legal_consequences.limitations_note
