import json

from adapter import convert_classifier_result
from g4_context_enrichment import build_enriched_context
with open("RAG_Violence_Detection/sample_classifier.json") as f:
    data = json.load(f)

classifier_input = convert_classifier_result(data)

context = build_enriched_context(classifier_input)
print(context)