import json

from main_pipeline import run_pipeline

with open(
    "RAG_Violence_Detection/sample_classifier.json",
    "r",
    encoding="utf-8",
) as f:
    classifier_result = json.load(f)

result = run_pipeline(classifier_result)

print("\n=== PIPELINE OUTPUT ===\n")

for key, value in result.items():
    print(f"\n{key.upper()}")
    print(value)