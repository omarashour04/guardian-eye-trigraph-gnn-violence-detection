import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import json
from adapter import convert_classifier_result

with open("RAG_Violence_Detection/sample_classifier.json") as f:
    data = json.load(f)

result = convert_classifier_result(data)

print(result)