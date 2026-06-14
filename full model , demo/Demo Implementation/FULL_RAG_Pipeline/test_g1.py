import json

from adapter import convert_classifier_result
from g1_evidence_packet import build_evidence_packet

with open("RAG_Violence_Detection/sample_classifier.json") as f:
    data = json.load(f)

classifier_input = convert_classifier_result(data)

packet = build_evidence_packet(classifier_input)

print(packet)