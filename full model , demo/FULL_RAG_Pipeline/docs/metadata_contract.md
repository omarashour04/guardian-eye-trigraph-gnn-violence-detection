# Guardian Eye RAG Metadata Contract

This document defines the exact JSON metadata contract expected by the Guardian Eye RAG layer.

## JSON Shape

```json
{
  "clip_id": "clip-001",
  "incident_id": "incident-001",
  "timestamp": "2026-06-08T20:30:00",
  "source": "camera-01",
  "verdict": "violence",
  "confidence": 0.91,
  "threshold": 0.75,
  "gate": {
    "skeleton": 0.87,
    "interaction": 0.83,
    "object": 0.42,
    "vit": 0.94
  },
  "gqs": {
    "q_skel": 0.88,
    "q_int": 0.81,
    "q_obj": 0.70,
    "q_po": 0.76,
    "valid_ratio": 0.93
  },
  "telemetry": {
    "people": 2,
    "peak_window": [24, 48],
    "weapon": {
      "flag": false,
      "class_name": null
    }
  },
  "frames_ref": [
    "frames/incident-001/frame_024.jpg",
    "frames/incident-001/frame_048.jpg"
  ]
}
```

## Field Contract

| Field | Type | Source | Meaning |
| --- | --- | --- | --- |
| `clip_id` | `str` | Tera clip processor | Unique identifier for the analyzed video clip. |
| `incident_id` | `str` | Tera incident tracker | Unique identifier linking the clip to a Guardian Eye incident. |
| `timestamp` | `datetime` as ISO 8601 string | Tera capture pipeline | Time when the clip or incident metadata was produced. |
| `source` | `str` | Tera camera or stream registry | Origin of the clip, such as camera ID, stream ID, or file source. |
| `verdict` | `str` | Tera classifier | Final classifier decision for the clip. |
| `confidence` | `float` | Tera classifier | Confidence score associated with the classifier verdict. |
| `threshold` | `float` | Tera classifier configuration | Decision threshold used to produce the verdict. |
| `gate.skeleton` | `float` | Tera skeleton stream | Gate score from skeleton-based motion or pose evidence. |
| `gate.interaction` | `float` | Tera interaction stream | Gate score from person-to-person interaction evidence. |
| `gate.object` | `float` | Tera object stream | Gate score from object-related evidence. |
| `gate.vit` | `float` | Tera ViT stream | Gate score from visual transformer classifier evidence. |
| `gqs.q_skel` | `float` | Tera quality scoring | Quality score for skeleton evidence. |
| `gqs.q_int` | `float` | Tera quality scoring | Quality score for interaction evidence. |
| `gqs.q_obj` | `float` | Tera quality scoring | Quality score for object evidence. |
| `gqs.q_po` | `float` | Tera quality scoring | Quality score for person-object evidence. |
| `gqs.valid_ratio` | `float` | Tera quality scoring | Ratio of valid frames or valid evidence used for the clip decision. |
| `telemetry.people` | `int` | Tera detection telemetry | Number of people detected in the relevant incident window. |
| `telemetry.peak_window` | `List[int]` | Tera temporal localization | Frame or time-index range where classifier evidence is strongest. |
| `telemetry.weapon.flag` | `bool` | Tera weapon detector | Whether a weapon-like object was detected. |
| `telemetry.weapon.class_name` | `Optional[str]` | Tera weapon detector | Detected weapon class name, or `null` when no class is available. |
| `frames_ref` | `List[str]` | Tera frame exporter | References to key frames available for evidence review or narrative grounding. |

## Fields Required From Tera

- `clip_id`
- `incident_id`
- `timestamp`
- `source`
- `verdict`
- `confidence`
- `threshold`
- `gate.skeleton`
- `gate.interaction`
- `gate.object`
- `gate.vit`
- `gqs.q_skel`
- `gqs.q_int`
- `gqs.q_obj`
- `gqs.q_po`
- `gqs.valid_ratio`
- `telemetry.people`
- `telemetry.peak_window`
- `telemetry.weapon.flag`
- `telemetry.weapon.class_name`
- `frames_ref`
