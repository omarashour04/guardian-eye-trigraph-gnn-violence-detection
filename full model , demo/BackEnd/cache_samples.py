"""
Guardian Eye — cache_samples.py
Pre-seeds the incident history with curated demo incidents so the
"tell me about the fight from a week ago" query always has something
compelling to retrieve during the defense.

Run ONCE after the server is up:
    python cache_samples.py
"""

from __future__ import annotations
import json, datetime, requests, io
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"

# ── Seed incidents (simulated past timestamps via explain then DB patch) ──────

SEED_CLIPS = [
    {
        "filename": "cam3_clip_07.mp4",
        "days_ago": 7,
        "label": "hockey fight with bottle",
    },
    {
        "filename": "cam1_clip_14.mp4",
        "days_ago": 3,
        "label": "two-person altercation",
    },
    {
        "filename": "cam2_clip_03.mp4",
        "days_ago": 1,
        "label": "routine non-violent activity",
    },
]


def _make_dummy_video(filename: str) -> bytes:
    """Return a tiny stub so the server has something to save."""
    return b"\x00\x01\x02" * 16


def seed():
    import sqlite3, os

    print("Seeding incident history…")
    for clip in SEED_CLIPS:
        fname = clip["filename"]
        dummy = _make_dummy_video(fname)

        # 1. /predict
        pred_resp = requests.post(
            f"{BASE_URL}/predict",
            files={"clip": (fname, io.BytesIO(dummy), "video/mp4")},
            data={"clip_id": fname},
        )
        pred_resp.raise_for_status()
        clip_id = pred_resp.json()["clip_id"]
        print(f"  ✓ /predict  {clip_id} → {pred_resp.json()['verdict']}")

        # 2. /explain  (writes incident record)
        exp_resp = requests.post(
            f"{BASE_URL}/explain",
            json={"clip_id": clip_id, "language": "en"},
        )
        exp_resp.raise_for_status()
        incident_id = exp_resp.json()["incident_id"]
        print(f"  ✓ /explain  incident_id={incident_id}")

        # 3. Back-date the timestamp so history queries are realistic
        db_path = Path("db/guardian_eye.db")
        if db_path.exists():
            ts = datetime.datetime.utcnow() - datetime.timedelta(days=clip["days_ago"])
            conn = sqlite3.connect(str(db_path))
            conn.execute(
                "UPDATE incidents SET timestamp=? WHERE incident_id=?",
                (ts.isoformat(), incident_id),
            )
            conn.commit()
            conn.close()
            print(f"  ✓ back-dated to {ts.date()} ({clip['days_ago']}d ago)")

    print("\nDone. Incident history seeded.")
    print("Try: GET /history  or  POST /ask  { \"question\": \"fight from a week ago\" }")


if __name__ == "__main__":
    seed()
