from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from rag.schemas import GQSScores, GateScores, IncidentRecord


DB_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "incidents.db"
)


def _connect() -> sqlite3.Connection:
    """Create a SQLite connection for the Guardian Eye incident database."""

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def create_tables() -> None:
    """Create the incidents table if it does not already exist."""

    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS incidents (
                incident_id TEXT PRIMARY KEY,
                timestamp TEXT,
                source TEXT,
                verdict TEXT,
                confidence REAL,
                gate TEXT,
                gqs TEXT,
                people_count INTEGER,
                peak_window TEXT,
                weapon_flag INTEGER,
                weapon_class TEXT,
                packet_summary TEXT,
                narrative TEXT,
                frames_ref TEXT
            )
            """
        )


def insert_incident(record: IncidentRecord) -> None:
    """Insert or replace an incident record in the database."""

    try:
        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO incidents (
                    incident_id,
                    timestamp,
                    source,
                    verdict,
                    confidence,
                    gate,
                    gqs,
                    people_count,
                    peak_window,
                    weapon_flag,
                    weapon_class,
                    packet_summary,
                    narrative,
                    frames_ref
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.incident_id,
                    record.timestamp.isoformat(),
                    record.source,
                    record.verdict,
                    record.confidence,
                    json.dumps(record.gate.model_dump()),
                    json.dumps(record.gqs.model_dump()),
                    record.people_count,
                    json.dumps(record.peak_window),
                    int(record.weapon_flag),
                    record.weapon_class,
                    record.packet_summary,
                    record.narrative,
                    json.dumps(record.frames_ref),
                ),
            )
    except sqlite3.IntegrityError as exc:
        raise ValueError(
            f"Incident {record.incident_id} already exists."
        ) from exc


def _row_to_incident(row: sqlite3.Row) -> IncidentRecord:
    """Convert a SQLite row into an IncidentRecord schema instance."""

    return IncidentRecord(
        incident_id=row["incident_id"],
        timestamp=datetime.fromisoformat(
            row["timestamp"]
        ),
        source=row["source"],
        verdict=row["verdict"],
        confidence=row["confidence"],
        gate=GateScores(**json.loads(row["gate"])),
        gqs=GQSScores(**json.loads(row["gqs"])),
        people_count=row["people_count"],
        peak_window=json.loads(row["peak_window"]),
        weapon_flag=bool(row["weapon_flag"]),
        weapon_class=row["weapon_class"],
        packet_summary=row["packet_summary"],
        narrative=row["narrative"],
        frames_ref=json.loads(row["frames_ref"]),
    )


def get_incident(incident_id: str) -> Optional[IncidentRecord]:
    """Retrieve one incident by incident ID."""

    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT *
            FROM incidents
            WHERE incident_id = ?
            """,
            (incident_id,),
        ).fetchone()

    if row is None:
        return None
    return _row_to_incident(row)


def list_incidents(limit: int = 100) -> List[IncidentRecord]:
    """List recent incidents, limited by the requested count."""

    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT *
            FROM incidents
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [_row_to_incident(row) for row in rows]


def delete_incident(incident_id: str) -> None:
    """Delete one incident by incident ID."""

    with _connect() as conn:
        conn.execute(
            """
            DELETE FROM incidents
            WHERE incident_id = ?
            """,
            (incident_id,),
        )


def count_incidents() -> int:
    """Return the total number of stored incidents."""

    with _connect() as conn:
        count = conn.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]

    return int(count)


def main() -> None:
    """Run a minimal smoke test for the incident database module."""

    create_tables()
    incident_id = str(uuid.uuid4())

    record = IncidentRecord(
        incident_id=incident_id,
        timestamp=datetime.fromisoformat("2026-06-08T20:30:00"),
        source="camera-01",
        verdict="violence",
        confidence=0.91,
        gate=GateScores(
            skeleton=0.87,
            interaction=0.83,
            object=0.42,
            vit=0.94,
        ),
        gqs=GQSScores(
            q_skel=0.88,
            q_int=0.81,
            q_obj=0.70,
            q_po=0.76,
            valid_ratio=0.93,
        ),
        people_count=2,
        peak_window=[24, 48],
        weapon_flag=False,
        weapon_class=None,
        packet_summary="Two people were detected in a high-confidence interaction window.",
        narrative="The classifier output indicates a likely violent interaction.",
        frames_ref=["frames/incident-001/frame_024.jpg", "frames/incident-001/frame_048.jpg"],
    )

    insert_incident(record)
    retrieved = get_incident(record.incident_id)

    print(retrieved)
    print(f"Total incidents: {count_incidents()}")


if __name__ == "__main__":
    main()
