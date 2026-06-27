"""
Guardian Eye — Database layer
SQLAlchemy model + session factory for the incident store.

Columns (full set):
  clip_id, incident_id, timestamp, source,
  verdict, confidence,
  gate_json  (skeleton/interaction/object/vit),
  gqs_json   (q_skel/q_int/q_obj/q_po/valid_ratio),
  people_count, peak_window_json, weapon_flag, weapon_class,
  thumbnail_path, overlay_path,
  packet_summary, narrative
"""

from __future__ import annotations
import datetime, json, uuid
from pathlib import Path
from sqlalchemy import (
    create_engine, Column, String, Float, Integer,
    Boolean, DateTime, Text, inspect, text,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from typing import Generator

# ── Engine ────────────────────────────────────────────────────────────────────

DB_DIR = Path("db")
DB_DIR.mkdir(parents=True, exist_ok=True)          # always safe on Windows too
DATABASE_URL = f"sqlite:///{DB_DIR / 'guardian_eye.db'}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ── ORM model ─────────────────────────────────────────────────────────────────

class IncidentRecord(Base):
    """
    One row per analysed clip.
    Written by incident_service.save_incident() on every /explain call.
    Schema mirrors EXPLANATION_RAG_SYSTEM.md §7.1 plus clip_id / overlay_path.
    """
    __tablename__ = "incidents"

    # ── Identity ──────────────────────────────────────────────────────────────
    incident_id   = Column(String,  primary_key=True,
                           default=lambda: str(uuid.uuid4()))
    clip_id       = Column(String,  nullable=False, index=True)   # stable handle from /predict
    timestamp     = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    source        = Column(String,  nullable=False)               # original filename / camera

    # ── Classifier outputs (authority) ────────────────────────────────────────
    verdict       = Column(String,  nullable=False, index=True)   # "violence" | "non-violence"
    confidence    = Column(Float,   nullable=False)
    threshold     = Column(Float,   nullable=True)

    # ── Fusion gate  [skeleton, interaction, object, vit]  (JSON array) ───────
    gate_json     = Column(Text, nullable=True)

    # ── GQS  [q_skel, q_int, q_obj, q_po, valid_ratio]  (JSON array) ─────────
    gqs_json      = Column(Text, nullable=True)

    # ── Geometry-derived telemetry ────────────────────────────────────────────
    people_count      = Column(Integer, default=0)
    peak_window_json  = Column(Text, nullable=True)   # [start_frame, end_frame]
    weapon_flag       = Column(Boolean, default=False, index=True)
    weapon_class      = Column(String,  nullable=True)

    # ── File paths ────────────────────────────────────────────────────────────
    thumbnail_path = Column(String, nullable=True)   # static/thumbnails/<clip_id>.jpg
    overlay_path   = Column(String, nullable=True)   # static/overlays/<clip_id>.mp4

    # ── Explanation ───────────────────────────────────────────────────────────
    packet_summary = Column(Text, nullable=True)     # deterministic evidence text
    narrative      = Column(Text, nullable=True)     # VLM output
    vlm_summary_json = Column(Text, nullable=True)   # structured VLM understanding
    model_route_json = Column(Text, nullable=True)   # dataset-router diagnostics

    # ── Helpers ───────────────────────────────────────────────────────────────

    def gate_dict(self) -> dict[str, float]:
        vals = json.loads(self.gate_json or "[0,0,0,0]")
        return dict(zip(["skeleton", "interaction", "object", "vit"], vals))

    def gqs_dict(self) -> dict[str, float]:
        vals = json.loads(self.gqs_json or "[0,0,0,0,0]")
        return dict(zip(["q_skel", "q_int", "q_obj", "q_po", "valid_ratio"], vals))

    def peak_window(self) -> list[int]:
        return json.loads(self.peak_window_json or "[0,0]")

    def vlm_summary(self) -> dict:
        try:
            value = json.loads(self.vlm_summary_json or "{}")
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}

    def model_route(self) -> dict:
        try:
            value = json.loads(self.model_route_json or "{}")
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}

    def __repr__(self) -> str:
        return (f"<Incident {self.incident_id[:8]}… "
                f"clip={self.clip_id} verdict={self.verdict} "
                f"conf={self.confidence:.2f}>")


# ── Public helpers ────────────────────────────────────────────────────────────

def create_tables() -> None:
    Base.metadata.create_all(bind=engine)
    _ensure_optional_columns()


def _ensure_optional_columns() -> None:
    inspector = inspect(engine)
    if "incidents" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("incidents")}
    if "vlm_summary_json" not in columns:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE incidents ADD COLUMN vlm_summary_json TEXT"))
    if "model_route_json" not in columns:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE incidents ADD COLUMN model_route_json TEXT"))


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
