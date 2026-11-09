"""SQLite + SQLModel persistence.

Schema:
  - Case: a single MRI upload + analysis session.
  - Prediction: a model output for a case (versioned by model_name).
  - FeatureBundle: serialized TumorFeaturesBundle.
  - EvalRun: aggregate metrics for a model on a dataset split.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship

from app.config import settings


class Base(DeclarativeBase):
    pass


class Case(Base):
    __tablename__ = "cases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    case_name: Mapped[str] = mapped_column(String(255), index=True)
    patient_id: Mapped[str | None] = mapped_column(String(255), index=True, nullable=True)
    modality_summary: Mapped[str | None] = mapped_column(String(255), nullable=True)
    nifti_path: Mapped[str] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    predictions: Mapped[list["Prediction"]] = relationship(back_populates="case", cascade="all, delete-orphan")


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    case_id: Mapped[int] = mapped_column(ForeignKey("cases.id"), index=True)
    model_name: Mapped[str] = mapped_column(String(64), index=True)
    model_version: Mapped[str] = mapped_column(String(64))
    inference_seconds: Mapped[float] = mapped_column(Float)
    mask_path: Mapped[str] = mapped_column(String(512))
    uncertainty_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    saliency_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    mesh_paths: Mapped[dict[str, Any]] = mapped_column(JSON)
    features: Mapped[dict[str, Any]] = mapped_column(JSON)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    risk_regions: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)
    risk_report: Mapped[str | None] = mapped_column(String, nullable=True)
    tumor_detection_status: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    case: Mapped[Case] = relationship(back_populates="predictions")

    @property
    def analysis_details(self) -> dict[str, Any] | None:
        if isinstance(self.features, dict) and "analysis_details" in self.features:
            return self.features["analysis_details"]
        return None


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(64), index=True)
    model_version: Mapped[str] = mapped_column(String(64))
    dataset_split: Mapped[str] = mapped_column(String(32))
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON)
    n_cases: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


_engine = create_engine(f"sqlite:///{settings.db_path}", echo=False, future=True)
settings.db_path.parent.mkdir(parents=True, exist_ok=True)
Base.metadata.create_all(_engine)

import sqlalchemy

with _engine.connect() as _conn:
    _inspector = sqlalchemy.inspect(_conn)
    if "predictions" in _inspector.get_table_names():
        _cols = [c["name"] for c in _inspector.get_columns("predictions")]
        if "tumor_detection_status" not in _cols:
            _conn.execute(sqlalchemy.text("ALTER TABLE predictions ADD COLUMN tumor_detection_status VARCHAR(64)"))
            _conn.commit()


def get_session() -> Session:
    return Session(_engine)


def list_cases() -> list[Case]:
    with get_session() as session:
        return list(session.query(Case).order_by(Case.created_at.desc()).all())


def get_case(case_id: int) -> Case | None:
    with get_session() as session:
        return session.get(Case, case_id)


def get_prediction(prediction_id: int) -> Prediction | None:
    with get_session() as session:
        return session.get(Prediction, prediction_id)


def get_latest_prediction(case_id: int) -> Prediction | None:
    with get_session() as session:
        return (
            session.query(Prediction)
            .filter(Prediction.case_id == case_id)
            .order_by(Prediction.created_at.desc())
            .first()
        )


def list_eval_runs() -> list[EvalRun]:
    with get_session() as session:
        return list(session.query(EvalRun).order_by(EvalRun.created_at.desc()).all())


def to_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, Base):
        return {
            c.key: (getattr(obj, c.key).isoformat() if isinstance(getattr(obj, c.key), datetime) else getattr(obj, c.key))
            for c in obj.__table__.columns
        }
    return json.loads(json.dumps(obj, default=str))
