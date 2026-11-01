"""
Phase 18 — Samsung Health REST API Router
Endpoints: /phase18/*
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

samsung_router = APIRouter(prefix="/phase18", tags=["Phase18-Samsung-Twin"])

# ---------------------------------------------------------------------------
# Config — paths resolved relative to working directory (backend/)
# ---------------------------------------------------------------------------
SAMSUNG_DATA_DIR = os.environ.get(
    "SAMSUNG_DATA_DIR",
    os.path.join(os.path.dirname(__file__), "../../../../samsung_healthdata")
)
MODEL_DIR = os.environ.get("MODEL_DIR", "models")

# ---------------------------------------------------------------------------
# Cached parsed record (in-memory, reset on re-ingest)
# ---------------------------------------------------------------------------
_cached_record = None
_ingest_summary: Dict[str, Any] = {}

def _get_record():
    return _cached_record

def _resolve_samsung_dir() -> str:
    """Resolve Samsung data directory, trying multiple fallback paths."""
    candidates = [
        SAMSUNG_DATA_DIR,
        os.path.join(os.getcwd(), "..", "samsung_healthdata"),
        "/app/samsung_healthdata",
        "/Users/abhishekadari/projects/bio-digital-twin/samsung_healthdata",
    ]
    for c in candidates:
        if c and os.path.isdir(c):
            return c
    return SAMSUNG_DATA_DIR  # return configured path even if missing (for clear error message)


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------
class TrainRequest(BaseModel):
    n_synthetic: int = Field(600, ge=100, le=5000,
                             description="Number of synthetic augmented samples")
    age_est: int = Field(30, ge=18, le=80, description="Estimated patient age")
    model_dir: Optional[str] = None

class PredictRequest(BaseModel):
    age_est: int = Field(30, ge=18, le=80)
    top_n_explain: int = Field(10, ge=3, le=20)

class FuseRequest(BaseModel):
    patient_id: str = Field("P_REAL_001", description="ID to assign to real patient")
    age: int = Field(25, ge=18, le=80)
    gender: str = Field("Male")


# ---------------------------------------------------------------------------
# Health check for this module
# ---------------------------------------------------------------------------
@samsung_router.get("/health")
def phase18_health():
    """Health check for Phase 18 Samsung integration."""
    from .training import get_predictor
    pred = get_predictor(MODEL_DIR)
    return {
        "phase": 18,
        "module": "Samsung Health Digital Twin",
        "models_loaded": pred.models_available(),
        "data_dir": _resolve_samsung_dir(),
        "data_dir_exists": os.path.isdir(_resolve_samsung_dir()),
        "cached_record": _cached_record is not None,
    }


# ---------------------------------------------------------------------------
# POST /phase18/ingest — parse Samsung CSV files
# ---------------------------------------------------------------------------
@samsung_router.post("/ingest")
def ingest_samsung_data(data_dir: Optional[str] = Query(None)):
    """
    Parse all Samsung Health CSV files and cache the patient record.
    Returns a summary of what was parsed.
    """
    global _cached_record, _ingest_summary
    resolved_dir = data_dir or _resolve_samsung_dir()
    if not os.path.isdir(resolved_dir):
        raise HTTPException(
            status_code=404,
            detail=f"Samsung data directory not found: {resolved_dir}"
        )
    try:
        from .csv_parser import SamsungCSVParser
        parser = SamsungCSVParser(resolved_dir)
        record = parser.parse_all()
        _cached_record = record
        agg = record.aggregates

        _ingest_summary = {
            "status": "success",
            "patient_id": record.patient_id,
            "data_dir": resolved_dir,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "data_counts": {
                "heart_rate_readings": len(record.heart_rate),
                "stress_readings": len(record.stress),
                "sleep_stage_records": len(record.sleep_stages),
                "pedometer_days": len(record.pedometer),
                "weight_readings": len(record.weight),
                "exercise_sessions": len(record.exercise),
                "spo2_readings": len(record.spo2),
                "hrv_readings": len(record.hrv),
                "floor_days": len(record.floors),
            },
            "date_range": {
                "heart_rate": {
                    "start": agg.get("hr_start", ""),
                    "end": agg.get("hr_end", ""),
                }
            },
            "aggregates": {
                k: round(float(v), 4) if isinstance(v, (int, float)) else v
                for k, v in agg.items()
                if not isinstance(v, (dict, list))
            },
        }
        return _ingest_summary

    except Exception as e:
        logger.error(f"Ingest failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingest failed: {str(e)}")


# ---------------------------------------------------------------------------
# GET /phase18/patient-record — return cached patient record summary
# ---------------------------------------------------------------------------
@samsung_router.get("/patient-record")
def get_patient_record():
    """Get the cached real-world patient digital twin summary."""
    if _cached_record is None:
        raise HTTPException(
            status_code=404,
            detail="No patient record cached. Call POST /phase18/ingest first."
        )
    agg = _cached_record.aggregates
    return {
        "patient_id": _cached_record.patient_id,
        "cached_at": _ingest_summary.get("ingested_at", ""),
        "data_counts": _ingest_summary.get("data_counts", {}),
        "aggregates": {
            k: round(float(v), 4) if isinstance(v, (int, float)) else v
            for k, v in agg.items()
            if not isinstance(v, (dict, list))
        },
        "vitals_snapshot": {
            "resting_hr_bpm": round(agg.get("hr_resting", 0), 1),
            "hr_mean_bpm": round(agg.get("hr_mean", 0), 1),
            "stress_mean": round(agg.get("stress_mean", 0), 1),
            "sleep_hours_per_night": round(
                agg.get("sleep_total_hours", 0) /
                max(len(set(s.start.date() for s in _cached_record.sleep_stages)), 1), 2
            ) if _cached_record.sleep_stages else 0,
            "steps_daily_mean": round(agg.get("steps_daily_mean", 0), 0),
            "weight_kg": round(agg.get("weight_kg_latest", 0), 1),
            "bmi": round(agg.get("bmi_est", 0), 2),
            "body_fat_pct": round(agg.get("body_fat_pct_latest", 0), 1),
            "spo2_mean": round(agg.get("spo2_mean", 0), 1),
            "vo2_max": round(agg.get("vo2_max", 0), 1),
        },
    }


# ---------------------------------------------------------------------------
# GET /phase18/features — extract feature vector
# ---------------------------------------------------------------------------
@samsung_router.get("/features")
def get_features(age_est: int = Query(30, ge=18, le=80)):
    """Extract the full feature vector from the cached Samsung record."""
    if _cached_record is None:
        raise HTTPException(
            status_code=404,
            detail="No patient record cached. Call POST /phase18/ingest first."
        )
    from .feature_engineering import engineer_features, FEATURE_NAMES
    vec = engineer_features(_cached_record, age_est=age_est)
    return {
        "patient_id": _cached_record.patient_id,
        "n_features": len(vec),
        "feature_names": FEATURE_NAMES,
        "feature_values": {
            name: round(float(val), 4)
            for name, val in zip(FEATURE_NAMES, vec)
        },
    }


# ---------------------------------------------------------------------------
# POST /phase18/train — run training pipeline
# ---------------------------------------------------------------------------
_training_result: Dict[str, Any] = {}

@samsung_router.post("/train")
def train_models(
    req: TrainRequest,
    background_tasks: BackgroundTasks,
    async_train: bool = Query(False, description="Run training in background"),
):
    """
    Run the Samsung Health ML training pipeline.
    Trains CardioRisk, SleepQuality, StressClassifier, AnomalyDetector models.
    """
    from .training import get_trainer, get_predictor

    data_dir = _resolve_samsung_dir()
    if not os.path.isdir(data_dir):
        raise HTTPException(
            status_code=404,
            detail=f"Samsung data directory not found: {data_dir}"
        )
    model_dir = req.model_dir or MODEL_DIR
    trainer = get_trainer()

    def _run_training():
        global _training_result
        try:
            result = trainer.train_all(
                data_dir=data_dir,
                model_dir=model_dir,
                n_synthetic=req.n_synthetic,
                age_est=req.age_est,
            )
            _training_result = result
            # Reload predictor with new models
            global _predictor_instance
            _predictor_instance = None
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            trainer._status["state"] = "error"
            trainer._status["error"] = str(e)

    if async_train:
        background_tasks.add_task(_run_training)
        return {
            "status": "training_started",
            "message": "Training running in background. Poll GET /phase18/training-status for progress.",
            "config": {"n_synthetic": req.n_synthetic, "age_est": req.age_est},
        }
    else:
        _run_training()
        return _training_result


# ---------------------------------------------------------------------------
# GET /phase18/training-status
# ---------------------------------------------------------------------------
@samsung_router.get("/training-status")
def training_status():
    """Get current training pipeline status and progress."""
    from .training import get_trainer
    trainer = get_trainer()
    status = trainer.status()
    # Attach last result if complete
    if status.get("state") == "complete" and _training_result:
        status["last_result"] = {
            "n_samples": _training_result.get("n_samples"),
            "elapsed_seconds": _training_result.get("elapsed_seconds"),
            "models_trained": _training_result.get("models_trained"),
            "metrics": _training_result.get("metrics"),
        }
    return status


# ---------------------------------------------------------------------------
# POST /phase18/predict — run predictions
# ---------------------------------------------------------------------------
@samsung_router.post("/predict")
def predict(req: PredictRequest):
    """
    Run all 4 model predictions on the cached Samsung patient record.
    Returns cardiovascular risk, sleep quality, stress pattern, and anomaly flags.
    """
    if _cached_record is None:
        raise HTTPException(
            status_code=404,
            detail="No patient record cached. Call POST /phase18/ingest first."
        )
    from .training import get_predictor
    predictor = get_predictor(MODEL_DIR)
    if not predictor.models_available():
        raise HTTPException(
            status_code=503,
            detail="Models not trained yet. Call POST /phase18/train first."
        )
    try:
        predictions = predictor.predict(_cached_record, age_est=req.age_est)
        explanation = predictor.explain(
            _cached_record, top_n=req.top_n_explain, age_est=req.age_est
        )
        predictions["explanation"] = explanation
        return predictions
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ---------------------------------------------------------------------------
# GET /phase18/health-timeline — chronological health events
# ---------------------------------------------------------------------------
@samsung_router.get("/health-timeline")
def health_timeline(
    limit: int = Query(200, ge=10, le=1000),
    event_types: Optional[str] = Query(None, description="Comma-separated: hr,stress,sleep,exercise,weight"),
):
    """
    Return chronological health timeline from all Samsung data sources.
    """
    if _cached_record is None:
        raise HTTPException(
            status_code=404,
            detail="No patient record cached. Call POST /phase18/ingest first."
        )
    types_filter = set(event_types.split(",")) if event_types else None
    events = []

    def should_include(t: str) -> bool:
        return types_filter is None or t in types_filter

    if should_include("hr"):
        for r in _cached_record.heart_rate[::5]:  # sample every 5th
            events.append({
                "type": "heart_rate", "timestamp": r.timestamp.isoformat(),
                "value": r.hr_bpm, "unit": "bpm", "device": r.device_id
            })

    if should_include("stress"):
        for r in _cached_record.stress[::3]:
            events.append({
                "type": "stress", "timestamp": r.timestamp.isoformat(),
                "value": r.score, "unit": "score (0-100)",
                "label": "high" if r.score > 60 else "medium" if r.score > 35 else "low"
            })

    if should_include("sleep"):
        stage_names = {40001: "awake", 40002: "light", 40003: "deep", 40004: "REM"}
        for r in _cached_record.sleep_stages:
            events.append({
                "type": "sleep_stage", "timestamp": r.start.isoformat(),
                "end": r.end.isoformat(),
                "value": r.stage_code, "label": stage_names.get(r.stage_code, "unknown"),
                "duration_min": round((r.end - r.start).total_seconds() / 60, 1)
            })

    if should_include("exercise"):
        ex_types = {1002: "Running", 1001: "Walking", 13001: "Cycling",
                    1003: "Hiking", 11007: "HIIT", 10003: "Strength"}
        for e in _cached_record.exercise:
            events.append({
                "type": "exercise", "timestamp": e.start.isoformat(),
                "end": e.end.isoformat(),
                "exercise_type": ex_types.get(e.exercise_type, f"type_{e.exercise_type}"),
                "duration_min": round(e.duration_ms / 60_000, 1),
                "mean_hr": round(e.mean_heart_rate, 1),
                "max_hr": round(e.max_heart_rate, 1),
                "calories": round(e.calorie, 1),
                "vo2_max": round(e.vo2_max, 1),
            })

    if should_include("weight"):
        for w in _cached_record.weight:
            events.append({
                "type": "weight", "timestamp": w.timestamp.isoformat(),
                "weight_kg": w.weight_kg, "bmi": round(w.bmi_est, 2),
                "body_fat_pct": w.body_fat_pct,
                "skeletal_muscle_kg": w.skeletal_muscle_mass_kg,
            })

    # Sort by timestamp and limit
    events.sort(key=lambda e: e["timestamp"], reverse=True)
    return {
        "patient_id": _cached_record.patient_id,
        "total_events": len(events),
        "returned": min(len(events), limit),
        "events": events[:limit],
    }


# ---------------------------------------------------------------------------
# POST /phase18/fuse-with-cohort — inject real patient into synthetic cohort
# ---------------------------------------------------------------------------
@samsung_router.post("/fuse-with-cohort")
def fuse_with_cohort(req: FuseRequest):
    """
    Inject the real Samsung patient as a row in data/synthetic_patients.csv,
    enabling all downstream phases (GNN, dynamics, PK/PD, etc.) to operate on real data.
    """
    if _cached_record is None:
        raise HTTPException(
            status_code=404,
            detail="No patient record cached. Call POST /phase18/ingest first."
        )
    from .csv_parser import samsung_record_to_patient_row

    cohort_path = "data/synthetic_patients.csv"
    if not os.path.exists(cohort_path):
        raise HTTPException(
            status_code=404,
            detail="Synthetic cohort not found. Call POST /generate-patients first."
        )

    df = pd.read_csv(cohort_path)

    # Remove existing real patient row if present
    df = df[df["patient_id"] != req.patient_id]

    # Build row from real data
    row = samsung_record_to_patient_row(_cached_record)
    row["patient_id"] = req.patient_id
    row["age"] = req.age
    row["gender"] = req.gender

    new_row_df = pd.DataFrame([row])
    # Align columns (add missing cols with NaN)
    for col in df.columns:
        if col not in new_row_df.columns:
            new_row_df[col] = np.nan
    new_row_df = new_row_df[df.columns]

    df_fused = pd.concat([df, new_row_df], ignore_index=True)
    df_fused.to_csv(cohort_path, index=False)

    return {
        "status": "success",
        "patient_id": req.patient_id,
        "message": f"Real patient {req.patient_id} fused into cohort ({len(df_fused)} total patients)",
        "real_patient_row": {k: row.get(k) for k in [
            "patient_id", "bmi", "hr", "hrv", "spo2", "steps_daily",
            "stress_score", "sleep_hours", "body_fat_pct", "vo2_max"
        ]},
        "next_steps": [
            "POST /phase2/build-graph to rebuild patient similarity graph",
            "POST /phase2/train-gnn to retrain GNN with real patient",
            "GET /phase2/similar-patients/P_REAL_001 to find synthetic twins",
        ]
    }


# ---------------------------------------------------------------------------
# GET /phase18/training-metrics — load saved training metrics
# ---------------------------------------------------------------------------
@samsung_router.get("/training-metrics")
def get_training_metrics():
    """Load and return the saved training metrics for all models."""
    from .training import get_predictor
    predictor = get_predictor(MODEL_DIR)
    metrics = predictor.load_training_metrics()
    if not metrics:
        raise HTTPException(
            status_code=404,
            detail="No training metrics found. Call POST /phase18/train first."
        )
    return {"models": metrics}
