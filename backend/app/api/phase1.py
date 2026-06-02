"""
Phase 1 router — synthetic patient generation.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.config import settings
from app.crud import patient as patient_crud
from app.models.patient import Patient
from app.schemas import PatientOut
from app.synthetic.generator import generate_patients

router = APIRouter(prefix="/api/phase1", tags=["phase1"])


@router.get("/health")
def health():
    return {"status": "ok", "phase": "1", "service": "synthetic-patient-generator"}


@router.post("/generate")
def generate(
    n: int = Query(500, ge=10, le=5000),
    seed: int = Query(42),
    persist: bool = Query(True, description="Persist to PostgreSQL"),
    db: Session = Depends(get_db),
):
    """Generate `n` synthetic patients, optionally persisting to PostgreSQL."""
    df = generate_patients(n=n, seed=seed)
    csv_path = os.path.join(settings.DATA_DIR, "synthetic_patients.csv")
    df.to_csv(csv_path, index=False)

    persisted = 0
    if persist:
        for _, row in df.iterrows():
            patient_crud.upsert_patient(db, row.to_dict())
        patient_crud.commit(db)
        persisted = df.shape[0]

    return {
        "status": "ok",
        "generated": int(df.shape[0]),
        "persisted": int(persisted),
        "csv_path": csv_path,
        "columns": df.columns.tolist(),
        "sample": df.head(3).to_dict(orient="records"),
    }


@router.get("/patients", response_model=list[PatientOut])
def list_patients(limit: int = Query(50, le=500),
                  offset: int = 0,
                  db: Session = Depends(get_db)):
    return patient_crud.list_patients(db, limit=limit, offset=offset)


@router.get("/patients/{patient_id}", response_model=PatientOut)
def get_patient(patient_id: str, db: Session = Depends(get_db)):
    obj = patient_crud.get_patient(db, patient_id)
    if obj is None:
        raise HTTPException(status_code=404, detail="patient not found")
    return obj


@router.get("/stats")
def stats(db: Session = Depends(get_db)):
    n = patient_crud.count_patients(db)
    return {"count": n}
