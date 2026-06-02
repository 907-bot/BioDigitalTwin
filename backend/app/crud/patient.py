import json
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.models.patient import Patient


def upsert_patient(db: Session, p: dict) -> Patient:
    obj = db.get(Patient, p["patient_id"])
    if obj is None:
        obj = Patient(patient_id=p["patient_id"])
        db.add(obj)
    for k, v in p.items():
        if k == "embedding" and v is not None:
            obj.embedding = json.dumps(v)
        elif hasattr(obj, k):
            setattr(obj, k, v)
    db.flush()
    return obj


def commit(db: Session):
    db.commit()


def get_patient(db: Session, patient_id: str) -> Optional[Patient]:
    return db.get(Patient, patient_id)


def list_patients(db: Session, limit: int = 100, offset: int = 0) -> list[Patient]:
    stmt = select(Patient).offset(offset).limit(limit)
    return list(db.scalars(stmt))


def count_patients(db: Session) -> int:
    return db.query(Patient).count()


def update_risk(db: Session, patient_id: str, score: float, label: str, embedding: list[float]):
    obj = db.get(Patient, patient_id)
    if obj is None:
        return None
    obj.risk_score = float(score)
    obj.risk_label = label
    obj.embedding = json.dumps(embedding)
    db.flush()
    return obj
