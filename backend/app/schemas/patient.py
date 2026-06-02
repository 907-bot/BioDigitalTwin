from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class PatientCreate(BaseModel):
    n: int = Field(default=500, ge=10, le=5000)
    seed: Optional[int] = 42


class PatientOut(BaseModel):
    patient_id: str
    age: int
    gender: str
    bmi: float
    hr: int
    hrv: int
    spo2: float
    glucose: int
    systolic_bp: int
    diastolic_bp: int
    risk_score: Optional[float] = None
    risk_label: Optional[str] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class PatientList(BaseModel):
    status: str
    count: int
    patients: list[PatientOut]
