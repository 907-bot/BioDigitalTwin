from pydantic import BaseModel, Field
from typing import Optional


class GraphBuildRequest(BaseModel):
    n_patients: int = Field(default=500, ge=10, le=5000)
    seed: int = 42


class GraphBuildResponse(BaseModel):
    status: str
    patients: int
    biomarkers: int
    organs: int
    diseases: int
    edges: int
    neo4j_loaded: bool


class PatientGraphState(BaseModel):
    patient_id: str
    risk_score: float
    risk_label: str
    embedding: list[float]
    abnormal_biomarkers: list[dict]
    affected_organs: list[dict]
    active_diseases: list[dict]
    neighbor_similar: list[dict] = []


class BiomarkerPrediction(BaseModel):
    patient_id: str
    target_biomarker: str
    predicted: float
    observed: Optional[float] = None
    confidence: float
