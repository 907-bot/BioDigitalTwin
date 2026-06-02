from pydantic import BaseModel, Field
from typing import Optional


class SimulationRequest(BaseModel):
    patient_id: str
    disease: str = Field(default="type_2_diabetes")
    horizon_days: int = Field(default=365, ge=7, le=1825)
    dt_hours: float = Field(default=6.0, gt=0)
    intervention: Optional[dict] = None


class BiomarkerTrajectory(BaseModel):
    name: str
    unit: str
    baseline: float
    healthy_range: list[float]
    trajectory: list[dict]


class SimulationResponse(BaseModel):
    patient_id: str
    disease: str
    horizon_days: int
    steps: int
    disease_state: str
    risk_evolution: list[dict]
    biomarkers: list[BiomarkerTrajectory]
    final_risk: float
    intervention_applied: Optional[dict] = None


class DiseaseAttractorResponse(BaseModel):
    disease: str
    description: str
    stable_states: list[dict]
    transitions: list[dict]
    bifurcation_params: dict
