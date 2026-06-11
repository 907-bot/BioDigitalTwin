"""Shared Pydantic schemas used across microservices."""
from pydantic import BaseModel, Field
from typing import Optional


class Patient(BaseModel):
    patient_id: str
    age: int
    gender: str
    bmi: float
    hr: float
    hrv: float
    spo2: float
    glucose: float
    systolic_bp: float
    diastolic_bp: float
    risk_score: Optional[float] = None
    risk_label: Optional[str] = None


class Simulation(BaseModel):
    disease: str
    horizon_days: int
    steps: int
    disease_state: str
    final_risk: float
    initial_risk: Optional[float] = None
    risk_evolution: list
    biomarkers: list
    spike_view: dict
    intervention_applied: Optional[dict] = None


class Counterfactual(BaseModel):
    disease: str
    horizon_days: int
    intervention_applied: Optional[dict] = None
    control: dict
    treated: dict
    counterfactual_effect: dict


class CausalNode(BaseModel):
    id: str
    kind: str
    name: str


class CausalEdge(BaseModel):
    src: str
    dst: str
    rel: str
    weight: float


class CausalGraph(BaseModel):
    n_nodes: int
    n_edges: int
    nodes: list
    edges: list


class ChatReply(BaseModel):
    session_id: str
    user_message: str
    reply: str
    backend: str
    tool_calls: list
    elapsed_s: float
    turn: int


class PatientCounterfactual(BaseModel):
    patient_id: str
    treatment: str
    outcome: str
    factual: float
    counterfactual: float
    effect: float
    effect_direction: str


class SimulateRequest(BaseModel):
    initial_state: dict
    disease: str = Field(..., description="One of: t2d, hypertension, cvd, copd")
    horizon_days: int = Field(180, ge=1, le=1825)
    dt_hours: float = Field(6.0, gt=0, le=48)
    intervention: Optional[dict] = Field(default=None)
    intervention_name: Optional[str] = Field(default=None)
    rng_seed: int = 0
    sample_every_hours: Optional[float] = Field(default=None)


class CounterfactualRequest(BaseModel):
    initial_state: dict
    disease: str
    horizon_days: int = 365
    intervention: Optional[dict] = None
    intervention_name: Optional[str] = None
    dt_hours: float = 6.0
    rng_seed: int = 0


class ATERequest(BaseModel):
    treatment: str
    outcome: str
    common_causes: Optional[list] = None


class CATERequest(BaseModel):
    treatment: str
    outcome: str
    effect_modifiers: list = ["bmi"]
    common_causes: Optional[list] = None


class RefuteRequest(BaseModel):
    treatment: str
    outcome: str
    common_causes: Optional[list] = None
    method: str = Field(default="random_common_cause", pattern="^(random_common_cause|placebo)$")
