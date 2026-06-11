"""
Phase 3 Router: Whole-body cellular digital twin API.
30-state, 25-parameter, 15-observation UKF twin with counterfactual
simulation, virtual cohorts, uncertainty quantification, and explainability.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import logging

from app.personalization.core import (
    PersonalizationEngine,
    create_personalization_engine,
    PHYSIO_DIM,
    PARAM_DIM,
    OBS_DIM,
)
from app.personalization.priors import STATE_NAMES, PARAMETER_NAMES
from app.personalization.biomarkers import compute_all_biomarkers

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/personalization/v2", tags=["Phase 3 — Digital Twin"])

_engines: Dict[str, PersonalizationEngine] = {}


# ── Schemas ──────────────────────────────────────────────────

class DemographicInfo(BaseModel):
    age: float = Field(35.0, ge=18, le=120)
    sex: str = Field("male", pattern="^(male|female)$")
    bmi: float = Field(24.0, ge=12, le=60)
    has_diabetes: bool = False
    has_hypertension: bool = False
    has_ckd: bool = False


class InitRequest(BaseModel):
    patient_id: str = Field(..., example="P000001")
    demographics: Optional[DemographicInfo] = None
    initial_obs: Optional[List[float]] = None  # 15-dim


class UpdateRequest(BaseModel):
    patient_id: str = Field(..., example="P000001")
    G: float = Field(95.0, ge=20, le=600)
    SBP: float = Field(120.0, ge=50, le=250)
    DBP: float = Field(80.0, ge=30, le=150)
    HR: float = Field(70.0, ge=30, le=220)
    HRV: float = Field(45.0, ge=0, le=200)
    GFR: float = Field(100.0, ge=0, le=200)
    Na: float = Field(140.0, ge=100, le=180)
    K: float = Field(4.2, ge=1.5, le=9.0)
    Osm: float = Field(290.0, ge=230, le=350)
    FFA: float = Field(0.5, ge=0.1, le=2.0)
    LDL: float = Field(100.0, ge=20, le=300)
    HDL: float = Field(50.0, ge=10, le=120)
    TG: float = Field(120.0, ge=20, le=800)
    cortisol: float = Field(350.0, ge=10, le=1000)
    sleep_pressure: float = Field(0.3, ge=0, le=1.0)
    light_level: Optional[float] = Field(0.5, ge=0, le=1)
    sleep: Optional[float] = Field(0.0, ge=0, le=1)
    meal_glucose: Optional[float] = Field(0.0, ge=0)
    exercise: Optional[float] = Field(0.0, ge=0, le=1)
    insulin_dose: Optional[float] = Field(0.0, ge=0)
    dietary_fat: Optional[float] = Field(0.0, ge=0)
    calorie_intake: Optional[float] = Field(0.0, ge=0)
    sodium_intake: Optional[float] = Field(0.0, ge=0)
    alcohol: Optional[float] = Field(0.0, ge=0, le=1)


class TwinStateResponse(BaseModel):
    patient_id: str
    twin_state: Dict[str, float]
    parameters: Dict[str, float]
    subsystems: Dict[str, Any]
    digital_biomarkers: Dict[str, float]
    uncertainty: Dict[str, Any]
    drift_status: Dict[str, Any]


# ── Response builder ──────────────────────────────────────────

def _build_response(patient_id: str, eng: PersonalizationEngine) -> TwinStateResponse:
    state = eng.get_twin_state()
    param_mean, param_cov = eng.get_parameters()

    state_dict = dict(zip(STATE_NAMES, (float(x) for x in state)))
    param_dict = dict(zip(PARAMETER_NAMES, (float(x) for x in param_mean)))

    biomarkers = compute_all_biomarkers(state, param_mean, eng._observation_buffer)

    # Subsystem breakdown
    subsystems = {
        "metabolic": {
            "G": float(state[0]), "I": float(state[1]), "HGP": float(state[2]),
            "PGU": float(state[3]), "IR": float(state[4]),
        },
        "cardiovascular": {
            "SBP": float(state[5]), "DBP": float(state[6]),
            "HR": float(state[7]), "HRV": float(state[8]),
        },
        "renal": {
            "GFR": float(state[9]), "Na": float(state[10]),
            "K": float(state[11]), "Osm": float(state[12]),
        },
        "inflammation": {"CRP": float(state[13])},
        "circadian": {
            "CLOCK_BMAL1": float(state[14]), "PER_CRY": float(state[15]),
            "cortisol": float(state[16]), "melatonin": float(state[17]),
            "circadian_phase": float(state[18]), "sleep_pressure": float(state[19]),
        },
        "adipose": {
            "fat_mass": float(state[20]), "FFA": float(state[21]),
            "LDL": float(state[22]), "HDL": float(state[23]),
            "TG": float(state[24]),
        },
        "immune": {
            "IL6": float(state[25]), "TNFa": float(state[26]),
            "M1_M2_ratio": float(state[27]), "NFkB": float(state[28]),
            "InflammatoryLoad": float(state[29]),
        },
    }

    return TwinStateResponse(
        patient_id=patient_id,
        twin_state=state_dict,
        parameters=param_dict,
        subsystems=subsystems,
        digital_biomarkers=biomarkers,
        uncertainty={
            "parameter_covariance": param_cov.tolist() if param_cov is not None else [],
            "twin_state_covariance": eng.get_twin_state_covariance().tolist(),
        },
        drift_status=eng.get_drift_status(),
    )


# ── Core Endpoints ────────────────────────────────────────────

@router.post("/initialize", response_model=TwinStateResponse)
async def initialize(req: InitRequest):
    if req.patient_id in _engines:
        raise HTTPException(status_code=400, detail=f"Engine exists for {req.patient_id}")

    demo = req.demographics or DemographicInfo()
    eng = create_personalization_engine(
        age=demo.age, sex=demo.sex, bmi=demo.bmi,
        has_diabetes=demo.has_diabetes,
        has_hypertension=demo.has_hypertension,
        has_ckd=demo.has_ckd,
    )

    if req.initial_obs and len(req.initial_obs) >= 1:
        obs = np.array(req.initial_obs[:OBS_DIM])
        if len(obs) < OBS_DIM:
            obs = np.pad(obs, (0, OBS_DIM - len(obs)), constant_values=0)
    else:
        obs = np.zeros(OBS_DIM)
        obs[0] = 95.0
        obs[13] = 300.0

    eng.initialize(obs)
    _engines[req.patient_id] = eng
    return _build_response(req.patient_id, eng)


@router.post("/update", response_model=TwinStateResponse)
async def update(req: UpdateRequest):
    eng = _engines.get(req.patient_id)
    if eng is None:
        raise HTTPException(status_code=404, detail=f"No engine for {req.patient_id}")

    obs = np.array([req.G, req.SBP, req.DBP, req.HR, req.HRV, req.GFR,
                    req.Na, req.K, req.Osm, req.FFA, req.LDL, req.HDL,
                    req.TG, req.cortisol, req.sleep_pressure])
    ctrl = {
        "light_level": req.light_level,
        "sleep": req.sleep,
        "meal_glucose": req.meal_glucose,
        "exercise": req.exercise,
        "insulin_dose": req.insulin_dose,
        "dietary_fat": req.dietary_fat,
        "calorie_intake": req.calorie_intake,
        "sodium_intake": req.sodium_intake,
        "alcohol": req.alcohol,
    }
    eng.update(obs, ctrl)
    return _build_response(req.patient_id, eng)


@router.get("/state/{patient_id}", response_model=TwinStateResponse)
async def get_state(patient_id: str):
    eng = _engines.get(patient_id)
    if eng is None:
        raise HTTPException(status_code=404, detail=f"No engine for {patient_id}")
    return _build_response(patient_id, eng)


@router.get("/{patient_id}/subsystem/{name}")
async def get_subsystem(patient_id: str, name: str):
    eng = _engines.get(patient_id)
    if eng is None:
        raise HTTPException(status_code=404, detail=f"No engine for {patient_id}")
    subsystems = {
        "metabolic": (eng.get_metabolic_state(), ["G","I","HGP","PGU","IR"]),
        "cardiovascular": (eng.get_cardio_state(), ["SBP","DBP","HR","HRV"]),
        "renal": (eng.get_renal_state(), ["GFR","Na","K","Osm"]),
        "circadian": (eng.get_circadian_state(), ["CLOCK_BMAL1","PER_CRY","cortisol",
                                                   "melatonin","phase","sleep_pressure"]),
        "adipose": (eng.get_adipose_state(), ["fat_mass","FFA","LDL","HDL","TG"]),
        "immune": (eng.get_immune_state(), ["IL6","TNFa","M1M2","NFkB","inflam_load"]),
    }
    if name not in subsystems:
        raise HTTPException(status_code=400, detail=f"Unknown: {name}")
    s, keys = subsystems[name]
    return {"subsystem": name, "state": dict(zip(keys, (float(x) for x in s)))}


# ── Biomarkers ────────────────────────────────────────────────

@router.get("/{patient_id}/biomarkers")
async def get_biomarkers(patient_id: str):
    eng = _engines.get(patient_id)
    if eng is None:
        raise HTTPException(status_code=404, detail=f"No engine for {patient_id}")
    state = eng.get_twin_state()
    params, _ = eng.get_parameters()
    return compute_all_biomarkers(state, params, eng._observation_buffer)


# ── Drift ─────────────────────────────────────────────────────

@router.get("/{patient_id}/drift")
async def get_drift(patient_id: str):
    eng = _engines.get(patient_id)
    if eng is None:
        raise HTTPException(status_code=404, detail=f"No engine for {patient_id}")
    return eng.get_drift_status()


# ── Uncertainty ───────────────────────────────────────────────

@router.get("/{patient_id}/uncertainty")
async def get_uncertainty(patient_id: str):
    eng = _engines.get(patient_id)
    if eng is None:
        raise HTTPException(status_code=404, detail=f"No engine for {patient_id}")
    from app.personalization.uncertainty import UncertaintyEngine
    ue = UncertaintyEngine(eng)
    report = ue.full_report()
    return {
        "parameter_uncertainty_90ci": report.parameter_uncertainty.tolist(),
        "measurement_uncertainty": report.measurement_uncertainty.tolist(),
        "intervention_cv_pct": report.intervention_uncertainty,
        "coverage_metrics": report.coverage_metrics,
    }


# ── Counterfactual ────────────────────────────────────────────

@router.post("/{patient_id}/counterfactual")
async def run_counterfactual(
    patient_id: str,
    program_name: str = Query("Combined: Diet + Exercise + Metformin"),
):
    eng = _engines.get(patient_id)
    if eng is None:
        raise HTTPException(status_code=404, detail=f"No engine for {patient_id}")
    if not eng.drift_detector.can_run_counterfactuals:
        raise HTTPException(status_code=400, detail="Drift too high for counterfactuals")

    from app.personalization.counterfactual import (
        CounterfactualEngine,
        MEDITERRANEAN_DIET, EXERCISE_PROGRAM, METFORMIN, COMBINED_THERAPY,
    )
    programs = {
        "Mediterranean Diet": MEDITERRANEAN_DIET,
        "Exercise 150min/week": EXERCISE_PROGRAM,
        "Metformin": METFORMIN,
        "Combined: Diet + Exercise + Metformin": COMBINED_THERAPY,
    }
    program = programs.get(program_name)
    if program is None:
        raise HTTPException(status_code=400, detail=f"Unknown program: {program_name}")

    ce = CounterfactualEngine(eng)
    traj = ce.simulate_program(program)
    summary = ce.program_summary(traj)
    return {
        "program": summary["program"],
        "final_glucose": summary["final_glucose"],
        "avg_glucose": summary["avg_glucose"],
        "estimated_hba1c": summary["estimated_hba1c"],
        "final_sbp": summary["final_sbp"],
        "weight_change_kg": summary["weight_change_kg"],
        "glucose_trajectory": traj.glucose,
        "sbp_trajectory": traj.sbp,
    }


# ── Virtual Cohort ────────────────────────────────────────────

@router.post("/virtual-cohort")
async def generate_cohort(
    n_patients: int = Query(100, ge=10, le=10000),
    seed: int = Query(42, ge=0),
):
    from app.personalization.cohort import VirtualCohortEngine
    vce = VirtualCohortEngine(seed=seed)
    cohort = vce.sample_from_priors(n_patients=n_patients)
    stats = vce.summary_stats()
    return {
        "n_patients": stats["n_patients"],
        "state_means": stats["state_means"],
        "param_means": stats["param_means"],
    }


# ── Explainability ────────────────────────────────────────────

@router.get("/{patient_id}/explain")
async def explain(
    patient_id: str,
    level: str = Query("patient", pattern="^(patient|clinician|scientist)$"),
):
    eng = _engines.get(patient_id)
    if eng is None:
        raise HTTPException(status_code=404, detail=f"No engine for {patient_id}")
    from app.personalization.explainability import ExplainabilityEngine
    ee = ExplainabilityEngine(eng)
    explanation = ee.explain(level)
    return {"level": explanation.level, "summary": explanation.summary, "details": explanation.details}


# ── RL Recommendation ─────────────────────────────────────────

@router.get("/{patient_id}/recommend")
async def recommend(patient_id: str):
    eng = _engines.get(patient_id)
    if eng is None:
        raise HTTPException(status_code=404, detail=f"No engine for {patient_id}")
    from app.personalization.rl import recommend_intervention
    program = recommend_intervention(eng)
    return {
        "recommendation": program.name,
        "duration_days": program.duration_days,
        "adherence_target": program.adherence,
    }


# ── Management ────────────────────────────────────────────────

@router.delete("/{patient_id}")
async def delete(patient_id: str):
    if patient_id not in _engines:
        raise HTTPException(status_code=404, detail=f"No engine for {patient_id}")
    del _engines[patient_id]
    return {"status": "deleted", "patient_id": patient_id}


@router.get("/engines")
async def list_engines():
    return {"active_engines": list(_engines.keys()), "count": len(_engines)}


@router.get("/health")
async def health():
    return {
        "status": "healthy",
        "layer": "personalization",
        "version": "3.0.0",
        "description": "Phase 3 Whole-body cellular digital twin — 30 states, 25 params, 15 obs",
        "capabilities": [
            "multi-organ state estimation",
            "hierarchical bayesian priors",
            "circadian rhythm modeling",
            "adipose-lipid metabolism",
            "immune-inflammatory signaling",
            "counterfactual simulation",
            "virtual cohort generation",
            "uncertainty quantification",
            "explainable AI (3 tiers)",
            "RL-based intervention recommendation",
        ],
    }
