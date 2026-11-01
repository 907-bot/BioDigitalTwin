"""
BioDigitalTwin — Physiological Core v2 Router.

Exposes the coupled multi-organ physiological world model:
1. GET  /physiology/v2/state-registry      — Canonical state and parameter schemas
2. POST /physiology/v2/simulation/run      — Coupled multi-organ simulation execution
3. POST /physiology/v2/challenge/meal      — Coordinated Oral Glucose Challenge (OGTT)
4. POST /physiology/v2/challenge/exercise  — Exercise stress challenge protocol
5. GET  /physiology/v2/conservation/audit  — Whole-body mass and flow conservation audit
6. POST /physiology/v2/anomaly/detect      — Telemetry anomaly and rate check
"""

from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import logging

from app.physiology.core.state_registry import get_global_registry
from app.physiology.core.simulation_engine import (
    CoupledSimulationEngine,
    SimulationEvent,
    WholeBodySnapshot,
)
from app.physiology.organs import create_standard_multi_organ_engine
from app.physiology.hybrid.residual_learner import (
    ConstrainedResidualLearner,
    PhysiologicalAnomalyDetector,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/physiology/v2", tags=["Physiology Core v2 — Multi-Organ World Model"])

# In-memory twin session store
_v2_engines: Dict[str, CoupledSimulationEngine] = {}
_anomaly_detector = PhysiologicalAnomalyDetector()


def _get_or_create_engine(patient_id: str, weight_kg: float = 70.0, height_cm: float = 175.0) -> CoupledSimulationEngine:
    if patient_id not in _v2_engines:
        _v2_engines[patient_id] = create_standard_multi_organ_engine(
            patient_weight_kg=weight_kg,
            patient_height_cm=height_cm,
        )
    return _v2_engines[patient_id]


# ── Pydantic Request Schemas ──────────────────────────────────────────

class SimulationEventModel(BaseModel):
    time_minutes: float = Field(..., ge=0.0, description="Event scheduled timestamp in minutes")
    event_type: str = Field(..., description="Event type: meal, exercise, drug_dose")
    payload: Dict[str, float] = Field(default_factory=dict, description="Event parameters")


class RunSimulationRequest(BaseModel):
    patient_id: str = Field("default", min_length=1, max_length=64)
    patient_weight_kg: float = Field(70.0, ge=30.0, le=250.0)
    patient_height_cm: float = Field(175.0, ge=100.0, le=230.0)
    duration_minutes: float = Field(60.0, ge=1.0, le=720.0)
    dt_minutes: float = Field(1.0, ge=0.05, le=10.0)
    solver: str = Field("rk4", pattern="^(rk4|euler)$")
    inputs: Dict[str, float] = Field(default_factory=dict)
    events: List[SimulationEventModel] = Field(default_factory=list)


class MealChallengeRequestV2(BaseModel):
    patient_id: str = Field("default", min_length=1, max_length=64)
    carbs_g: float = Field(75.0, ge=5.0, le=250.0, description="Carbohydrate intake (75g for standard OGTT)")
    duration_minutes: float = Field(180.0, ge=30.0, le=360.0)
    dt_minutes: float = Field(1.0, ge=0.1, le=5.0)
    patient_weight_kg: float = Field(70.0, ge=30.0, le=250.0)


class ExerciseChallengeRequestV2(BaseModel):
    patient_id: str = Field("default", min_length=1, max_length=64)
    exercise_intensity: float = Field(0.6, ge=0.1, le=1.0)
    exercise_duration_minutes: float = Field(30.0, ge=5.0, le=120.0)
    total_observation_minutes: float = Field(60.0, ge=15.0, le=180.0)
    dt_minutes: float = Field(1.0, ge=0.1, le=5.0)
    patient_weight_kg: float = Field(70.0, ge=30.0, le=250.0)


class AnomalyDetectionRequest(BaseModel):
    current_telemetry: Dict[str, float]
    previous_telemetry: Dict[str, float]
    dt_minutes: float = Field(1.0, gt=0.0, le=60.0)


def _serialize_snapshot(snap: WholeBodySnapshot) -> Dict[str, Any]:
    return {
        "time_minutes": snap.time_minutes,
        "glucose_mg_dL": snap.circulating_glucose_mg_dL,
        "insulin_uU_mL": snap.circulating_insulin_uU_mL,
        "glucagon_pg_mL": snap.circulating_glucagon_pg_mL,
        "heart_rate_bpm": snap.heart_rate_bpm,
        "cardiac_output_L_min": snap.cardiac_output_L_min,
        "mean_arterial_pressure_mmHg": snap.mean_arterial_pressure_mmHg,
        "systolic_bp_mmHg": snap.systolic_bp_mmHg,
        "diastolic_bp_mmHg": snap.diastolic_bp_mmHg,
        "arterial_paO2_mmHg": snap.arterial_paO2_mmHg,
        "arterial_paCO2_mmHg": snap.arterial_paCO2_mmHg,
        "arterial_pH": snap.arterial_pH,
        "arterial_saO2_pct": snap.arterial_saO2_pct,
        "extracellular_fluid_L": snap.extracellular_fluid_L,
        "organ_states": snap.organ_states,
        "active_fluxes": snap.active_fluxes,
        "is_physiologically_valid": snap.is_physiologically_valid,
        "violations": snap.violations,
    }


# ── API Endpoints ────────────────────────────────────────────────────

@router.get("/state-registry")
def get_canonical_state_registry():
    """
    Returns the canonical state and parameter schemas for the whole-body digital twin.
    Includes SI/clinical units, physiological bounds, and medical literature provenance.
    """
    # Ensure canonical organ schemas are registered
    _get_or_create_engine("default")
    reg = get_global_registry()
    states_out = {
        k: {
            "name": v.name,
            "display_name": v.display_name,
            "unit": v.unit,
            "compartment": v.compartment,
            "min_bound": v.min_bound,
            "max_bound": v.max_bound,
            "nominal": v.nominal,
            "initial_condition_source": v.initial_condition_source,
            "is_observable": v.is_observable,
            "owning_module": v.owning_module,
            "description": v.description,
        }
        for k, v in reg._states.items()
    }
    params_out = {
        k: {
            "name": v.name,
            "display_name": v.display_name,
            "unit": v.unit,
            "default_value": v.default_value,
            "min_bound": v.min_bound,
            "max_bound": v.max_bound,
            "provenance": v.provenance,
            "is_personalized": v.is_personalized,
            "description": v.description,
        }
        for k, v in reg._parameters.items()
    }
    return {
        "status": "success",
        "total_states": len(states_out),
        "total_parameters": len(params_out),
        "states": states_out,
        "parameters": params_out,
    }


@router.post("/simulation/run")
def run_coupled_simulation(req: RunSimulationRequest):
    """
    Executes a multi-step coupled whole-body simulation with registered discrete events.
    Returns the full time-series trajectory and conservation audit.
    """
    engine = _get_or_create_engine(req.patient_id, req.patient_weight_kg, req.patient_height_cm)

    # Schedule requested events
    for evt in req.events:
        engine.schedule_event(
            SimulationEvent(
                time_minutes=evt.time_minutes,
                event_type=evt.event_type,
                payload=evt.payload,
            )
        )

    trajectory = engine.run_simulation(
        duration_minutes=req.duration_minutes,
        dt_minutes=req.dt_minutes,
        inputs=req.inputs,
        solver=req.solver,
    )

    serialized_snaps = [_serialize_snapshot(s) for s in trajectory]
    final_snap = serialized_snaps[-1] if serialized_snaps else None

    # Conservation audit
    ok_hemo, hemo_issues = engine.conservation_validator.verify_hemodynamic_conservation(engine.environment)
    ok_bounds, bound_issues = engine.conservation_validator.verify_physical_bounds(
        engine.environment, {k: v.get_state() for k, v in engine.organs.items()}
    )
    violations = hemo_issues + bound_issues

    return {
        "status": "success",
        "patient_id": req.patient_id,
        "duration_minutes": req.duration_minutes,
        "total_steps": len(serialized_snaps),
        "final_snapshot": final_snap,
        "trajectory": serialized_snaps,
        "conservation_audit": {
            "is_conserved": len(violations) == 0,
            "violations": violations,
        },
    }


@router.post("/challenge/meal")
def run_meal_challenge(req: MealChallengeRequestV2):
    """
    Standard Oral Glucose Tolerance Test (OGTT) challenge.
    Demonstrates coordinated multi-organ dynamics:
    - GI gastric emptying & SGLT1 absorption
    - Pancreatic insulin & incretin GLP-1 response
    - Hepatic HGP suppression
    - Renal glycosuria threshold clearance
    """
    engine = create_standard_multi_organ_engine(patient_weight_kg=req.patient_weight_kg)
    engine.initialize()

    # Schedule oral meal at t = 5.0 min
    engine.schedule_event(
        SimulationEvent(
            time_minutes=5.0,
            event_type="meal",
            payload={"carbs_g": req.carbs_g},
        )
    )

    history = engine.simulate(
        total_minutes=req.duration_minutes,
        dt_minutes=req.dt_minutes,
    )

    glucose_profile = [s.circulating_glucose_mg_dL for s in history]
    insulin_profile = [s.circulating_insulin_uU_mL for s in history]

    g_base = glucose_profile[4] if len(glucose_profile) > 4 else glucose_profile[0]
    g_max = max(glucose_profile)
    t_peak = history[glucose_profile.index(g_max)].time_minutes
    g_final = glucose_profile[-1]

    gi_organ = engine.organs.get("gastrointestinal")
    absorbed_g = getattr(gi_organ, "cumulative_absorbed_g", 0.0)

    return {
        "status": "success",
        "challenge_type": "oral_glucose_tolerance_test",
        "meal_carbs_g": req.carbs_g,
        "baseline_glucose_mg_dL": round(g_base, 1),
        "peak_glucose_mg_dL": round(g_max, 1),
        "peak_time_minutes": round(t_peak, 1),
        "final_glucose_mg_dL": round(g_final, 1),
        "peak_insulin_uU_mL": round(max(insulin_profile), 1),
        "cumulative_absorbed_g": round(absorbed_g, 2),
        "trajectory": [_serialize_snapshot(s) for s in history],
    }


@router.post("/challenge/exercise")
def run_exercise_challenge(req: ExerciseChallengeRequestV2):
    """
    Exercise Stress Challenge:
    Simulates cardiovascular, metabolic, and respiratory adaptation to physical exertion.
    """
    engine = create_standard_multi_organ_engine(patient_weight_kg=req.patient_weight_kg)
    engine.initialize()

    # 1. Pre-exercise baseline (10 min)
    pre_snaps = engine.simulate(total_minutes=10.0, dt_minutes=req.dt_minutes)
    base_hr = pre_snaps[-1].heart_rate_bpm
    base_co = pre_snaps[-1].cardiac_output_L_min

    # 2. Exercise bout
    ex_snaps = engine.simulate(
        total_minutes=req.exercise_duration_minutes,
        dt_minutes=req.dt_minutes,
        inputs={"exercise": req.exercise_intensity},
    )
    peak_ex = ex_snaps[-1]

    # 3. Post-exercise recovery
    recovery_minutes = max(5.0, req.total_observation_minutes - 10.0 - req.exercise_duration_minutes)
    rec_snaps = engine.simulate(total_minutes=recovery_minutes, dt_minutes=req.dt_minutes, inputs={})

    all_snaps = pre_snaps + ex_snaps + rec_snaps

    return {
        "status": "success",
        "challenge_type": "exercise_stress_challenge",
        "exercise_intensity": req.exercise_intensity,
        "exercise_duration_minutes": req.exercise_duration_minutes,
        "baseline_hr_bpm": round(base_hr, 1),
        "peak_hr_bpm": round(peak_ex.heart_rate_bpm, 1),
        "peak_cardiac_output_L_min": round(peak_ex.cardiac_output_L_min, 2),
        "peak_mean_arterial_pressure_mmHg": round(peak_ex.mean_arterial_pressure_mmHg, 1),
        "nadir_arterial_pH": round(min(s.arterial_pH for s in all_snaps), 3),
        "trajectory": [_serialize_snapshot(s) for s in all_snaps],
    }


@router.get("/conservation/audit")
def audit_conservation_laws(patient_id: str = Query("default")):
    """
    Audits whole-body physical conservation laws on the active twin:
    - Hemodynamic flow conservation (sum(Q_regional) == CO)
    - Pulmonary blood flow == Systemic cardiac output
    - Physical and biological bound compliance
    """
    engine = _get_or_create_engine(patient_id)
    env = engine.environment

    ok_hemo, hemo_issues = engine.conservation_validator.verify_hemodynamic_conservation(env)
    ok_bounds, bound_issues = engine.conservation_validator.verify_physical_bounds(
        env, {k: v.get_state() for k, v in engine.organs.items()}
    )
    violations = hemo_issues + bound_issues

    return {
        "status": "success",
        "patient_id": patient_id,
        "is_conserved": len(violations) == 0,
        "violations": violations,
        "cardiac_output_L_min": round(env.circulation.cardiac_output_L_min, 2),
        "regional_flows_L_min": {k: round(v, 3) for k, v in env.circulation.regional_flows_L_min.items()},
        "total_body_water_L": env.fluids.total_body_water_L,
        "extracellular_fluid_L": env.fluids.extracellular_fluid_L,
    }


@router.post("/anomaly/detect")
def detect_telemetry_anomalies(req: AnomalyDetectionRequest):
    """
    Evaluates telemetry jumps against maximum plausible physiological rates of change:
    - CGM rate-of-change <= 4.5 mg/dL/min
    - Heart rate jump <= 25 bpm/min
    - Blood pressure jump <= 30 mmHg/min
    - Hypoglycemia and severe bradycardia limits
    """
    is_valid, anomalies = _anomaly_detector.check_telemetry(
        current_telemetry=req.current_telemetry,
        previous_telemetry=req.previous_telemetry,
        dt_minutes=req.dt_minutes,
    )
    return {
        "status": "success",
        "is_physiologically_plausible": is_valid,
        "detected_anomalies": anomalies,
    }
