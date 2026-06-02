"""FastAPI router for Phase 10 — PK/PD simulation."""
from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from .compartments import (
    DosingRegimen,
    PatientCovariates,
    PKResult,
    population_simulation,
    population_summary,
    simulate_pk,
)
from .pd_models import PDParams, PDModel, effect_at_concentration, simulate_effect_compartment
from .registry import DRUG_REGISTRY, get_drug, list_drugs

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/phase10", tags=["Phase 10 — PK/PD"])


PATIENTS_CSV = "data/synthetic_patients.csv"


def _patient_covariates_from_id(patient_id: str) -> PatientCovariates:
    if not os.path.exists(PATIENTS_CSV):
        raise HTTPException(404, "no cohort — call POST /generate-patients first")
    df = pd.read_csv(PATIENTS_CSV)
    row = df[df["patient_id"] == patient_id]
    if row.empty:
        raise HTTPException(404, f"patient '{patient_id}' not found")
    r = row.iloc[0]
    sex = "F" if str(r.get("gender", "Male")).lower().startswith("f") else "M"
    return PatientCovariates(
        age=float(r.get("age", 45)),
        weight=float(r.get("weight", 70) if "weight" in df.columns else 70),
        sex=sex,
        serum_creatinine=float(r.get("serum_creatinine", 1.0)
                                if "serum_creatinine" in df.columns else 1.0),
    )


@router.get("/drugs")
def list_pkpd_drugs():
    return {"total": len(DRUG_REGISTRY), "drugs": list_drugs()}


@router.get("/drugs/{name}")
def get_drug_detail(name: str):
    rec = get_drug(name)
    return {
        "name": rec.name,
        "drug_class": rec.drug_class,
        "typical_dose_mg": rec.typical_dose_mg,
        "typical_interval_h": rec.typical_interval_h,
        "target_biomarker": rec.target_biomarker,
        "effect_direction": rec.effect_direction,
        "notes": rec.notes,
        "pk": {
            "ka": rec.pk.ka, "CL": rec.pk.CL, "Vc": rec.pk.Vc,
            "Vp": rec.pk.Vp, "Q": rec.pk.Q, "F": rec.pk.F,
            "route": rec.pk.route,
        },
        "pd": {
            "model": rec.pd.model.value,
            "E0": rec.pd.E0, "Emax": rec.pd.Emax,
            "EC50": rec.pd.EC50, "gamma": rec.pd.gamma,
            "ke0": rec.pd.ke0,
            "target_unit": rec.pd.target_unit,
        },
    }


class PKRequest(BaseModel):
    drug: str
    dose_mg: float
    n_doses: int = 1
    interval_h: float = 24.0
    route: str = "oral"
    patient_id: Optional[str] = None
    age: Optional[float] = None
    weight: Optional[float] = None
    sex: Optional[str] = None
    serum_creatinine: Optional[float] = None
    t_end_h: Optional[float] = None
    seed: int = 0


@router.post("/pk/simulate")
def pk_simulate(req: PKRequest):
    """Run a 2-compartment PK simulation with covariate adjustment and BSV."""
    rec = get_drug(req.drug)
    if req.patient_id:
        cov = _patient_covariates_from_id(req.patient_id)
    else:
        cov = PatientCovariates(
            age=req.age or 45.0,
            weight=req.weight or 70.0,
            sex=(req.sex or "M"),
            serum_creatinine=req.serum_creatinine or 1.0,
        )
    regimen = DosingRegimen(
        dose_mg=req.dose_mg,
        n_doses=max(1, int(req.n_doses)),
        interval_h=float(req.interval_h),
        route=req.route,
    )
    res = simulate_pk(rec.pk, regimen, cov=cov, t_end_h=req.t_end_h, seed=req.seed)
    return _format_pk_result(res, drug=rec.name, regimen=regimen, cov=cov)


def _format_pk_result(res: PKResult, drug: str, regimen: DosingRegimen,
                      cov: PatientCovariates) -> dict:
    # Downsample to ~150 points for the response payload
    step = max(1, len(res.times_h) // 150)
    t_dense = res.times_h
    c_dense = res.c_central
    return {
        "drug": drug,
        "regimen": {
            "dose_mg": regimen.dose_mg,
            "n_doses": regimen.n_doses,
            "interval_h": regimen.interval_h,
            "route": regimen.route,
        },
        "patient": {
            "age": cov.age, "weight": cov.weight, "sex": cov.sex,
            "serum_creatinine": cov.serum_creatinine,
        },
        "pk_metrics": {
            "cmax":   round(res.cmax, 4),
            "tmax":   round(res.tmax, 2),
            "auc_0_t": round(res.auc_0_t, 3),
            "auc_0_inf": round(res.auc_0_inf, 3),
            "half_life_h": round(res.half_life, 2),
            "clearance_L_per_h": round(res.clearance, 3),
            "vd_ss_L": round(res.vd_ss, 2),
            "cmin_ss":  round(res.cmin_ss, 4) if res.cmin_ss is not None else None,
            "cmax_ss":  round(res.cmax_ss, 4) if res.cmax_ss is not None else None,
            "accumulation_ratio": round(res.accumulation_ratio, 2) if res.accumulation_ratio else None,
            "time_to_steady_state_h": round(res.time_to_steady_state_h, 1) if res.time_to_steady_state_h else None,
        },
        "concentration_curve": [
            {"t_h": round(float(t), 3), "c_mg_per_L": round(float(c), 4)}
            for t, c in zip(t_dense[::step], c_dense[::step])
        ],
        "validation_checks": _validate_pk(res, regimen),
    }


def _validate_pk(res: PKResult, regimen: DosingRegimen) -> list[dict]:
    """Industry-style sanity checks for a PK profile."""
    checks = []
    # 1) Linear PK: doubling dose should ~double Cmax
    checks.append({
        "check": "linear_pk_cmax",
        "description": "Doubling dose should ~double Cmax (linear PK)",
        "status": "info",
        "value": res.cmax,
    })
    # 2) Half-life positive
    checks.append({
        "check": "half_life_positive",
        "description": "Half-life is positive and finite",
        "status": "pass" if (0 < res.half_life < 1000) else "fail",
        "value": round(res.half_life, 2),
    })
    # 3) Multi-dose accumulation
    if regimen.n_doses >= 2 and res.accumulation_ratio:
        if 1.0 < res.accumulation_ratio < 5.0:
            status = "pass"
        else:
            status = "warn"
        checks.append({
            "check": "accumulation_ratio",
            "description": "Steady-state Cmax / first-dose Cmax should be 1-5x for typical dosing",
            "status": status,
            "value": round(res.accumulation_ratio, 2),
        })
    return checks


class PDRequest(BaseModel):
    drug: str
    dose_mg: float
    n_doses: int = 1
    interval_h: float = 24.0
    patient_id: Optional[str] = None
    age: Optional[float] = None
    weight: Optional[float] = None
    sex: Optional[str] = None
    serum_creatinine: Optional[float] = None
    target_biomarker: Optional[str] = None
    t_end_h: Optional[float] = None
    seed: int = 0


@router.post("/pd/simulate")
def pd_simulate(req: PDRequest):
    """
    Run a coupled PK+PD simulation.

    Uses the drug's PK params to compute a plasma-concentration curve,
    then drives a sigmoid-Emax (or other) PD model to predict the
    effect on the target biomarker.
    """
    rec = get_drug(req.drug)
    if req.patient_id:
        cov = _patient_covariates_from_id(req.patient_id)
    else:
        cov = PatientCovariates(
            age=req.age or 45.0,
            weight=req.weight or 70.0,
            sex=(req.sex or "M"),
            serum_creatinine=req.serum_creatinine or 1.0,
        )
    regimen = DosingRegimen(
        dose_mg=req.dose_mg, n_doses=req.n_doses,
        interval_h=req.interval_h, route=rec.pk.route,
    )
    pk_res = simulate_pk(rec.pk, regimen, cov=cov, t_end_h=req.t_end_h, seed=req.seed)
    effect = simulate_effect_compartment(pk_res.c_central, pk_res.times_h, rec.pd)

    step = max(1, len(pk_res.times_h) // 150)
    target = req.target_biomarker or rec.target_biomarker

    return {
        "drug": rec.name,
        "target_biomarker": target,
        "pd_model": rec.pd.model.value,
        "pd_unit": rec.pd.target_unit,
        "patient": {
            "age": cov.age, "weight": cov.weight, "sex": cov.sex,
            "serum_creatinine": cov.serum_creatinine,
        },
        "effect_at_tmax": round(float(effect[pk_res.c_central.argmax()]), 3),
        "max_effect": round(float(effect.min() if rec.effect_direction == "decrease"
                                    else effect.max()), 3),
        "min_effect": round(float(effect.max() if rec.effect_direction == "decrease"
                                    else effect.min()), 3),
        "effect_curve": [
            {"t_h": round(float(t), 3),
             "c_mg_per_L": round(float(c), 4),
             "effect": round(float(e), 3)}
            for t, c, e in zip(pk_res.times_h[::step],
                                pk_res.c_central[::step], effect[::step])
        ],
    }


class PopRequest(BaseModel):
    drug: str
    dose_mg: float
    n_doses: int = 1
    interval_h: float = 24.0
    n_subjects: int = 50
    seed: int = 0


@router.post("/population")
def population_sim(req: PopRequest):
    """Monte Carlo PK simulation across a virtual population."""
    rec = get_drug(req.drug)
    regimen = DosingRegimen(
        dose_mg=req.dose_mg, n_doses=req.n_doses,
        interval_h=req.interval_h, route=rec.pk.route,
    )
    results = population_simulation(rec.pk, regimen, n_subjects=req.n_subjects,
                                     seed=req.seed)
    summary = population_summary(results)
    return {
        "drug": rec.name,
        "regimen": {
            "dose_mg": req.dose_mg, "n_doses": req.n_doses,
            "interval_h": req.interval_h,
        },
        **summary,
    }
