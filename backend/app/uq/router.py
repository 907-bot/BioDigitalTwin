"""FastAPI router for Phase 11 — Uncertainty Quantification."""
from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.causal.scm import (
    fit_cohort_scm,
    get_dag,
    patient_counterfactual,
    reset_scm,
)
from .bootstrap import bootstrap_ate, bootstrap_patient_counterfactual

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/phase11", tags=["Phase 11 — Uncertainty Quantification"])

PATIENTS_CSV = "data/synthetic_patients.csv"


def _cohort_df() -> pd.DataFrame:
    if not os.path.exists(PATIENTS_CSV):
        raise HTTPException(404, "no cohort — call POST /generate-patients first")
    return pd.read_csv(PATIENTS_CSV)


class UQRequest(BaseModel):
    patient_id: str
    treatment: str
    outcome: str
    value: float = 1.0
    n_bootstrap: int = 200
    confidence: float = 0.90
    seed: int = 42


@router.post("/patient-counterfactual")
def uq_patient_counterfactual(req: UQRequest):
    """Patient counterfactual with bootstrap 90% CI over the effect."""
    df = _cohort_df()
    row = df[df["patient_id"] == req.patient_id]
    if row.empty:
        raise HTTPException(404, f"patient '{req.patient_id}' not found")
    r = row.iloc[0]
    from app.graph.ontology import BIOMARKERS
    observed = {b.id: float(r[b.id]) for b in BIOMARKERS}
    observed["bmi"] = float(r["bmi"])
    observed["age"] = float(r["age"])

    return bootstrap_patient_counterfactual(
        df, observed=observed,
        treatment=req.treatment, value=req.value, outcome=req.outcome,
        n_bootstrap=req.n_bootstrap, confidence=req.confidence, seed=req.seed,
    )


@router.post("/ate")
def uq_ate(treatment: str = Query(...),
           outcome: str = Query(...),
           common_causes: Optional[str] = Query(None, description="comma-separated"),
           n_bootstrap: int = Query(100, ge=10, le=500),
           confidence: float = Query(0.95, ge=0.5, le=0.99)):
    """ATE with bootstrap CI."""
    df = _cohort_df()
    cc = [c.strip() for c in (common_causes or "").split(",") if c.strip()]
    return bootstrap_ate(df, treatment=treatment, outcome=outcome,
                         common_causes=cc, n_bootstrap=n_bootstrap,
                         confidence=confidence)
