"""
Phase 5 — Agent tools.

Each tool wraps a phase 1-4 endpoint into a single string-returning
function the LLM can call. We deliberately use the local CSV/JSON files
instead of HTTP to avoid a self-call inside the container.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import pandas as pd

from app.causal.scm import (
    OUTCOMES_FOR_DISEASE, TREATMENT_TARGETS, fit_cohort_scm, get_dag,
    patient_counterfactual, reset_scm, cate_estimate, ate_estimate,
)
from app.dynamics.disease_model import (
    DISEASE_FORCINGS, INTERVENTIONS, classify_risk, bifurcation_summary,
)
from app.graph.builder import (
    _normalise, compute_risk_score, _risk_label, load_patients,
)
from app.graph.ontology import BIOMARKERS, DISEASES, ORGANS
from app.graph.trainer import (
    load_embeddings, get_embedding, get_top_k_similar_by_embedding,
)


PATIENTS_CSV = "data/synthetic_patients.csv"


def _load_patients() -> pd.DataFrame:
    if not os.path.exists(PATIENTS_CSV):
        raise FileNotFoundError(
            f"{PATIENTS_CSV} not found. Call POST /generate-patients first.")
    return pd.read_csv(PATIENTS_CSV)


# --- patient lookup -----------------------------------------------------
def get_patient_summary(patient_id: str) -> str:
    df = _load_patients()
    row = df[df["patient_id"] == patient_id]
    if row.empty:
        return f"Patient '{patient_id}' not found in the cohort."
    r = row.iloc[0].to_dict()
    score = compute_risk_score(row.iloc[0])
    label = _risk_label(score)
    abnormal = []
    for b in BIOMARKERS:
        v = r.get(b.id)
        if v is None or pd.isna(v):
            continue
        if not (b.healthy_lo <= float(v) <= b.healthy_hi):
            abnormal.append(f"{b.name}={float(v):.1f} {b.unit} "
                            f"(healthy {b.healthy_lo}-{b.healthy_hi})")
    out = (
        f"Patient {patient_id}: {int(r['age'])}yo {r['gender']}, "
        f"BMI={float(r['bmi']):.1f}. Risk={label} (score={score:.2f})."
    )
    if abnormal:
        out += " Abnormal biomarkers: " + "; ".join(abnormal) + "."
    else:
        out += " All biomarkers in healthy range."
    return out


# --- similarity ---------------------------------------------------------
def find_similar_patients(patient_id: str, k: int = 5) -> str:
    try:
        results = get_top_k_similar_by_embedding(patient_id, k=k)
    except (FileNotFoundError, KeyError) as e:
        return f"Could not load embeddings: {e}. Run POST /phase2/train-gnn first."
    df = _load_patients().set_index("patient_id")
    lines = [f"Top {len(results)} patients similar to {patient_id}:"]
    for pid, sim in results:
        if pid in df.index:
            r = df.loc[pid]
            lines.append(f"  {pid} (sim={sim:.3f}): age={int(r['age'])}, "
                         f"BMI={float(r['bmi']):.1f}, glu={int(r['glucose'])}, "
                         f"sbp={int(r['systolic_bp'])}")
    return "\n".join(lines)


# --- disease trajectory -------------------------------------------------
def simulate_disease(patient_id: str, disease: str,
                     horizon_days: int = 180,
                     intervention: Optional[str] = None) -> str:
    if disease not in DISEASE_FORCINGS:
        return f"Unknown disease '{disease}'. Available: {list(DISEASE_FORCINGS.keys())}"
    df = _load_patients()
    row = df[df["patient_id"] == patient_id]
    if row.empty:
        return f"Patient '{patient_id}' not found."
    r = row.iloc[0]
    initial = {b.id: float(r[b.id]) for b in BIOMARKERS}
    interv = INTERVENTIONS.get(intervention, {}) if intervention else {}
    from app.dynamics.disease_model import DiseaseSimulator
    sim = DiseaseSimulator()
    res = sim.simulate(initial_state=initial, disease=disease,
                       horizon_days=horizon_days, intervention=interv, rng_seed=0)
    interv_str = f" with {intervention}" if intervention else ""
    return (
        f"Disease simulation{interv_str} for {patient_id} over {horizon_days} days: "
        f"risk {res['risks'][0]:.2f} -> {res['final_risk']:.2f} "
        f"({classify_risk(res['final_risk'])}). "
        f"Final glucose={res['final_state']['glucose']:.1f}, "
        f"SBP={res['final_state']['systolic_bp']:.1f}, "
        f"BMI={res['final_state']['bmi']:.1f}."
    )


# --- counterfactual -----------------------------------------------------
def counterfactual_for_patient(patient_id: str, treatment: str,
                               outcome: str, value: float = 1.0) -> str:
    df = _load_patients()
    row = df[df["patient_id"] == patient_id]
    if row.empty:
        return f"Patient '{patient_id}' not found."
    r = row.iloc[0]
    observed = {b.id: float(r[b.id]) for b in BIOMARKERS}

    scm = fit_cohort_scm(df)
    res = patient_counterfactual(scm, observed=observed,
                                 treatment=treatment, value=float(value),
                                 outcome=outcome)
    if "error" in res:
        return f"Counterfactual failed: {res['error']}"
    return (
        f"Counterfactual for {patient_id}: do({treatment}={value}) -> "
        f"{outcome}: factual={res['factual']:.2f}, "
        f"counterfactual={res['counterfactual']:.2f} "
        f"({res['effect_direction']} of {abs(res['effect']):.2f})."
    )


# --- ATE / CATE ---------------------------------------------------------
def estimate_treatment_effect(treatment: str, outcome: str) -> str:
    df = _load_patients()
    # If treatment is a named intervention, treat as a binary indicator
    # (the synthetic cohort doesn't have an exposure column, so we
    # simulate it by binarising a related biomarker).
    INTERVENTION_TO_COL = {
        "metformin": "glucose",
        "losartan":  "systolic_bp",
        "statin":    "systolic_bp",
        "exercise_30m": "hrv",
        "weight_loss":  "bmi",
        "smoking_cessation": "spo2",
    }
    if treatment in INTERVENTION_TO_COL and treatment not in df.columns:
        col = INTERVENTION_TO_COL[treatment]
        if col not in df.columns:
            return f"ATE failed: column '{col}' not in cohort"
        median = float(df[col].median())
        df = df.copy()
        df[treatment] = (df[col] > median).astype(float)
    common = ["age", "bmi", "hr", "hrv", "spo2"]
    res = ate_estimate(df, treatment=treatment, outcome=outcome,
                       common_causes=common)
    if "error" in res:
        return f"ATE failed: {res['error']}"
    return (f"ATE of {treatment} on {outcome} across {res['n_samples']} patients: "
            f"{res['ate']:.4f}. {res['ate_interpretation']} "
            f"(R^2={res['r2']}).")


def estimate_heterogeneous_effect(treatment: str, outcome: str,
                                  modifier: str = "bmi") -> str:
    df = _load_patients()
    INTERVENTION_TO_COL = {
        "metformin": "glucose",
        "losartan":  "systolic_bp",
        "statin":    "systolic_bp",
        "exercise_30m": "hrv",
        "weight_loss":  "bmi",
        "smoking_cessation": "spo2",
    }
    if treatment in INTERVENTION_TO_COL and treatment not in df.columns:
        col = INTERVENTION_TO_COL[treatment]
        if col not in df.columns:
            return f"CATE failed: column '{col}' not in cohort"
        median = float(df[col].median())
        df = df.copy()
        df[treatment] = (df[col] > median).astype(float)
    res = cate_estimate(df, treatment=treatment, outcome=outcome,
                        effect_modifiers=[modifier])
    if "error" in res:
        return f"CATE failed: {res['error']}"
    if "bins" in res:
        bin_str = "; ".join(f"{b['bin']}: ATE={b['ate']:.3f} (n={b['n']})"
                            for b in res["bins"])
        return (f"CATE of {treatment} on {outcome} by {modifier} quartile "
                f"(method={res['method']}): {bin_str}. "
                f"Mean CATE={res['mean_cate']:.4f}.")
    return (f"CATE of {treatment} on {outcome} by {modifier} "
            f"(method={res['method']}): mean={res['mean_cate']:.4f}, "
            f"std={res['std_cate']:.4f}, range=[{res['min_cate']:.3f}, "
            f"{res['max_cate']:.3f}]. N={res['n_samples']}.")


# --- cohort-level -------------------------------------------------------
def cohort_overview() -> str:
    df = _load_patients()
    n = len(df)
    avg_age = float(df["age"].mean())
    pct_female = float((df["gender"] == "Female").mean() * 100)
    risks = []
    for _, row in df.iterrows():
        risks.append(compute_risk_score(row))
    risks = np.array(risks)
    bands = {
        "low":      int(((risks < 0.25)).sum()),
        "moderate": int(((risks >= 0.25) & (risks < 0.55)).sum()),
        "high":     int(((risks >= 0.55) & (risks < 0.80)).sum()),
        "critical": int((risks >= 0.80).sum()),
    }
    return (f"Cohort has {n} patients. Avg age {avg_age:.1f}, "
            f"{pct_female:.1f}% female. Risk distribution: {bands}. "
            f"Mean risk={risks.mean():.3f}, max={risks.max():.3f}.")


def list_diseases() -> str:
    return "Available diseases: " + ", ".join(
        f"{k} ({v.name})" for k, v in DISEASE_FORCINGS.items())


def list_interventions() -> str:
    return "Available interventions: " + ", ".join(INTERVENTIONS.keys())


# --- tool registry ------------------------------------------------------
TOOL_REGISTRY = {
    "get_patient_summary": {
        "fn": get_patient_summary,
        "description": "Look up a single patient's demographics, biomarkers, and risk band. Use when the user asks about a specific patient_id.",
        "args": {"patient_id": "string, e.g. 'P000001'"},
    },
    "find_similar_patients": {
        "fn": find_similar_patients,
        "description": "Find the k most similar patients to a given patient using the trained GNN embeddings.",
        "args": {"patient_id": "string", "k": "integer, default 5"},
    },
    "simulate_disease": {
        "fn": simulate_disease,
        "description": "Run a forward disease-trajectory simulation for a patient. Returns how their biomarkers and risk evolve over time.",
        "args": {
            "patient_id": "string",
            "disease": "one of t2d, hypertension, cvd, copd",
            "horizon_days": "integer, default 180",
            "intervention": "optional intervention name (metformin, losartan, statin, exercise_30m, weight_loss, smoking_cessation)",
        },
    },
    "counterfactual_for_patient": {
        "fn": counterfactual_for_patient,
        "description": "Answer 'what would happen to this patient if we set treatment to value?' using the 3-step SCM-based counterfactual (abduction -> do -> prediction).",
        "args": {
            "patient_id": "string",
            "treatment": "biomarker id, e.g. 'glucose' or 'systolic_bp'",
            "outcome": "biomarker id to predict, e.g. 'hrv'",
            "value": "numeric do-value, default 1.0",
        },
    },
    "estimate_treatment_effect": {
        "fn": estimate_treatment_effect,
        "description": "Estimate the Average Treatment Effect (ATE) of a treatment on an outcome across the cohort using backdoor adjustment.",
        "args": {"treatment": "biomarker column", "outcome": "biomarker column"},
    },
    "estimate_heterogeneous_effect": {
        "fn": estimate_heterogeneous_effect,
        "description": "Estimate Conditional ATE (CATE) — how the treatment effect varies across patients (e.g. by BMI quartile).",
        "args": {"treatment": "biomarker column", "outcome": "biomarker column",
                 "modifier": "effect modifier, default 'bmi'"},
    },
    "cohort_overview": {
        "fn": cohort_overview,
        "description": "Get aggregate statistics for the whole patient cohort (count, demographics, risk distribution).",
        "args": {},
    },
    "list_diseases": {
        "fn": list_diseases,
        "description": "List the disease models that can be simulated.",
        "args": {},
    },
    "list_interventions": {
        "fn": list_interventions,
        "description": "List the named interventions (drugs, lifestyle changes) that can be applied in a simulation.",
        "args": {},
    },
}
