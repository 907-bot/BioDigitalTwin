"""
Phase 3 — Disease dynamical system.

We model disease progression as a coupled ODE/SNN system where each
biomarker is a slow-timescale variable driven by:

    dx_i/dt = f_i(x) - lambda_i * (x_i - x_i_baseline) + sigma_i * eta(t)
              ─────────   ──────────────────────────   ───────────
              forcing     homeostatic pull             noise
              (e.g.       (returns to healthy          (process
               BMI→glu)    range if no driver)          noise)

For the LIF/SNN view, we discretise the ODE with Euler and emit a
"spike" whenever a biomarker crosses a clinical threshold. The spike
train is the temporal fingerprint of the disease trajectory.

Disease attractors:
  - "healthy"     (low risk, all biomarkers in range)
  - "preclinical" (sub-threshold drift, e.g. impaired fasting glucose)
  - "clinical"    (overt disease: T2D / hypertension / CVD)
  - "decompensated" (multi-organ, high risk)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from app.dynamics.lif_neuron import BiologicalLIFNeuron
from app.graph.ontology import BIOMARKERS


# --- time-scale conversion -------------------------------------------
# 1 simulator step = `dt_hours`. The "spike" view is on a fast timescale
# (1ms), the ODE view is on a slow timescale (hours-days).
def step_to_spike_timestep(dt_hours: float, dt_ms: float = 1.0) -> int:
    return max(1, int((dt_hours * 3600 * 1000) / dt_ms))


# --- disease forcing models ------------------------------------------
@dataclass
class Forcing:
    name: str
    rate_fn: Callable[[dict[str, float]], dict[str, float]]
    """Given current state {biomarker: value}, return {biomarker: forcing}.
    Units of forcing = (biomarker unit) / hour."""


def forcing_t2d(state: dict[str, float]) -> dict[str, float]:
    """Type 2 diabetes forcing: BMI drives glucose; glucose degrades HRV."""
    bmi = state.get("bmi", 26.5)
    glu = state.get("glucose", 100)
    out = {
        "glucose": 0.05 * max(0.0, bmi - 25) - 0.02 * max(0.0, glu - 100),
        "bmi":     0.0,
        "hrv":    -0.03 * max(0.0, glu - 110),
    }
    return out


def forcing_hypertension(state: dict[str, float]) -> dict[str, float]:
    """Age + BMI driven BP increase."""
    bmi = state.get("bmi", 26.5)
    out = {
        "systolic_bp":  0.04 * max(0.0, bmi - 25) + 0.02 * max(0.0, bmi - 30),
        "diastolic_bp": 0.02 * max(0.0, bmi - 25),
    }
    return out


def forcing_cvd(state: dict[str, float]) -> dict[str, float]:
    """BP + glucose jointly damage vasculature -> HRV drop."""
    sbp = state.get("systolic_bp", 120)
    glu = state.get("glucose", 100)
    out = {
        "hrv":     -0.05 * max(0.0, sbp - 130) - 0.04 * max(0.0, glu - 110),
        "spo2":    -0.01 * max(0.0, sbp - 150),
    }
    return out


def forcing_copd(state: dict[str, float]) -> dict[str, float]:
    """Mild spo2 decline (in absence of smoking info, just age-driven)."""
    return {
        "spo2": -0.005,
    }


DISEASE_FORCINGS: dict[str, Forcing] = {
    "t2d":          Forcing("Type 2 diabetes",        forcing_t2d),
    "hypertension": Forcing("Hypertension",           forcing_hypertension),
    "cvd":          Forcing("Cardiovascular disease", forcing_cvd),
    "copd":         Forcing("COPD",                   forcing_copd),
}

ALL_BIOMARKER_NAMES = [b.id for b in BIOMARKERS]


# --- disease attractor definition ------------------------------------
@dataclass
class Attractor:
    name: str
    description: str
    risk_lo: float
    risk_hi: float


ATTRACTORS: list[Attractor] = [
    Attractor("healthy",       "All biomarkers in healthy range",                  0.00, 0.25),
    Attractor("preclinical",   "Sub-threshold drift, e.g. impaired fasting glu",  0.25, 0.55),
    Attractor("clinical",      "Overt disease state",                              0.55, 0.80),
    Attractor("decompensated", "Multi-organ involvement, critical",               0.80, 1.01),
]


def classify_risk(score: float) -> str:
    for a in ATTRACTORS:
        if a.risk_lo <= score < a.risk_hi:
            return a.name
    return "decompensated"


# --- interventions ----------------------------------------------------
INTERVENTIONS: dict[str, dict] = {
    "metformin":     {"glucose": -8.0,  "bmi": -0.3},
    "losartan":      {"systolic_bp": -6.0, "diastolic_bp": -3.0},
    "statin":        {"systolic_bp": -2.0, "glucose": -1.0},
    "exercise_30m":  {"hrv": +3.0, "systolic_bp": -2.0, "glucose": -4.0, "bmi": -0.1},
    "weight_loss":   {"bmi": -1.5, "glucose": -10.0, "systolic_bp": -5.0, "hrv": +2.0},
    "smoking_cessation": {"spo2": +0.4, "hrv": +2.0},
}


# --- core simulator ---------------------------------------------------
class DiseaseSimulator:
    """
    Forward-Euler simulator of coupled biomarker dynamics.
    Optional LIF encoder at the end emits the spike train view.
    """

    def __init__(self, dt_hours: float = 6.0, sigma: float = 0.05) -> None:
        self.dt_hours = dt_hours
        self.sigma = sigma
        self.biomarker_names = ALL_BIOMARKER_NAMES
        self.biomarker_index = {n: i for i, n in enumerate(self.biomarker_names)}
        self.baseline = {
            "hr": 72.0, "hrv": 45.0, "spo2": 97.0, "glucose": 95.0,
            "systolic_bp": 120.0, "diastolic_bp": 78.0, "bmi": 24.0,
        }
        # homeostatic pull strengths (per hour)
        self.lam = {
            "hr": 0.05, "hrv": 0.05, "spo2": 0.20, "glucose": 0.04,
            "systolic_bp": 0.04, "diastolic_bp": 0.04, "bmi": 0.005,
        }

    def _risk_from_state(self, state: dict[str, float]) -> float:
        from app.graph.builder import _normalise
        deltas = []
        for name in self.biomarker_names:
            v = state.get(name, self.baseline[name])
            b = next(b for b in BIOMARKERS if b.id == name)
            deltas.append(abs(_normalise(v, b.healthy_lo, b.healthy_hi)))
        return float(np.clip(np.mean(deltas), 0.0, 1.0))

    def simulate(
        self,
        initial_state: dict[str, float],
        disease: str,
        horizon_days: int,
        intervention: dict | None = None,
        rng_seed: int = 0,
    ) -> dict:
        rng = np.random.default_rng(rng_seed)
        forcing = DISEASE_FORCINGS.get(disease)
        if forcing is None:
            raise ValueError(f"Unknown disease: {disease}")

        # intervention: {biomarker: delta_per_day}
        daily_intervention = intervention or {}

        steps = max(1, int((horizon_days * 24) / self.dt_hours))
        state = {k: float(v) for k, v in initial_state.items()}

        times_h = []
        risks = []
        series = {n: [] for n in self.biomarker_names}

        for s in range(steps):
            t_h = s * self.dt_hours
            times_h.append(t_h)

            f = forcing.rate_fn(state)
            for name in self.biomarker_names:
                v = state.get(name, self.baseline[name])
                # homeostatic pull toward baseline
                homeo = -self.lam[name] * (v - self.baseline[name])
                drive = f.get(name, 0.0)
                # intervention scales by dt_hours/24 (rate is per day)
                interv = sum(delta for k, delta in daily_intervention.items() if k == name) * (self.dt_hours / 24.0)
                noise = self.sigma * rng.normal() * (v * 0.01)
                v_new = v + (homeo + drive) * self.dt_hours + interv + noise
                # physiological clamps
                v_new = max(0.0, v_new)
                state[name] = v_new
                series[name].append(v_new)

            risk = self._risk_from_state(state)
            risks.append(risk)

        # final spike-train view (LIF) for the dominant biomarker
        dominant = "glucose" if disease == "t2d" else \
                   "systolic_bp" if disease in ("hypertension", "cvd") else "spo2"
        lif = BiologicalLIFNeuron(n_features=1)
        # resample state to 1ms steps in chunks of 50ms windows, then run LIF
        # For tractability we run a small T window on the normalised residual
        T = 200
        signal = np.array(series[dominant]) - self.baseline[dominant]
        signal = signal / (np.std(signal) + 1e-6)  # z-score
        # take last T points (or pad)
        if len(signal) < T:
            signal = np.pad(signal, (T - len(signal), 0))
        else:
            signal = signal[-T:]
        I = torch.tensor(signal, dtype=torch.float32).view(T, 1, 1) * 5.0
        with torch.no_grad():
            V, S = lif(I)
        spike_count = int(S.sum().item())
        spike_rate_hz = spike_count / (T * 0.001)  # 1ms timesteps

        return {
            "times_h": times_h,
            "risks": risks,
            "series": series,
            "final_risk": float(risks[-1]),
            "final_state": {n: state[n] for n in self.biomarker_names},
            "disease_state": classify_risk(risks[-1]),
            "spike_count": spike_count,
            "spike_rate_hz": round(spike_rate_hz, 2),
            "lif_dominant_biomarker": dominant,
        }


# --- bifurcation helper ----------------------------------------------
def bifurcation_summary(disease: str) -> dict:
    """Return the parameter ranges where the disease bifurcates between
    attractors. Useful for the Phase 3 explainability endpoint."""
    if disease == "t2d":
        return {
            "bifurcation_param": "bmi",
            "healthy":  [18.5, 25.0],
            "preclinical": [25.0, 30.0],
            "clinical":    [30.0, 35.0],
            "decompensated": [35.0, 45.0],
        }
    if disease == "hypertension":
        return {
            "bifurcation_param": "systolic_bp",
            "healthy":  [90, 130],
            "preclinical": [130, 140],
            "clinical":    [140, 160],
            "decompensated": [160, 180],
        }
    if disease == "cvd":
        return {
            "bifurcation_param": "systolic_bp",
            "healthy":  [90, 130],
            "preclinical": [130, 150],
            "clinical":    [150, 170],
            "decompensated": [170, 200],
        }
    if disease == "copd":
        return {
            "bifurcation_param": "spo2",
            "healthy":  [95, 100],
            "preclinical": [92, 95],
            "clinical":    [88, 92],
            "decompensated": [80, 88],
        }
    return {}
