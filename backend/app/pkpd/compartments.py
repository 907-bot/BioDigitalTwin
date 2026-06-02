"""
2-compartment IV / 1st-order absorption PK solver.

State variables:
  A_gut  — drug in absorption compartment (mg)
  A_cen  — drug in central compartment (mg)
  A_per  — drug in peripheral compartment (mg)

ODEs:
  dA_gut/dt = -ka * A_gut                           (oral)
  dA_cen/dt = ka*A_gut - (CL/Vc + Q/Vc)*A_cen + (Q/Vp)*A_per
  dA_per/dt = (Q/Vc)*A_cen - (Q/Vp)*A_per

Concentration:
  C_central = A_cen / Vc                            (mg/L)

Industry-standard:
  - Allometric scaling on weight
  - Cockcroft-Gault eGFR on clearance
  - BSV (between-subject variability) on CL, Vc, ka
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp

logger = logging.getLogger(__name__)


@dataclass
class PKParams:
    """Population PK parameters for a drug (literature-derived)."""
    ka: float            # 1/h    — absorption rate constant
    CL: float            # L/h    — clearance
    Vc: float            # L      — central volume of distribution
    Vp: float            # L      — peripheral volume
    Q: float             # L/h    — inter-compartmental clearance
    F: float = 0.85      # oral bioavailability (1.0 for IV)
    route: str = "oral"  # "oral" or "iv"
    # BSV (omega, fraction; e.g. 0.4 means 40% CV)
    omega_ka: float = 0.30
    omega_CL: float = 0.30
    omega_Vc: float = 0.25
    omega_Vp: float = 0.25
    omega_Q:  float = 0.30


@dataclass
class PatientCovariates:
    age: float = 45.0         # years
    weight: float = 70.0       # kg
    sex: str = "M"             # "M" or "F"
    serum_creatinine: float = 1.0   # mg/dL  (for Cockcroft-Gault)
    height: float = 1.70       # m (used to estimate ideal body weight)


def allometric_scale(value: float, ref_weight: float, actual_weight: float,
                     exponent: float = 0.75) -> float:
    """Allometric scaling of PK parameters by body weight."""
    return value * (actual_weight / ref_weight) ** exponent


def cockcroft_gault_egfr(age: float, weight: float, sex: str,
                          serum_cr: float) -> float:
    """Cockcroft-Gault creatinine clearance (mL/min)."""
    crcl = ((140 - age) * weight) / (72 * serum_cr)
    if sex.upper() == "F":
        crcl *= 0.85
    return crcl


def adjust_for_renal(CL: float, crcl: float, frac_renal: float = 0.5) -> float:
    """
    Scale clearance by renal function. crcl in mL/min; normal ~100.
    frac_renal: fraction of total clearance that is renal (0..1).
    """
    renal_cl = CL * frac_renal
    non_renal_cl = CL * (1 - frac_renal)
    new_renal = renal_cl * (crcl / 100.0)
    return new_renal + non_renal_cl


def adjust_params_for_patient(params: PKParams, cov: PatientCovariates,
                              frac_renal: float = 0.5) -> PKParams:
    """Apply allometric + renal adjustment for a specific patient."""
    scaled = PKParams(
        ka=params.ka,
        CL=allometric_scale(params.CL, 70.0, cov.weight, exponent=0.75),
        Vc=allometric_scale(params.Vc, 70.0, cov.weight, exponent=1.0),
        Vp=allometric_scale(params.Vp, 70.0, cov.weight, exponent=1.0),
        Q=allometric_scale(params.Q, 70.0, cov.weight, exponent=0.75),
        F=params.F,
        route=params.route,
        omega_ka=params.omega_ka,
        omega_CL=params.omega_CL,
        omega_Vc=params.omega_Vc,
        omega_Vp=params.omega_Vp,
        omega_Q=params.omega_Q,
    )
    crcl = cockcroft_gault_egfr(cov.age, cov.weight, cov.sex, cov.serum_creatinine)
    scaled.CL = adjust_for_renal(scaled.CL, crcl, frac_renal)
    return scaled


def _ode_system(t, y, ka, CL, Vc, Vp, Q, F, route):
    A_gut, A_cen, A_per = y
    if route == "iv":
        dA_gut = 0.0
        A_input = 0.0
    else:
        dA_gut = -ka * A_gut
        A_input = ka * A_gut
    dA_cen = A_input - (CL / Vc) * A_cen - (Q / Vc) * A_cen + (Q / Vp) * A_per
    dA_per = (Q / Vc) * A_cen - (Q / Vp) * A_per
    return [dA_gut, dA_cen, dA_per]


@dataclass
class DosingRegimen:
    """A multi-dose schedule."""
    dose_mg: float
    n_doses: int
    interval_h: float            # q4h, q6h, q8h, q12h, q24h
    route: str = "oral"          # "oral" or "iv"
    infusion_h: float = 0.0      # for IV infusion duration (0 = bolus)


@dataclass
class PKResult:
    times_h: np.ndarray
    c_central: np.ndarray        # mg/L (= µg/mL)
    cmax: float
    tmax: float
    auc_0_t: float               # mg·h/L
    auc_0_inf: float
    half_life: float             # h
    clearance: float             # L/h
    vd_ss: float                 # L (steady-state Vd = Vc + Vp)
    cmin_ss: Optional[float] = None
    cmax_ss: Optional[float] = None
    accumulation_ratio: Optional[float] = None
    time_to_steady_state_h: Optional[float] = None
    params_used: dict = field(default_factory=dict)


def simulate_pk(params: PKParams, regimen: DosingRegimen,
                cov: Optional[PatientCovariates] = None,
                t_end_h: Optional[float] = None,
                n_points: int = 600,
                seed: int = 0) -> PKResult:
    """
    Run a single PK simulation.

    Handles both single-dose and multi-dose regimens. For multi-dose, also
    computes steady-state Cmax/Cmin and time-to-90%-steady-state.
    """
    rng = np.random.default_rng(seed)
    if cov is not None:
        p = adjust_params_for_patient(params, cov, frac_renal=_frac_renal(params))
    else:
        p = params

    # Add BSV (log-normal)
    ka = p.ka * np.exp(rng.normal(0, p.omega_ka))
    CL = p.CL * np.exp(rng.normal(0, p.omega_CL))
    Vc = p.Vc * np.exp(rng.normal(0, p.omega_Vc))
    Vp = p.Vp * np.exp(rng.normal(0, p.omega_Vp))
    Q  = p.Q  * np.exp(rng.normal(0, p.omega_Q))

    if t_end_h is None:
        t_end_h = regimen.n_doses * regimen.interval_h + 48.0

    t_eval = np.linspace(0, t_end_h, n_points)
    c = np.zeros_like(t_eval)
    init = [0.0, 0.0, 0.0]

    for d in range(regimen.n_doses):
        t0 = d * regimen.interval_h
        t1 = t_end_h
        if t0 >= t1:
            break
        if regimen.route == "iv" and regimen.infusion_h > 0:
            # For now, treat IV infusion as bolus at dose time (simplification)
            y0 = [0.0, init[1] + regimen.dose_mg * p.F, init[2]]
        else:
            gut_dose = regimen.dose_mg * p.F if regimen.route == "oral" else 0.0
            y0 = [init[0] + gut_dose, init[1], init[2]]

        sol = solve_ivp(
            _ode_system,
            (t0, t1),
            y0,
            args=(ka, CL, Vc, Vp, Q, p.F, regimen.route),
            t_eval=t_eval[t_eval >= t0],
            method="LSODA",
            rtol=1e-8,
            atol=1e-10,
        )
        if not sol.success:
            logger.warning("PK solver failed: %s", sol.message)
            continue
        c_t = sol.y[1] / Vc
        mask = t_eval >= t0
        c[mask] = c_t

    cmax = float(c.max())
    tmax = float(t_eval[c.argmax()])
    auc_0_t = float(np.trapz(c, t_eval))
    # AUC_0_inf ≈ AUC_0_t + C_last / ke; use last non-zero / k
    half_life = (np.log(2.0) * (Vc + Vp)) / CL if CL > 0 else float("inf")
    ke = CL / (Vc + Vp) if (Vc + Vp) > 0 else 0.0
    c_last = float(c[-1])
    auc_0_inf = auc_0_t + c_last / ke if ke > 0 else auc_0_t

    # Steady-state metrics for multi-dose
    cmin_ss = cmax_ss = accumulation_ratio = time_to_ss = None
    if regimen.n_doses >= 2:
        # Last full interval
        interval_mask = t_eval >= (regimen.n_doses - 1) * regimen.interval_h
        if interval_mask.sum() > 1:
            c_interval = c[interval_mask]
            cmax_ss = float(c_interval.max())
            cmin_ss = float(c_interval.min())
            # Single-dose Cmax from first dose (rough)
            first_mask = t_eval < regimen.interval_h
            if first_mask.sum() > 1:
                cmax_first = float(c[first_mask].max())
                if cmax_first > 0:
                    accumulation_ratio = cmax_ss / cmax_first
        # Time to 90% steady state: ~3.3 * half_life
        if half_life > 0:
            time_to_ss = 3.3 * half_life

    return PKResult(
        times_h=t_eval,
        c_central=c,
        cmax=cmax,
        tmax=tmax,
        auc_0_t=auc_0_t,
        auc_0_inf=auc_0_inf,
        half_life=half_life,
        clearance=CL,
        vd_ss=Vc + Vp,
        cmin_ss=cmin_ss,
        cmax_ss=cmax_ss,
        accumulation_ratio=accumulation_ratio,
        time_to_steady_state_h=time_to_ss,
        params_used={"ka": ka, "CL": CL, "Vc": Vc, "Vp": Vp, "Q": Q, "F": p.F},
    )


def _frac_renal(params: PKParams) -> float:
    """
    Fraction of clearance that is renal — drug class heuristic.
    Used by adjust_params_for_patient.
    """
    return 0.5  # default; per-drug overrides happen in registry


def population_simulation(params: PKParams, regimen: DosingRegimen,
                          n_subjects: int = 50,
                          seed: int = 0) -> list[PKResult]:
    """Run a virtual population (Monte Carlo PK)."""
    rng = np.random.default_rng(seed)
    results = []
    for i in range(n_subjects):
        # Sample a typical covariate set
        cov = PatientCovariates(
            age=float(rng.normal(50, 15)),
            weight=float(rng.normal(72, 15)),
            sex="M" if rng.random() < 0.5 else "F",
            serum_creatinine=float(rng.normal(1.0, 0.25)),
        )
        results.append(simulate_pk(params, regimen, cov, seed=seed + i))
    return results


def population_summary(results: list[PKResult]) -> dict:
    """5/50/95 percentiles for Cmax and AUC."""
    arr_cmax = np.array([r.cmax for r in results])
    arr_auc  = np.array([r.auc_0_inf for r in results])
    arr_t12  = np.array([r.half_life for r in results])
    return {
        "n_subjects": len(results),
        "cmax_p05": float(np.percentile(arr_cmax, 5)),
        "cmax_p50": float(np.percentile(arr_cmax, 50)),
        "cmax_p95": float(np.percentile(arr_cmax, 95)),
        "auc_p05":  float(np.percentile(arr_auc, 5)),
        "auc_p50":  float(np.percentile(arr_auc, 50)),
        "auc_p95":  float(np.percentile(arr_auc, 95)),
        "half_life_p50": float(np.percentile(arr_t12, 50)),
    }
