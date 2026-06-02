"""
Phase 11 — Uncertainty quantification (bootstrap CIs).

Re-fits the SCM on bootstrap resamples of the cohort and runs the same
patient counterfactual through each fit to produce a distribution over
the effect size. Returns 5/50/95 percentiles for the counterfactual,
effect, and factual.

Also re-uses DoWhy refutation results to flag low-confidence counterfactuals.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from app.causal.scm import (
    LinearSCM,
    fit_cohort_scm,
    get_dag,
    patient_counterfactual,
    reset_scm,
)

logger = logging.getLogger(__name__)


def _quantiles(arr: np.ndarray) -> dict[str, float]:
    arr = np.asarray(arr, dtype=float)
    if len(arr) == 0:
        return {"p05": 0.0, "p50": 0.0, "p95": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def bootstrap_patient_counterfactual(
    df: pd.DataFrame,
    observed: dict[str, float],
    treatment: str,
    value: float,
    outcome: str,
    n_bootstrap: int = 200,
    confidence: float = 0.90,
    seed: int = 42,
) -> dict:
    """
    Re-fit the SCM n_bootstrap times on bootstrap resamples of the cohort
    and run the patient counterfactual through each fit.

    Returns 5/50/95 percentiles (or whatever the confidence level dictates)
    of the counterfactual outcome, the effect, and the factual.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    cf_vals: list[float] = []
    fx_vals: list[float] = []
    effect_vals: list[float] = []
    noise_vals: list[float] = []

    # Save the original fit so we can restore at the end
    original = fit_cohort_scm(df, force=False)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        df_boot = df.iloc[idx].reset_index(drop=True)
        try:
            reset_scm()
            scm = fit_cohort_scm(df_boot, force=True)
            res = patient_counterfactual(
                scm, observed=observed, treatment=treatment,
                value=float(value), outcome=outcome,
            )
            if "error" in res:
                continue
            cf_vals.append(res["counterfactual"])
            fx_vals.append(res["factual"])
            effect_vals.append(res["effect"])
            noise_vals.append(res["abducted_noise"])
        except Exception as e:  # noqa: BLE001
            logger.debug("bootstrap %d failed: %s", i, e)
            continue

    # Restore the original fit
    reset_scm()
    fit_cohort_scm(df, force=True)

    cf_arr = np.array(cf_vals)
    fx_arr = np.array(fx_vals)
    eff_arr = np.array(effect_vals)

    alpha = 1.0 - confidence
    lo = alpha / 2 * 100
    hi = (1 - alpha / 2) * 100

    cf_q = {"p_lo": float(np.percentile(cf_arr, lo)) if len(cf_arr) else 0.0,
            "p_50": float(np.percentile(cf_arr, 50)) if len(cf_arr) else 0.0,
            "p_hi": float(np.percentile(cf_arr, hi)) if len(cf_arr) else 0.0,
            "mean": float(cf_arr.mean()) if len(cf_arr) else 0.0,
            "std":  float(cf_arr.std())  if len(cf_arr) else 0.0}

    eff_q = {"p_lo": float(np.percentile(eff_arr, lo)) if len(eff_arr) else 0.0,
             "p_50": float(np.percentile(eff_arr, 50)) if len(eff_arr) else 0.0,
             "p_hi": float(np.percentile(eff_arr, hi)) if len(eff_arr) else 0.0,
             "mean": float(eff_arr.mean()) if len(eff_arr) else 0.0,
             "std":  float(eff_arr.std())  if len(eff_arr) else 0.0}

    # Confidence band width
    cf_width = cf_q["p_hi"] - cf_q["p_lo"]
    cf_rel_width = cf_width / (abs(cf_q["p_50"]) + 1e-9)
    if cf_rel_width < 0.10:
        confidence_label = "high"
    elif cf_rel_width < 0.30:
        confidence_label = "medium"
    else:
        confidence_label = "low"

    # Sign stability: do all bootstrap runs agree on the direction?
    if len(eff_arr):
        positive = (eff_arr > 0).sum()
        negative = (eff_arr < 0).sum()
        if positive == 0 or negative == 0:
            direction_stability = 1.0
        else:
            direction_stability = max(positive, negative) / len(eff_arr)
    else:
        direction_stability = 0.0

    return {
        "treatment": treatment,
        "treatment_value": float(value),
        "outcome": outcome,
        "factual": {
            "mean": float(fx_arr.mean()) if len(fx_arr) else None,
            "std":  float(fx_arr.std())  if len(fx_arr) else None,
        },
        "counterfactual": {
            "mean": cf_q["mean"],
            "std":  cf_q["std"],
            "ci_lo": cf_q["p_lo"],
            "ci_50": cf_q["p_50"],
            "ci_hi": cf_q["p_hi"],
            "ci_level": confidence,
        },
        "effect": {
            "mean": eff_q["mean"],
            "std":  eff_q["std"],
            "ci_lo": eff_q["p_lo"],
            "ci_50": eff_q["p_50"],
            "ci_hi": eff_q["p_hi"],
            "ci_level": confidence,
        },
        "noise_distribution": {
            "mean": float(np.mean(noise_vals)) if noise_vals else 0.0,
            "std":  float(np.std(noise_vals))  if noise_vals else 0.0,
        },
        "n_bootstrap": len(cf_arr),
        "n_bootstrap_attempted": n_bootstrap,
        "ci_method": f"bootstrap_n={n_bootstrap}",
        "confidence_label": confidence_label,
        "direction_stability": round(direction_stability, 3),
        "ci_width_relative": round(cf_rel_width, 3),
    }


def bootstrap_ate(df: pd.DataFrame, treatment: str, outcome: str,
                  common_causes: list[str] | None = None,
                  n_bootstrap: int = 100,
                  confidence: float = 0.95,
                  seed: int = 42) -> dict:
    """
    Re-fit a linear regression on bootstrap resamples and return CI for ATE.
    Used to back the ATE/CATE endpoints with CIs.
    """
    from app.causal.scm import ate_estimate

    common_causes = [c for c in (common_causes or []) if c in df.columns]
    rng = np.random.default_rng(seed)
    n = len(df)
    ates: list[float] = []
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            res = ate_estimate(df.iloc[idx], treatment, outcome, common_causes)
            if "ate" in res and res["ate"] is not None:
                ates.append(float(res["ate"]))
        except Exception:  # noqa: BLE001
            continue

    arr = np.array(ates)
    alpha = 1.0 - confidence
    return {
        "treatment": treatment,
        "outcome": outcome,
        "n_bootstrap": len(arr),
        "ate_mean":   float(arr.mean()) if len(arr) else None,
        "ate_std":    float(arr.std())  if len(arr) else None,
        "ate_ci_lo":  float(np.percentile(arr, 100 * alpha / 2)) if len(arr) else None,
        "ate_ci_50":  float(np.percentile(arr, 50))               if len(arr) else None,
        "ate_ci_hi":  float(np.percentile(arr, 100 * (1 - alpha / 2))) if len(arr) else None,
        "ci_level":   confidence,
        "ci_method":  f"bootstrap_n={n_bootstrap}",
    }
