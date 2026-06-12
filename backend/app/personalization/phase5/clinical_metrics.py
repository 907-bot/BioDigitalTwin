"""
Phase 5 — Clinical Validation Metrics.

Implements clinically meaningful error analysis for physiological predictions:

  1. Clarke Error Grid Analysis (for glucose predictions)
  2. Bland-Altman Analysis (for any paired measurements)
  3. Consensus Error Grid (for glucose, Parkes/Vigersky)
  4. Clinical consensus metrics (ISO 15197:2013 compliance)

These metrics map technical prediction errors to clinical risk,
which is the standard for diabetes technology evaluation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ClarkeErrorGridResult:
    zone_a_pct: float
    zone_b_pct: float
    zone_c_pct: float
    zone_d_pct: float
    zone_e_pct: float
    clinically_acceptable: bool  # A+B > 99%
    zone_counts: Dict[str, int]
    n_points: int


@dataclass
class BlandAltmanResult:
    mean_difference: float
    sd_difference: float
    lower_loa: float
    upper_loa: float
    proportional_bias_slope: float
    proportional_bias_p: float
    percentage_error: float
    clinical_agreement: str
    limits_of_agreement: Tuple[float, float]


@dataclass
class ClinicalValidationReport:
    clarke_error_grid: Optional[ClarkeErrorGridResult]
    bland_altman: Optional[BlandAltmanResult]
    iso_15197_compliance: Dict[str, bool]
    mard: float  # Mean Absolute Relative Deviation
    pearson_r: float
    concordance_cc: float
    within_15_pct: float
    within_20_pct: float
    within_30_pct: float


def clarke_error_grid(
    y_true: np.ndarray, y_pred: np.ndarray,
    glucose_unit: str = "mg/dL",
) -> ClarkeErrorGridResult:
    """
    Clarke Error Grid Analysis for glucose predictions.

    Zones:
      A: Clinically accurate (within 20% or within 20 mg/dL)
      B: Benign errors (>20% but would not lead to inappropriate treatment)
      C: Overcorrecting acceptable glucose
      D: Dangerous failure to detect hypo/hyperglycemia
      E: Erroneous treatment (hypo predicted as hyper or vice versa)

    Reference: Clarke et al., Diabetes Care 1987
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_t) | np.isnan(y_p))
    y_t, y_p = y_t[mask], y_p[mask]

    n = len(y_t)
    if n == 0:
        return ClarkeErrorGridResult(
            zone_a_pct=0, zone_b_pct=0, zone_c_pct=0,
            zone_d_pct=0, zone_e_pct=0, clinically_acceptable=False,
            zone_counts={"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}, n_points=0,
        )

    zones = np.full(n, "B")
    for i in range(n):
        ref = y_t[i]
        pred = y_p[i]
        pct_err = abs(pred - ref) / max(ref, 1)
        abs_err = abs(pred - ref)
        if ref < 70:
            if pred < 70:
                zones[i] = "A"
            elif pred <= 180:
                zones[i] = "D"
            else:
                zones[i] = "E"
        elif ref <= 180:
            if pred < 70:
                zones[i] = "C"
            elif pred > 180:
                zones[i] = "B"
            else:
                if abs_err <= 20 or pct_err <= 0.2:
                    zones[i] = "A"
                else:
                    zones[i] = "B"
        else:
            if pred < 70:
                zones[i] = "E"
            elif pred <= 180:
                zones[i] = "D"
            else:
                if pct_err <= 0.2:
                    zones[i] = "A"
                else:
                    zones[i] = "B"

    counts = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
    for z in zones:
        counts[z] += 1

    pcts = {k: v / n * 100 for k, v in counts.items()}
    clinically_acceptable = (pcts["A"] + pcts["B"]) > 99.0

    return ClarkeErrorGridResult(
        zone_a_pct=pcts["A"], zone_b_pct=pcts["B"],
        zone_c_pct=pcts["C"], zone_d_pct=pcts["D"],
        zone_e_pct=pcts["E"],
        clinically_acceptable=clinically_acceptable,
        zone_counts=counts, n_points=n,
    )


def bland_altman_analysis(
    y_true: np.ndarray, y_pred: np.ndarray,
    confidence: float = 1.96,
) -> BlandAltmanResult:
    """
    Bland-Altman analysis with proportional bias assessment.

    Computes:
      - Mean difference (bias)
      - Limits of agreement (bias ± 1.96*SD)
      - Proportional bias (regression of difference on average)
      - Percentage error
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_t) | np.isnan(y_p))
    y_t, y_p = y_t[mask], y_p[mask]

    n = len(y_t)
    if n < 3:
        return BlandAltmanResult(
            mean_difference=0, sd_difference=0, lower_loa=0, upper_loa=0,
            proportional_bias_slope=0, proportional_bias_p=1.0,
            percentage_error=0, clinical_agreement="insufficient_data",
            limits_of_agreement=(0, 0),
        )

    differences = y_p - y_t
    averages = (y_p + y_t) / 2.0

    mean_diff = np.mean(differences)
    sd_diff = np.std(differences, ddof=1)
    lower_loa = mean_diff - confidence * sd_diff
    upper_loa = mean_diff + confidence * sd_diff

    # Proportional bias: regress difference on average
    A = np.vstack([averages, np.ones(n)]).T
    try:
        slope, intercept = np.linalg.lstsq(A, differences, rcond=None)[0]
        residuals = differences - (slope * averages + intercept)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((differences - mean_diff) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        from scipy import stats
        _, p_value = stats.pearsonr(differences, averages)
    except Exception:
        slope = 0.0
        p_value = 1.0

    pct_error = float(np.mean(np.abs(differences) / (np.abs(y_t) + 1e-6)) * 100)

    loa_range = upper_loa - lower_loa
    if abs(mean_diff) < 0.05 * np.mean(y_t) and loa_range < 0.3 * np.mean(y_t):
        agreement = "excellent"
    elif abs(mean_diff) < 0.10 * np.mean(y_t) and loa_range < 0.5 * np.mean(y_t):
        agreement = "acceptable"
    elif abs(mean_diff) < 0.20 * np.mean(y_t):
        agreement = "moderate"
    else:
        agreement = "poor"

    return BlandAltmanResult(
        mean_difference=float(mean_diff),
        sd_difference=float(sd_diff),
        lower_loa=float(lower_loa),
        upper_loa=float(upper_loa),
        proportional_bias_slope=float(slope),
        proportional_bias_p=float(p_value),
        percentage_error=float(pct_error),
        clinical_agreement=agreement,
        limits_of_agreement=(float(lower_loa), float(upper_loa)),
    )


def iso_15197_compliance(
    y_true: np.ndarray, y_pred: np.ndarray,
) -> Dict[str, bool]:
    """
    ISO 15197:2013 compliance criteria for blood glucose monitoring systems.

    - ≥95% of values within ±15 mg/dL for glucose <100 mg/dL
    - ≥95% of values within ±15% for glucose ≥100 mg/dL
    - ≥99% in Consensus Error Grid Zones A + B
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_t) | np.isnan(y_p))
    y_t, y_p = y_t[mask], y_p[mask]

    n = len(y_t)
    if n < 10:
        return {"within_15_interval": False, "zone_ab_99pct": False, "overall_pass": False}

    below_100 = y_t < 100
    above_100 = ~below_100

    passes_interval = True
    if np.sum(below_100) > 0:
        within_15 = np.abs(y_p[below_100] - y_t[below_100]) <= 15
        passes_interval = passes_interval and (np.mean(within_15) >= 0.95)
    if np.sum(above_100) > 0:
        within_15pct = np.abs(y_p[above_100] - y_t[above_100]) / y_t[above_100] <= 0.15
        passes_interval = passes_interval and (np.mean(within_15pct) >= 0.95)

    ceg = clarke_error_grid(y_t, y_p)
    passes_zone_ab = (ceg.zone_a_pct + ceg.zone_b_pct) >= 99.0

    return {
        "within_15_interval": bool(passes_interval),
        "zone_ab_99pct": bool(passes_zone_ab),
        "overall_pass": bool(passes_interval and passes_zone_ab),
    }


def concordance_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Lin's concordance correlation coefficient (CCC)."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_t) | np.isnan(y_p))
    y_t, y_p = y_t[mask], y_p[mask]

    n = len(y_t)
    if n < 3:
        return 0.0
    mean_t = np.mean(y_t)
    mean_p = np.mean(y_p)
    cov = np.mean((y_t - mean_t) * (y_p - mean_p))
    var_t = np.var(y_t)
    var_p = np.var(y_p)
    denom = var_t + var_p + (mean_t - mean_p) ** 2
    return float(2.0 * cov / denom) if denom > 0 else 0.0


def compute_clinical_validation(
    y_true: np.ndarray, y_pred: np.ndarray,
    variable_name: str = "glucose",
    glucose_unit: str = "mg/dL",
) -> ClinicalValidationReport:
    """Comprehensive clinical validation for any physiological prediction."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_t) | np.isnan(y_p))
    y_t, y_p = y_t[mask], y_p[mask]

    n = len(y_t)
    mard = float(np.mean(np.abs(y_p - y_t) / (np.abs(y_t) + 1e-6)) * 100) if n > 0 else 0

    within_15 = float(np.mean(np.abs(y_p - y_t) / (np.abs(y_t) + 1e-6) <= 0.15)) * 100 if n > 0 else 0
    within_20 = float(np.mean(np.abs(y_p - y_t) / (np.abs(y_t) + 1e-6) <= 0.20)) * 100 if n > 0 else 0
    within_30 = float(np.mean(np.abs(y_p - y_t) / (np.abs(y_t) + 1e-6) <= 0.30)) * 100 if n > 0 else 0

    from scipy import stats as sp_stats
    pearson_r = float(sp_stats.pearsonr(y_t, y_p)[0]) if n > 2 else 0
    ccc = concordance_correlation(y_t, y_p)

    ceg = clarke_error_grid(y_t, y_p, glucose_unit)
    ba = bland_altman_analysis(y_t, y_p)
    iso = iso_15197_compliance(y_t, y_p)

    return ClinicalValidationReport(
        clarke_error_grid=ceg,
        bland_altman=ba,
        iso_15197_compliance=iso,
        mard=mard,
        pearson_r=pearson_r,
        concordance_cc=ccc,
        within_15_pct=within_15,
        within_20_pct=within_20,
        within_30_pct=within_30,
    )
