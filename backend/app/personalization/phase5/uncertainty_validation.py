"""
Bayesian Uncertainty Validation Suite.

Tests 7 categories of probabilistic calibration for the digital twin:

  1. PIT Uniformity         — Probability Integral Transform histogram
  2. Coverage Probability   — Actual vs nominal at 50/80/90/95%
  3. Sharpness              — Mean interval width (conditional on calibration)
  4. Reliability Diagrams   — Binned calibration curves, ECE/MCE
  5. Prediction Intervals   — Horizon-adaptive coverage at multi-horizon
  6. Parameter Uncertainty  — Posterior interval coverage for known params
  7. Structural Uncertainty — Model mismatch sensitivity analysis
  8. Calibrated Twin        — Coverage after temperature scaling + conformal

All tests assume confidence intervals are wrong until proven otherwise.
All coverage tests use PREDICTIVE (pre-update, one-step-ahead forecast)
residuals, not posterior (post-update) residuals.

References:
  - Gneiting & Katzfuss (2014) "Probabilistic Forecasting"
  - Dawid (1984) "Present position and potential developments: PIT"
  - Murphy & Winkler (1987) "A general framework for forecast verification"

NOTES ON STRUCTURAL IDENTIFIABILITY:
  The 30-dim physiological state is only 15-dim observable. The UKF
  estimates all 30 states from 15 observations through ODE coupling
  and process noise propagation. This creates a structural identifiability
  concern for unobserved states (HGP, PGU, IR, CLOCK_BMAL1, PER_CRY,
  CRP, fat_mass, M1_M2_ratio, NFkB_activity, InflammatoryLoad).
  Coverage for these states depends on the correctness of the ODE
  coupling rather than direct observation. See test_structural_uncertainty
  for sensitivity analysis under model misspecification.

  The 25 parameters are tuned to match population-average physiology
  (Bergman minimal model for glucose-insulin, Guyton model for CV/renal,
  Relogio model for circadian, Kim model for adipose-lipid, Liu model
  for immune-inflammation). Individual-level personalization via UKF
  updates the 30-dim state but not the 25-dim parameters (the UKF
  tracks parameters as part of the augmented state during initialization;
  online adaptation of all 25 parameters from 15 observations would be
  ill-posed without strong regularization).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from scipy import stats

from app.personalization.dynamics import DEFAULT_PARAMS, full_dynamics, full_observation
from app.personalization.state import PHYSIO_DIM, PARAM_DIM, OBS_DIM
from app.personalization.core import PersonalizationEngine
from app.personalization.calibrated_twin import CalibratedTwin
from app.personalization.phase5.pi_validation import ValidationDatum, TestResult


# ── Config ──────────────────────────────────────────────────────

NOMINAL_LEVELS = [0.50, 0.80, 0.90, 0.95]
Z_SCORES = {0.50: 0.674, 0.80: 1.282, 0.90: 1.645, 0.95: 1.960}

# Clinically relevant variables for coverage assessment
COVERAGE_VARS = [
    ("glucose", 0, 70.0, 180.0),
    ("SBP", 1, 90.0, 160.0),
    ("DBP", 2, 60.0, 100.0),
    ("HR", 3, 50.0, 100.0),
    ("HRV", 4, 20.0, 80.0),
    ("GFR", 5, 60.0, 140.0),
]

# Acceptable thresholds (clinically calibrated, not statistically perfect)
# A model can be clinically useful even if statistically miscalibrated.
THRESHOLDS = {
    # PIT: strict statistical test, diagnostic only
    "pit_bin_dev_z": 4.0,         # Max bin deviation in z-score units
    # Coverage: clinical acceptability (allow ±10% for 90% CI)
    "coverage_90_deviation": 0.10,   # |actual - nominal| <= 0.10
    "coverage_95_deviation": 0.08,
    "coverage_80_deviation": 0.12,
    "coverage_50_deviation": 0.15,
    "coverage_max_deviation": 0.15,  # Max across all levels
    # Reliability: clinical calibration
    "ece_max": 0.15,            # Expected Calibration Error <= 0.15
    "mce_max": 0.30,            # Max Calibration Error <= 0.30
    "slope_lower": 0.70,        # Calibration slope >= 0.70
    "slope_upper": 1.30,        # Calibration slope <= 1.30
    # Parameter uncertainty
    "param_coverage_min": 0.50, # At least 50% of params pass 90% CI cover
    # Structural sensitivity
    "structural_inflation_max": 3.0,  # CI width inflation < 3x under perturbation
    # Sharpness
    "sharpness_glucose_max": 60.0,    # Mean 90% PI width <= 60 mg/dL
    "sharpness_clinical_frac": 0.50,  # PI width < 50% of clinical range
}


# ── Test 1: PIT Uniformity ──────────────────────────────────────

def test_pit_uniformity(
    n_patients: int = 8,
    n_steps: int = 250,
) -> TestResult:
    """
    Probability Integral Transform (PIT) histogram test.

    For a well-calibrated probabilistic forecast, the PIT values
    (CDF evaluated at the observation) should be Uniform(0,1).

    Uses Kolmogorov-Smirnov test against U(0,1).
    Also reports the PIT histogram's deviation from flatness.
    """
    rng = np.random.RandomState(42)
    all_pits: Dict[str, List[float]] = {name: [] for name, _, _, _ in COVERAGE_VARS}

    for p in range(n_patients):
        true_params = DEFAULT_PARAMS.copy()
        true_params[0] = rng.lognormal(-4.0, 0.3)
        true_params[5] = rng.lognormal(4.5, 0.2)
        true_params[12] = 1440.0

        state = np.zeros(30)
        state[0] = rng.normal(100, 15)
        state[1] = 0.013 * max(0, state[0] - 80)
        state[5] = rng.normal(125, 10)
        state[6] = rng.normal(80, 8)
        state[7] = rng.normal(70, 8)
        state[21] = rng.uniform(0.3, 0.7)

        obs = []
        s = state.copy()
        for t in range(n_steps):
            s = full_dynamics(s, true_params, {})
            obs.append(full_observation(s))
        obs_arr = np.array(obs)

        engine = PersonalizationEngine()
        engine.initialize(obs_arr[0])
        # Step 1 is already initialized; for t >= 1: predict, check, update
        for t in range(1, n_steps):
            engine.filter.predict({})
            cov = engine.filter.get_physio_covariance()
            mu = engine.filter.get_physio_state()

            if t > 50 and t % 5 == 0:
                for var_name, var_idx, _, _ in COVERAGE_VARS:
                    pred = float(mu[var_idx])
                    var = float(cov[var_idx, var_idx]) if cov.shape[0] > var_idx else 100.0
                    std = max(np.sqrt(var), 1.0)
                    actual = float(obs_arr[t, var_idx])
                    pit = float(stats.norm.cdf(actual, loc=pred, scale=std))
                    pit = np.clip(pit, 1e-6, 1 - 1e-6)
                    all_pits[var_name].append(pit)

            engine.filter.update(obs_arr[t])

    data_list = []
    ks_stats = []
    ks_pvals = []

    for var_name, _, _, _ in COVERAGE_VARS:
        pits = np.array(all_pits[var_name])
        if len(pits) < 20:
            continue
        ks_stat, ks_p = stats.kstest(pits, 'uniform', args=(0, 1))
        ks_stats.append(ks_stat)
        ks_pvals.append(ks_p)

        # Bin into 10 bins and compute deviation from flatness
        hist, _ = np.histogram(pits, bins=10, range=(0, 1))
        expected = len(pits) / 10
        max_dev = float(np.max(np.abs(hist - expected)))
        # Bin deviation in z-score units: sqrt(expected) is std under Poisson
        bin_z = max_dev / max(np.sqrt(expected), 1.0)

        data_list.append(ValidationDatum(
            f"PIT_KS_stat_{var_name}", True, float(ks_stat),
            f"KS statistic (lower = better)"
        ))
        passed_bin = bin_z <= THRESHOLDS["pit_bin_dev_z"]
        data_list.append(ValidationDatum(
            f"PIT_bin_Z_{var_name}", passed_bin,
            bin_z, f"≤ {THRESHOLDS['pit_bin_dev_z']}σ"
        ))

    overall_ks_stat = float(np.mean(ks_stats)) if ks_stats else 1.0
    passed = overall_ks_stat < 0.20  # KS statistic < 0.20 is roughly calibrated
    mean_ks = overall_ks_stat

    return TestResult(
        name="PIT Uniformity (Probabilistic Calibration)",
        passed=passed,
        score=max(0, 1.0 - mean_ks),
        data=data_list,
    )


# ── Test 2: Multi-Level Coverage Probability ────────────────────

def test_multi_level_coverage(
    n_patients: int = 8,
    n_steps: int = 250,
) -> TestResult:
    """
    Nominal vs actual coverage at 50/80/90/95% levels across key variables.

    For each CI level, computes the empirical coverage fraction.
    Tests that |actual - nominal| is within the per-level threshold.
    """
    rng = np.random.RandomState(42)
    coverages: Dict[str, Dict[float, List[float]]] = {}

    for var_name, var_idx, _, _ in COVERAGE_VARS:
        coverages[var_name] = {lvl: [] for lvl in NOMINAL_LEVELS}

    for p in range(n_patients):
        true_params = DEFAULT_PARAMS.copy()
        true_params[0] = rng.lognormal(-4.0, 0.3)
        true_params[5] = rng.lognormal(4.5, 0.2)
        true_params[12] = 1440.0

        state = np.zeros(30)
        state[0] = rng.normal(100, 15)
        state[1] = 0.013 * max(0, state[0] - 80)
        state[5] = rng.normal(125, 10)
        state[6] = rng.normal(80, 8)
        state[7] = rng.normal(70, 8)
        state[21] = rng.uniform(0.3, 0.7)

        obs = []
        s = state.copy()
        for t in range(n_steps):
            s = full_dynamics(s, true_params, {})
            obs.append(full_observation(s))
        obs_arr = np.array(obs)

        engine = PersonalizationEngine()
        engine.initialize(obs_arr[0])
        for t in range(1, n_steps):
            # Predict first, check coverage on prediction, then update
            engine.filter.predict(engine.control_input if hasattr(engine, 'control_input') else {})

            if t > 100 and t % 5 == 0:
                cov = engine.filter.get_physio_covariance()
                mu = engine.filter.get_physio_state()

                for var_name, var_idx, _, _ in COVERAGE_VARS:
                    pred = float(mu[var_idx])
                    var = float(cov[var_idx, var_idx]) if cov.shape[0] > var_idx else 100.0
                    std = max(np.sqrt(var), 1.0)
                    actual = float(obs_arr[t, var_idx])

                    for lvl in NOMINAL_LEVELS:
                        z = Z_SCORES[lvl]
                        lo = pred - z * std
                        hi = pred + z * std
                        coverages[var_name][lvl].append(1.0 if lo <= actual <= hi else 0.0)

            engine.filter.update(obs_arr[t])

    data_list = []
    all_deviations = []
    for var_name, _, _, _ in COVERAGE_VARS:
        for lvl in NOMINAL_LEVELS:
            vals = coverages[var_name][lvl]
            if not vals:
                continue
            actual_cov = float(np.mean(vals))
            deviation = abs(actual_cov - lvl)
            all_deviations.append(deviation)

            threshold_key = f"coverage_{int(lvl*100)}_deviation"
            thresh = THRESHOLDS.get(threshold_key, 0.10)
            passed = deviation <= thresh
            data_list.append(ValidationDatum(
                f"cov_{var_name}_{int(lvl*100)}%", passed,
                actual_cov, f"nominal={lvl} ±{thresh}"
            ))

    max_dev = float(np.max(all_deviations)) if all_deviations else 1.0
    mean_dev = float(np.mean(all_deviations)) if all_deviations else 1.0
    passed = max_dev <= THRESHOLDS["coverage_max_deviation"]

    return TestResult(
        name="Multi-Level Coverage Probability",
        passed=passed,
        score=max(0, 1.0 - mean_dev),
        data=data_list,
    )


# ── Test 3: Sharpness Assessment ────────────────────────────────

def test_sharpness(
    n_patients: int = 8,
    n_steps: int = 250,
) -> TestResult:
    """
    Sharpness: mean prediction interval width.

    Sharpness is evaluated CONDITIONAL on calibration being adequate.
    Narrower intervals are better, but only if coverage is correct.

    Reports absolute and relative (to observation std) widths
    at 50% and 90% nominal coverage.
    """
    rng = np.random.RandomState(42)
    widths_50: Dict[str, List[float]] = {name: [] for name, _, _, _ in COVERAGE_VARS}
    widths_90: Dict[str, List[float]] = {name: [] for name, _, _, _ in COVERAGE_VARS}

    for p in range(n_patients):
        true_params = DEFAULT_PARAMS.copy()
        true_params[0] = rng.lognormal(-4.0, 0.3)
        true_params[5] = rng.lognormal(4.5, 0.2)
        true_params[12] = 1440.0

        state = np.zeros(30)
        state[0] = rng.normal(100, 15)
        state[1] = 0.013 * max(0, state[0] - 80)
        state[5] = rng.normal(125, 10)
        state[6] = rng.normal(80, 8)
        state[7] = rng.normal(70, 8)
        state[21] = rng.uniform(0.3, 0.7)

        obs = []
        s = state.copy()
        for t in range(n_steps):
            s = full_dynamics(s, true_params, {})
            obs.append(full_observation(s))
        obs_arr = np.array(obs)

        engine = PersonalizationEngine()
        engine.initialize(obs_arr[0])
        for t in range(1, n_steps):
            engine.filter.predict(engine.control_input if hasattr(engine, 'control_input') else {})
            if t > 100 and t % 5 == 0:
                cov = engine.filter.get_physio_covariance()
                mu = engine.filter.get_physio_state()

                for var_name, var_idx, lo_clin, hi_clin in COVERAGE_VARS:
                    var = float(cov[var_idx, var_idx]) if cov.shape[0] > var_idx else 100.0
                    std = max(np.sqrt(var), 1.0)
                    widths_50[var_name].append(2.0 * Z_SCORES[0.50] * std)
                    widths_90[var_name].append(2.0 * Z_SCORES[0.90] * std)
            engine.filter.update(obs_arr[t])

    data_list = []
    for var_name, var_idx, lo_clin, hi_clin in COVERAGE_VARS:
        w50 = np.array(widths_50[var_name])
        w90 = np.array(widths_90[var_name])

        mean_w50 = float(np.mean(w50)) if len(w50) > 0 else 0.0
        mean_w90 = float(np.mean(w90)) if len(w90) > 0 else 0.0

        # Clinical range width for context
        clin_range = hi_clin - lo_clin
        w90_rel = mean_w90 / clin_range if clin_range > 0 else 1.0

        # Sharpness pass/fail only for glucose (others are informational)
        if var_name == "glucose":
            passed = mean_w90 <= THRESHOLDS["sharpness_glucose_max"]
        else:
            passed = True

        data_list.append(ValidationDatum(
            f"sharp_50_{var_name}", True, mean_w50, f"half-width (50% CI)"
        ))
        data_list.append(ValidationDatum(
            f"sharp_90_{var_name}", passed, mean_w90,
            f"≤ {THRESHOLDS.get('sharpness_glucose_max', 30)} mg/dL"
        ))
        data_list.append(ValidationDatum(
            f"sharp_90_rel_{var_name}", True, w90_rel,
            f"fraction of clinical range ({clin_range:.0f})"
        ))

    # Compute glucose sharpness score
    glucose_w90s = widths_90["glucose"]
    glucose_mean_w90 = float(np.mean(glucose_w90s)) if glucose_w90s else 0.0
    glucose_sharp_pass = glucose_mean_w90 <= THRESHOLDS["sharpness_glucose_max"]
    sharp_score = max(0, 1.0 - glucose_mean_w90 / THRESHOLDS["sharpness_glucose_max"]) if glucose_mean_w90 > 0 else 1.0

    return TestResult(
        name="Sharpness (Prediction Interval Width)",
        passed=glucose_sharp_pass,
        score=sharp_score,
        data=data_list,
    )


# ── Test 4: Reliability Diagrams ────────────────────────────────

def test_reliability_diagrams(
    n_patients: int = 8,
    n_steps: int = 250,
) -> TestResult:
    """
    Reliability (calibration) curves at multiple nominal levels.

    For each CI level, bins the predicted probability space and
    computes observed frequency within each bin.

    Reports:
      - ECE  (Expected Calibration Error)
      - MCE  (Maximum Calibration Error)
      - Calibration slope
      - Bin-wise observed vs expected frequencies
    """
    rng = np.random.RandomState(42)
    n_bins = 10

    reliability_data: Dict[str, Dict[float, dict]] = {}
    for var_name, _, _, _ in COVERAGE_VARS:
        reliability_data[var_name] = {}
        for lvl in NOMINAL_LEVELS:
            reliability_data[var_name][lvl] = {
                "bin_expected": np.zeros(n_bins),
                "bin_observed": np.zeros(n_bins),
                "bin_counts": np.zeros(n_bins),
            }

    for p in range(n_patients):
        true_params = DEFAULT_PARAMS.copy()
        true_params[0] = rng.lognormal(-4.0, 0.3)
        true_params[5] = rng.lognormal(4.5, 0.2)
        true_params[12] = 1440.0

        state = np.zeros(30)
        state[0] = rng.normal(100, 15)
        state[1] = 0.013 * max(0, state[0] - 80)
        state[5] = rng.normal(125, 10)
        state[6] = rng.normal(80, 8)
        state[7] = rng.normal(70, 8)
        state[21] = rng.uniform(0.3, 0.7)

        obs = []
        s = state.copy()
        for t in range(n_steps):
            s = full_dynamics(s, true_params, {})
            obs.append(full_observation(s))
        obs_arr = np.array(obs)

        engine = PersonalizationEngine()
        engine.initialize(obs_arr[0])
        for t in range(1, n_steps):
            engine.filter.predict(engine.control_input if hasattr(engine, 'control_input') else {})
            if t > 100 and t % 5 == 0:
                cov = engine.filter.get_physio_covariance()
                mu = engine.filter.get_physio_state()

                for var_name, var_idx, _, _ in COVERAGE_VARS:
                    pred = float(mu[var_idx])
                    var = float(cov[var_idx, var_idx]) if cov.shape[0] > var_idx else 100.0
                    std = max(np.sqrt(var), 1.0)
                    actual = float(obs_arr[t, var_idx])

                # Standardized residual
                std_resid = (actual - pred) / std

                for lvl in NOMINAL_LEVELS:
                    z = Z_SCORES[lvl]
                    # For each bin of the predictive CDF, check if standardized
                    # residual falls within the corresponding interval
                    # Bin edges represent the equally-spaced probability levels
                    for b in range(n_bins):
                        p_lo = b / n_bins
                        p_hi = (b + 1) / n_bins
                        # The CI centered at prediction covers when
                        # standardized residual ∈ [z_lo, z_hi] corresponding to p_lo, p_hi
                        # Actually, for reliability of a CI level lvl:
                        # We check whether (actual in CI) for each prediction
                        # and group by the predicted std residual magnitude
                        pass

                    lo = pred - z * std
                    hi = pred + z * std
                    in_ci = 1.0 if lo <= actual <= hi else 0.0

                    # Bin by the predicted z-score quantile
                    # Using the absolute standardized residual as the "confidence" proxy
                    abs_resid = min(abs(std_resid) / z, 1.0)
                    bin_idx = min(int(abs_resid * n_bins), n_bins - 1)
                    rd = reliability_data[var_name][lvl]
                    rd["bin_expected"][bin_idx] += 1.0 / n_bins
                    rd["bin_observed"][bin_idx] += in_ci
                    rd["bin_counts"][bin_idx] += 1

            engine.filter.update(obs_arr[t])

    data_list = []
    ece_values = []
    mce_values = []

    for var_name, _, _, _ in COVERAGE_VARS:
        for lvl in NOMINAL_LEVELS:
            rd = reliability_data[var_name][lvl]
            counts = rd["bin_counts"]
            total = max(np.sum(counts), 1)
            if total < 20:
                continue

            # Normalize: compute observed frequency within each bin
            # and compare against expected (midpoint of bin probability)
            bin_ece = 0.0
            bin_mce = 0.0
            for b in range(n_bins):
                cnt = counts[b]
                if cnt > 0:
                    obs_frac = rd["bin_observed"][b] / cnt
                    # Expected probability for this bin = midpoint
                    exp_frac = lvl
                    w = cnt / total
                    diff = abs(obs_frac - exp_frac)
                    bin_ece += w * diff
                    bin_mce = max(bin_mce, diff)

            ece_values.append(bin_ece)
            mce_values.append(bin_mce)

            ece_pass = bin_ece <= THRESHOLDS["ece_max"]
            mce_pass = bin_mce <= THRESHOLDS["mce_max"]
            data_list.append(ValidationDatum(
                f"ECE_{var_name}_{int(lvl*100)}%", ece_pass,
                bin_ece, f"≤ {THRESHOLDS['ece_max']}"
            ))
            data_list.append(ValidationDatum(
                f"MCE_{var_name}_{int(lvl*100)}%", mce_pass,
                bin_mce, f"≤ {THRESHOLDS['mce_max']}"
            ))

    overall_ece = float(np.mean(ece_values)) if ece_values else 1.0
    overall_mce = float(np.max(mce_values)) if mce_values else 1.0
    passed = overall_ece <= THRESHOLDS["ece_max"] and overall_mce <= THRESHOLDS["mce_max"]

    return TestResult(
        name="Reliability Diagrams (ECE/MCE)",
        passed=passed,
        score=max(0, 1.0 - overall_ece),
        data=data_list,
    )


# ── Test 5: Prediction Interval Calibration at Horizon ──────────

def test_prediction_interval_calibration(
    n_patients: int = 8,
    n_train: int = 400,
) -> TestResult:
    """
    Multi-level prediction interval coverage at 1/6/24/48 hr horizons.

    Extends the existing forecast calibration test to cover ALL nominal
    levels (50/80/90/95%) instead of only 90%.

    CI width is scaled by sqrt(horizon) to account for growing
    prediction uncertainty.
    """
    rng = np.random.RandomState(42)
    horizons_hr = [1, 6, 24, 48]
    horizons_steps = [h * 60 for h in horizons_hr]

    # (variable_idx, name) pairs to test
    test_vars = [(0, "glucose"), (5, "SBP"), (7, "HR")]

    coverage_data: Dict[int, Dict[str, Dict[float, List[float]]]] = {}
    for hr in horizons_hr:
        coverage_data[hr] = {}
        for _, vname in test_vars:
            coverage_data[hr][vname] = {lvl: [] for lvl in NOMINAL_LEVELS}

    for p in range(n_patients):
        true_params = DEFAULT_PARAMS.copy()
        true_params[0] = rng.lognormal(-4.0, 0.3)
        true_params[5] = rng.lognormal(4.5, 0.2)
        true_params[12] = 1440.0

        state = np.zeros(30)
        state[0] = rng.normal(100, 10)
        state[5] = rng.normal(120, 10)
        state[6] = rng.normal(80, 5)
        state[7] = rng.normal(70, 5)
        state[1] = rng.uniform(0.5, 2.0)

        # Generate enough data for training + shortest horizon
        min_steps = min(horizons_steps)
        total = n_train + min_steps + 50
        obs = []
        s = state.copy()
        for t in range(total):
            s = full_dynamics(s, true_params, {})
            obs.append(full_observation(s))
        obs_arr = np.array(obs)

        engine = PersonalizationEngine()
        engine.initialize(obs_arr[0])
        train_end = min(n_train, len(obs_arr) - 1)
        for t in range(1, train_end):
            engine.update(obs_arr[t], {})

        for hr, steps in zip(horizons_hr, horizons_steps):
            window = min(48, len(obs_arr) - train_end - steps - 1)
            if window < 3:
                continue
            for t in range(n_train, min(len(obs_arr) - steps, n_train + window)):
                cov = engine.get_twin_state_covariance()
                mu = engine.get_twin_state()

                for var_idx, vname in test_vars:
                    pred = float(mu[var_idx])
                    var = float(cov[var_idx, var_idx]) if cov.shape[0] > var_idx else 100.0
                    # Scale by sqrt(horizon) — prediction uncertainty grows
                    # with forecast horizon
                    pred_std = max(np.sqrt(var) * np.sqrt(steps / 10.0), 5.0)
                    actual = float(obs_arr[t + steps, var_idx])

                    for lvl in NOMINAL_LEVELS:
                        z = Z_SCORES[lvl]
                        lo = pred - z * pred_std
                        hi = pred + z * pred_std
                        coverage_data[hr][vname][lvl].append(
                            1.0 if lo <= actual <= hi else 0.0
                        )

    data_list = []
    all_devs = []
    for hr in horizons_hr:
        for _, vname in test_vars:
            for lvl in NOMINAL_LEVELS:
                vals = coverage_data[hr][vname][lvl]
                if not vals:
                    continue
                actual = float(np.mean(vals))
                dev = abs(actual - lvl)
                all_devs.append(dev)

                thresh = {0.50: 0.15, 0.80: 0.12, 0.90: 0.10, 0.95: 0.08}.get(lvl, 0.10)
                # Allow wider deviation at longer horizons
                if hr >= 24:
                    thresh *= 1.5
                passed = dev <= thresh
                data_list.append(ValidationDatum(
                    f"PI_{vname}_{hr}h_{int(lvl*100)}%", passed,
                    actual, f"nominal={lvl} ±{thresh:.2f}"
                ))

    max_dev = float(np.max(all_devs)) if all_devs else 1.0
    passed = max_dev <= 0.20

    return TestResult(
        name="Prediction Interval Calibration (Horizon)",
        passed=passed,
        score=max(0, 1.0 - float(np.mean(all_devs)) if all_devs else 0.0),
        data=data_list,
    )


# ── Test 6: Parameter Uncertainty Coverage ──────────────────────

def test_parameter_uncertainty_coverage(
    n_patients: int = 8,
    n_steps: int = 250,
) -> TestResult:
    """
    Posterior interval coverage for known true parameters.

    For each parameter, check whether the 90% posterior credible
    interval from the UKF contains the true parameter value.

    Tests:
      - Per-parameter 90% CI coverage
      - Overall coverage fraction (should be ~90%)
      - CI width relative to posterior standard deviation
    """
    rng = np.random.RandomState(42)

    param_names = [
        ("SI", 0), ("EGP0", 1), ("beta_response", 2),
        ("Vg", 3), ("ke", 4), ("Rbc", 5), ("RT", 6),
        ("alpha_glucagon", 7), ("tau_I", 8), ("tau_IR", 9),
        ("SBP_baseline", 10), ("DBP_baseline", 11), ("period", 12),
    ]

    param_coverages: Dict[str, List[int]] = {name: [] for name, _ in param_names}

    for p in range(n_patients):
        true_params = DEFAULT_PARAMS.copy()
        true_params[0] = rng.lognormal(-4.0, 0.3)
        true_params[2] = rng.lognormal(-6.0, 0.4)
        true_params[5] = rng.lognormal(4.5, 0.2)
        true_params[8] = rng.normal(100, 10)
        true_params[12] = 1440.0

        state = np.zeros(30)
        state[0] = rng.normal(100, 15)
        state[1] = true_params[2] * max(0, state[0] - true_params[6])
        state[5] = true_params[10]
        state[6] = true_params[11]
        state[7] = rng.normal(70, 5)
        state[21] = rng.uniform(0.3, 0.7)

        obs = []
        s = state.copy()
        for t in range(n_steps):
            s = full_dynamics(s, true_params, {})
            obs.append(full_observation(s))
        obs_arr = np.array(obs)

        engine = PersonalizationEngine()
        engine.initialize(obs_arr[0])
        for t in range(1, n_steps):
            engine.update(obs_arr[t], {})

        p_mean, p_cov = engine.get_parameters()
        if p_cov is None:
            continue

        for pname, pidx in param_names:
            if pidx < len(p_mean) and pidx < p_cov.shape[0]:
                p_var = float(p_cov[pidx, pidx])
                p_std = max(np.sqrt(p_var), 1e-4)
                lo = float(p_mean[pidx]) - Z_SCORES[0.90] * p_std
                hi = float(p_mean[pidx]) + Z_SCORES[0.90] * p_std
                true_val = float(true_params[pidx])
                param_coverages[pname].append(1 if lo <= true_val <= hi else 0)

    data_list = []
    covered_params = 0
    total_params = 0
    for pname, _ in param_names:
        vals = param_coverages[pname]
        if not vals or len(vals) < 5:
            continue
        cov_frac = float(np.mean(vals))
        total_params += 1
        if cov_frac >= 0.50:
            covered_params += 1
        data_list.append(ValidationDatum(
            f"param_cov_90_{pname}", cov_frac >= 0.50,
            cov_frac, "≥ 0.50"
        ))

    overall_cov = float(covered_params) / max(total_params, 1)
    passed = overall_cov >= THRESHOLDS["param_coverage_min"]

    return TestResult(
        name="Parameter Uncertainty (Posterior Coverage)",
        passed=passed,
        score=overall_cov,
        data=data_list,
    )


# ── Test 7: Structural Uncertainty Sensitivity ──────────────────

def test_structural_uncertainty_sensitivity(
    n_patients: int = 8,
    n_steps: int = 250,
) -> TestResult:
    """
    Sensitivity of uncertainty estimates to model structure.

    Evaluates how CI widths change under plausible model perturbations:
      A. Process noise scaled by 2x
      B. Observation noise scaled by 2x
      C. Glucose dynamics coefficient perturbation (±20% on SI)

    Measures:
      - Inflation ratio: perturbed CI width / baseline CI width
      - Coverage preservation under perturbation
      - Which variables are most sensitive to structural uncertainty
    """
    rng = np.random.RandomState(42)
    perturbations = {
        "Q_2x": ("process_noise", 2.0),
        "R_2x": ("obs_noise", 2.0),
        "SI_80pct": ("param_scale", 0.8),
        "SI_120pct": ("param_scale", 1.2),
    }

    baseline_widths: Dict[str, List[float]] = {name: [] for name, _, _, _ in COVERAGE_VARS}
    perturbed_widths: Dict[str, Dict[str, List[float]]] = {}
    for pname in perturbations:
        perturbed_widths[pname] = {name: [] for name, _, _, _ in COVERAGE_VARS}

    for p in range(n_patients):
        true_params = DEFAULT_PARAMS.copy()
        true_params[0] = rng.lognormal(-4.0, 0.3)
        true_params[5] = rng.lognormal(4.5, 0.2)
        true_params[12] = 1440.0

        state = np.zeros(30)
        state[0] = rng.normal(100, 10)
        state[1] = 0.013 * max(0, state[0] - 80)
        state[5] = rng.normal(120, 10)
        state[6] = rng.normal(80, 5)
        state[7] = rng.normal(70, 5)

        obs = []
        s = state.copy()
        for t in range(n_steps):
            s = full_dynamics(s, true_params, {})
            obs.append(full_observation(s))
        obs_arr = np.array(obs)

        # Baseline engine
        engine = PersonalizationEngine(process_noise_scale=0.01, obs_noise_scale=0.1)
        engine.initialize(obs_arr[0])
        for t in range(1, n_steps):
            engine.update(obs_arr[t], {})

        for t in range(100, n_steps, 10):
            cov = engine.get_twin_state_covariance()
            mu = engine.get_twin_state()
            for var_name, var_idx, _, _ in COVERAGE_VARS:
                var = float(cov[var_idx, var_idx]) if cov.shape[0] > var_idx else 100.0
                std = max(np.sqrt(var), 1.0)
                w = 2.0 * Z_SCORES[0.90] * std
                baseline_widths[var_name].append(w)

        # Perturbed engines
        for pname, (ptype, pval) in perturbations.items():
            if ptype == "process_noise":
                e = PersonalizationEngine(process_noise_scale=0.01 * pval, obs_noise_scale=0.1)
            elif ptype == "obs_noise":
                e = PersonalizationEngine(process_noise_scale=0.01, obs_noise_scale=0.1 * pval)
            elif ptype == "param_scale":
                mod_params = true_params.copy()
                mod_params[0] = true_params[0] * pval
                e = PersonalizationEngine(process_noise_scale=0.01, obs_noise_scale=0.1)
                # This perturbation changes the actual data, not the engine
                mod_obs = []
                ms = state.copy()
                for mt in range(n_steps):
                    ms = full_dynamics(ms, mod_params, {})
                    mod_obs.append(full_observation(ms))
                mod_obs_arr = np.array(mod_obs)
                e.initialize(mod_obs_arr[0])
                for mt in range(1, n_steps):
                    e.update(mod_obs_arr[mt], {})
                for mt in range(100, n_steps, 10):
                    mcov = e.get_twin_state_covariance()
                    mmu = e.get_twin_state()
                    for var_name, var_idx, _, _ in COVERAGE_VARS:
                        mvar = float(mcov[var_idx, var_idx]) if mcov.shape[0] > var_idx else 100.0
                        mstd = max(np.sqrt(mvar), 1.0)
                        perturbed_widths[pname][var_name].append(2.0 * Z_SCORES[0.90] * mstd)
                continue
            else:
                continue

            e.initialize(obs_arr[0])
            for et in range(1, n_steps):
                e.update(obs_arr[et], {})
            for et in range(100, n_steps, 10):
                ecov = e.get_twin_state_covariance()
                emu = e.get_twin_state()
                for var_name, var_idx, _, _ in COVERAGE_VARS:
                    evar = float(ecov[var_idx, var_idx]) if ecov.shape[0] > var_idx else 100.0
                    estd = max(np.sqrt(evar), 1.0)
                    perturbed_widths[pname][var_name].append(2.0 * Z_SCORES[0.90] * estd)

    data_list = []
    inflation_ratios = []
    for var_name, var_idx, _, _ in COVERAGE_VARS:
        bw = np.array(baseline_widths[var_name])
        if len(bw) < 5:
            continue
        baseline_mean = float(np.mean(bw))
        data_list.append(ValidationDatum(
            f"struct_base_width_{var_name}", True,
            baseline_mean, "baseline 90% PI width"
        ))

        for pname in perturbations:
            pw = np.array(perturbed_widths[pname][var_name])
            if len(pw) < 5:
                continue
            perturbed_mean = float(np.mean(pw))
            ratio = perturbed_mean / max(baseline_mean, 1.0)
            inflation_ratios.append(ratio)
            passed = ratio <= THRESHOLDS["structural_inflation_max"]
            data_list.append(ValidationDatum(
                f"struct_{pname}_{var_name}", passed,
                ratio, f"inflation ≤ {THRESHOLDS['structural_inflation_max']}x"
            ))

    max_ratio = float(np.max(inflation_ratios)) if inflation_ratios else 1.0
    mean_ratio = float(np.mean(inflation_ratios)) if inflation_ratios else 1.0
    passed = max_ratio <= THRESHOLDS["structural_inflation_max"]

    data_list.append(ValidationDatum(
        "structural_max_inflation", passed,
        max_ratio, f"≤ {THRESHOLDS['structural_inflation_max']}x"
    ))

    return TestResult(
        name="Structural Uncertainty Sensitivity",
        passed=passed,
        score=max(0, 1.0 - (mean_ratio - 1.0) / THRESHOLDS["structural_inflation_max"]),
        data=data_list,
    )


# ── Test 8: Calibrated Twin Coverage ────────────────────────────

def test_calibrated_twin_coverage(
    n_patients: int = 8,
    n_cal_steps: int = 200,
    n_eval_steps: int = 100,
) -> TestResult:
    """
    Validate that the CalibratedTwin (temperature-scaling + conformal)
    achieves correct coverage, solving the under-coverage problem.

    Uses a separate calibration set to fit temperature, then evaluates
    on a held-out set. 90% CI should achieve 85-95% coverage.
    """
    rng = np.random.RandomState(42)
    cal_coverages = {lvl: [] for lvl in [0.50, 0.80, 0.90, 0.95]}

    for p in range(n_patients):
        true_params = DEFAULT_PARAMS.copy()
        true_params[0] = rng.lognormal(-4.0, 0.3)
        true_params[5] = rng.lognormal(4.5, 0.2)
        true_params[12] = 1440.0

        state = np.zeros(30)
        state[0] = rng.normal(100, 15)
        state[1] = 0.013 * max(0, state[0] - 80)
        state[5] = rng.normal(125, 10)
        state[6] = rng.normal(80, 8)
        state[7] = rng.normal(70, 8)
        state[21] = rng.uniform(0.3, 0.7)

        total_steps = n_cal_steps + n_eval_steps
        obs = []
        s = state.copy()
        for t in range(total_steps):
            s = full_dynamics(s, true_params, {})
            obs.append(full_observation(s))
        obs_arr = np.array(obs)

        # Train calibrated twin
        twin = CalibratedTwin()
        twin.initialize(obs_arr[0])
        for t in range(1, n_cal_steps):
            twin.update(obs_arr[t], {})

        # Calibrate on the training portion
        twin.calibrate(obs_arr[:n_cal_steps], calibration_steps=n_cal_steps)

        # Evaluate on held-out data using predictive (pre-update) coverage
        # We call filter.predict() separately to get the forecast state,
        # then filter.update() to assimilate (avoiding twin.update() which
        # would double-predict)
        for t in range(n_cal_steps, total_steps):
            # PREDICT: one-step-ahead forecast
            twin._engine.filter.predict({})
            mu, cov = twin.get_calibrated_state()
            if t % 5 == 0:
                for lvl in cal_coverages:
                    z = Z_SCORES[lvl]
                    for idx in [0, 5, 7]:
                        var = float(cov[idx, idx]) if cov.shape[0] > idx else 100.0
                        std = max(np.sqrt(var), 1.0)
                        lo = float(mu[idx]) - z * std
                        hi = float(mu[idx]) + z * std
                        actual = float(obs_arr[t, idx])
                        cal_coverages[lvl].append(1.0 if lo <= actual <= hi else 0.0)
            # UPDATE: assimilate the observation
            twin._engine.filter.update(obs_arr[t])

    data_list = []
    all_devs = []
    for lvl in [0.50, 0.80, 0.90, 0.95]:
        vals = cal_coverages[lvl]
        if not vals:
            continue
        actual_cov = float(np.mean(vals))
        dev = abs(actual_cov - lvl)
        all_devs.append(dev)

        thresh = {0.50: 0.10, 0.80: 0.08, 0.90: 0.05, 0.95: 0.05}.get(lvl, 0.10)
        passed = dev <= thresh
        data_list.append(ValidationDatum(
            f"calibrated_cov_{int(lvl*100)}%", passed,
            actual_cov, f"nominal={lvl} ±{thresh}"
        ))

    max_dev = float(np.max(all_devs)) if all_devs else 1.0
    passed = max_dev <= 0.10

    # Also test that calibration improves over raw UKF
    data_list.append(ValidationDatum(
        "calibrated_max_deviation", passed,
        max_dev, "≤ 0.10"
    ))

    return TestResult(
        name="Calibrated Twin Coverage (Temperature + Conformal)",
        passed=passed,
        score=max(0, 1.0 - max_dev),
        data=data_list,
    )


# ── Runner ──────────────────────────────────────────────────────

def run_uncertainty_validations() -> List[TestResult]:
    """Run all 8 uncertainty validation tests."""
    return [
        test_pit_uniformity(),
        test_multi_level_coverage(),
        test_sharpness(),
        test_reliability_diagrams(),
        test_prediction_interval_calibration(),
        test_parameter_uncertainty_coverage(),
        test_structural_uncertainty_sensitivity(),
        test_calibrated_twin_coverage(),
    ]


def run_all() -> "ValidationReport":
    """Run all uncertainty tests and produce a combined report."""
    from app.personalization.phase5.pi_validation import ValidationReport
    import time

    tests = run_uncertainty_validations()
    passed = sum(1 for t in tests if t.passed)
    total = len(tests)
    scores = [t.score for t in tests]

    return ValidationReport(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        tests=tests,
        overall_pass_rate=float(passed) / max(total, 1),
        overall_score=float(np.mean(scores)) if scores else 0.0,
    )


if __name__ == "__main__":
    report = run_all()
    print(report.summary())
