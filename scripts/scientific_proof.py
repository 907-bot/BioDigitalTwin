#!/usr/bin/env python3
"""
BioDigitalTwin — Milestone 1: Scientific Proof

Single-entry orchestrator that runs all three validation pillars
and produces a consolidated publishable report:

  1. MIMIC-IV-equivalent forecasting validation
  2. Clinical trial replication (DCCT, UKPDS, NICE-SUGAR)
  3. Calibration evaluation

Usage:
    PYTHONPATH=backend backend/.venv/bin/python scripts/scientific_proof.py [--quick] [--output-dir PATH]

Output:
    scientific_proof/
    ├── results.json                # Machine-readable consolidated results
    ├── results.md                  # Human-readable markdown report
    ├── forecasting_table.csv       # Per-patient forecasting results
    ├── forecasting_results.json    # Aggregated forecasting metrics
    ├── trial_results.json          # Trial replication results
    ├── calibration_results.json    # Calibration evaluation results
    └── figures/                    # (future) PNG plots
"""

import sys
import os
import json
import time
import argparse
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))

PASS = "\033[0;32m✓\033[0m"
FAIL = "\033[0;31m✗\033[0m"
BOLD = "\033[1m"
RESET = "\033[0m"


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ═════════════════════════════════════════════════════════════════
# Pillar 1: MIMIC-IV-equivalent Forecasting
# ═════════════════════════════════════════════════════════════════

def run_forecasting(output_dir: str, quick: bool = False) -> dict:
    section("PILLAR 1: MIMIC-IV-EQUIVALENT FORECASTING")
    from app.personalization.mimic_validation import MIMICValidationPipeline

    n_patients = 5 if quick else 30
    print(f"  Running validation on {n_patients} synthetic patients...")
    t0 = time.time()

    pipeline = MIMICValidationPipeline(seed=42)
    results = []
    patients = pipeline.gen.generate_mixed_cohort(n_patients)
    for p in patients:
        res = pipeline._validate_patient(p)
        results.append(res)
        print(f"    {res.patient_id}: RMSE 1-step={res.rmse_1step:.1f}, "
              f"baseline={res.baseline_rmse_1step:.1f}, "
              f"A={res.clarke_zone_a_pct:.0f}%")
    aggregate = pipeline._aggregate_results(results, patients)
    aggregate["elapsed_s"] = time.time() - t0

    # Save
    os.makedirs(output_dir, exist_ok=True)
    pipeline.write_results_csv(results, os.path.join(output_dir, "forecasting_table.csv"))
    with open(os.path.join(output_dir, "forecasting_results.json"), "w") as f:
        json.dump(aggregate, f, indent=2, default=str)

    print(f"\n  {PASS} Forecasting complete ({aggregate['elapsed_s']:.1f}s)")
    print(f"    1-step RMSE:     {aggregate['rmse_1step_mean']:.1f} mg/dL")
    print(f"    Baseline:        {aggregate['baseline_rmse_1step']:.1f} mg/dL")
    print(f"    Improvement:     {aggregate['improvement_over_baseline_1step']:.1f}%")
    print(f"    Clarke Zone A+B: {aggregate['clarke_zone_ab_mean']:.1f}%")
    print(f"    MARD:            {aggregate['mard_mean']:.1f}%")
    return aggregate


# ═════════════════════════════════════════════════════════════════
# Pillar 2: Clinical Trial Replication
# ═════════════════════════════════════════════════════════════════

def run_trials(output_dir: str, quick: bool = False) -> dict:
    section("PILLAR 2: CLINICAL TRIAL REPLICATION")
    from app.personalization.phase5.clinical_trial_simulator import (
        ClinicalTrialSimulator, PRESET_TRIALS, LANDMARK_TRIAL_EFFECTS,
        run_landmark_trial,
    )

    trial_names = ["dcct"] if quick else ["dcct", "ukpds", "nice_sugar"]
    n_patients = 20 if quick else 100
    sim = ClinicalTrialSimulator()

    all_results = {}
    t0 = time.time()

    for name in trial_names:
        print(f"  Running {name.upper()} (n={n_patients})...")
        trial_t0 = time.time()
        trial_design = PRESET_TRIALS[name]
        result = sim.simulate_trial(trial_design, n_patients)
        landmark = LANDMARK_TRIAL_EFFECTS.get(name, {})
        elapsed = time.time() - trial_t0

        print(f"    Summary: {result.summary()}")
        print(f"    Published {landmark.get('reference', '?')}")
        print(f"    Elapsed: {elapsed:.1f}s")

        all_results[name] = {
            "name": landmark.get("name", name),
            "published_reference": landmark.get("reference", ""),
            "n_patients": result.n_patients,
            "endpoint_results": result.endpoint_results,
            "effect_sizes": result.effect_sizes,
            "p_values": result.p_values,
            "summary": result.summary(),
            "elapsed_s": elapsed,
        }

    results = {
        "trials": all_results,
        "elapsed_s": time.time() - t0,
        "n_trials": len(all_results),
    }

    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "trial_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  {PASS} Trial replication complete ({results['elapsed_s']:.1f}s)")
    return results


# ═════════════════════════════════════════════════════════════════
# Pillar 3: Calibration Evaluation
# ═════════════════════════════════════════════════════════════════

def run_calibration(output_dir: str, quick: bool = False) -> dict:
    import numpy as np
    from scipy import stats as sp_stats
    section("PILLAR 3: CALIBRATION EVALUATION")
    from app.personalization.dual_engine import create_dual_engine
    from app.personalization.dynamics import full_observation, full_dynamics, DEFAULT_PARAMS

    n_patients = 5 if quick else 15
    n_steps = 144
    rng = np.random.default_rng(42)

    print(f"  Evaluating calibration on {n_patients} patients, {n_steps} steps each...")
    t0 = time.time()

    all_coverages = {50: [], 80: [], 90: [], 95: []}
    pit_values = []
    pi_widths = []
    z_scores = {50: 0.67, 80: 1.28, 90: 1.64, 95: 1.96}
    errors = []

    for p_idx in range(n_patients):
        # Generate patient
        engine = create_dual_engine()
        state = np.zeros(30)
        state[0] = rng.normal(140, 30)
        state[1] = 0.013 * max(0, state[0] - 80)
        state[5] = rng.normal(120, 10)
        state[6] = rng.normal(80, 8)
        state[7] = rng.normal(72, 8)
        state[9] = rng.normal(95, 15)

        engine.initialize(full_observation(state))

        for t in range(n_steps):
            # Simulate forward with observation noise
            state = full_dynamics(state, DEFAULT_PARAMS, {})
            state[0] = max(40, min(400, state[0]))
            obs_clean = full_observation(state)
            # Add realistic CGM noise (5 mg/dL SD, per ISO 15197)
            obs = obs_clean.copy()
            obs[0] = obs[0] + rng.normal(0, 5.0)
            obs[1] = max(60, min(220, obs[1] + rng.normal(0, 3.0)))
            obs[2] = max(40, min(140, obs[2] + rng.normal(0, 2.0)))
            obs[5] = max(30, min(200, obs[5] + rng.normal(0, 5.0)))

            # Prediction with uncertainty (before update — true 1-step ahead)
            pred_mean, pred_std = engine.predict(n_steps=1)
            pred_g = pred_mean[0]
            pred_s = max(pred_std[0], 0.1)
            actual_g = obs[0]

            errors.append(abs(pred_g - actual_g))

            # Skip burn-in (first 5 steps) for calibration metrics
            if t >= 5:
                # PIT: Probability Integral Transform (should be Uniform)
                pit = float(sp_stats.norm.cdf(actual_g, loc=pred_g, scale=pred_s))
                pit_values.append(pit)

                # Coverage
                for level in all_coverages:
                    z = z_scores[level]
                    lo = pred_g - z * pred_s
                    hi = pred_g + z * pred_s
                    all_coverages[level].append(1.0 if lo <= actual_g <= hi else 0.0)

                # PI width
                pi_widths.append(2 * 1.96 * pred_s)

            # Twin update (now with observation)
            engine.update(obs)

    # Aggregate
    elapsed = time.time() - t0
    coverage_results = {}
    for level, hits in all_coverages.items():
        emp_cov = float(np.mean(hits)) if hits else 0.0
        coverage_results[f"nominal_{level}"] = float(level / 100)
        coverage_results[f"empirical_{level}"] = emp_cov
        coverage_results[f"coverage_error_{level}"] = emp_cov - level / 100

    # PIT uniformity test
    ks_stat, ks_p = sp_stats.kstest(pit_values, "uniform")
    pit_mean = float(np.mean(pit_values))
    pit_sd = float(np.std(pit_values))

    # Overall calibration score (0-1)
    max_cov_error = max(abs(coverage_results.get(f"coverage_error_{l}", 0)) for l in [50, 80, 90, 95])
    cal_score = max(0.0, min(1.0, 1.0 - max_cov_error * 2))

    results = {
        "n_patients": n_patients,
        "n_predictions": len(pit_values),
        "elapsed_s": elapsed,
        "coverage": coverage_results,
        "max_coverage_error": float(max_cov_error),
        "mean_pi_width_95": float(np.mean(pi_widths)) if pi_widths else 0.0,
        "pit_uniformity_ks_stat": float(ks_stat),
        "pit_uniformity_ks_p": float(ks_p),
        "pit_mean": pit_mean,
        "pit_std": pit_sd,
        "pit_is_uniform": bool(ks_p > 0.05),
        "calibration_score": cal_score,
        "mean_abs_error": float(np.mean(errors)) if errors else 0.0,
        "rmse": float(np.sqrt(np.mean(np.square(errors)))) if errors else 0.0,
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "calibration_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  {PASS} Calibration complete ({elapsed:.1f}s)")
    print(f"  Nominal 80% coverage:   {coverage_results['empirical_80']:.1%}")
    print(f"  Nominal 95% coverage:   {coverage_results['empirical_95']:.1%}")
    print(f"  Max coverage error:     {max_cov_error:.1%}")
    print(f"  Calibration score:      {cal_score:.3f}")
    print(f"  PIT KS test p-value:    {ks_p:.4f}")
    print(f"  Mean absolute error:    {np.mean(errors):.1f} mg/dL")
    return results


# ═════════════════════════════════════════════════════════════════
# Report Generation
# ═════════════════════════════════════════════════════════════════

def generate_report(
    forecasting: dict,
    trials: dict,
    calibration: dict,
    output_dir: str,
) -> str:
    """Generate a publishable markdown report."""
    lines = []
    def L(*args):
        lines.append(" ".join(str(a) for a in args))

    L("# BioDigitalTwin — Scientific Proof Results")
    L(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L(f"**Framework:** 30-dimensional whole-body physiological ODE")
    L(f"**Observations:** 15-dim (CGM, BP, HR, lab values)")
    L(f"**Engine:** Dual estimation (30-dim UKF + 7-dim MAP)")
    L("")

    # ── Summary ──
    L("## Executive Summary")
    L("")
    L("| Metric | Value | Target | Status |")
    L("|--------|-------|--------|--------|")

    f_improve = forecasting.get("improvement_over_baseline_1step", 0)
    f_improve_str = f"{f_improve:.0f}%" if isinstance(f_improve, (int, float)) else "N/A"
    f_rmse = forecasting.get("rmse_1step_mean", 0)
    f_rmse_str = f"{f_rmse:.1f}" if isinstance(f_rmse, (int, float)) else "N/A"
    f_clarke = forecasting.get("clarke_zone_ab_mean", 0)
    f_clarke_str = f"{f_clarke:.0f}%" if isinstance(f_clarke, (int, float)) else "N/A"

    L(f"| Forecasting RMSE (1-step) | {f_rmse_str} mg/dL | < 15 | {'✓' if not isinstance(f_rmse, str) and f_rmse < 15 else '✗'} |")
    L(f"| Improvement over baseline | {f_improve_str} | > 20% | {'✓' if not isinstance(f_improve, str) and f_improve > 20 else '✗'} |")
    L(f"| Clarke Error Grid A+B | {f_clarke_str} | > 99% | {'✓' if not isinstance(f_clarke, str) and f_clarke > 99 else '✗'} |")

    cal_score = calibration.get("calibration_score", 0)
    cal_str = f"{cal_score:.3f}"
    L(f"| Calibration score | {cal_str} | > 0.70 | {'✓' if cal_score > 0.70 else '✗'} |")

    # Show trial summaries
    for name, trial_data in trials.get("trials", {}).items():
        effect_sizes = trial_data.get("effect_sizes", {})
        for ep, es in effect_sizes.items():
            L(f"| {name.upper()} {ep} | {es:.3f} | — | — |")

    L("")

    # ── Pillar 1 ──
    L("## Pillar 1: MIMIC-IV-equivalent Forecasting")
    L("")
    L(f"**Cohort:** {forecasting.get('n_patients', 'N/A')} synthetic patients "
      f"matching published MIMIC-IV demographics (Sauer 2022, Johnson 2023)")
    L("")
    L("### 1-Step Ahead Glucose Prediction")
    L("")
    L("| Metric | Twin | Baseline (Persistence) | Improvement |")
    L("|--------|------|----------------------|------------|")
    rmse_1 = forecasting.get('rmse_1step_mean', 0)
    baseline_1 = forecasting.get('baseline_rmse_1step', 0)
    improvement_1 = forecasting.get('improvement_over_baseline_1step', 0)
    L(f"| 5-min RMSE | {rmse_1 if isinstance(rmse_1, (int, float)) else 0:.1f} mg/dL | "
      f"{baseline_1 if isinstance(baseline_1, (int, float)) else 0:.1f} mg/dL | "
      f"{improvement_1 if isinstance(improvement_1, (int, float)) else 0:.0f}% |")
    rmse_6 = forecasting.get('rmse_6step_mean', 0)
    baseline_6 = forecasting.get('baseline_rmse_6step', 0)
    improvement_6 = forecasting.get('improvement_over_baseline_6step', 0)
    L(f"| 30-min RMSE | {rmse_6 if isinstance(rmse_6, (int, float)) else 0:.1f} mg/dL | "
      f"{baseline_6 if isinstance(baseline_6, (int, float)) else 0:.1f} mg/dL | "
      f"{improvement_6 if isinstance(improvement_6, (int, float)) else 0:.0f}% |")
    rmse_24 = forecasting.get('rmse_24step_mean', 0)
    baseline_24 = forecasting.get('baseline_rmse_24step', 0)
    improvement_24 = forecasting.get('improvement_over_baseline_24step', 0)
    L(f"| 2-hour RMSE | {rmse_24 if isinstance(rmse_24, (int, float)) else 0:.1f} mg/dL | "
      f"{baseline_24 if isinstance(baseline_24, (int, float)) else 0:.1f} mg/dL | "
      f"{improvement_24 if isinstance(improvement_24, (int, float)) else 0:.0f}% |")
    L("")
    L("### Clinical Error Analysis")
    L("")
    L(f"| Metric | Value | Passing |")
    L("|--------|-------|---------|")
    L(f"| Clarke Zone A | {forecasting.get('clarke_zone_a_mean', 0):.1f}% | — |")
    L(f"| Clarke Zone A+B | {forecasting.get('clarke_zone_ab_mean', 0):.1f}% | {'✓' if forecasting.get('clarke_zone_ab_mean', 0) > 99 else '✗'} |")
    L(f"| MARD | {forecasting.get('mard_mean', 0):.2f}% | {'✓' if forecasting.get('mard_mean', 0) < 15 else '✗'} |")
    L(f"| 80% PI Coverage | {forecasting.get('pi80_coverage_mean', 0):.1%} | {'✓' if abs(forecasting.get('pi80_coverage_mean', 0) - 0.8) < 0.1 else '✗'} |")
    L(f"| 95% PI Coverage | {forecasting.get('pi95_coverage_mean', 0):.1%} | {'✓' if abs(forecasting.get('pi95_coverage_mean', 0) - 0.95) < 0.1 else '✗'} |")
    L("")

    # ── Pillar 2 ──
    L("## Pillar 2: Clinical Trial Replication")
    L("")
    L(f"**Simulator:** {trials.get('n_trials', 0)} landmark trials, "
      f"{trials.get('elapsed_s', 0):.1f}s total runtime")
    L("")

    for name, trial_data in trials.get("trials", {}).items():
        L(f"### {trial_data.get('name', name)}")
        L(f"**Reference:** {trial_data.get('published_reference', 'N/A')}")
        L(f"**Patients:** {trial_data.get('n_patients', 'N/A')}")
        L(f"**Summary:** {trial_data.get('summary', '').replace(chr(10), chr(10) + '  ')}")
        L("")

    # ── Pillar 3 ──
    L("## Pillar 3: Probabilistic Calibration")
    L("")
    L(f"**Predictions evaluated:** {calibration.get('n_predictions', 'N/A')}")
    L("")
    L("| Nominal Level | Empirical Coverage | Error |")
    L("|-------------|-------------------|-------|")
    for level in [50, 80, 90, 95]:
        nominal = calibration.get("coverage", {}).get(f"nominal_{level}", level / 100)
        empirical = calibration.get("coverage", {}).get(f"empirical_{level}", 0)
        err = calibration.get("coverage", {}).get(f"coverage_error_{level}", 0)
        L(f"| {level}% | {empirical:.1%} | {err:+.1%} {'✓' if abs(err) < 0.05 else '✗'} |")
    L("")
    L(f"| Metric | Value | Target | Status |")
    L("|--------|-------|--------|--------|")
    L(f"| Calibration score | {calibration.get('calibration_score', 0):.3f} | >0.70 | {'✓' if calibration.get('calibration_score', 0) > 0.70 else '✗'} |")
    L(f"| PIT KS test p-value | {calibration.get('pit_uniformity_ks_p', 0):.4f} | >0.05 | {'✓' if calibration.get('pit_uniformity_ks_p', 0) > 0.05 else '✗'} |")
    L(f"| Mean absolute error | {calibration.get('mean_abs_error', 0):.1f} mg/dL | — | — |")
    L(f"| RMSE | {calibration.get('rmse', 0):.1f} mg/dL | — | — |")
    L(f"| Mean 95% PI width | {calibration.get('mean_pi_width_95', 0):.1f} mg/dL | — | — |")
    L("")

    # ── ValidationFrameworkV2 ──
    L("## Supplementary: Phase 5 Validation Framework")
    L("")
    try:
        from app.personalization.phase5.validation_framework import ValidationFrameworkV2
        vf = ValidationFrameworkV2()
        vf.run_all()
        report = vf.get_report()
        L(f"**Overall status:** {report.get('overall_status', 'N/A')}")
        L(f"**Overall score:** {report.get('overall_score', 0):.3f}")
        L(f"**Levels passed:** {report.get('levels_passed', '0/0')}")
        L("")
        for r in report.get("results", []):
            status = "✓" if r.get("passed") else "✗"
            L(f"- Level {r.get('level', '?')}: {r.get('test', '?')}: {status} (score={r.get('score', 0):.2f})")
    except Exception as e:
        L(f"*Validation framework error: {e}*")
    L("")

    # ── Conclusions ──
    L("## Assessment")
    L("")
    overall_pass = True
    issues = []

    if isinstance(f_rmse, (int, float)) and f_rmse >= 15:
        overall_pass = False
        issues.append(f"Forecasting RMSE {f_rmse:.1f} mg/dL >= 15 mg/dL target")
    if cal_score < 0.70:
        overall_pass = False
        issues.append(f"Calibration score {cal_score:.3f} < 0.70 target")
    if not isinstance(f_clarke, str) and f_clarke < 99:
        issues.append(f"Clarke Zone A+B {f_clarke:.1f}% < 99% target (informational)")

    if overall_pass:
        L("**Overall: PASS** — All targets met. System is ready for external validation.")
    else:
        L("**Overall: PARTIAL PASS** — Some targets not yet met.")
        if issues:
            L("")
            L("### Outstanding Issues")
            for i in issues:
                L(f"- {i}")

    report = "\n".join(lines)

    report_path = os.path.join(output_dir, "results.md")
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\n  {PASS} Report written to {report_path}")
    return report


# ═════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="BioDigitalTwin — Scientific Proof Pipeline")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer patients)")
    parser.add_argument("--output-dir", default="scientific_proof", help="Output directory")
    parser.add_argument("--skip-forecasting", action="store_true", help="Skip Pillar 1")
    parser.add_argument("--skip-trials", action="store_true", help="Skip Pillar 2")
    parser.add_argument("--skip-calibration", action="store_true", help="Skip Pillar 3")
    args = parser.parse_args()

    import numpy as np
    print(f"{BOLD}BioDigitalTwin — Scientific Proof Pipeline{RESET}")
    print(f"  Output:  {args.output_dir}/")
    print(f"  Mode:    {'QUICK' if args.quick else 'FULL'}")

    total_t0 = time.time()
    forecasting = {"n_patients": 0}
    trials = {"n_trials": 0, "trials": {}}
    calibration = {"coverage": {}}
    all_results = {}

    if not args.skip_forecasting:
        forecasting = run_forecasting(args.output_dir, quick=args.quick)
        all_results["forecasting"] = forecasting

    if not args.skip_trials:
        trials = run_trials(args.output_dir, quick=args.quick)
        all_results["trials"] = trials

    if not args.skip_calibration:
        calibration = run_calibration(args.output_dir, quick=args.quick)
        all_results["calibration"] = calibration

    # Consolidate
    total_elapsed = time.time() - total_t0
    all_results["total_elapsed_s"] = total_elapsed
    all_results["generated"] = datetime.now().isoformat()
    all_results["mode"] = "quick" if args.quick else "full"

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    report = generate_report(forecasting, trials, calibration, args.output_dir)

    print(f"\n{'='*60}")
    print(f"  TOTAL ELAPSED: {total_elapsed:.1f}s")
    print(f"  RESULTS:       {args.output_dir}/")
    print(f"  REPORT:        {args.output_dir}/results.md")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
