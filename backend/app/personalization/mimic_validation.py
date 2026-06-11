"""
MIMIC-IV-equivalent validation pipeline.

For each synthetic patient:
1. Split observations into train (70%) and test (30%)
2. Run dual estimation engine on train
3. Predict 1-step, 6-step, and 24-step ahead
4. Compute RMSE, calibration, and counterfactual validity

Reports:
- 1-step RMSE
- 6-step (30-min) RMSE
- 24-step (2-hour) RMSE
- 80% PI coverage
- 95% PI coverage
- Counterfactual ATE accuracy on synthetic ground truth
"""

import math
import logging
import json
import os
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

from .mimic_equivalent import MIMICEquivalentGenerator, MIMICPatient, compute_validation_metrics
from .dual_engine import create_dual_engine, DualEstimationEngine
from .do_calculus import DoCalculusCounterfactual, InterventionSpec, InterventionType
from .dynamics import DEFAULT_PARAMS
from .phase5.clinical_metrics import clarke_error_grid, bland_altman_analysis, compute_clinical_validation

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    patient_id: str
    diabetes_type: str
    n_train: int
    n_test: int
    rmse_1step: float
    rmse_6step: float
    rmse_24step: float
    pi80_coverage: float
    pi95_coverage: float
    tir_predicted: float
    tir_actual: float
    counterfactual_ate_predicted: float
    counterfactual_ate_actual: float
    convergence_achieved: bool
    # New fields
    baseline_rmse_1step: float = 0.0
    baseline_rmse_6step: float = 0.0
    baseline_rmse_24step: float = 0.0
    clarke_zone_a_pct: float = 0.0
    clarke_zone_b_pct: float = 0.0
    clarke_acceptable: bool = False
    mard: float = 0.0
    mean_glucose: float = 0.0
    glucose_std: float = 0.0


class MIMICValidationPipeline:
    """Validation pipeline on MIMIC-IV-equivalent synthetic cohort."""

    def __init__(self, seed: int = 42, train_frac: float = 0.7):
        self.seed = seed
        self.train_frac = train_frac
        self.gen = MIMICEquivalentGenerator(seed=seed)

    def run_full_validation(
        self,
        n_patients: int = 20,
        cohort_type: str = "mixed",
    ) -> Dict:
        """Run full validation on synthetic cohort."""
        if cohort_type == "icu":
            patients = self.gen.generate_icu_cohort(n_patients)
        elif cohort_type == "t1dm":
            patients = self.gen.generate_t1dm_outpatient_cohort(n_patients)
        else:
            patients = self.gen.generate_mixed_cohort(n_patients)
        # Validate each patient
        results = []
        for p in patients:
            res = self._validate_patient(p)
            results.append(res)
        # Aggregate
        return self._aggregate_results(results, patients)

    def _validate_patient(self, patient: MIMICPatient) -> ValidationResult:
        """Run personalization + prediction + counterfactual on one patient."""
        n_total = len(patient.observations)
        n_train = int(n_total * self.train_frac)
        n_test = n_total - n_train
        # Train
        engine = create_dual_engine()
        engine.initialize(patient.observations[0])
        for t in range(1, n_train):
            engine.update(patient.observations[t])
        # Predict 1-step, 6-step, 24-step on test
        preds_1, preds_6, preds_24, pred_stds_1 = [], [], [], []
        for t in range(n_test - 24):
            p1_mean, p1_std = engine.predict(n_steps=1)
            p6_mean, p6_std = engine.predict(n_steps=6)
            p24_mean, p24_std = engine.predict(n_steps=24)
            preds_1.append(p1_mean[0])
            preds_6.append(p6_mean[0])
            preds_24.append(p24_mean[0])
            pred_stds_1.append(p1_std[0])
            engine.update(patient.observations[n_train + t])
        preds_1 = np.array(preds_1)
        preds_6 = np.array(preds_6)
        preds_24 = np.array(preds_24)
        pred_stds_1 = np.array(pred_stds_1)
        # Actual
        actual_1 = patient.observations[n_train:n_train + len(preds_1), 0]
        actual_6_start = n_train + 5
        actual_6_end = actual_6_start + len(preds_6)
        if actual_6_end > n_total:
            actual_6_end = n_total
            actual_6 = patient.observations[actual_6_start:actual_6_end, 0]
            preds_6 = preds_6[:len(actual_6)]
        else:
            actual_6 = patient.observations[actual_6_start:actual_6_end, 0]
        actual_24_start = n_train + 23
        actual_24_end = actual_24_start + len(preds_24)
        if actual_24_end > n_total:
            actual_24_end = n_total
            actual_24 = patient.observations[actual_24_start:actual_24_end, 0]
            preds_24 = preds_24[:len(actual_24)]
        else:
            actual_24 = patient.observations[actual_24_start:actual_24_end, 0]
        # Compute RMSE
        rmse_1 = float(np.sqrt(np.mean((preds_1 - actual_1) ** 2)))
        rmse_6 = float(np.sqrt(np.mean((preds_6 - actual_6) ** 2))) if len(actual_6) > 0 else float('nan')
        rmse_24 = float(np.sqrt(np.mean((preds_24 - actual_24) ** 2))) if len(actual_24) > 0 else float('nan')

        # ── Persistence baseline ─────────────────────────────────
        # Baseline: predict glucose[t+h] = glucose[t]
        baseline_preds_1 = actual_1[:-1] if len(actual_1) > 1 else actual_1
        baseline_actual_1 = actual_1[1:] if len(actual_1) > 1 else actual_1
        baseline_rmse_1 = float(np.sqrt(np.mean((baseline_preds_1 - baseline_actual_1) ** 2))) if len(baseline_actual_1) > 0 else float('nan')

        baseline_preds_6 = patient.observations[n_train:n_train + len(preds_6) - 6, 0] if len(preds_6) > 6 else preds_6
        baseline_actual_6 = actual_6[6:] if len(actual_6) > 6 else actual_6
        baseline_rmse_6 = float(np.sqrt(np.mean((baseline_preds_6[:len(baseline_actual_6)] - baseline_actual_6) ** 2))) if len(baseline_actual_6) > 0 else float('nan')

        baseline_preds_24 = patient.observations[n_train:n_train + len(preds_24) - 24, 0] if len(preds_24) > 24 else preds_24
        baseline_actual_24 = actual_24[24:] if len(actual_24) > 24 else actual_24
        baseline_rmse_24 = float(np.sqrt(np.mean((baseline_preds_24[:len(baseline_actual_24)] - baseline_actual_24) ** 2))) if len(baseline_actual_24) > 0 else float('nan')

        # ── Clarke Error Grid ────────────────────────────────────
        ceg = clarke_error_grid(actual_1, preds_1)

        # ── MARD ─────────────────────────────────────────────────
        mard = float(np.mean(np.abs(preds_1 - actual_1) / (np.abs(actual_1) + 1e-6)) * 100)

        # PI coverage (80% nominal)
        z80 = 1.28
        in_pi80 = (actual_1 >= preds_1 - z80 * pred_stds_1) & (actual_1 <= preds_1 + z80 * pred_stds_1)
        coverage80 = float(np.mean(in_pi80)) if len(in_pi80) > 0 else 0.0
        z95 = 1.96
        in_pi95 = (actual_1 >= preds_1 - z95 * pred_stds_1) & (actual_1 <= preds_1 + z95 * pred_stds_1)
        coverage95 = float(np.mean(in_pi95)) if len(in_pi95) > 0 else 0.0
        # TIR
        tir_p = float(np.mean((preds_1 >= 70) & (preds_1 <= 180)))
        tir_a = float(np.mean((actual_1 >= 70) & (actual_1 <= 180)))
        # Counterfactual validation: predict ATE of insulin bolus
        cf_pred_ate, cf_actual_ate = self._validate_counterfactual(patient, engine)
        return ValidationResult(
            patient_id=patient.profile.patient_id,
            diabetes_type=patient.profile.diabetes_type,
            n_train=n_train,
            n_test=n_test,
            rmse_1step=rmse_1,
            rmse_6step=rmse_6,
            rmse_24step=rmse_24,
            pi80_coverage=coverage80,
            pi95_coverage=coverage95,
            tir_predicted=tir_p,
            tir_actual=tir_a,
            counterfactual_ate_predicted=cf_pred_ate,
            counterfactual_ate_actual=cf_actual_ate,
            convergence_achieved=engine._is_converged,
            baseline_rmse_1step=baseline_rmse_1,
            baseline_rmse_6step=baseline_rmse_6,
            baseline_rmse_24step=baseline_rmse_24,
            clarke_zone_a_pct=ceg.zone_a_pct,
            clarke_zone_b_pct=ceg.zone_b_pct,
            clarke_acceptable=ceg.clinically_acceptable,
            mard=mard,
            mean_glucose=float(np.mean(actual_1)),
            glucose_std=float(np.std(actual_1)),
        )

    def _validate_counterfactual(
        self,
        patient: MIMICPatient,
        engine: DualEstimationEngine,
    ) -> tuple:
        """Validate counterfactual prediction against ground truth."""
        # Get current state estimate
        estimated_params, _ = engine.get_estimated_params()
        # Predict counterfactual
        dc = DoCalculusCounterfactual()
        intervention = InterventionSpec(
            intervention_type=InterventionType.INSULIN_BOLUS,
            magnitude=5.0,
            duration_steps=6,
            start_step=0,
        )
        # Predicted: use estimated parameters
        state_estimate = engine.filter.get_state()
        cf_predicted = dc.evaluate_intervention(
            state_estimate, engine._build_full_params(), intervention, n_total_steps=12
        )
        # Actual: use true parameters
        cf_actual = dc.evaluate_intervention(
            patient.initial_state, patient.true_params, intervention, n_total_steps=12
        )
        return cf_predicted.ate_glucose, cf_actual.ate_glucose

    def _aggregate_results(
        self,
        results: List[ValidationResult],
        patients: List[MIMICPatient],
    ) -> Dict:
        """Aggregate validation results into a summary report."""
        # Filter valid results
        valid = [r for r in results if not np.isnan(r.rmse_1step)]
        # Compute aggregate metrics
        rmse_1 = [r.rmse_1step for r in valid]
        rmse_6 = [r.rmse_6step for r in valid if not np.isnan(r.rmse_6step)]
        rmse_24 = [r.rmse_24step for r in valid if not np.isnan(r.rmse_24step)]
        cov80 = [r.pi80_coverage for r in valid]
        cov95 = [r.pi95_coverage for r in valid]
        cf_pred = [r.counterfactual_ate_predicted for r in valid]
        cf_actual = [r.counterfactual_ate_actual for r in valid]
        cf_bias = [p - a for p, a in zip(cf_pred, cf_actual)]
        # Baseline comparison
        bl_1 = [r.baseline_rmse_1step for r in valid if not np.isnan(r.baseline_rmse_1step)]
        bl_6 = [r.baseline_rmse_6step for r in valid if not np.isnan(r.baseline_rmse_6step)]
        bl_24 = [r.baseline_rmse_24step for r in valid if not np.isnan(r.baseline_rmse_24step)]
        # Clarke Error Grid
        clarke_a = [r.clarke_zone_a_pct for r in valid]
        clarke_ab = [r.clarke_zone_a_pct + r.clarke_zone_b_pct for r in valid]
        # MARD
        mards = [r.mard for r in valid]
        return {
            "n_patients": len(valid),
            "n_total": len(results),
            "rmse_1step_mean": float(np.mean(rmse_1)) if rmse_1 else 0.0,
            "rmse_1step_std": float(np.std(rmse_1)) if rmse_1 else 0.0,
            "rmse_6step_mean": float(np.mean(rmse_6)) if rmse_6 else 0.0,
            "rmse_24step_mean": float(np.mean(rmse_24)) if rmse_24 else 0.0,
            "pi80_coverage_mean": float(np.mean(cov80)) if cov80 else 0.0,
            "pi95_coverage_mean": float(np.mean(cov95)) if cov95 else 0.0,
            "counterfactual_ate_rmse": float(np.sqrt(np.mean(np.array(cf_bias) ** 2))) if cf_bias else 0.0,
            "counterfactual_ate_bias_mean": float(np.mean(cf_bias)) if cf_bias else 0.0,
            "convergence_rate": float(np.mean([r.convergence_achieved for r in valid])) if valid else 0.0,
            # New fields
            "baseline_rmse_1step": float(np.mean(bl_1)) if bl_1 else 0.0,
            "baseline_rmse_6step": float(np.mean(bl_6)) if bl_6 else 0.0,
            "baseline_rmse_24step": float(np.mean(bl_24)) if bl_24 else 0.0,
            "improvement_over_baseline_1step": float((1 - np.mean(rmse_1) / max(np.mean(bl_1), 1e-6)) * 100) if rmse_1 and bl_1 else 0.0,
            "improvement_over_baseline_6step": float((1 - np.mean(rmse_6) / max(np.mean(bl_6), 1e-6)) * 100) if rmse_6 and bl_6 else 0.0,
            "improvement_over_baseline_24step": float((1 - np.mean(rmse_24) / max(np.mean(bl_24), 1e-6)) * 100) if rmse_24 and bl_24 else 0.0,
            "clarke_zone_a_mean": float(np.mean(clarke_a)) if clarke_a else 0.0,
            "clarke_zone_ab_mean": float(np.mean(clarke_ab)) if clarke_ab else 0.0,
            "mard_mean": float(np.mean(mards)) if mards else 0.0,
            "by_diabetes_type": self._group_by_diabetes(valid),
        }

    def write_results_csv(self, results: List[ValidationResult], filepath: str):
        """Write per-patient validation results as CSV."""
        rows = []
        for r in results:
            rows.append({
                "patient_id": r.patient_id,
                "diabetes_type": r.diabetes_type,
                "n_train": r.n_train,
                "n_test": r.n_test,
                "rmse_1step": r.rmse_1step,
                "rmse_6step": r.rmse_6step,
                "rmse_24step": r.rmse_24step,
                "baseline_rmse_1step": r.baseline_rmse_1step,
                "baseline_rmse_6step": r.baseline_rmse_6step,
                "baseline_rmse_24step": r.baseline_rmse_24step,
                "pi80_coverage": r.pi80_coverage,
                "pi95_coverage": r.pi95_coverage,
                "clarke_zone_a_pct": r.clarke_zone_a_pct,
                "clarke_zone_b_pct": r.clarke_zone_b_pct,
                "clarke_acceptable": r.clarke_acceptable,
                "mard": r.mard,
                "tir_predicted": r.tir_predicted,
                "tir_actual": r.tir_actual,
                "mean_glucose": r.mean_glucose,
                "glucose_std": r.glucose_std,
                "converged": r.convergence_achieved,
            })
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        logger.info(f"Wrote {len(rows)} validation results to {filepath}")

    def run_and_save(
        self,
        n_patients: int = 20,
        cohort_type: str = "mixed",
        output_dir: str = "scientific_proof",
    ) -> Dict:
        """Run validation, save results, and return aggregate report."""
        if cohort_type == "icu":
            patients = self.gen.generate_icu_cohort(n_patients)
        elif cohort_type == "t1dm":
            patients = self.gen.generate_t1dm_outpatient_cohort(n_patients)
        else:
            patients = self.gen.generate_mixed_cohort(n_patients)
        results = []
        for p in patients:
            res = self._validate_patient(p)
            results.append(res)
        aggregate = self._aggregate_results(results, patients)
        # Save
        os.makedirs(output_dir, exist_ok=True)
        self.write_results_csv(results, os.path.join(output_dir, "forecasting_table.csv"))
        with open(os.path.join(output_dir, "forecasting_results.json"), "w") as f:
            json.dump(aggregate, f, indent=2, default=str)
        logger.info(f"Saved forecasting results to {output_dir}/")
        return aggregate

    def _group_by_diabetes(self, results: List[ValidationResult]) -> Dict:
        groups = {}
        for r in results:
            t = r.diabetes_type
            if t not in groups:
                groups[t] = []
            groups[t].append(r)
        return {
            t: {
                "n": len(rs),
                "rmse_1step": float(np.mean([r.rmse_1step for r in rs])),
                "pi80_coverage": float(np.mean([r.pi80_coverage for r in rs])),
            } for t, rs in groups.items()
        }
