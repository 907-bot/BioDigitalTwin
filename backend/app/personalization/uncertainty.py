"""
Phase 3: Uncertainty Engine V2.

Quantifies four layers of uncertainty:
  1. Parameter uncertainty  — from UKF posterior covariance
  2. Structural uncertainty  — across different model formulations
  3. Measurement uncertainty — sensor noise models
  4. Intervention uncertainty — patient adherence variability
"""

import numpy as np
from typing import List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass
from .core import PersonalizationEngine, PHYSIO_DIM, PARAM_DIM, OBS_DIM


@dataclass
class UncertaintyReport:
    """Comprehensive uncertainty quantification for a prediction."""
    parameter_uncertainty: np.ndarray     # 90% CI half-width for each state dim
    structural_uncertainty: float         # % contribution
    measurement_uncertainty: np.ndarray   # per-observable
    intervention_uncertainty: float       # adherence-adjusted spread
    posterior_predictive_samples: np.ndarray  # n_samples × state_dim
    coverage_metrics: Dict


class UncertaintyEngine:
    """
    Quantifies and communicates all sources of uncertainty.
    """

    def __init__(self, engine: PersonalizationEngine):
        self.engine = engine

    def parameter_uncertainty(self, n_sigma: float = 1.645) -> np.ndarray:
        """
        90% CI half-width for each physiological state dimension.
        Returns: array of shape (PHYSIO_DIM,)
        """
        cov = self.engine.get_twin_state_covariance()
        return n_sigma * np.sqrt(np.maximum(np.diag(cov), 1e-10))

    def posterior_predictive_samples(
        self, n_samples: int = 500,
        horizon: int = 60,
    ) -> np.ndarray:
        """
        Generate posterior predictive trajectories.
        Samples from the UKF posterior and simulates forward.

        Returns: array of shape (n_samples, horizon, PHYSIO_DIM)
        """
        from .dynamics import full_dynamics

        mu = self.engine.get_twin_state()
        cov = self.engine.get_twin_state_covariance()
        param_mean, param_cov = self.engine.get_parameters()

        trajectories = np.zeros((n_samples, horizon, PHYSIO_DIM))
        for i in range(n_samples):
            reg_cov_state = cov + np.eye(PHYSIO_DIM) * 1e-4
            try:
                state = np.random.multivariate_normal(mu, reg_cov_state)
            except (np.linalg.LinAlgError, ValueError):
                state = mu + np.random.randn(PHYSIO_DIM) * np.sqrt(np.diag(cov))
            reg_cov_param = param_cov + np.eye(PARAM_DIM) * 1e-4
            try:
                params = np.random.multivariate_normal(param_mean, reg_cov_param)
            except (np.linalg.LinAlgError, ValueError):
                params = param_mean + np.random.randn(PARAM_DIM) * np.sqrt(np.diag(param_cov))
            for t in range(horizon):
                state = full_dynamics(state, params, {})
                trajectories[i, t] = state

        return trajectories

    def measurement_uncertainty(
        self, sensor_noise: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """
        Per-observable measurement noise standard deviation.

        Default sensor models:
          - CGM glucose: ~5-15 mg/dL
          - BP monitor: ~3-8 mmHg
          - HR wearable: ~2-5 bpm
          - HRV: ~5-15 ms
          - Lab values: small (1-5%)
        """
        default_noise = {
            "glucose": 8.0, "SBP": 5.0, "DBP": 4.0, "HR": 3.0, "HRV": 10.0,
            "GFR": 5.0, "Na": 2.0, "K": 0.2, "Osm": 3.0,
            "FFA": 0.05, "LDL": 5.0, "HDL": 3.0, "TG": 10.0,
            "cortisol": 30.0, "sleep_pressure": 0.1,
        }
        if sensor_noise:
            default_noise.update(sensor_noise)
        return np.array(list(default_noise.values()))

    def structural_uncertainty(
        self,
        alternate_dynamics: List[Callable],
        n_samples: int = 50,
    ) -> float:
        """
        Estimate structural uncertainty by running alternate model formulations.
        Returns coefficient of variation across models as a percentage.
        """
        from .dynamics import full_dynamics
        base = self.engine.get_twin_state()
        base_params, _ = self.engine.get_parameters()

        predictions = []
        predictions.append(full_dynamics(base, base_params, {}))
        for alt_fn in alternate_dynamics:
            try:
                pred = alt_fn(base, base_params, {})
                predictions.append(pred)
            except Exception:
                continue

        if len(predictions) < 2:
            return 0.0

        preds = np.array(predictions)
        cv = np.std(preds, axis=0) / (np.mean(preds, axis=0) + 1e-10)
        return float(np.mean(cv) * 100.0)

    def intervention_adherence_uncertainty(
        self,
        base_adherence: float = 0.8,
        n_scenarios: int = 100,
    ) -> float:
        """
        Estimate intervention outcome spread due to variable adherence.
        Samples adherence rates and simulates the effect.
        """
        from .dynamics import full_dynamics

        state = self.engine.get_twin_state()
        params, _ = self.engine.get_parameters()

        outcomes = []
        for _ in range(n_scenarios):
            adherence = np.random.beta(
                base_adherence * 10, (1 - base_adherence) * 10
            )
            inputs = {
                "exercise": adherence * 0.5,
                "meal_glucose": (1 - 0.3 * adherence) * 50.0,
            }
            new_state = full_dynamics(state, params, inputs)
            outcomes.append(new_state[0])  # glucose outcome

        outcomes = np.array(outcomes)
        return float(np.std(outcomes) / max(np.mean(outcomes), 1.0) * 100.0)

    def coverage_assessment(
        self, observed_trajectory: np.ndarray,
        n_samples: int = 200, horizon: int = 30,
    ) -> Dict[str, float]:
        """
        Assess calibration of uncertainty intervals.
        Computes empirical coverage of 50%, 80%, 90%, 95% CIs.
        """
        pp = self.posterior_predictive_samples(n_samples, horizon)
        coverage = {}
        for nominal in [0.50, 0.80, 0.90, 0.95]:
            lower_p = (1.0 - nominal) / 2.0
            upper_p = 1.0 - lower_p
            lower = np.percentile(pp, lower_p * 100, axis=0)
            upper = np.percentile(pp, upper_p * 100, axis=0)
            n_forecast = min(horizon, len(observed_trajectory))
            covered = 0
            total = 0
            for t in range(n_forecast):
                for v in range(PHYSIO_DIM):
                    if lower[t, v] <= observed_trajectory[t, v] <= upper[t, v]:
                        covered += 1
                    total += 1
            empirical = covered / max(total, 1)
            coverage[f"nominal_{nominal:.0%}"] = float(empirical)
            coverage[f"calibration_error_{nominal:.0%}"] = float(abs(empirical - nominal))
        coverage["mean_calibration_error"] = float(np.mean([
            coverage.get(f"calibration_error_{nominal:.0%}", 0)
            for nominal in [0.50, 0.80, 0.90, 0.95]
        ]))
        return coverage

    def full_report(
        self,
        n_pp_samples: int = 100,
        horizon: int = 30,
        observed_trajectory: Optional[np.ndarray] = None,
    ) -> UncertaintyReport:
        """Generate comprehensive uncertainty report."""
        param_unc = self.parameter_uncertainty()
        pp_samples = self.posterior_predictive_samples(n_pp_samples, horizon)
        meas_unc = self.measurement_uncertainty()
        interv_unc = self.intervention_adherence_uncertainty()

        coverage_metrics = {
            "90p_ci_halfwidth": float(np.mean(param_unc)),
            "measurement_noise_avg": float(np.mean(meas_unc)),
            "intervention_cv_pct": float(interv_unc),
        }
        if observed_trajectory is not None:
            coverage_metrics["coverage_assessment"] = self.coverage_assessment(
                observed_trajectory, n_samples=n_pp_samples, horizon=horizon
            )

        return UncertaintyReport(
            parameter_uncertainty=param_unc,
            structural_uncertainty=0.0,
            measurement_uncertainty=meas_unc,
            intervention_uncertainty=interv_unc,
            posterior_predictive_samples=pp_samples,
            coverage_metrics=coverage_metrics,
        )
