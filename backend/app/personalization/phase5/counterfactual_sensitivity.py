"""
Counterfactual Sensitivity Analysis.

Quantifies how counterfactual conclusions change under variation in
adherence rates, dose-response curves, intervention parameter values,
and unmeasured confounders.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Callable


@dataclass
class AdherenceSensitivityPoint:
    adherence: float
    outcome_mean: float
    outcome_std: float
    effect_size_vs_baseline: float
    n_simulations: int


@dataclass
class DoseResponsePoint:
    dose: float
    response_mean: float
    response_std: float
    is_toxic: bool


@dataclass
class CounterfactualSensitivityReport:
    adherence_sensitivity: List[AdherenceSensitivityPoint]
    dose_response: Dict[str, List[DoseResponsePoint]]
    parameter_sensitivity: Dict[str, float]
    unmeasured_confounding: Dict[str, float]
    robustness_score: float


class CounterfactualSensitivityAnalyzer:
    """
    Sensitivity analysis for counterfactual intervention simulations.

    Tests:
      1. Adherence sensitivity: vary adherence from 0→1 at 0.1 increments
      2. Dose-response: vary each intervention parameter across its range
      3. Parameter sensitivity: perturb physiological parameters
      4. Unmeasured confounding: add residual confounder with varying strength
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def adherence_sensitivity(self, simulate_fn: Callable,
                               adherence_range: Optional[List[float]] = None,
                               n_per_point: int = 50) -> List[AdherenceSensitivityPoint]:
        if adherence_range is None:
            adherence_range = np.linspace(0, 1, 11).tolist()
        results = []
        for adh in adherence_range:
            outcomes = []
            for _ in range(n_per_point):
                try:
                    outcome = simulate_fn(adherence=adh)
                    if isinstance(outcome, dict):
                        outcome = outcome.get("glucose", outcome.get("outcome", 0))
                    outcomes.append(float(outcome) if outcome is not None else 0.0)
                except Exception:
                    outcomes.append(0.0)
            results.append(AdherenceSensitivityPoint(
                adherence=adh,
                outcome_mean=float(np.mean(outcomes)),
                outcome_std=float(np.std(outcomes)),
                effect_size_vs_baseline=float(np.mean(outcomes) - outcomes[0] if len(outcomes) > 1 else 0),
                n_simulations=n_per_point,
            ))
        return results

    def dose_response_analysis(self, simulate_fn: Callable,
                                parameter_name: str,
                                dose_range: Optional[List[float]] = None,
                                toxicity_threshold: Optional[float] = None,
                                n_per_point: int = 30) -> List[DoseResponsePoint]:
        if dose_range is None:
            dose_range = np.linspace(0, 2, 11).tolist()
        results = []
        for dose in dose_range:
            responses = []
            for _ in range(n_per_point):
                try:
                    response = simulate_fn(**{parameter_name: dose})
                    if isinstance(response, dict):
                        response = response.get("glucose", response.get("outcome", response.get("effect", 0)))
                    responses.append(float(response) if response is not None else 0.0)
                except Exception:
                    responses.append(0.0)
            is_toxic = False
            if toxicity_threshold is not None:
                is_toxic = np.mean(responses) > toxicity_threshold
            results.append(DoseResponsePoint(
                dose=dose,
                response_mean=float(np.mean(responses)),
                response_std=float(np.std(responses)),
                is_toxic=is_toxic,
            ))
        return results

    def parameter_sensitivity(self, simulate_fn: Callable,
                               param_ranges: Dict[str, Tuple[float, float]],
                               n_samples: int = 100) -> Dict[str, float]:
        sensitivities = {}
        for param_name, (lo, hi) in param_ranges.items():
            base_val = (lo + hi) / 2
            try:
                base_outcome = simulate_fn(**{param_name: base_val})
                if isinstance(base_outcome, dict):
                    base_outcome = base_outcome.get("glucose", base_outcome.get("outcome", 0))
                base_val = float(base_outcome) if base_outcome is not None else 0.0

                perturbed_outcome = simulate_fn(**{param_name: hi})
                if isinstance(perturbed_outcome, dict):
                    perturbed_outcome = perturbed_outcome.get("glucose", perturbed_outcome.get("outcome", 0))
                perturbed_val = float(perturbed_outcome) if perturbed_outcome is not None else 0.0
                delta = perturbed_val - base_val
                sensitivities[param_name] = float(delta / max(abs(hi - lo), 0.01))
            except Exception:
                sensitivities[param_name] = 0.0
        return sensitivities

    def unmeasured_confounding_analysis(self, simulate_fn: Callable,
                                         effect_estimator: Callable,
                                         confounder_strengths: Optional[List[float]] = None,
                                         ) -> Dict[str, float]:
        if confounder_strengths is None:
            confounder_strengths = np.linspace(0, 0.5, 6).tolist()
        base_effect = None
        effects = []
        for strength in confounder_strengths:
            try:
                effect = effect_estimator(confounder_strength=strength)
                if base_effect is None:
                    base_effect = effect
                effects.append(effect)
            except Exception:
                effects.append(0.0)
        if base_effect is None or len(effects) < 2:
            return {"base_effect": 0.0, "max_bias": 0.0, "e_value": 0.0}
        max_bias = max(abs(e - base_effect) for e in effects) if effects else 0.0
        e_value = 1.0 + max_bias / max(abs(base_effect), 0.01) if abs(base_effect) > 0.01 else 1.0
        return {
            "base_effect": float(base_effect),
            "max_bias": float(max_bias),
            "e_value": float(e_value),
        }

    def full_sensitivity_report(self, simulate_fn: Callable,
                                 dose_response_params: Optional[Dict[str, List[float]]] = None,
                                 param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
                                 effect_estimator: Optional[Callable] = None,
                                 ) -> CounterfactualSensitivityReport:
        adhere = self.adherence_sensitivity(simulate_fn)
        dose_resp = {}
        if dose_response_params:
            for param_name, dose_range in dose_response_params.items():
                dose_resp[param_name] = self.dose_response_analysis(simulate_fn, param_name, dose_range)
        param_sens = {}
        if param_ranges:
            param_sens = self.parameter_sensitivity(simulate_fn, param_ranges)
        unmeasured = {}
        if effect_estimator:
            unmeasured = self.unmeasured_confounding_analysis(simulate_fn, effect_estimator)

        n_robust = sum(1 for a in adhere if abs(a.effect_size_vs_baseline) < 0.5 * max(abs(a.effect_size_vs_baseline) for a in adhere) + 1e-10)
        robustness = float(n_robust / max(len(adhere), 1))
        return CounterfactualSensitivityReport(
            adherence_sensitivity=adhere,
            dose_response=dose_resp,
            parameter_sensitivity=param_sens,
            unmeasured_confounding=unmeasured,
            robustness_score=robustness,
        )


def run_counterfactual_sensitivity(simulate_fn: Callable,
                                    n_adherence_points: int = 5) -> Dict:
    analyzer = CounterfactualSensitivityAnalyzer()
    report = analyzer.full_sensitivity_report(simulate_fn)
    return {
        "adherence_sensitivity": [
            {"adherence": p.adherence, "outcome": p.outcome_mean,
             "std": p.outcome_std, "effect_size": p.effect_size_vs_baseline}
            for p in report.adherence_sensitivity
        ],
        "robustness_score": report.robustness_score,
        "n_dose_response_curves": len(report.dose_response),
        "n_parameters_tested": len(report.parameter_sensitivity),
    }
