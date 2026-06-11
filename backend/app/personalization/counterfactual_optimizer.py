"""
Counterfactual Treatment Optimizer.

Uses the ODE to find the optimal personalized treatment policy
via simulation-based optimization (parameter perturbation approach).

For each patient, we simulate:
  - What if insulin sensitivity were increased by 20%? (SI adjustment)
  - What if exercise were added post-meal? (exercise timing)
  - What if meal timing shifted earlier? (meal schedule)
  - What if the renal threshold were modified? (SGLT2 mechanism)

The optimizer searches over the intervention space and returns
the policy that maximizes a clinical composite score.

NOTE: This implements parameter perturbation, not Pearl's do-operator
(graph surgery). Causal interpretation requires the assumption that
the ODE parameter encodes the causal mechanism for that intervention.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from itertools import product

from app.personalization.dynamics import full_dynamics, DEFAULT_PARAMS
from app.personalization.state import PHYSIO_DIM
from app.personalization.phase5.causal_inference import StructuralCausalModel
from app.personalization.clinical_outcomes import (
    compute_clinical_outcomes, ClinicalOutcomes, RCTSimulator, TrialResult
)


@dataclass
class TreatmentPolicy:
    name: str
    description: str
    intervention_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    parameter_changes: Dict[str, float] = field(default_factory=dict)

    def apply(self, state: np.ndarray, params: np.ndarray) -> np.ndarray:
        return self.intervention_fn(state, params)


@dataclass
class OptimizerResult:
    best_policy: TreatmentPolicy
    baseline_outcomes: ClinicalOutcomes
    best_outcomes: ClinicalOutcomes
    all_results: List[Tuple[str, ClinicalOutcomes]]
    tir_improvement: float
    a1c_reduction: float
    n_policies_evaluated: int


# ── Pre-built Intervention Policies ───────────────────────

def _no_intervention(state: np.ndarray, params: np.ndarray) -> np.ndarray:
    return full_dynamics(state, params, {})


def _exercise_post_meal(state: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Add moderate exercise, increasing glucose utilization."""
    mod_state = state.copy()
    return full_dynamics(mod_state, params, {"exercise": 0.5})


def _meal_shift_earlier(
    morning_meal: float = 40.0,
    evening_meal: float = 30.0,
) -> Callable:
    """Shift carbohydrate load earlier in the day."""
    def fn(state: np.ndarray, params: np.ndarray) -> np.ndarray:
        return full_dynamics(state, params, {"meal": morning_meal, "meal_timing": -1.0})
    return fn


def _sglt2_therapy(state: np.ndarray, params: np.ndarray) -> np.ndarray:
    """SGLT2 inhibition: lower renal threshold to increase glucosuria."""
    mod_params = params.copy()
    mod_params[3] = 120.0  # RT lowered from 180 to 120 (param index 3)
    return full_dynamics(state, mod_params, {})


def _glp1_agonist(state: np.ndarray, params: np.ndarray) -> np.ndarray:
    """GLP-1 RA: increase insulin secretion, delay gastric emptying."""
    mod_params = params.copy()
    mod_params[2] *= 2.0  # beta_response doubled
    return full_dynamics(state, mod_params, {})


def _metformin_mechanism(state: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Metformin: reduce HGP, increase insulin sensitivity."""
    mod_params = params.copy()
    mod_params[0] *= 1.3  # SI increased by 30%
    mod_params[1] *= 0.8  # EGP0 reduced by 20%
    return full_dynamics(state, mod_params, {})


def _combined_therapy(state: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Combination: metformin + SGLT2i."""
    mod_params = params.copy()
    mod_params[0] *= 1.3
    mod_params[1] *= 0.8
    mod_params[3] = 120.0  # RT lowered from 180 to 120
    mod_params[2] *= 1.2
    return full_dynamics(state, mod_params, {})


# ── Built-in Policies ────────────────────────────────────

BUILTIN_POLICIES = [
    TreatmentPolicy("baseline", "Standard care (no intervention)", _no_intervention),
    TreatmentPolicy("exercise", "Post-meal moderate exercise", _exercise_post_meal),
    TreatmentPolicy("sglt2i", "SGLT2 inhibition (RT lowered to 120)", _sglt2_therapy),
    TreatmentPolicy("glp1_ra", "GLP-1 RA (beta_response x2)", _glp1_agonist),
    TreatmentPolicy("metformin", "Metformin (SI+30%, EGP0-20%)", _metformin_mechanism),
    TreatmentPolicy("combination", "Metformin + SGLT2i + GLP1", _combined_therapy),
]


class CounterfactualOptimizer:
    """
    Personalized treatment optimizer using ODE counterfactual simulation.

    For a given patient state, evaluates all available treatment policies
    via simulation and returns the one maximizing clinical outcomes.

    Usage:
        state = twin.get_twin_state()
        params = twin.get_parameters()
        opt = CounterfactualOptimizer()
        result = opt.optimize(state, params[0], n_steps=288)
        print(f"Best: {result.best_policy.name} (TIR +{result.tir_improvement:.1f}%)")
    """

    def __init__(self, policies: Optional[List[TreatmentPolicy]] = None):
        self._policies = policies or BUILTIN_POLICIES
        self._scm = StructuralCausalModel()

    def optimize(
        self,
        state: np.ndarray,
        params: np.ndarray,
        n_steps: int = 288,
    ) -> OptimizerResult:
        baseline_outcomes = self._evaluate_policy(
            state, params, _no_intervention, n_steps
        )

        results: List[Tuple[str, ClinicalOutcomes]] = [
            ("baseline", baseline_outcomes)
        ]

        best_score = baseline_outcomes.composite_score()
        best_policy = self._policies[0]
        best_outcomes = baseline_outcomes

        for policy in self._policies:
            if policy.name == "baseline":
                continue
            outcomes = self._evaluate_policy(
                state, params, policy.intervention_fn, n_steps
            )
            results.append((policy.name, outcomes))
            score = outcomes.composite_score()
            if score > best_score:
                best_score = score
                best_policy = policy
                best_outcomes = outcomes

        tir_improvement = best_outcomes.tir - baseline_outcomes.tir
        a1c_reduction = baseline_outcomes.eA1c - best_outcomes.eA1c

        return OptimizerResult(
            best_policy=best_policy,
            baseline_outcomes=baseline_outcomes,
            best_outcomes=best_outcomes,
            all_results=results,
            tir_improvement=tir_improvement,
            a1c_reduction=a1c_reduction,
            n_policies_evaluated=len(self._policies),
        )

    def _evaluate_policy(
        self,
        state: np.ndarray,
        params: np.ndarray,
        intervention_fn: Callable,
        n_steps: int,
    ) -> ClinicalOutcomes:
        s = state.copy()
        g_trace = []
        sbp_trace = []
        for t in range(n_steps):
            s = intervention_fn(s, params)
            g_trace.append(s[0])
            sbp_trace.append(s[5])
        return compute_clinical_outcomes(
            np.array(g_trace), np.array(sbp_trace)
        )

    def dose_response_curve(
        self,
        state: np.ndarray,
        params: np.ndarray,
        param_index: int,
        dose_range: np.ndarray,
        n_steps: int = 288,
    ) -> List[Tuple[float, ClinicalOutcomes]]:
        """
        Compute dose-response curve for a single parameter.

        Args:
            param_index: Which parameter to vary
            dose_range: Array of values to test

        Returns:
            List of (dose, outcomes) pairs
        """
        results = []
        for dose in dose_range:
            mod_params = params.copy()
            mod_params[param_index] = dose
            s = state.copy()
            g_trace = []
            for t in range(n_steps):
                s = full_dynamics(s, mod_params, {})
                g_trace.append(s[0])
            outcomes = compute_clinical_outcomes(np.array(g_trace))
            results.append((float(dose), outcomes))
        return results

    def run_virtual_trial(
        self,
        intervention_fn: Callable,
        n_arm: int = 50,
        n_steps: int = 288,
    ) -> TrialResult:
        """Run a virtual RCT comparing the given intervention vs baseline."""
        sim = RCTSimulator(seed=42)
        return sim.run_trial(
            control_fn=_no_intervention,
            intervention_fn=intervention_fn,
            n_arm=n_arm,
            n_steps=n_steps,
        )
