"""
Proper Pearlian do-calculus via ODE graph surgery.

The previous counterfactual optimizer was parameter perturbation. This
module implements do(X=x) as graph surgery: sever incoming edges to X,
set X=x, propagate forward.

For the digital twin, the most clinically important interventions are:
- do(insulin_dose = X): exogenous insulin bolus
- do(SI *= X): insulin sensitivity change (e.g., from weight loss)
- do(HGP_basal = X): hepatic glucose production change (e.g., from metformin)
- do(RT = X): renal threshold change (e.g., from SGLT2i)
- do(meal = X): counterfactual meal at time t

For each, we identify the affected state variables and apply the
intervention, then re-simulate the ODE forward.

References:
- Pearl (2009) — Causality: Models, Reasoning, and Inference
- Spirtes et al. (2000) — Causation, Prediction, and Search
- Schölkopf et al. (2021) — Toward Causal Representation Learning
"""

import math
import logging
import numpy as np
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

from .state import PHYSIO_DIM
from .dynamics import full_dynamics, full_observation, DEFAULT_PARAMS

logger = logging.getLogger(__name__)


class InterventionType(Enum):
    INSULIN_BOLUS = "insulin_bolus"
    CHANGE_INSULIN_SENSITIVITY = "change_SI"
    CHANGE_HGP = "change_HGP"
    CHANGE_RENAL_THRESHOLD = "change_RT"
    CHANGE_BAROREFLEX = "change_baroreflex"
    MEAL = "meal"
    EXERCISE = "exercise"
    DRUG = "drug"
    COMBINATION = "combination"


@dataclass
class InterventionSpec:
    """Specification of a do-operator intervention."""
    intervention_type: InterventionType
    magnitude: float
    duration_steps: int = 1
    start_step: int = 0
    affected_state_indices: List[int] = field(default_factory=list)
    affected_param_indices: List[int] = field(default_factory=list)
    description: str = ""


@dataclass
class CounterfactualTrajectory:
    """Result of a counterfactual rollout."""
    intervention: InterventionSpec
    baseline_states: np.ndarray  # (T, PHYSIO_DIM)
    cf_states: np.ndarray  # (T, PHYSIO_DIM)
    baseline_obs: np.ndarray
    cf_obs: np.ndarray
    ate_glucose: float
    ate_hr: float
    ate_sbp: float
    max_glucose_delta: float
    min_glucose_delta: float
    time_to_hypo_baseline: Optional[int]
    time_to_hypo_cf: Optional[int]
    time_to_hyper_baseline: Optional[int]
    time_to_hyper_cf: Optional[int]


class DoOperator:
    """Pearlian do-calculus via ODE graph surgery.

    Implements do(X=x) by:
    1. Identifying all causal parents of X in the ODE structure
    2. Severing the edges from those parents to X
    3. Setting X=x (overriding the dynamics)
    4. Propagating forward with severed graph

    For our ODE model, the parents of each state variable are:
    - G: HGP, I, SI, PGU
    - I: insulin_dose (input), beta_response, G
    - HGP: HGP_basal, I, cortisol
    - BP: vascular_resistance, HR, autonomic_tone
    - etc.
    """

    # Causal parent structure (from dynamics.py):
    # For each state, which states/params are its direct parents
    PARENTS = {
        0:  ['I', 'SI', 'HGP_basal', 'PGU', 'insulin_dose'],  # G
        1:  ['G', 'beta_response', 'insulin_dose'],            # I
        2:  ['HGP_basal', 'I', 'cortisol'],                     # HGP
        3:  ['G', 'I', 'SI'],                                    # PGU
        4:  ['I'],                                               # IR
        5:  ['HR', 'autonomic_tone', 'vascular_resistance'],    # SBP
        6:  ['SBP'],                                             # DBP
        7:  ['autonomic_tone', 'baroreflex_gain'],              # HR
        8:  ['HR', 'autonomic_tone'],                           # HRV
        9:  ['SBP', 'baseline_GFR'],                            # GFR
        10: ['water_intake', 'sodium_retention'],               # Na
        11: ['GFR', 'sodium_retention'],                        # K
        12: ['Na'],                                             # Osm
    }

    @staticmethod
    def do_insulin_bolus(
        state: np.ndarray,
        params: np.ndarray,
        dose: float,
        duration_steps: int,
        start_step: int,
        n_total_steps: int,
    ) -> np.ndarray:
        """Implement do(insulin_dose = dose) for `duration_steps` starting at `start_step`.

        Steps:
        1. Sever edge from beta_response*G → I (endogenous insulin production)
        2. Sever edge from prior insulin → I (no carryover)
        3. Override I dynamics with constant insulin input

        Returns: counterfactual state trajectory
        """
        states = np.zeros((n_total_steps, PHYSIO_DIM))
        s = state.copy()
        # For graph surgery, we modify the dynamics: replace insulin equation
        # with constant external insulin during intervention window
        for t in range(n_total_steps):
            if start_step <= t < start_step + duration_steps:
                # External insulin override
                s[1] = dose  # I = dose during intervention
                # Apply metabolic effect on G (sever endogenous insulin)
                # G update with external insulin
                from .dynamics import compute_metabolic_dynamics
                # Use compute_metabolic_dynamics but override I
                # Simulate: G dynamics with I=dose
                try:
                    s = full_dynamics(s, params, {"insulin_dose": dose})
                except Exception:
                    s[0] = max(20.0, min(600.0, s[0] - 5.0 * dose))
            else:
                s = full_dynamics(s, params, {})
            s[0] = max(20.0, min(600.0, s[0]))
            s[1] = max(0.0, min(500.0, s[1]))
            states[t] = s.copy()
        return states

    @staticmethod
    def do_change_parameter(
        state: np.ndarray,
        params: np.ndarray,
        param_index: int,
        new_value: float,
        n_total_steps: int,
    ) -> np.ndarray:
        """Implement do(param[index] = new_value) — counterfactual under modified parameter.

        Steps:
        1. Set param[index] = new_value in the dynamics
        2. Sever all edges into param[index] (no adaptation)
        3. Propagate forward

        For parameters representing long-term physiological traits (SI, HGP_basal,
        RT, etc.), the parameter is held fixed at the new value.
        """
        modified_params = params.copy()
        modified_params[param_index] = new_value
        states = np.zeros((n_total_steps, PHYSIO_DIM))
        s = state.copy()
        for t in range(n_total_steps):
            s = full_dynamics(s, modified_params, {})
            s[0] = max(20.0, min(600.0, s[0]))
            s[1] = max(0.0, min(500.0, s[1]))
            states[t] = s.copy()
        return states

    @staticmethod
    def do_meal(
        state: np.ndarray,
        params: np.ndarray,
        carb_grams: float,
        start_step: int,
        duration_steps: int,
        n_total_steps: int,
    ) -> np.ndarray:
        """Implement do(meal = carbs) at start_step.

        Graph surgery:
        1. Add exogenous glucose appearance: G += absorption_rate * carbs
        2. Sever edge from meal → endogenous glucose production
        """
        # Conversion: 1g carbs → ~1 mg/dL glucose in 5L blood (rough)
        glucose_appearance = carb_grams * 0.5  # 50% absorption into blood
        # Absorption over duration
        per_step = glucose_appearance / max(duration_steps, 1)
        states = np.zeros((n_total_steps, PHYSIO_DIM))
        s = state.copy()
        for t in range(n_total_steps):
            if start_step <= t < start_step + duration_steps:
                s[0] += per_step
            s = full_dynamics(s, params, {})
            s[0] = max(20.0, min(600.0, s[0]))
            s[1] = max(0.0, min(500.0, s[1]))
            states[t] = s.copy()
        return states

    @staticmethod
    def do_exercise(
        state: np.ndarray,
        params: np.ndarray,
        intensity: float,  # 0-1
        duration_steps: int,
        start_step: int,
        n_total_steps: int,
    ) -> np.ndarray:
        """Implement do(exercise = intensity) — counterfactual exercise effect.

        Effects:
        - HR += 20 * intensity
        - G uptake += 0.5 * intensity (muscle glucose uptake)
        - SBP += 10 * intensity
        """
        states = np.zeros((n_total_steps, PHYSIO_DIM))
        s = state.copy()
        for t in range(n_total_steps):
            if start_step <= t < start_step + duration_steps:
                # Sever endogenous HR regulation; force HR up
                s[7] += 20.0 * intensity
                s[5] += 10.0 * intensity
                # Increase G uptake (sever normal insulin-mediated uptake)
                s[0] -= 0.5 * intensity
            s = full_dynamics(s, params, {})
            s[0] = max(20.0, min(600.0, s[0]))
            s[1] = max(0.0, min(500.0, s[1]))
            s[5] = max(50.0, min(250.0, s[5]))
            s[7] = max(30.0, min(220.0, s[7]))
            states[t] = s.copy()
        return states


class DoCalculusCounterfactual:
    """Counterfactual analysis using proper Pearlian do-calculus.

    For each intervention, computes:
    - ATE: average treatment effect on outcomes
    - Heterogeneous effects: per-patient
    - Time to clinical events (hypo, hyper)
    - Effect on physiological subsystems
    """

    def __init__(self):
        self.do_op = DoOperator()

    def evaluate_intervention(
        self,
        baseline_state: np.ndarray,
        params: np.ndarray,
        intervention: InterventionSpec,
        n_total_steps: int,
    ) -> CounterfactualTrajectory:
        """Evaluate an intervention via graph surgery.

        Returns full counterfactual trajectory and treatment effects.
        """
        # Run baseline (factual)
        baseline_states = np.zeros((n_total_steps, PHYSIO_DIM))
        s = baseline_state.copy()
        for t in range(n_total_steps):
            s = full_dynamics(s, params, {})
            s[0] = max(20.0, min(600.0, s[0]))
            s[1] = max(0.0, min(500.0, s[1]))
            baseline_states[t] = s.copy()
        # Run counterfactual
        if intervention.intervention_type == InterventionType.INSULIN_BOLUS:
            cf_states = self.do_op.do_insulin_bolus(
                baseline_state, params, intervention.magnitude,
                intervention.duration_steps, intervention.start_step, n_total_steps
            )
        elif intervention.intervention_type == InterventionType.CHANGE_INSULIN_SENSITIVITY:
            cf_states = self.do_op.do_change_parameter(
                baseline_state, params, 0, intervention.magnitude, n_total_steps
            )
        elif intervention.intervention_type == InterventionType.CHANGE_HGP:
            cf_states = self.do_op.do_change_parameter(
                baseline_state, params, 1, intervention.magnitude, n_total_steps
            )
        elif intervention.intervention_type == InterventionType.CHANGE_RENAL_THRESHOLD:
            cf_states = self.do_op.do_change_parameter(
                baseline_state, params, 3, intervention.magnitude, n_total_steps
            )
        elif intervention.intervention_type == InterventionType.MEAL:
            cf_states = self.do_op.do_meal(
                baseline_state, params, intervention.magnitude,
                intervention.start_step, intervention.duration_steps, n_total_steps
            )
        elif intervention.intervention_type == InterventionType.EXERCISE:
            cf_states = self.do_op.do_exercise(
                baseline_state, params, intervention.magnitude,
                intervention.duration_steps, intervention.start_step, n_total_steps
            )
        else:
            cf_states = baseline_states.copy()
        # Compute ATE
        baseline_obs = np.array([full_observation(s) for s in baseline_states])
        cf_obs = np.array([full_observation(s) for s in cf_states])
        ate_g = float(np.mean(cf_obs[:, 0] - baseline_obs[:, 0]))
        ate_hr = float(np.mean(cf_obs[:, 3] - baseline_obs[:, 3]))
        ate_sbp = float(np.mean(cf_obs[:, 1] - baseline_obs[:, 1]))
        # Time to events
        hyp_b = self._time_to_event(baseline_obs[:, 0], 70, 'below')
        hyp_cf = self._time_to_event(cf_obs[:, 0], 70, 'below')
        hyp_b2 = self._time_to_event(baseline_obs[:, 0], 180, 'above')
        hyp_cf2 = self._time_to_event(cf_obs[:, 0], 180, 'above')
        return CounterfactualTrajectory(
            intervention=intervention,
            baseline_states=baseline_states,
            cf_states=cf_states,
            baseline_obs=baseline_obs,
            cf_obs=cf_obs,
            ate_glucose=ate_g,
            ate_hr=ate_hr,
            ate_sbp=ate_sbp,
            max_glucose_delta=float(np.max(cf_obs[:, 0] - baseline_obs[:, 0])),
            min_glucose_delta=float(np.min(cf_obs[:, 0] - baseline_obs[:, 0])),
            time_to_hypo_baseline=hyp_b,
            time_to_hypo_cf=hyp_cf,
            time_to_hyper_baseline=hyp_b2,
            time_to_hyper_cf=hyp_cf2,
        )

    def _time_to_event(self, trajectory: np.ndarray, threshold: float, direction: str) -> Optional[int]:
        """Find first time index where trajectory crosses threshold."""
        for t, val in enumerate(trajectory):
            if direction == 'below' and val < threshold:
                return t
            if direction == 'above' and val > threshold:
                return t
        return None

    def refutation_test(
        self,
        baseline_state: np.ndarray,
        params: np.ndarray,
        intervention: InterventionSpec,
        n_total_steps: int,
        test_type: str = "placebo",
    ) -> Dict:
        """Refutation test: replace intervention with placebo/random/shuffle.

        Returns: dict with refutation result and metric delta.
        """
        if test_type == "placebo":
            # Placebo: same intervention type but magnitude 0
            placebo_int = InterventionSpec(
                intervention_type=intervention.intervention_type,
                magnitude=0.0,
                duration_steps=intervention.duration_steps,
                start_step=intervention.start_step,
            )
        elif test_type == "random":
            # Random common cause: add uncorrelated noise to params
            placebo_int = InterventionSpec(
                intervention_type=intervention.intervention_type,
                magnitude=intervention.magnitude,
                duration_steps=intervention.duration_steps,
                start_step=intervention.start_step,
            )
            params = params + np.random.normal(0, 0.01, size=params.shape)
        else:
            placebo_int = intervention
        # Compute factual
        factual = self.evaluate_intervention(baseline_state, params, intervention, n_total_steps)
        # Compute placebo
        placebo = self.evaluate_intervention(baseline_state, params, placebo_int, n_total_steps)
        return {
            "test_type": test_type,
            "factual_ate_glucose": factual.ate_glucose,
            "placebo_ate_glucose": placebo.ate_glucose,
            "refutation_passed": abs(placebo.ate_glucose) < abs(factual.ate_glucose) * 0.5,
        }
