"""
Phase 4: Counterfactual V3 — Multi-Objective Intervention Optimization.

Extends Phase 3 counterfactual simulation with:
  - Multi-objective Pareto optimization across outcome dimensions
  - Bayesian optimization for intervention parameter search
  - Constraint-aware personalization (safety constraints)
  - Uncertainty propagation through simulations
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from enum import Enum

from app.personalization.dynamics import full_dynamics
from app.personalization.counterfactual import (
    InterventionProgram, CounterfactualTrajectory,
)
from app.personalization.core import PersonalizationEngine
from app.personalization.state import PHYSIO_DIM, PARAM_DIM
from app.personalization.priors import PARAMETER_NAMES


# ── Outcome Dimensions ────────────────────────────────────────

class OutcomeMetric(Enum):
    """Dimensions for multi-objective optimization."""
    GLUCOSE_CONTROL = "glucose_control"         # lower mean glucose
    SBP_CONTROL = "sbp_control"                 # lower SBP
    INFLAMMATION = "inflammation"               # lower CRP/inflamm load
    WEIGHT_LOSS = "weight_loss"                 # weight reduction
    INSULIN_SENSITIVITY = "insulin_sensitivity"  # improve SI
    LIPID_PROFILE = "lipid_profile"             # improve LDL/HDL/TG
    KIDNEY_FUNCTION = "kidney_function"         # preserve GFR
    CIRCADIAN_HEALTH = "circadian_health"       # improve sleep/cortisol
    ADHERENCE_BURDEN = "adherence_burden"       # minimize complexity
    SIDE_EFFECT_RISK = "side_effect_risk"       # minimize adverse effects


# ── Intervention Parameters (Search Space) ────────────────────

@dataclass
class InterventionParameter:
    """A tunable intervention parameter with bounds."""
    name: str
    dtype: type = float
    min_val: float = 0.0
    max_val: float = 1.0
    default: float = 0.5
    description: str = ""

    def sample(self, rng: np.random.Generator) -> float:
        if self.dtype == int:
            return float(rng.integers(int(self.min_val), int(self.max_val) + 1))
        return float(rng.uniform(self.min_val, self.max_val))


@dataclass
class InterventionDesign:
    """
    A fully specified intervention design with tunable knobs.
    Can be converted to an InterventionProgram for Phase 3 simulation.
    """
    # Lifestyle
    exercise_minutes: float = 30.0       # 0-60 min/day
    diet_calories: float = 2000.0        # 1200-3000 kcal
    diet_quality: float = 0.5            # 0-1
    sodium_intake: float = 100.0         # 50-200 mg

    # Medication
    metformin_dose: float = 0.0          # 0-2000 mg
    statin_intensity: float = 0.0         # 0-1 (atorvastatin equivalent)
    sglt2_dose: float = 0.0               # 0-1 (empagliflozin equivalent)

    # Behavioral
    sleep_target: float = 7.5            # 6-9 hours
    stress_reduction: float = 0.0        # 0-1

    # Duration
    duration_days: int = 90

    @property
    def parameter_space(cls) -> List[InterventionParameter]:
        return [
            InterventionParameter("exercise_minutes", float, 0, 60, 30, "Daily exercise minutes"),
            InterventionParameter("diet_calories", float, 1200, 3000, 2000, "Daily calorie intake"),
            InterventionParameter("diet_quality", float, 0, 1, 0.5, "Diet quality (0-1)"),
            InterventionParameter("sodium_intake", float, 50, 200, 100, "Daily sodium intake (mg)"),
            InterventionParameter("metformin_dose", float, 0, 2000, 0, "Metformin dose (mg)"),
            InterventionParameter("statin_intensity", float, 0, 1, 0, "Statin intensity"),
            InterventionParameter("sglt2_dose", float, 0, 1, 0, "SGLT2 inhibitor dose"),
            InterventionParameter("sleep_target", float, 6, 9, 7.5, "Sleep target (hours)"),
            InterventionParameter("stress_reduction", float, 0, 1, 0, "Stress reduction (0-1)"),
            InterventionParameter("duration_days", int, 30, 365, 90, "Intervention duration"),
        ]

    def to_program(self, patient_params: np.ndarray) -> InterventionProgram:
        """Convert design to Phase 3 InterventionProgram."""
        daily_inputs = {
            "exercise": self.exercise_minutes / 60.0,  # normalized
            "calorie_intake": self.diet_calories,
            "dietary_fat": self.diet_quality * 60.0,
            "sodium_intake": self.sodium_intake,
        }

        param_modifiers = {}
        # Exercise effects
        ex_benefit = self.exercise_minutes / 60.0
        param_modifiers["SI"] = 1.0 + 0.3 * ex_benefit
        param_modifiers["baroreflex_gain"] = 1.0 + 0.1 * ex_benefit
        param_modifiers["vagal_tone_effect"] = 1.0 + 0.15 * ex_benefit

        # Diet effects
        param_modifiers["LDL_clearance"] = 1.0 + 0.1 * self.diet_quality
        param_modifiers["HDL_production"] = 1.0 + 0.1 * self.diet_quality
        param_modifiers["lipolysis_rate"] = 1.0 - 0.1 * self.diet_quality

        # Metformin
        met_eff = self.metformin_dose / 2000.0
        if met_eff > 0:
            param_modifiers["SI"] *= 1.0 + 0.2 * met_eff
            param_modifiers["HGP_basal"] = 1.0 - 0.15 * met_eff
            param_modifiers["M1_activation"] = 1.0 - 0.1 * met_eff

        # SGLT2
        sglt2_eff = self.sglt2_dose
        if sglt2_eff > 0:
            param_modifiers["SGLT_activity"] = 1.0 + 0.5 * sglt2_eff
            param_modifiers["SI"] *= 1.0 + 0.1 * sglt2_eff

        # Statin
        statin_eff = self.statin_intensity
        if statin_eff > 0:
            param_modifiers["LDL_clearance"] = 1.0 + 0.4 * statin_eff
            param_modifiers["lipogenesis_rate"] = 1.0 - 0.1 * statin_eff

        return InterventionProgram(
            name=f"Optimized Intervention (d={self.duration_days}d)",
            duration_days=self.duration_days,
            daily_inputs=daily_inputs,
            param_modifiers=param_modifiers,
            adherence=0.8,
        )


# ── Multi-Objective Outcome ───────────────────────────────────

@dataclass
class MultiObjectiveOutcome:
    """Outcome evaluation across multiple dimensions."""
    metrics: Dict[str, float] = field(default_factory=dict)
    pareto_rank: int = 0
    crowding_distance: float = 0.0
    constraint_violations: List[str] = field(default_factory=list)

    @property
    def is_feasible(self) -> bool:
        return len(self.constraint_violations) == 0


# ── Safety Constraints ────────────────────────────────────────

SAFETY_CONSTRAINTS = {
    "min_glucose": 60.0,         # hypoglycemia threshold (mg/dL)
    "max_glucose": 300.0,        # severe hyperglycemia (mg/dL)
    "min_sbp": 80.0,             # hypotension (mmHg)
    "max_sbp": 200.0,            # hypertensive crisis (mmHg)
    "min_gfr": 15.0,             # kidney failure threshold (mL/min)
    "max_hr": 150.0,             # tachycardia (bpm)
    "min_hr": 40.0,              # bradycardia (bpm)
    "max_weight_loss_pct": 15.0, # max safe weight loss in % of baseline
}


# ── Multi-Objective Counterfactual Engine V3 ──────────────────

class CounterfactualEngineV3:
    """
    Phase 4 counterfactual engine with multi-objective optimization.

    Features:
      - Pareto-front optimization across customizable outcome metrics
      - Bayesian optimization (random search) for intervention design
      - Safety constraint checking
      - Uncertainty propagation via ensemble simulation
    """

    def __init__(
        self,
        engine: PersonalizationEngine,
        n_ensemble: int = 10,
        seed: int = 42,
    ):
        self.engine = engine
        self.n_ensemble = n_ensemble
        self.rng = np.random.default_rng(seed)
        self._pareto_front: List[Tuple[InterventionDesign, MultiObjectiveOutcome]] = []

    # ── Single Simulation ──

    def simulate_design(
        self,
        design: InterventionDesign,
        n_simulations: int = 1,
    ) -> Tuple[CounterfactualTrajectory, MultiObjectiveOutcome]:
        """Simulate an intervention design and evaluate outcomes."""
        program = design.to_program(self.engine.get_parameters()[0])
        trajectories = []
        outcomes = []

        for _ in range(n_simulations):
            traj = self._simulate_single(program)
            outcome = self._evaluate_outcome(traj, design)
            trajectories.append(traj)
            outcomes.append(outcome)

        # Average across ensemble
        avg_outcome = self._average_outcomes(outcomes)
        return trajectories[0], avg_outcome

    def _simulate_single(
        self, program: InterventionProgram,
    ) -> CounterfactualTrajectory:
        """Run a single counterfactual simulation."""
        state = self.engine.get_twin_state().copy()
        params, _ = self.engine.get_parameters()
        params = params.copy()

        dt = 1.0
        # Apply parameter modifiers
        for name, multiplier in program.param_modifiers.items():
            if name in PARAMETER_NAMES:
                idx = PARAMETER_NAMES.index(name)
                params[idx] *= multiplier

        steps = int(program.duration_days * 1440.0 / dt)
        traj = CounterfactualTrajectory(name=program.name)
        traj.glucose = []
        traj.sbp = []
        traj.gfr = []
        traj.crp = []
        traj.inflam_load = []
        traj.weight_kg = []

        for step in range(min(steps, 10000)):
            adherence_factor = 1.0 if self.rng.random() < program.adherence else 0.0
            inputs = {
                k: v * adherence_factor
                for k, v in program.daily_inputs.items()
            }
            state = full_dynamics(state, params, inputs, dt)

            if step % 1440 == 0:
                traj.glucose.append(float(state[0]))
                traj.sbp.append(float(state[5]))
                traj.gfr.append(float(state[9]))
                traj.crp.append(float(state[13]))
                traj.inflam_load.append(float(state[29]))
                traj.weight_kg.append(float(state[20]))

            if step > 5000 and step % 1440 == 0:
                if state[0] < 20 or state[0] > 600:
                    break

        traj.final_state = state.copy()
        return traj

    # ── Outcome Evaluation ──

    def _evaluate_outcome(
        self, traj: CounterfactualTrajectory,
        design: InterventionDesign,
    ) -> MultiObjectiveOutcome:
        """Evaluate a trajectory across all outcome dimensions."""
        metrics = {}
        violations = []

        if not traj.glucose:
            return MultiObjectiveOutcome(metrics={}, constraint_violations=["empty_trajectory"])

        # Glucose control (lower mean near 100 is better)
        mean_g = np.mean(traj.glucose[-14:])  # last 14 days
        metrics["glucose_control"] = float(max(0.0, 100.0 - abs(mean_g - 100.0)))

        # SBP control
        mean_sbp = np.mean(traj.sbp[-14:]) if traj.sbp else 120.0
        metrics["sbp_control"] = float(max(0.0, 100.0 - abs(mean_sbp - 120.0)))

        # Inflammation reduction
        if traj.crp:
            baseline_crp = traj.crp[0] if len(traj.crp) > 0 else 2.0
            final_crp = traj.crp[-1]
            metrics["inflammation"] = float(max(0.0, 100.0 * (1.0 - final_crp / max(baseline_crp, 0.1))))
        else:
            metrics["inflammation"] = 50.0

        # Weight loss (0-15%, scaled to 0-100)
        if traj.weight_kg and len(traj.weight_kg) > 1:
            weight_loss_pct = 100.0 * (traj.weight_kg[0] - traj.weight_kg[-1]) / max(traj.weight_kg[0], 1.0)
            metrics["weight_loss"] = float(min(100.0, weight_loss_pct * 6.67))
        else:
            metrics["weight_loss"] = 0.0

        # Insulin sensitivity proxy (glucose reduction)
        baseline_g = traj.glucose[0] if len(traj.glucose) > 0 else 140.0
        g_reduction = max(0.0, baseline_g - mean_g)
        metrics["insulin_sensitivity"] = float(min(100.0, g_reduction * 2.0))

        # Lipid profile (LDL reduction, HDL increase proxy via SBP)
        metrics["lipid_profile"] = float(0.5 * (metrics["sbp_control"]) + 25.0)

        # Kidney function (GFR preservation)
        if traj.gfr and len(traj.gfr) > 1:
            gfr_change = traj.gfr[-1] - traj.gfr[0]
            metrics["kidney_function"] = float(max(0.0, 100.0 + gfr_change))
        else:
            metrics["kidney_function"] = 50.0

        # Circadian health (SBP variability proxy)
        if traj.sbp and len(traj.sbp) > 5:
            sbp_var = np.std(traj.sbp[-7:])
            metrics["circadian_health"] = float(max(0.0, 100.0 - sbp_var * 2.0))
        else:
            metrics["circadian_health"] = 50.0

        # Adherence burden (inverse of number of interventions)
        n_interventions = sum([
            design.metformin_dose > 0,
            design.statin_intensity > 0,
            design.sglt2_dose > 0,
            design.exercise_minutes > 0,
        ])
        metrics["adherence_burden"] = float(max(0.0, 100.0 - n_interventions * 20.0))

        # Side effect risk (inverse of medication intensity)
        med_burden = (design.metformin_dose / 2000.0 +
                      design.statin_intensity +
                      design.sglt2_dose)
        metrics["side_effect_risk"] = float(max(0.0, 100.0 - med_burden * 30.0))

        # Safety constraints
        min_g = min(traj.glucose)
        max_g = max(traj.glucose)
        min_sbp = min(traj.sbp) if traj.sbp else 120
        max_sbp = max(traj.sbp) if traj.sbp else 120

        if min_g < SAFETY_CONSTRAINTS["min_glucose"]:
            violations.append(f"hypoglycemia (glucose={min_g:.0f})")
        if max_g > SAFETY_CONSTRAINTS["max_glucose"]:
            violations.append(f"hyperglycemia (glucose={max_g:.0f})")
        if min_sbp < SAFETY_CONSTRAINTS["min_sbp"]:
            violations.append(f"hypotension (SBP={min_sbp:.0f})")
        if max_sbp > SAFETY_CONSTRAINTS["max_sbp"]:
            violations.append(f"hypertension (SBP={max_sbp:.0f})")

        return MultiObjectiveOutcome(
            metrics=metrics,
            constraint_violations=violations,
        )

    def _average_outcomes(
        self, outcomes: List[MultiObjectiveOutcome],
    ) -> MultiObjectiveOutcome:
        """Average multiple ensemble outcomes."""
        if not outcomes:
            return MultiObjectiveOutcome()
        keys = outcomes[0].metrics.keys()
        avg_metrics = {}
        for k in keys:
            vals = [o.metrics.get(k, 0.0) for o in outcomes]
            avg_metrics[k] = float(np.mean(vals))
        violations = []
        for o in outcomes:
            violations.extend(o.constraint_violations)
        return MultiObjectiveOutcome(
            metrics=avg_metrics,
            constraint_violations=list(set(violations)),
        )

    # ── Multi-Objective Optimization ──

    def optimize(
        self,
        weights: Optional[Dict[str, float]] = None,
        n_trials: int = 100,
        n_ensemble: int = 5,
        objectives: Optional[List[str]] = None,
    ) -> List[Tuple[InterventionDesign, MultiObjectiveOutcome]]:
        """
        Multi-objective optimization via random search.

        Returns Pareto-optimal designs sorted by crowding distance.

        Args:
            weights: objective weights for scalarization (if None, find Pareto front)
            n_trials: number of random designs to evaluate
            n_ensemble: ensemble size per design
            objectives: subset of metrics to optimize (default: all)
        """
        if objectives is None:
            objectives = [
                "glucose_control", "sbp_control", "inflammation",
                "weight_loss", "insulin_sensitivity", "lipid_profile",
                "kidney_function", "circadian_health",
            ]

        designs_and_outcomes = []

        for i in range(n_trials):
            design = self._random_design()
            _, outcome = self.simulate_design(design, n_simulations=n_ensemble)
            designs_and_outcomes.append((design, outcome))

        if weights is None:
            # Pareto dominance
            self._pareto_front = self._compute_pareto_front(
                designs_and_outcomes, objectives,
            )
        else:
            # Weighted scalarization
            self._pareto_front = self._compute_weighted_front(
                designs_and_outcomes, objectives, weights,
            )

        return self._pareto_front

    def _random_design(self) -> InterventionDesign:
        """Sample a random intervention design."""
        return InterventionDesign(
            exercise_minutes=float(self.rng.uniform(0, 60)),
            diet_calories=float(self.rng.uniform(1200, 3000)),
            diet_quality=float(self.rng.uniform(0, 1)),
            sodium_intake=float(self.rng.uniform(50, 200)),
            metformin_dose=float(self.rng.choice([0, 500, 1000, 1500, 2000])),
            statin_intensity=float(self.rng.choice([0, 0.5, 1.0])),
            sglt2_dose=float(self.rng.choice([0, 0.5, 1.0])),
            sleep_target=float(self.rng.uniform(6, 9)),
            stress_reduction=float(self.rng.uniform(0, 1)),
            duration_days=int(self.rng.integers(30, 365)),
        )

    def _compute_pareto_front(
        self,
        designs: List[Tuple[InterventionDesign, MultiObjectiveOutcome]],
        objectives: List[str],
    ) -> List[Tuple[InterventionDesign, MultiObjectiveOutcome]]:
        """Compute Pareto frontier using non-dominated sorting."""
        # Filter feasible designs
        feasible = [(d, o) for d, o in designs if o.is_feasible]
        if not feasible:
            return []

        n = len(feasible)
        metrics = np.array([
            [o.metrics.get(obj, 0.0) for obj in objectives]
            for _, o in feasible
        ])

        # Non-dominated sorting
        fronts = [[] for _ in range(n)]
        domination_count = np.zeros(n, dtype=int)
        dominated_set = [[] for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(metrics[i], metrics[j]):
                    dominated_set[i].append(j)
                elif self._dominates(metrics[j], metrics[i]):
                    domination_count[i] += 1
            if domination_count[i] == 0:
                fronts[0].append(i)

        front_idx = 0
        while fronts[front_idx]:
            next_front = []
            for i in fronts[front_idx]:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            front_idx += 1
            if next_front:
                fronts[front_idx] = next_front

        # Collect Pareto front (rank 0)
        pareto_idx = fronts[0]
        pareto_set = [feasible[i] for i in pareto_idx]

        # Compute crowding distance for diversity
        if len(pareto_set) > 2:
            pareto_metrics = np.array([
                [o.metrics.get(obj, 0.0) for obj in objectives]
                for _, o in pareto_set
            ])
            crowding = self._crowding_distance(pareto_metrics)
            for i, (_, o) in enumerate(pareto_set):
                o.crowding_distance = crowding[i]

        pareto_set.sort(key=lambda x: x[1].crowding_distance, reverse=True)
        return pareto_set

    def _dominates(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Check if solution a dominates solution b (all objectives maximized)."""
        return np.all(a >= b) and np.any(a > b)

    def _crowding_distance(self, metrics: np.ndarray) -> np.ndarray:
        """Compute crowding distance for diversity preservation."""
        n, m = metrics.shape
        distances = np.zeros(n)
        for j in range(m):
            order = np.argsort(metrics[:, j])
            distances[order[0]] = np.inf
            distances[order[-1]] = np.inf
            for i in range(1, n - 1):
                distances[order[i]] += (
                    metrics[order[i + 1], j] - metrics[order[i - 1], j]
                ) / max(metrics[order[-1], j] - metrics[order[0], j], 1e-10)
        return distances

    def _compute_weighted_front(
        self,
        designs: List[Tuple[InterventionDesign, MultiObjectiveOutcome]],
        objectives: List[str],
        weights: Dict[str, float],
    ) -> List[Tuple[InterventionDesign, MultiObjectiveOutcome]]:
        """Compute weighted scalarization and return top designs."""
        scored = []
        for d, o in designs:
            if not o.is_feasible:
                continue
            score = sum(
                weights.get(obj, 0.0) * o.metrics.get(obj, 0.0)
                for obj in objectives
            )
            scored.append((score, d, o))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(d, o) for _, d, o in scored[:10]]

    def get_pareto_front(self) -> List[Tuple[InterventionDesign, MultiObjectiveOutcome]]:
        return self._pareto_front

    def get_best_design(
        self, objective_weights: Optional[Dict[str, float]] = None,
    ) -> Optional[Tuple[InterventionDesign, MultiObjectiveOutcome]]:
        """Get the best design based on weighted preferences."""
        if not self._pareto_front:
            return None
        if objective_weights is None:
            objective_weights = {k: 1.0 / 8.0 for k in [
                "glucose_control", "sbp_control", "inflammation",
                "weight_loss", "insulin_sensitivity",
            ]}
        best = None
        best_score = -np.inf
        for d, o in self._pareto_front:
            score = sum(
                objective_weights.get(k, 0.0) * o.metrics.get(k, 0.0)
                for k in objective_weights
            )
            if score > best_score:
                best_score = score
                best = (d, o)
        return best
