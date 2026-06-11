"""
Phase 3: Counterfactual Engine V2.

Multi-intervention counterfactual simulation with full uncertainty.
Supports intervention programs (diet + exercise + medication combos).
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from .core import PersonalizationEngine, PHYSIO_DIM


@dataclass
class InterventionProgram:
    """A multi-component intervention program."""
    name: str
    duration_days: int
    daily_inputs: Dict[str, float]         # applied every step
    param_modifiers: Dict[str, float]       # parameter multipliers
    adherence: float = 0.8                  # base adherence rate


@dataclass
class CounterfactualTrajectory:
    """Simulated trajectory under an intervention."""
    name: str
    glucose: List[float]
    sbp: List[float]
    hba1c_estimated: Optional[List[float]] = None
    weight_kg: Optional[List[float]] = None
    gfr: Optional[List[float]] = None
    crp: Optional[List[float]] = None
    inflam_load: Optional[List[float]] = None
    final_state: Optional[np.ndarray] = None


class CounterfactualEngine:
    """
    Phase 3 counterfactual simulator for multi-intervention programs.
    """

    def __init__(self, engine: PersonalizationEngine):
        self.engine = engine

    def simulate_program(
        self,
        program: InterventionProgram,
        dt: float = 1.0,
    ) -> CounterfactualTrajectory:
        """
        Simulate a full intervention program forward in time.
        Applies daily inputs and parameter modifications.
        """
        from .dynamics import full_dynamics

        state = self.engine.get_twin_state().copy()
        params, param_cov = self.engine.get_parameters()
        params = params.copy()

        # Apply parameter modifiers
        for name, multiplier in program.param_modifiers.items():
            from .priors import PARAMETER_NAMES
            if name in PARAMETER_NAMES:
                idx = PARAMETER_NAMES.index(name)
                params[idx] *= multiplier

        steps = int(program.duration_days * 1440.0 / dt)
        glucose_traj = []
        sbp_traj = []
        gfr_traj = []
        crp_traj = []
        inflam_traj = []
        weight_traj = []

        for step in range(min(steps, 10000)):
            adherence_factor = 1.0 if np.random.random() < program.adherence else 0.0
            inputs = {
                k: v * adherence_factor
                for k, v in program.daily_inputs.items()
            }
            state = full_dynamics(state, params, inputs, dt)

            if step % 1440 == 0:  # once per day
                glucose_traj.append(float(state[0]))
                sbp_traj.append(float(state[5]))
                gfr_traj.append(float(state[9]))
                crp_traj.append(float(state[13]))
                inflam_traj.append(float(state[29]))
                weight_traj.append(float(state[20]))

            if step > 5000 and step % 1440 == 0:
                # Check for physiological collapse
                if state[0] < 20 or state[0] > 600:
                    break

        return CounterfactualTrajectory(
            name=program.name,
            glucose=glucose_traj,
            sbp=sbp_traj,
            gfr=gfr_traj,
            crp=crp_traj,
            inflam_load=inflam_traj,
            weight_kg=weight_traj,
            final_state=state,
        )

    def compare_programs(
        self,
        programs: List[InterventionProgram],
    ) -> Dict[str, CounterfactualTrajectory]:
        """Run multiple programs and return all trajectories."""
        results = {}
        base_state = self.engine.get_twin_state().copy()
        for program in programs:
            traj = self.simulate_program(program)
            results[program.name] = traj
            # Reset state for each simulation
            self.engine.filter._mu[:PHYSIO_DIM] = base_state
        return results

    def estimate_hba1c(self, glucose_trajectory: List[float]) -> float:
        """Estimate HbA1c from average glucose (ADAG formula)."""
        if not glucose_trajectory:
            return 5.5
        avg_g = np.mean(glucose_trajectory[-30:])  # last 30 days
        return (avg_g + 46.7) / 28.7  # ADAG formula

    def program_summary(self, traj: CounterfactualTrajectory) -> Dict:
        """Generate summary statistics for a counterfactual trajectory."""
        if not traj.glucose:
            return {"error": "empty trajectory"}
        return {
            "program": traj.name,
            "final_glucose": traj.glucose[-1] if traj.glucose else None,
            "avg_glucose": float(np.mean(traj.glucose)),
            "final_sbp": traj.sbp[-1] if traj.sbp else None,
            "avg_sbp": float(np.mean(traj.sbp)),
            "estimated_hba1c": self.estimate_hba1c(traj.glucose),
            "weight_change_kg": (
                traj.weight_kg[-1] - traj.weight_kg[0]
                if traj.weight_kg and len(traj.weight_kg) > 1
                else 0.0
            ),
        }


# ── Pre-built intervention programs ────────────────────────────

MEDITERRANEAN_DIET = InterventionProgram(
    name="Mediterranean Diet",
    duration_days=90,
    daily_inputs={
        "dietary_fat": 60.0,      # g/day (~35% from healthy fats)
        "calorie_intake": 1800.0,  # kcal/day
        "sodium_intake": 70.0,    # mg/day (low sodium)
    },
    param_modifiers={
        "SI": 1.15,               # 15% improvement
        "LDL_clearance": 1.10,
        "HDL_production": 1.15,
        "lipolysis_rate": 0.90,
    },
    adherence=0.75,
)

EXERCISE_PROGRAM = InterventionProgram(
    name="Exercise 150min/week",
    duration_days=90,
    daily_inputs={
        "exercise": 0.25,          # ~30min moderate exercise
        "calorie_intake": 2100.0,
    },
    param_modifiers={
        "SI": 1.30,
        "lipolysis_rate": 1.20,
        "HDL_production": 1.10,
        "LDL_clearance": 1.05,
        "vagal_tone_effect": 1.20,
        "baroreflex_gain": 1.10,
    },
    adherence=0.70,
)

METFORMIN = InterventionProgram(
    name="Metformin",
    duration_days=90,
    daily_inputs={},
    param_modifiers={
        "SI": 1.25,
        "HGP_basal": 0.85,
        "lipogenesis_rate": 0.90,
        "M1_activation": 0.85,
    },
    adherence=0.90,
)

COMBINED_THERAPY = InterventionProgram(
    name="Combined: Diet + Exercise + Metformin",
    duration_days=180,
    daily_inputs={
        "dietary_fat": 55.0,
        "calorie_intake": 1750.0,
        "sodium_intake": 65.0,
        "exercise": 0.30,
    },
    param_modifiers={
        "SI": 1.50,
        "HGP_basal": 0.80,
        "LDL_clearance": 1.15,
        "HDL_production": 1.20,
        "lipolysis_rate": 1.15,
        "lipogenesis_rate": 0.85,
        "vagal_tone_effect": 1.25,
        "baroreflex_gain": 1.15,
        "M1_activation": 0.80,
        "NFkB_sensitivity": 0.85,
    },
    adherence=0.75,
)
