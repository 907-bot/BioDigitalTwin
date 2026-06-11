"""
Phase 3: Reinforcement Learning Layer.

Goal: Learn optimal intervention policies from the digital twin.

State: 30-dim physiological state + time + past interventions
Action: exercise dose, medication dose, dietary composition
Reward: composite of HbA1c↓, BP↓, risk↓, adverse events↓

Framework architecture only — offline RL training requires
large batches of counterfactual simulations before deployment.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field
from .core import PersonalizationEngine, PHYSIO_DIM, PARAM_DIM
from .counterfactual import InterventionProgram, CounterfactualTrajectory, MEDITERRANEAN_DIET, EXERCISE_PROGRAM, METFORMIN


@dataclass
class Experience:
    """A single (state, action, reward, next_state) transition."""
    state: np.ndarray
    action: np.ndarray      # [exercise, diet_adherence, medication]
    reward: float
    next_state: np.ndarray
    done: bool


@dataclass
class RLPolicy:
    """Learned intervention policy."""
    name: str
    description: str
    action_dim: int = 3  # exercise, diet, medication
    policy_params: Dict = field(default_factory=dict)


class RLTwinEnvironment:
    """
    Gym-like environment wrapping the digital twin.
    Used for training RL agents.

    Action space (3-dim):
      [0]: exercise dose (0-1, fraction of 150min/week target)
      [1]: diet quality (0-1, 0=standard, 1=optimal Mediterranean)
      [2]: medication adherence (0-1)

    Reward components:
      - glucose regulation: -0.1 * |G - 90|
      - BP control: -0.05 * max(0, SBP - 130)
      - inflammation: -0.1 * InflammatoryLoad / 100
      - metabolic flexibility: +0.2 * flexibility_score / 100
      - intervention cost: -0.05 * action_cost
    """

    def __init__(self, engine: PersonalizationEngine):
        self.engine = engine
        self.state = engine.get_twin_state().copy()
        self.params, _ = engine.get_parameters()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take a step (1 day = 1440 min of dt=1)."""
        from .dynamics import full_dynamics

        exercise = np.clip(action[0], 0, 1) * 0.5
        diet_quality = np.clip(action[1], 0, 1)
        med_adh = np.clip(action[2], 0, 1)

        # Simulate 1 day
        for _ in range(1440):
            inputs = {
                "exercise": exercise,
                "meal_glucose": 50.0 * (1.0 - diet_quality * 0.4),
                "dietary_fat": 80.0 * (1.0 - diet_quality * 0.4),
                "calorie_intake": 2000.0 * (1.0 - diet_quality * 0.15),
                "sodium_intake": 100.0 * (1.0 - diet_quality * 0.3),
            }
            self.state = full_dynamics(self.state, self.params, inputs)

        G = self.state[0]
        SBP = self.state[5]
        infl_load = self.state[29]
        flex = self._compute_flexibility()

        glucose_reward = -0.1 * abs(G - 90.0)
        bp_reward = -0.05 * max(0, SBP - 130.0)
        inflam_reward = -0.1 * infl_load / 100.0
        flex_reward = 0.2 * flex / 100.0
        action_cost = -0.05 * (exercise + diet_quality + med_adh)

        reward = glucose_reward + bp_reward + inflam_reward + flex_reward + action_cost
        done = bool(G < 30 or G > 500 or SBP > 250 or SBP < 40)
        info = {
            "G": float(G), "SBP": float(SBP),
            "inflam_load": float(infl_load), "flexibility": float(flex),
        }
        return self.state.copy(), reward, done, info

    def reset(self) -> np.ndarray:
        self.state = self.engine.get_twin_state().copy()
        return self.state.copy()

    @staticmethod
    def _compute_flexibility(ir: float = 5.0, ffa: float = 0.5) -> float:
        ir_pen = max(0, min(1, (ir - 3) / 10)) * 40
        ffa_score = max(0, min(1, (ffa - 0.3) / 0.8)) * 20
        return max(0, 100 - ir_pen - ffa_score)


class ConservativeQLearning:
    """
    Conservative Q-Learning for offline RL.
    Architecture reference — full training requires offline dataset.

    Key idea: penalize Q-values for out-of-distribution actions
    to avoid overestimating untested interventions.
    """

    def __init__(self, state_dim: int = PHYSIO_DIM, action_dim: int = 3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.Q = np.zeros((state_dim, action_dim))  # placeholder

    def get_action(self, state: np.ndarray, epsilon: float = 0.1) -> np.ndarray:
        """Epsilon-greedy action selection."""
        if np.random.random() < epsilon:
            return np.random.uniform(0, 1, self.action_dim)
        # Placeholder: return moderate intervention
        return np.array([0.3, 0.5, 0.8])

    def update(self, batch: List[Experience]) -> None:
        """CQL update step — requires large offline dataset."""
        pass


def recommend_intervention(
    engine: PersonalizationEngine,
    risk_tolerance: str = "moderate",
) -> InterventionProgram:
    """
    Rule-based intervention recommendation based on twin state.
    Acts as a safe fallback before RL policy is trained.
    """
    state = engine.get_twin_state()
    params, _ = engine.get_parameters()

    G = state[0]
    SBP = state[5]
    infl_load = state[29]
    ir = state[4]
    ldl = state[22]
    hdl = state[23]
    bmi_est = state[20] / 0.8 + 22  # rough BMI from fat_mass

    components = []

    if G > 110 or ir > 5:
        components.append(METFORMIN)
    if SBP > 135:
        pass  # would add BP med in production
    if bmi_est > 27:
        components.append(MEDITERRANEAN_DIET)
    if True:
        components.append(EXERCISE_PROGRAM)

    if len(components) == 0:
        return InterventionProgram("Maintenance", 30, {}, {})
    if len(components) == 1:
        return components[0]

    # Combine into a single program
    combined_inputs = {}
    combined_modifiers = {}
    for c in components:
        combined_inputs.update(c.daily_inputs)
        for k, v in c.param_modifiers.items():
            if k in combined_modifiers:
                combined_modifiers[k] = (combined_modifiers[k] + v) / 2
            else:
                combined_modifiers[k] = v

    return InterventionProgram(
        name="Recommended: " + " + ".join(c.name for c in components),
        duration_days=90,
        daily_inputs=combined_inputs,
        param_modifiers=combined_modifiers,
        adherence=0.75,
    )



