"""
Phase 2 Hierarchical Drift Detection — Multi-Organ.

Level 1 (Warning):         3 consecutive  |residual| > 2 * uncertainty
Level 2 (Recalibrate):     5 consecutive
Level 3 (Twin Invalid):   10 consecutive

Each subsystem (metabolic, CV, renal) has its own detector.
Global drift = max(subsystem level).
"""

from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass, field
from enum import IntEnum

logger = logging.getLogger(__name__)

LEVEL_1_THRESHOLD = 3
LEVEL_2_THRESHOLD = 5
LEVEL_3_THRESHOLD = 10


class DriftLevel(IntEnum):
    NORMAL = 0
    WARNING = 1
    RECALIBRATE = 2
    INVALID = 3


LABEL_MAP = {0: "normal", 1: "warning", 2: "recalibrate", 3: "invalid"}


@dataclass
class SubsystemDrift:
    name: str
    _violations: List[bool] = field(default_factory=list)
    _residuals: List[float] = field(default_factory=list)
    _level: int = 0

    def check(self, observed: float, predicted_mean: float, uncertainty: float) -> None:
        residual = observed - predicted_mean
        self._residuals.append(residual)
        is_violation = abs(residual) > 2.0 * max(uncertainty, 1e-6)
        self._violations.append(is_violation)
        self._evaluate()

    def _evaluate(self) -> None:
        consecutive = 0
        for v in reversed(self._violations):
            if v:
                consecutive += 1
            else:
                break
        if consecutive >= LEVEL_3_THRESHOLD:
            self._level = 3
        elif consecutive >= LEVEL_2_THRESHOLD:
            self._level = 2
        elif consecutive >= LEVEL_1_THRESHOLD:
            self._level = 1
        else:
            self._level = 0

    @property
    def level(self) -> int:
        return self._level

    @property
    def label(self) -> str:
        return LABEL_MAP.get(self._level, "unknown")

    def status(self) -> Dict[str, Any]:
        return {
            "level": self._level,
            "label": self.label,
            "consecutive_violations": self._count_consecutive(),
            "total_violations": sum(self._violations),
        }

    def _count_consecutive(self) -> int:
        c = 0
        for v in reversed(self._violations):
            if v:
                c += 1
            else:
                break
        return c

    def reset(self) -> None:
        self._violations.clear()
        self._residuals.clear()
        self._level = 0


class DriftDetector:

    def __init__(self):
        self.subsystems: Dict[str, SubsystemDrift] = {
            "metabolic": SubsystemDrift("metabolic"),
            "cardiovascular": SubsystemDrift("cardiovascular"),
            "renal": SubsystemDrift("renal"),
        }
        self._global_level: int = 0

    def check(
        self,
        observed: float,
        predicted_mean: float,
        prediction_uncertainty: float,
        subsystem: str = "metabolic",
    ) -> None:
        sub = self.subsystems.get(subsystem)
        if sub is None:
            logger.warning(f"Unknown subsystem '{subsystem}' — defaulting to metabolic")
            sub = self.subsystems["metabolic"]
        sub.check(observed, predicted_mean, prediction_uncertainty)
        self._rebuild_global()

    def check_multi(
        self,
        observations: Dict[str, float],
        predictions: Dict[str, float],
        uncertainties: Dict[str, float],
    ) -> None:
        for key in observations:
            sub = self.subsystems.get(key)
            if sub:
                sub.check(
                    observations[key],
                    predictions.get(key, 0.0),
                    uncertainties.get(key, 1.0),
                )
        self._rebuild_global()

    def _rebuild_global(self) -> None:
        self._global_level = max(s.level for s in self.subsystems.values())

    @property
    def level(self) -> int:
        return self._global_level

    @property
    def label(self) -> str:
        return LABEL_MAP.get(self._global_level, "unknown")

    @property
    def can_run_counterfactuals(self) -> bool:
        return self._global_level < 3

    def status(self) -> Dict[str, Any]:
        return {
            "level": self._global_level,
            "label": self.label,
            "subsystems": {k: v.status() for k, v in self.subsystems.items()},
        }

    def subsystem_status(self, name: str) -> Optional[Dict[str, Any]]:
        sub = self.subsystems.get(name)
        return sub.status() if sub else None

    def reset(self) -> None:
        for s in self.subsystems.values():
            s.reset()
        self._global_level = 0


# ── Counterfactual Simulator ───────────────────────────────────

@dataclass
class CounterfactualResult:
    intervention: str
    baseline_outcome: float
    cf_outcome: float
    delta: float


class CounterfactualSimulator:
    """
    Simulate alternative trajectories by modifying UKF state/params.

    Only available when drift < Level 3.
    """

    def __init__(self, engine: Any):
        self.engine = engine

    def simulate_insulin_sensitivity_change(
        self, SI_multiplier: float = 1.5, steps: int = 5
    ) -> CounterfactualResult:
        base_state = self.engine.get_twin_state().copy()
        base_params, _ = self.engine.get_parameters()
        from .dynamics import compute_metabolic_dynamics
        from .state import MetabolicState

        base_meta = MetabolicState.from_array(base_state)
        base_future = base_meta
        for _ in range(steps):
            base_future = compute_metabolic_dynamics(base_future, {}, {})
        base_G = base_future.G

        cf_params = base_params.copy()
        cf_params[0] *= SI_multiplier
        cf_meta = MetabolicState.from_array(base_state)
        cf_future = cf_meta
        for _ in range(steps):
            cf_future = compute_metabolic_dynamics(cf_future, {}, {})
        cf_G = cf_future.G

        return CounterfactualResult(
            intervention=f"SI × {SI_multiplier}",
            baseline_outcome=float(base_G),
            cf_outcome=float(cf_G),
            delta=float(cf_G - base_G),
        )
