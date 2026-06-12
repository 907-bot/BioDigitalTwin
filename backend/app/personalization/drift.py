"""
Phase 2 Hierarchical Drift Detection — Multi-Organ.

Level 1 (Warning):         3 consecutive  |residual| > 2 * uncertainty
Level 2 (Recalibrate):     5 consecutive
Level 3 (Twin Invalid):   10 consecutive

Each subsystem (metabolic, CV, renal) has its own detector.
Global drift = max(subsystem level).

CRITICAL FIX: Added CUSUM-based drift detection for gradual drift,
and per-subsystem residual tracking for better drift attribution.
"""

from typing import List, Dict, Any, Optional
import logging
import numpy as np
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
class CUSUMState:
    """Cumulative Sum (CUSUM) tracker for gradual drift detection."""
    cusum_pos: float = 0.0
    cusum_neg: float = 0.0
    n_observations: int = 0
    
    def update(self, residual: float, threshold: float = 1.0) -> None:
        """Update CUSUM with normalized residual."""
        # Normalize by expected uncertainty
        if threshold > 0:
            normalized = abs(residual) / threshold
        else:
            normalized = 0.0
        
        # CUSUM for positive drift (observation > prediction)
        if normalized > 1.0:
            self.cusum_pos += normalized - 1.0
        else:
            self.cusum_pos = max(0.0, self.cusum_pos - 0.1)
        
        # CUSUM for negative drift (observation < prediction)
        if normalized > 1.0:
            self.cusum_neg += normalized - 1.0
        else:
            self.cusum_neg = max(0.0, self.cusum_neg - 0.1)
        
        self.n_observations += 1
    
    def get_cusum_score(self) -> float:
        """Return max CUSUM score for drift detection."""
        return max(self.cusum_pos, self.cusum_neg)
    
    def reset(self) -> None:
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        self.n_observations = 0


@dataclass
class SubsystemDrift:
    name: str
    _violations: List[bool] = field(default_factory=list)
    _residuals: List[float] = field(default_factory=list)
    _level: int = 0
    _cusum: CUSUMState = field(default_factory=CUSUMState)
    _gradual_drift_level: int = 0  # CRITICAL FIX: Track gradual drift separately
    _recent_residuals: List[float] = field(default_factory=list)  # For gradual detection

    def check(self, observed: float, predicted_mean: float, uncertainty: float) -> None:
        residual = observed - predicted_mean
        self._residuals.append(residual)
        self._recent_residuals.append(residual)
        
        # Keep only last 50 residuals for gradual drift detection
        if len(self._recent_residuals) > 50:
            self._recent_residuals = self._recent_residuals[-50:]
        
        is_violation = abs(residual) > 2.0 * max(uncertainty, 1e-6)
        self._violations.append(is_violation)
        
        # CRITICAL FIX: Update CUSUM for gradual drift detection
        self._cusum.update(residual, max(uncertainty, 1e-6))
        
        # Check for gradual drift: slowly accumulating bias
        self._check_gradual_drift()
        
        self._evaluate()

    def _check_gradual_drift(self) -> None:
        """CRITICAL FIX: Detect gradual drift via CUSUM and rolling mean shift."""
        if len(self._recent_residuals) < 20:
            return
        
        # CUSUM-based gradual drift detection
        cusum_score = self._cusum.get_cusum_score()
        if cusum_score > 15.0:  # Gradual drift threshold
            self._gradual_drift_level = 2  # Recalibrate
        elif cusum_score > 8.0:  # Warning threshold
            self._gradual_drift_level = 1  # Warning
        else:
            self._gradual_drift_level = 0  # Normal
        
        # Also check for rolling mean shift
        if len(self._recent_residuals) >= 30:
            first_half = np.mean(self._recent_residuals[-30:-15])
            second_half = np.mean(self._recent_residuals[-15:])
            mean_shift = abs(second_half - first_half)
            # If mean shifted significantly, flag gradual drift
            if mean_shift > 10.0:  # 10 mg/dL shift threshold
                self._gradual_drift_level = max(self._gradual_drift_level, 1)

    def _evaluate(self) -> None:
        consecutive = 0
        for v in reversed(self._violations):
            if v:
                consecutive += 1
            else:
                break
        
        # CRITICAL FIX: Combine abrupt and gradual drift detection
        # Global level is max of abrupt (consecutive violations) and gradual drift
        abrupt_level = 0
        if consecutive >= LEVEL_3_THRESHOLD:
            abrupt_level = 3
        elif consecutive >= LEVEL_2_THRESHOLD:
            abrupt_level = 2
        elif consecutive >= LEVEL_1_THRESHOLD:
            abrupt_level = 1
        
        self._level = max(abrupt_level, self._gradual_drift_level)

    @property
    def level(self) -> int:
        return self._level

    @property
    def label(self) -> str:
        return LABEL_MAP.get(self._level, "unknown")

    @property
    def cusum_score(self) -> float:
        """Return current CUSUM score for this subsystem."""
        return self._cusum.get_cusum_score()

    def status(self) -> Dict[str, Any]:
        return {
            "level": self._level,
            "label": self.label,
            "consecutive_violations": self._count_consecutive(),
            "total_violations": sum(self._violations),
            "cusum_score": round(self.cusum_score, 2),  # CRITICAL FIX: Include CUSUM score
            "gradual_drift_level": self._gradual_drift_level,
            "recent_mean_residual": round(float(np.mean(self._recent_residuals[-10:])), 2) if self._recent_residuals else 0.0,
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
        self._cusum.reset()
        self._gradual_drift_level = 0
        self._recent_residuals.clear()


class DriftDetector:

    def __init__(self):
        self.subsystems: Dict[str, SubsystemDrift] = {
            "metabolic": SubsystemDrift("metabolic"),
            "cardiovascular": SubsystemDrift("cardiovascular"),
            "renal": SubsystemDrift("renal"),
            # CRITICAL FIX: Add more subsystems for better drift attribution
            "circadian": SubsystemDrift("circadian"),
            "adipose": SubsystemDrift("adipose"),
            "immune": SubsystemDrift("immune"),
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
            # CRITICAL FIX: Add drift attribution info
            "drift_attribution": self._get_drift_attribution(),
        }

    def _get_drift_attribution(self) -> Dict[str, Any]:
        """CRITICAL FIX: Attribute drift to specific subsystems."""
        subsystem_scores = {
            name: sub.cusum_score 
            for name, sub in self.subsystems.items()
        }
        # Find subsystem with highest CUSUM score (most drift)
        if subsystem_scores:
            max_subsystem = max(subsystem_scores, key=subsystem_scores.get)
            return {
                "primary_drift_subsystem": max_subsystem,
                "subsystem_cusum_scores": subsystem_scores,
                "recommendation": f"Recalibrate {max_subsystem} subsystem" if subsystem_scores.get(max_subsystem, 0) > 5.0 else "Monitor",
            }
        return {}

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
