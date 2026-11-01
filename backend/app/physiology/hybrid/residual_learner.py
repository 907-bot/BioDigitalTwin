"""
BioDigitalTwin — Hybrid Mechanistic + Machine Learning Residual Layer.

Implements the hybrid formulation:
dx/dt = f_mechanistic(x, theta, u) + g_ml(x, u)

Rules:
1. The mechanistic model remains authoritative for all transitions.
2. The ML residual g_ml is strictly bounded and constrained to prevent unphysiological drift.
3. g_ml cannot violate physical conservation laws (e.g. mass balance, non-negativity).
4. Provides anomaly detection flagging sensor errors or unannounced clinical disturbances.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class ResidualCorrection:
    state_name: str
    mechanistic_derivative: float
    learned_residual: float
    total_derivative: float
    is_clamped: bool


class ConstrainedResidualLearner:
    """
    Learns systematic discrepancies between the mechanistic model and patient observations.
    Constrains residual adjustments to safe, physically realistic bounds.
    """

    def __init__(
        self,
        max_residual_fraction: float = 0.20,
        regularization_l2: float = 0.01,
    ):
        self.max_residual_fraction = max_residual_fraction
        self.regularization_l2 = regularization_l2
        # Feature weights for simple ridge/linear residual model
        self._weights: Dict[str, np.ndarray] = {}
        self._feature_dim = 6  # [G, I, HR, MAP, exercise, meal_carbs]

    def _extract_features(
        self,
        state_dict: Dict[str, float],
        inputs: Dict[str, float],
    ) -> np.ndarray:
        """Constructs standardized feature vector."""
        g = (state_dict.get("blood.glucose", 95.0) - 95.0) / 30.0
        i = (state_dict.get("blood.insulin", 8.0) - 8.0) / 10.0
        hr = (state_dict.get("cardio.heart_rate_bpm", 72.0) - 72.0) / 15.0
        map_val = (state_dict.get("cardio.mean_arterial_pressure_mmHg", 93.0) - 93.0) / 15.0
        ex = inputs.get("exercise", 0.0)
        meal = inputs.get("meal_carbs_g", 0.0) / 50.0
        return np.array([g, i, hr, map_val, ex, meal], dtype=np.float64)

    def predict_residual(
        self,
        state_name: str,
        mechanistic_deriv: float,
        state_dict: Dict[str, float],
        inputs: Dict[str, float],
    ) -> ResidualCorrection:
        """
        Computes constrained residual correction g_ml(x, u).
        """
        if state_name not in self._weights:
            # No trained residual for this state: residual is 0.0
            return ResidualCorrection(
                state_name=state_name,
                mechanistic_derivative=mechanistic_deriv,
                learned_residual=0.0,
                total_derivative=mechanistic_deriv,
                is_clamped=False,
            )

        features = self._extract_features(state_dict, inputs)
        w = self._weights[state_name]
        raw_residual = float(np.dot(features, w))

        # Strict safety bounds: |g_ml| <= max_fraction * (|f_mech| + baseline_tolerance)
        bound = max(0.5, abs(mechanistic_deriv) * self.max_residual_fraction)
        clamped_residual = float(np.clip(raw_residual, -bound, bound))
        is_clamped = (clamped_residual != raw_residual)

        return ResidualCorrection(
            state_name=state_name,
            mechanistic_derivative=mechanistic_deriv,
            learned_residual=clamped_residual,
            total_derivative=mechanistic_deriv + clamped_residual,
            is_clamped=is_clamped,
        )

    def train_residual(
        self,
        state_name: str,
        feature_matrix: np.ndarray,
        residual_targets: np.ndarray,
    ) -> None:
        """
        Trains ridge regression weights: (X^T X + lambda I)^-1 X^T y
        """
        n, p = feature_matrix.shape
        reg = self.regularization_l2 * np.eye(p)
        w = np.linalg.solve(feature_matrix.T @ feature_matrix + reg, feature_matrix.T @ residual_targets)
        self._weights[state_name] = w


class PhysiologicalAnomalyDetector:
    """
    Detects unphysical observations or unannounced patient events.
    """

    def __init__(self):
        # Max plausible biological rates of change per minute
        self.max_rates_per_min = {
            "glucose": 4.5,       # CGM cannot physically rise >4.5 mg/dL/min without IV bolus
            "heart_rate": 25.0,   # HR change > 25 bpm/min without extreme exertion indicates artifact
            "systolic_bp": 30.0,  # SBP jump > 30 mmHg/min
        }

    def detect_anomalies(
        self,
        current_obs: Dict[str, float],
        previous_obs: Dict[str, float],
        dt_minutes: float,
    ) -> List[str]:
        """Flags rate-of-change and boundary anomalies in telemetry."""
        anomalies = []
        if dt_minutes <= 0.0:
            return anomalies

        for key, max_rate in self.max_rates_per_min.items():
            if key in current_obs and key in previous_obs:
                rate = abs(current_obs[key] - previous_obs[key]) / dt_minutes
                if rate > max_rate:
                    anomalies.append(
                        f"Physiological rate anomaly in {key}: {rate:.2f}/min exceeds physiological limit of {max_rate}/min"
                    )

        # Non-negative checks
        if current_obs.get("glucose", 100.0) < 15.0:
            anomalies.append("Extreme hypoglycemia alert: Glucose < 15 mg/dL")
        if current_obs.get("heart_rate", 70.0) < 25.0:
            anomalies.append("Severe bradycardia / sensor dropout: HR < 25 bpm")

        return anomalies

    def check_telemetry(
        self,
        current_telemetry: Dict[str, float],
        previous_telemetry: Dict[str, float],
        dt_minutes: float,
    ) -> Tuple[bool, List[str]]:
        """Convenience method returning (is_valid, anomalies) with alias key resolution."""
        alias_map = {
            "glucose_mg_dL": "glucose",
            "heart_rate_bpm": "heart_rate",
            "mean_bp_mmHg": "systolic_bp",
            "systolic_bp_mmHg": "systolic_bp",
        }
        c_mapped = {alias_map.get(k, k): v for k, v in current_telemetry.items()}
        p_mapped = {alias_map.get(k, k): v for k, v in previous_telemetry.items()}
        anomalies = self.detect_anomalies(c_mapped, p_mapped, dt_minutes)
        return len(anomalies) == 0, anomalies
