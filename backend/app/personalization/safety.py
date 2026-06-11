"""
Clinical safety and trustworthiness layer.

Components:
- OODDetector: Mahalanobis-distance based out-of-distribution detection
- HypoglycemiaEarlyWarning: predicts impending hypo from predictive distribution
- SafetyGuardrails: abstention mechanism for low-confidence predictions
- DriftAttributor: per-subsystem drift attribution with CUSUM statistics
- AdversarialDetector: detects anomalous input patterns

References:
- Mahalanobis (1936) — distance metric for multivariate outliers
- Page (1954) — CUSUM for change detection
- Ambrosino et al. (2018) — hypoglycemic event prediction
"""

import math
import logging
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

logger = logging.getLogger(__name__)


class ConfidenceLevel(IntEnum):
    HIGH = 0       # safe to act
    MEDIUM = 1     # proceed with caution
    LOW = 2        # recommend human review
    ABSTAIN = 3    # do not make prediction


class SafetyVerdict(IntEnum):
    SAFE = 0
    CAUTION = 1
    UNSAFE = 2
    ABSTAIN = 3


@dataclass
class OODResult:
    is_ood: bool
    distance: float
    threshold: float
    p_in_dist: float


class OODDetector:
    """Mahalanobis-distance based OOD detection on the observation space.

    Tracks running mean and covariance of the in-distribution observations.
    A new observation is OOD if its Mahalanobis distance exceeds a
    chi-squared threshold at the configured percentile.
    """

    def __init__(self, percentile: float = 0.99, regularization: float = 1e-3):
        self.percentile = percentile
        self.regularization = regularization
        self._mean: Optional[np.ndarray] = None
        self._cov: Optional[np.ndarray] = None
        self._inv_cov: Optional[np.ndarray] = None
        self._threshold: Optional[float] = None
        self._n_obs = 0
        self._distances: List[float] = []

    def fit(self, observations: np.ndarray) -> None:
        """Fit the in-distribution statistics from a calibration set.

        observations: shape (N, d)
        """
        self._mean = np.mean(observations, axis=0)
        self._cov = np.cov(observations, rowvar=False)
        if self._cov.ndim == 0:
            self._cov = np.array([[float(self._cov)]])
        d = self._cov.shape[0]
        self._cov += np.eye(d) * self.regularization
        self._inv_cov = np.linalg.inv(self._cov)
        dists = self._compute_distances(observations)
        self._distances = dists.tolist()
        self._threshold = float(np.percentile(dists, self.percentile * 100))
        self._n_obs = len(observations)
        logger.info(
            f"OODDetector fit on {self._n_obs} obs, dim {d}, "
            f"threshold (p{self.percentile*100:.0f})={self._threshold:.2f}"
        )

    def _compute_distances(self, observations: np.ndarray) -> np.ndarray:
        if self._mean is None or self._inv_cov is None:
            return np.zeros(len(observations))
        diff = observations - self._mean
        mahal = np.einsum("ij,jk,ik->i", diff, self._inv_cov, diff)
        return mahal

    def predict(self, observation: np.ndarray) -> OODResult:
        if self._mean is None or self._inv_cov is None:
            return OODResult(False, 0.0, np.inf, 1.0)
        diff = observation - self._mean
        d2 = float(diff @ self._inv_cov @ diff)
        if self._threshold is None:
            return OODResult(d2 > d2, d2, d2, 1.0)
        is_ood = d2 > self._threshold
        # Probability of being in-distribution via survival function
        # Use exponential tail approximation
        p_in = float(np.exp(-0.5 * d2))
        return OODResult(
            is_ood=is_ood,
            distance=d2,
            threshold=self._threshold,
            p_in_dist=min(max(p_in, 0.0), 1.0),
        )

    def update(self, observation: np.ndarray, learning_rate: float = 0.05) -> None:
        """Online update of the running statistics.

        Uses exponential moving average; bounded by learning_rate.
        """
        if self._mean is None:
            self._mean = observation.copy()
            return
        self._mean = (1 - learning_rate) * self._mean + learning_rate * observation
        diff_old = self._cov.diagonal() if self._cov is not None else np.ones_like(observation)
        if self._cov is None:
            self._cov = np.eye(len(observation)) * 0.1
        deviation = (observation - self._mean)
        outer = np.outer(deviation, deviation)
        self._cov = (1 - learning_rate) * self._cov + learning_rate * outer
        d = self._cov.shape[0]
        self._cov += np.eye(d) * self.regularization
        self._inv_cov = np.linalg.inv(self._cov)


@dataclass
class HypoglycemiaAlert:
    predicted: bool
    probability: float
    time_to_event_hours: float
    predicted_glucose_mg_dL: float
    threshold_mg_dL: float = 70.0
    severity: int = 0  # 0=ok, 1=mild, 2=moderate, 3=severe


class HypoglycemiaEarlyWarning:
    """Predicts impending hypoglycemia from predictive distribution.

    Uses the lower tail of the predicted glucose distribution. If
    P(G < 70 mg/dL | observations) > threshold, alert.
    """

    def __init__(
        self,
        threshold_mg_dL: float = 70.0,
        alert_probability: float = 0.30,
        lead_time_hours: float = 0.5,
    ):
        self.threshold = threshold_mg_dL
        self.alert_prob = alert_probability
        self.lead_time = lead_time_hours

    def evaluate(
        self,
        predicted_mean: float,
        predicted_std: float,
        horizon_steps: int,
        step_duration_hours: float = 0.0833,  # 5 min
    ) -> HypoglycemiaAlert:
        """Evaluate hypoglycemia risk at the prediction horizon.

        predicted_mean, predicted_std: from twin's predictive distribution
        horizon_steps: number of steps ahead
        step_duration_hours: hours per step
        """
        horizon_hours = horizon_steps * step_duration_hours
        if predicted_std <= 1e-6:
            return HypoglycemiaAlert(False, 0.0, horizon_hours, predicted_mean, self.threshold, 0)
        # P(G < threshold) under N(mean, std^2)
        z = (self.threshold - predicted_mean) / predicted_std
        p = 0.5 * (1 + math.erf(z / math.sqrt(2)))
        p = max(0.0, min(1.0, p))
        predicted = p > self.alert_prob
        if not predicted:
            severity = 0
        elif p < 0.50:
            severity = 1
        elif p < 0.80:
            severity = 2
        else:
            severity = 3
        return HypoglycemiaAlert(
            predicted=predicted,
            probability=p,
            time_to_event_hours=horizon_hours,
            predicted_glucose_mg_dL=predicted_mean,
            threshold_mg_dL=self.threshold,
            severity=severity,
        )


@dataclass
class GuardrailVerdict:
    safe: bool
    verdict: SafetyVerdict
    confidence_level: ConfidenceLevel
    reasons: List[str] = field(default_factory=list)
    abstention_required: bool = False


class SafetyGuardrails:
    """Abstention mechanism for low-confidence predictions.

    Combines: twin-state health, OOD detection, drift level, and
    prediction interval width to determine whether to make a clinical
    recommendation.
    """

    def __init__(
        self,
        ood_detector: Optional[OODDetector] = None,
        min_confidence: float = 0.50,
        max_drift_level: int = 2,
    ):
        self.ood_detector = ood_detector or OODDetector()
        self.min_confidence = min_confidence
        self.max_drift_level = max_drift_level

    def evaluate(
        self,
        twin_state: np.ndarray,
        twin_cov: np.ndarray,
        observation: np.ndarray,
        drift_level: int = 0,
    ) -> GuardrailVerdict:
        reasons: List[str] = []
        confidence = 1.0
        # Check physiological plausibility
        G = twin_state[0] if len(twin_state) > 0 else np.nan
        if not (40 <= G <= 500):
            reasons.append(f"Glucose out of plausible range: {G:.1f}")
            confidence *= 0.0
        # Check covariance size — large variance = low confidence
        var_G = twin_cov[0, 0] if twin_cov.shape[0] > 0 else 0
        std_G = math.sqrt(max(var_G, 0))
        if std_G > 50:
            reasons.append(f"Glucose std={std_G:.1f} mg/dL too high")
            confidence *= 0.5
        if std_G > 100:
            reasons.append(f"Glucose std={std_G:.1f} mg/dL catastrophic")
            confidence *= 0.0
        # Check drift
        if drift_level > self.max_drift_level:
            reasons.append(f"Drift level {drift_level} > max {self.max_drift_level}")
            confidence *= 0.3
        # Check OOD
        ood_result = self.ood_detector.predict(observation)
        if ood_result.is_ood:
            reasons.append(
                f"OOD: Mahalanobis d²={ood_result.distance:.1f} > {ood_result.threshold:.1f}"
            )
            confidence *= 0.5
        # Determine confidence level
        if confidence >= 0.85:
            level = ConfidenceLevel.HIGH
        elif confidence >= 0.65:
            level = ConfidenceLevel.MEDIUM
        elif confidence >= self.min_confidence:
            level = ConfidenceLevel.LOW
        else:
            level = ConfidenceLevel.ABSTAIN
        abstention = level == ConfidenceLevel.ABSTAIN
        safe = level in (ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM) and \
               not abstention
        if abstention:
            verdict = SafetyVerdict.ABSTAIN
        elif level == ConfidenceLevel.LOW:
            verdict = SafetyVerdict.CAUTION
        elif len(reasons) == 0:
            verdict = SafetyVerdict.SAFE
        else:
            verdict = SafetyVerdict.CAUTION
        return GuardrailVerdict(
            safe=safe,
            verdict=verdict,
            confidence_level=level,
            reasons=reasons,
            abstention_required=abstention,
        )


@dataclass
class DriftAttribution:
    subsystem: str
    drift_score: float
    cusum: float
    threshold_exceeded: bool
    direction: str  # 'increase', 'decrease', 'stable'
    suspected_state: Optional[int] = None


class DriftAttributor:
    """Per-subsystem drift attribution using CUSUM statistics.

    Tracks residuals for each subsystem and computes a cumulative sum
    to detect gradual changes. Identifies which subsystem is responsible
    for global drift.
    """

    def __init__(self, slack: float = 5.0, threshold: float = 30.0):
        self.slack = slack
        self.threshold = threshold
        self._cusum_pos: Dict[str, float] = {}
        self._cusum_neg: Dict[str, float] = {}
        self._history: Dict[str, List[float]] = {}
        self._residuals: Dict[str, List[float]] = {}

    def update(self, subsystem: str, residual: float) -> DriftAttribution:
        if subsystem not in self._cusum_pos:
            self._cusum_pos[subsystem] = 0.0
            self._cusum_neg[subsystem] = 0.0
            self._history[subsystem] = []
            self._residuals[subsystem] = []
        # One-sided CUSUM for positive and negative shifts
        self._cusum_pos[subsystem] = max(
            0.0, self._cusum_pos[subsystem] + residual - self.slack
        )
        self._cusum_neg[subsystem] = max(
            0.0, self._cusum_neg[subsystem] - residual - self.slack
        )
        self._residuals[subsystem].append(residual)
        cusum = max(self._cusum_pos[subsystem], self._cusum_neg[subsystem])
        threshold_exceeded = cusum > self.threshold
        if self._cusum_pos[subsystem] > self._cusum_neg[subsystem]:
            direction = "increase"
        elif self._cusum_neg[subsystem] > self._cusum_pos[subsystem]:
            direction = "decrease"
        else:
            direction = "stable"
        # Normalize by history variance
        hist = self._residuals[subsystem]
        if len(hist) > 5:
            std = np.std(hist) + 1e-6
            drift_score = cusum / (std * math.sqrt(len(hist)))
        else:
            drift_score = 0.0
        self._history[subsystem].append(cusum)
        return DriftAttribution(
            subsystem=subsystem,
            drift_score=float(drift_score),
            cusum=float(cusum),
            threshold_exceeded=threshold_exceeded,
            direction=direction,
        )

    def dominant_subsystem(self) -> Optional[DriftAttribution]:
        """Return the subsystem with the highest drift score."""
        best = None
        for sub in self._cusum_pos.keys():
            hist = self._residuals[sub]
            if len(hist) > 5:
                std = np.std(hist) + 1e-6
                score = max(self._cusum_pos[sub], self._cusum_neg[sub]) / \
                        (std * math.sqrt(len(hist)))
                if best is None or score > best.drift_score:
                    best = DriftAttribution(
                        subsystem=sub,
                        drift_score=float(score),
                        cusum=float(max(self._cusum_pos[sub], self._cusum_neg[sub])),
                        threshold_exceeded=max(
                            self._cusum_pos[sub], self._cusum_neg[sub]
                        ) > self.threshold,
                        direction="increase" if self._cusum_pos[sub] > self._cusum_neg[sub] else "decrease",
                    )
        return best

    def reset(self) -> None:
        self._cusum_pos.clear()
        self._cusum_neg.clear()
        self._history.clear()
        self._residuals.clear()


class AdversarialDetector:
    """Detects anomalous input patterns that may indicate sensor fault
    or adversarial attack.

    Uses a combination of:
    - Range checks (physiologically impossible)
    - Rate-of-change checks (sudden jumps)
    - Staleness check (no data)
    """

    def __init__(
        self,
        glucose_max_delta: float = 60.0,
        sbp_max_delta: float = 40.0,
        hr_max_delta: float = 50.0,
        max_stale_steps: int = 12,
    ):
        self.glucose_max_delta = glucose_max_delta
        self.sbp_max_delta = sbp_max_delta
        self.hr_max_delta = hr_max_delta
        self.max_stale_steps = max_stale_steps
        self._last_obs: Optional[np.ndarray] = None
        self._stale_count = 0

    def update(self, observation: np.ndarray) -> List[str]:
        """Check observation for anomalies. Returns list of anomaly types."""
        anomalies: List[str] = []
        obs = observation.copy()
        # Staleness
        if self._last_obs is not None and np.allclose(obs, self._last_obs, atol=1e-6):
            self._stale_count += 1
            if self._stale_count > self.max_stale_steps:
                anomalies.append("stale_data")
        else:
            self._stale_count = 0
        # Range checks (glucose is index 0 in SAMPLE_OBS)
        if obs[0] < 30 or obs[0] > 500:
            anomalies.append(f"glucose_out_of_range:{obs[0]:.1f}")
        if obs[5] < 50 or obs[5] > 250:
            anomalies.append(f"sbp_out_of_range:{obs[5]:.1f}")
        if obs[7] < 30 or obs[7] > 220:
            anomalies.append(f"hr_out_of_range:{obs[7]:.1f}")
        # Rate-of-change checks
        if self._last_obs is not None:
            if abs(obs[0] - self._last_obs[0]) > self.glucose_max_delta:
                anomalies.append(
                    f"glucose_jump:{obs[0]-self._last_obs[0]:.1f}"
                )
            if abs(obs[5] - self._last_obs[5]) > self.sbp_max_delta:
                anomalies.append(f"sbp_jump:{obs[5]-self._last_obs[5]:.1f}")
            if abs(obs[7] - self._last_obs[7]) > self.hr_max_delta:
                anomalies.append(f"hr_jump:{obs[7]-self._last_obs[7]:.1f}")
        self._last_obs = obs.copy()
        return anomalies

    def is_adversarial(self, observation: np.ndarray) -> Tuple[bool, List[str]]:
        anomalies = self.update(observation)
        return len(anomalies) > 0, anomalies

    def reset(self) -> None:
        self._last_obs = None
        self._stale_count = 0


def create_default_safety_layer() -> Dict:
    """Factory for the full safety layer."""
    ood = OODDetector(percentile=0.99)
    guard = SafetyGuardrails(ood_detector=ood)
    hypo = HypoglycemiaEarlyWarning()
    drift_attr = DriftAttributor()
    adv = AdversarialDetector()
    return {
        "ood_detector": ood,
        "guardrails": guard,
        "hypo_warning": hypo,
        "drift_attributor": drift_attr,
        "adversarial_detector": adv,
    }
