"""
Tests for clinical safety and trustworthiness layer.
"""

import math
import numpy as np
import pytest

from app.personalization.safety import (
    OODDetector,
    HypoglycemiaEarlyWarning,
    SafetyGuardrails,
    SafetyVerdict,
    ConfidenceLevel,
    DriftAttributor,
    AdversarialDetector,
    create_default_safety_layer,
)


class TestOODDetector:
    def test_unfitted_returns_no_ood(self):
        ood = OODDetector()
        result = ood.predict(np.array([100.0] * 15))
        assert result.is_ood is False

    def test_fit_then_in_dist(self):
        rng = np.random.RandomState(42)
        obs = rng.normal(100, 10, size=(100, 15))
        obs[:, 5] = rng.normal(120, 15, size=100)
        ood = OODDetector(percentile=0.95)
        ood.fit(obs)
        result = ood.predict(np.array([100.0] * 15))
        assert result.is_ood is False

    def test_fit_then_out_of_dist(self):
        rng = np.random.RandomState(42)
        obs = rng.normal(100, 10, size=(100, 15))
        ood = OODDetector(percentile=0.95)
        ood.fit(obs)
        # Severe outlier: glucose at 400 (vs in-dist mean 100)
        result = ood.predict(np.array([400.0] * 15))
        assert result.is_ood is True

    def test_distance_increases_with_distance(self):
        rng = np.random.RandomState(42)
        obs = rng.normal(100, 10, size=(100, 15))
        ood = OODDetector(percentile=0.95)
        ood.fit(obs)
        r1 = ood.predict(np.array([100.0] * 15))
        r2 = ood.predict(np.array([200.0] * 15))
        r3 = ood.predict(np.array([400.0] * 15))
        assert r1.distance < r2.distance < r3.distance


class TestHypoglycemiaEarlyWarning:
    def test_no_alert_at_normal_glucose(self):
        hypo = HypoglycemiaEarlyWarning(threshold_mg_dL=70, alert_probability=0.30)
        # Glucose 120 mg/dL with std 10 — essentially no hypo risk
        alert = hypo.evaluate(120, 10, horizon_steps=6)
        assert alert.predicted is False
        assert alert.probability < 0.10

    def test_alert_when_mean_below_threshold(self):
        hypo = HypoglycemiaEarlyWarning(threshold_mg_dL=70, alert_probability=0.30)
        # Glucose 60 mg/dL with std 5 — definitely hypo
        # P(G<70 | mean=60, std=5) = Φ((70-60)/5) = Φ(2) ≈ 0.977
        alert = hypo.evaluate(60, 5, horizon_steps=6)
        assert alert.predicted is True
        assert alert.probability > 0.95
        assert alert.severity >= 2

    def test_alert_when_lower_tail_below_threshold(self):
        hypo = HypoglycemiaEarlyWarning(threshold_mg_dL=70, alert_probability=0.30)
        # Glucose 90 mg/dL with std 15 — P(G<70) ≈ 9% — borderline
        alert = hypo.evaluate(90, 15, horizon_steps=6)
        # P(G<70) = Φ((70-90)/15) = Φ(-1.33) ≈ 0.092
        assert 0.05 < alert.probability < 0.20

    def test_high_uncertainty_high_risk(self):
        hypo = HypoglycemiaEarlyWarning(threshold_mg_dL=70, alert_probability=0.30)
        # Mean 90, std 30 — P(G<70) ≈ 0.25
        alert = hypo.evaluate(90, 30, horizon_steps=6)
        # P(G<70) = Φ(-0.67) ≈ 0.25
        assert alert.probability > 0.10


class TestSafetyGuardrails:
    def test_safe_state(self):
        guard = SafetyGuardrails()
        twin_state = np.zeros(30)
        twin_state[0] = 120.0  # glucose
        twin_cov = np.eye(30) * 0.5
        observation = np.array([120.0] * 15)
        verdict = guard.evaluate(twin_state, twin_cov, observation, drift_level=0)
        assert verdict.safe
        assert verdict.verdict == SafetyVerdict.SAFE

    def test_unsafe_extreme_glucose(self):
        guard = SafetyGuardrails()
        twin_state = np.zeros(30)
        twin_state[0] = 600.0  # impossible
        twin_cov = np.eye(30) * 0.5
        observation = np.array([120.0] * 15)
        verdict = guard.evaluate(twin_state, twin_cov, observation, drift_level=0)
        assert not verdict.safe
        assert any("Glucose" in r for r in verdict.reasons)

    def test_unsafe_high_drift(self):
        guard = SafetyGuardrails(max_drift_level=2)
        twin_state = np.zeros(30)
        twin_state[0] = 120.0
        twin_cov = np.eye(30) * 0.5
        observation = np.array([120.0] * 15)
        verdict = guard.evaluate(twin_state, twin_cov, observation, drift_level=3)
        assert any("Drift" in r for r in verdict.reasons)
        assert verdict.confidence_level in (ConfidenceLevel.LOW, ConfidenceLevel.ABSTAIN)

    def test_unsafe_high_variance(self):
        guard = SafetyGuardrails()
        twin_state = np.zeros(30)
        twin_state[0] = 120.0
        twin_cov = np.eye(30) * 0.5
        twin_cov[0, 0] = 10000  # std = 100
        observation = np.array([120.0] * 15)
        verdict = guard.evaluate(twin_state, twin_cov, observation, drift_level=0)
        assert any("std" in r for r in verdict.reasons)

    def test_abstention_when_all_fail(self):
        guard = SafetyGuardrails()
        twin_state = np.zeros(30)
        twin_state[0] = 1200.0  # totally impossible
        twin_cov = np.eye(30) * 0.5
        observation = np.array([120.0] * 15)
        verdict = guard.evaluate(twin_state, twin_cov, observation, drift_level=3)
        assert verdict.abstention_required
        assert verdict.confidence_level == ConfidenceLevel.ABSTAIN


class TestDriftAttributor:
    def test_no_drift_at_baseline(self):
        attr = DriftAttributor()
        for _ in range(20):
            res = attr.update("metabolic", np.random.normal(0, 1))
        assert res.drift_score < 1.0
        assert not res.threshold_exceeded

    def test_drift_detected(self):
        attr = DriftAttributor(slack=1.0, threshold=10.0)
        for _ in range(30):
            res = attr.update("metabolic", 2.0)  # systematic positive shift
        assert res.threshold_exceeded
        assert res.direction == "increase"

    def test_dominant_subsystem(self):
        attr = DriftAttributor(slack=1.0, threshold=10.0)
        for _ in range(30):
            attr.update("metabolic", 2.0)
        for _ in range(30):
            attr.update("cardiovascular", 0.1)
        dom = attr.dominant_subsystem()
        assert dom is not None
        assert dom.subsystem == "metabolic"

    def test_negative_drift(self):
        attr = DriftAttributor(slack=1.0, threshold=10.0)
        for _ in range(30):
            res = attr.update("metabolic", -2.0)
        assert res.direction == "decrease"


class TestAdversarialDetector:
    def test_normal_observation_clean(self):
        adv = AdversarialDetector()
        obs = np.array([120.0] * 15)
        anomalies = adv.update(obs)
        assert len(anomalies) == 0

    def test_glucose_out_of_range(self):
        adv = AdversarialDetector()
        obs = np.array([120.0] * 15)
        adv.update(obs)
        obs[0] = 600.0
        anomalies = adv.update(obs)
        assert any("glucose_out_of_range" in a for a in anomalies)

    def test_glucose_jump(self):
        adv = AdversarialDetector(glucose_max_delta=60.0)
        obs1 = np.array([120.0] * 15)
        adv.update(obs1)
        obs2 = obs1.copy()
        obs2[0] = 250.0  # jump of 130
        anomalies = adv.update(obs2)
        assert any("glucose_jump" in a for a in anomalies)

    def test_stale_detection(self):
        adv = AdversarialDetector(max_stale_steps=3)
        obs = np.array([120.0] * 15)
        anomalies_list = []
        for i in range(10):
            anomalies_list.append(adv.update(obs))
        assert any("stale_data" in a for a in anomalies_list)

    def test_reset(self):
        adv = AdversarialDetector()
        obs = np.array([120.0] * 15)
        adv.update(obs)
        adv.reset()
        # After reset, no anomaly for normal obs
        anomalies = adv.update(obs)
        assert len(anomalies) == 0


class TestFactory:
    def test_create_default(self):
        layer = create_default_safety_layer()
        assert "ood_detector" in layer
        assert "guardrails" in layer
        assert "hypo_warning" in layer
        assert "drift_attributor" in layer
        assert "adversarial_detector" in layer
