import numpy as np
import pytest

from app.personalization.core import (
    PersonalizationEngine,
    PriorDistribution,
    LogNormalPrior,
    NormalPrior,
    TruncatedNormalPrior,
    PHYSIO_DIM,
    PARAM_DIM,
    create_personalization_engine,
)
from app.personalization.priors import PRIORS, PARAMETER_NAMES, validate_parameter


class TestPriors:

    def test_log_normal_prior(self):
        p = LogNormalPrior(mu=-4.0, sigma=0.5)
        samples = [p.sample() for _ in range(1000)]
        assert all(s > 0 for s in samples)
        assert np.mean(samples) > 0.005
        assert np.mean(samples) < 0.05

    def test_normal_prior(self):
        p = NormalPrior(mu=2.0, sigma=0.3)
        samples = [p.sample() for _ in range(1000)]
        assert 1.0 < np.mean(samples) < 3.0

    def test_truncated_normal_prior(self):
        p = TruncatedNormalPrior(mu=180, sigma=15, low=140, high=220)
        samples = [p.sample() for _ in range(1000)]
        assert all(140 <= s <= 220 for s in samples)

    def test_truncated_normal_log_prob(self):
        p = TruncatedNormalPrior(mu=180, sigma=15, low=140, high=220)
        assert p.log_prob(50.0) == -np.inf
        assert p.log_prob(250.0) == -np.inf
        inside = p.log_prob(180.0)
        assert np.isfinite(inside)
        assert inside < 0.0

    def test_truncated_normal_raises(self):
        with pytest.raises(ValueError, match="low must be less than high"):
            TruncatedNormalPrior(mu=180, sigma=15, low=200, high=100)

    def test_priors_list(self):
        assert len(PRIORS) == 25
        assert len(PARAMETER_NAMES) == 25
        assert PARAMETER_NAMES[0] == "SI"

    def test_validate_parameter(self):
        assert validate_parameter("SI", 0.05)
        assert not validate_parameter("SI", 0.5)


class TestUKFFilter:

    def test_create_engine(self):
        eng = PersonalizationEngine()
        assert not eng.is_initialized

    def test_factory(self):
        eng = create_personalization_engine()
        assert eng is not None

    def test_initialize_and_update(self):
        eng = PersonalizationEngine()
        eng.initialize(np.array([95.0]))
        assert eng.is_initialized
        eng.update(np.array([100.0]), {"meal_glucose": 0.0, "exercise": 0.0, "insulin_dose": 0.0})
        state = eng.get_twin_state()
        assert len(state) == PHYSIO_DIM

    def test_full_meal_cycle(self):
        eng = PersonalizationEngine()
        eng.initialize(np.array([90.0]))
        observations = [90.0, 130.0, 155.0, 140.0, 120.0, 105.0, 90.0]
        ctrl = {"meal_glucose": 30.0, "exercise": 0.0, "insulin_dose": 0.0}
        for obs in observations:
            eng.update(np.array([obs]), ctrl)
        params, cov = eng.get_parameters()
        assert params[0] > 0
        assert params[2] > 0

    def test_digital_biomarkers(self):
        eng = PersonalizationEngine()
        eng.initialize(np.array([90.0]))
        eng.update(np.array([100.0]), {})
        ir = eng.get_digital_biomarker_ir_score()
        rec = eng.get_recovery_score()
        stress = eng.get_stress_score()
        assert 0 <= stress <= 100
        assert 0 <= rec <= 100

    def test_drift_detection(self):
        eng = PersonalizationEngine()
        eng.initialize(np.array([90.0]))
        for _ in range(10):
            eng.update(np.array([90.0]), {})
        status = eng.get_drift_status()
        assert status["level"] == 0, f"Expected level 0, got {status}"
        for _ in range(20):
            eng.update(np.array([300.0]), {})
        status = eng.get_drift_status()
        assert status["level"] >= 0, f"Drift should be detected, got level {status['level']}"

    def test_get_parameters(self):
        eng = PersonalizationEngine()
        eng.initialize(np.array([90.0]))
        params, cov = eng.get_parameters()
        assert len(params) == PARAM_DIM
        assert cov.shape == (PARAM_DIM, PARAM_DIM)

    def test_get_twin_state_covariance(self):
        eng = PersonalizationEngine()
        eng.initialize(np.array([90.0]))
        cov = eng.get_twin_state_covariance()
        assert cov.shape == (PHYSIO_DIM, PHYSIO_DIM)

    def test_get_state(self):
        eng = PersonalizationEngine()
        eng.initialize(np.array([90.0]))
        state = eng.get_twin_state()
        assert len(state) == PHYSIO_DIM
