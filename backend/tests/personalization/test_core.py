import numpy as np
import pytest

from app.personalization.core import (
    PersonalizationEngine,
    ParticleFilter,
    Particle,
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
        assert len(PRIORS) == 4
        assert len(PARAMETER_NAMES) == 4
        assert PARAMETER_NAMES[0] == "SI"

    def test_validate_parameter(self):
        assert validate_parameter("SI", 0.05)
        assert not validate_parameter("SI", 0.5)


class TestParticleFilter:

    def test_initialize(self):
        state_dim = PHYSIO_DIM + PARAM_DIM
        Q = np.eye(state_dim) * 0.01
        R = np.eye(1) * 0.1

        def dummy_dynamics(physio, params, u):
            return physio

        def dummy_obs(physio):
            return np.array([physio[0]])

        pf = ParticleFilter(100, state_dim, Q, R, PRIORS, dummy_dynamics, dummy_obs)
        assert len(pf.particles) == 100
        assert pf.state_dim == state_dim
        assert pf.physio_dim == PHYSIO_DIM
        assert np.allclose(pf.weights, np.ones(100) / 100)

    def test_predict_update_cycle(self):
        state_dim = PHYSIO_DIM + PARAM_DIM
        Q = np.eye(state_dim) * 0.01
        R = np.eye(1) * 0.1

        def dynamics(physio, params, u):
            return physio

        def obs(physio):
            return np.array([physio[0]])

        pf = ParticleFilter(100, state_dim, Q, R, PRIORS, dynamics, obs)
        pf.predict({})
        pf.update(np.array([100.0]))
        state = pf.get_state()
        assert len(state) == state_dim

    def test_get_predicted_obs_stats(self):
        state_dim = PHYSIO_DIM + PARAM_DIM
        Q = np.eye(state_dim) * 0.01
        R = np.eye(1) * 0.1

        def dynamics(physio, params, u):
            return physio

        def obs(physio):
            return np.array([physio[0]])

        pf = ParticleFilter(100, state_dim, Q, R, PRIORS, dynamics, obs)
        mean, unc = pf.get_predicted_obs_stats()
        assert np.isfinite(mean)
        assert unc >= 0


class TestPersonalizationEngine:

    def test_create(self):
        eng = PersonalizationEngine(num_particles=100)
        assert eng.num_particles == 100
        assert len(eng.priors) == 4
        assert not eng.is_initialized

    def test_factory(self):
        eng = create_personalization_engine(num_particles=50)
        assert eng.num_particles == 50

    def test_initialize_and_update(self):
        eng = PersonalizationEngine(num_particles=100)
        eng.initialize(np.array([95.0]))
        assert eng.is_initialized
        eng.update(np.array([100.0]), {"meal_glucose": 0.0, "exercise": 0.0, "insulin_dose": 0.0})
        state = eng.get_twin_state()
        assert len(state) == PHYSIO_DIM

    def test_full_meal_cycle(self):
        eng = PersonalizationEngine(num_particles=200)
        eng.initialize(np.array([90.0]))
        observations = [90.0, 130.0, 155.0, 140.0, 120.0, 105.0, 90.0]
        ctrl = {"meal_glucose": 30.0, "exercise": 0.0, "insulin_dose": 0.0}
        for obs in observations:
            eng.update(np.array([obs]), ctrl)
        params, cov = eng.get_parameters()
        assert params[0] > 0
        assert params[2] > 0

    def test_digital_biomarkers(self):
        eng = PersonalizationEngine(num_particles=100)
        eng.initialize(np.array([90.0]))
        eng.update(np.array([100.0]), {})
        ir = eng.get_digital_biomarker_ir_score()
        rec = eng.get_recovery_score()
        stress = eng.get_stress_score()
        assert 0 <= stress <= 100
        assert 0 <= rec <= 100

    def test_drift_detection(self):
        eng = PersonalizationEngine(num_particles=100)
        eng.initialize(np.array([90.0]))
        for _ in range(5):
            eng.update(np.array([90.0]), {})
        assert eng.get_drift_status()["level"] == 0
        for _ in range(5):
            eng.update(np.array([300.0]), {})
        assert eng.get_drift_status()["level"] >= 1

    def test_get_parameters(self):
        eng = PersonalizationEngine(num_particles=100)
        eng.initialize(np.array([90.0]))
        params, cov = eng.get_parameters()
        assert len(params) == PARAM_DIM
        assert cov.shape == (PARAM_DIM, PARAM_DIM)

    def test_get_twin_state_covariance(self):
        eng = PersonalizationEngine(num_particles=100)
        eng.initialize(np.array([90.0]))
        cov = eng.get_twin_state_covariance()
        assert cov.shape == (PHYSIO_DIM, PHYSIO_DIM)

    def test_needs_resampling(self):
        eng = PersonalizationEngine(num_particles=100)
        eng.initialize(np.array([90.0]))
        assert isinstance(eng.needs_resampling(), bool)

    def test_is_weight_degenerate(self):
        eng = PersonalizationEngine(num_particles=100)
        eng.initialize(np.array([90.0]))
        assert isinstance(eng.is_weight_degenerate(), bool)
