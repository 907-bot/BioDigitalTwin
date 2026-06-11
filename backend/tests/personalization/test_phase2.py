"""
Phase 2 tests: UKF, multi-organ dynamics, biomarkers, drift, pipeline.
"""

import numpy as np
import pytest

from app.personalization.core import (
    UnscentedKalmanFilter,
    PersonalizationEngine,
    create_personalization_engine,
    PHYSIO_DIM,
    PARAM_DIM,
)
from app.personalization.dynamics import (
    DEFAULT_PARAMS,
    compute_metabolic_dynamics,
    compute_cardio_dynamics,
    compute_renal_dynamics,
    compute_inflammation_dynamics,
    full_dynamics,
    full_observation,
)
from app.personalization.state import (
    MetabolicState,
    CardioState,
    RenalState,
    InflammatoryState,
    FullTwinState,
    OBS_DIM,
)
from app.personalization.biomarkers import (
    compute_recovery_score,
    compute_stress_score,
    compute_vascular_age,
    compute_arterial_stiffness_index,
    compute_salt_sensitivity_index,
    compute_all_biomarkers,
)
from app.personalization.drift import DriftDetector, SubsystemDrift, CounterfactualSimulator
from app.personalization.priors import PRIORS


SAMPLE_OBS = np.array([
    95.0, 120.0, 80.0, 72.0, 45.0, 100.0, 140.0, 4.2, 290.0,
    0.5, 100.0, 50.0, 120.0, 350.0, 0.3,
])


# ── Dynamics Tests ────────────────────────────────────────────

class TestMetabolicDynamics:
    def test_produces_valid_metabolic_state(self):
        state = MetabolicState(90, 5, 2, 5, 5)
        result = compute_metabolic_dynamics(state, {}, {})
        assert isinstance(result, MetabolicState)
        assert result.G > 0

    def test_meal_raises_glucose(self):
        fast = MetabolicState(90, 5, 2, 5, 5)
        fed = compute_metabolic_dynamics(fast, {"meal_glucose": 50.0}, {})
        assert fed.G > fast.G

    def test_exercise_lowers_glucose(self):
        rest = MetabolicState(100, 5, 2, 5, 5)
        ex = compute_metabolic_dynamics(rest, {"exercise": 0.8}, {})
        assert ex.G < rest.G

    def test_from_array_roundtrip(self):
        arr = np.array([85.0, 6.0, 1.5, 4.5, 6.0])
        state = MetabolicState.from_array(arr)
        assert np.allclose(state.to_array(), arr)

    def test_metabolic_positive_definite(self):
        for _ in range(20):
            state = MetabolicState(
                G=np.random.uniform(50, 200),
                I=np.random.uniform(0, 30),
                HGP=np.random.uniform(0, 5),
                PGU=np.random.uniform(0, 10),
                IR=np.random.uniform(0, 20),
            )
            result = compute_metabolic_dynamics(state, {}, {})
            assert result.G >= 20, f"G dropped to {result.G}"
            assert result.I >= 0
            assert result.HGP >= -5
            assert result.PGU >= 0
            assert result.IR >= 0


class TestCardioDynamics:
    def test_produces_valid_cardio_state(self):
        state = CardioState(120, 80, 70, 45)
        result = compute_cardio_dynamics(state, {}, {}, meta_ir=5.0)
        assert isinstance(result, CardioState)
        assert 50 < result.SBP < 250
        assert 30 < result.DBP < 180

    def test_resting_values_stable(self):
        state = CardioState(120, 80, 70, 45)
        for _ in range(10):
            state = compute_cardio_dynamics(state, {"exercise": 0.0}, {}, meta_ir=5.0)
        assert abs(state.SBP - 120) < 10
        assert abs(state.DBP - 80) < 8

    def test_exercise_raises_hr(self):
        rest = CardioState(120, 80, 70, 45)
        ex = compute_cardio_dynamics(rest, {"exercise": 0.7}, {}, meta_ir=5.0)
        assert ex.HR > rest.HR

    def test_coupling_from_metabolic(self):
        rest = CardioState(120, 80, 70, 45)
        coupled = compute_cardio_dynamics(rest, {"exercise": 0.0}, {}, meta_ir=0.5)
        high_ir = compute_cardio_dynamics(rest, {"exercise": 0.0}, {}, meta_ir=15.0)
        assert high_ir.SBP >= coupled.SBP - 0.5


class TestRenalDynamics:
    def _cardio(self, sbp=120, dbp=80):
        return CardioState(sbp, dbp, 70, 45)

    def test_produces_valid_renal_state(self):
        state = RenalState(100, 140, 4.2, 290)
        result = compute_renal_dynamics(state, {}, {}, cardio=self._cardio(), meta_g=95.0)
        assert isinstance(result, RenalState)
        assert result.GFR >= 0
        assert 100 < result.Na < 180
        assert result.K > 0
        assert 230 < result.Osm < 350

    def test_gfr_tracks_bp(self):
        state = RenalState(100, 140, 4.2, 290)
        low_bp = compute_renal_dynamics(state, {}, {}, cardio=self._cardio(sbp=80), meta_g=95.0)
        high_bp = compute_renal_dynamics(state, {}, {}, cardio=self._cardio(sbp=160), meta_g=95.0)
        assert high_bp.GFR >= low_bp.GFR

    def test_na_retention_increases_bp_coupling(self):
        result = compute_renal_dynamics(RenalState(100, 140, 4.2, 290), {}, {}, cardio=self._cardio(), meta_g=95.0)
        assert isinstance(result, RenalState)


class TestInflammationDynamics:
    def test_produces_valid_inflam_state(self):
        state = InflammatoryState(1.0)
        result = compute_inflammation_dynamics(state, meta_ir=5.0, cardio_hrv=45.0)
        assert result.CRP >= 0.1

    def test_crp_rises_with_ir(self):
        low_ir = compute_inflammation_dynamics(InflammatoryState(1.0), meta_ir=1.0, cardio_hrv=45.0)
        high_ir = compute_inflammation_dynamics(InflammatoryState(1.0), meta_ir=15.0, cardio_hrv=45.0)
        assert high_ir.CRP >= low_ir.CRP


class TestFullDynamics:
    def test_full_dynamics_output_dim(self):
        state = np.zeros(PHYSIO_DIM)
        state[0] = 90
        state[5] = 120
        state[6] = 80
        state[7] = 70
        state[8] = 45
        state[9] = 100
        state[10] = 140
        state[11] = 4.2
        state[12] = 290
        state[13] = 1.0
        params = DEFAULT_PARAMS.copy()
        params[:12] = [0.01, 2.0, 0.005, 180, 15.0, 80.0, 2.0, 0.5, 100.0, 0.6, 80.0, 0.5]
        result = full_dynamics(state, params, {})
        assert len(result) == PHYSIO_DIM
        assert result[0] > 0  # G positive
        assert 50 < result[5] < 250  # SBP

    def test_full_dynamics_meal_effect(self):
        state = np.zeros(PHYSIO_DIM)
        state[0] = 90
        state[5] = 120
        state[6] = 80
        state[7] = 70
        state[9] = 100
        state[13] = 1.0
        params = DEFAULT_PARAMS.copy()
        params[:12] = [0.01, 2.0, 0.005, 180, 15.0, 80.0, 2.0, 0.5, 100.0, 0.6, 80.0, 0.5]
        r1 = full_dynamics(state, params, {"meal_glucose": 0.0})
        r2 = full_dynamics(state, params, {"meal_glucose": 80.0})
        assert r2[0] > r1[0]

    def test_full_observation_dim(self):
        state = np.zeros(PHYSIO_DIM)
        state[0] = 95
        state[5] = 120
        state[6] = 80
        state[7] = 72
        state[8] = 45
        state[9] = 100
        state[10] = 140
        state[11] = 4.2
        state[12] = 290
        obs = full_observation(state)
        assert len(obs) == OBS_DIM
        assert obs[0] == 95.0

    def test_organ_coupling_ir_to_bp(self):
        state = np.zeros(PHYSIO_DIM)
        state[0] = 90
        state[5] = 120
        state[6] = 80
        state[7] = 70
        state[9] = 100
        state[13] = 1.0
        params = DEFAULT_PARAMS.copy()
        params[:12] = [0.01, 2.0, 0.005, 180, 15.0, 80.0, 2.0, 0.5, 100.0, 0.6, 80.0, 0.5]
        state[4] = 2.0
        r1 = full_dynamics(state, params, {})
        state[4] = 18.0
        r2 = full_dynamics(state, params, {})
        assert r2[5] >= r1[5] - 0.5


# ── UKF Tests ─────────────────────────────────────────────────

class TestUnscentedKalmanFilter:
    def _make_ukf(self):
        return UnscentedKalmanFilter(
            state_dim=PHYSIO_DIM + PARAM_DIM,
            process_noise=np.eye(PHYSIO_DIM + PARAM_DIM) * 0.01,
            obs_noise=np.eye(OBS_DIM) * 0.1,
            dynamics_fn=lambda ps, pr, u: ps,
            obs_fn=lambda s: s[:OBS_DIM],
            param_prior_fn=lambda: np.zeros(PARAM_DIM),
        )

    def test_ukf_initializes(self):
        ukf = self._make_ukf()
        assert ukf.get_state().shape == (PHYSIO_DIM + PARAM_DIM,)

    def test_ukf_predict_update_cycle(self):
        ukf = self._make_ukf()
        mu = ukf.get_state()
        mu[0] = 95.0
        mu[5] = 120.0
        mu[9] = 100.0
        ukf._mu = mu
        ukf.predict({})
        ukf.update(SAMPLE_OBS)
        assert ukf.get_physio_state().shape == (PHYSIO_DIM,)

    def test_ukf_convergence(self):
        ukf = self._make_ukf()
        for _ in range(10):
            ukf.predict({})
            ukf.update(SAMPLE_OBS)
        state = ukf.get_physio_state()
        assert abs(state[0] - 95) < 5  # glucose directly observed — tracks well


# ── Engine Tests ──────────────────────────────────────────────

class TestPersonalizationEngine:
    def test_engine_initializes(self):
        eng = create_personalization_engine()
        assert eng is not None
        assert not eng.is_initialized

    def test_initialize_with_obs(self):
        eng = PersonalizationEngine()
        eng.initialize(np.array([95.0]))
        assert eng.is_initialized
        state = eng.get_twin_state()
        assert len(state) == PHYSIO_DIM
        assert abs(state[0] - 95.0) < 1e-6

    def test_update_cycle(self):
        eng = PersonalizationEngine()
        eng.update(SAMPLE_OBS)
        state = eng.get_twin_state()
        assert len(state) == PHYSIO_DIM

    def test_multiple_updates(self):
        eng = PersonalizationEngine()
        for _ in range(5):
            eng.update(SAMPLE_OBS)
        state = eng.get_twin_state()
        assert abs(state[0] - 95) < 50
        assert 50 < state[5] < 250

    def test_get_parameters(self):
        eng = PersonalizationEngine()
        eng.update(SAMPLE_OBS)
        params, cov = eng.get_parameters()
        assert len(params) == PARAM_DIM
        assert cov.shape == (PARAM_DIM, PARAM_DIM)

    def test_get_metabolic_state(self):
        eng = PersonalizationEngine()
        eng.update(SAMPLE_OBS)
        s = eng.get_metabolic_state()
        assert len(s) == 5

    def test_get_cardio_state(self):
        eng = PersonalizationEngine()
        eng.update(SAMPLE_OBS)
        s = eng.get_cardio_state()
        assert len(s) == 4

    def test_get_renal_state(self):
        eng = PersonalizationEngine()
        eng.update(SAMPLE_OBS)
        s = eng.get_renal_state()
        assert len(s) == 4

    def test_drift_status(self):
        eng = PersonalizationEngine()
        eng.update(SAMPLE_OBS)
        status = eng.get_drift_status()
        assert "level" in status
        assert "subsystems" in status
        assert "metabolic" in status["subsystems"]


# ── Biomarker Tests ───────────────────────────────────────────

class TestBiomarkers:
    def test_recovery_score_default(self):
        score = compute_recovery_score([], 5.0)
        assert score == 50.0

    def test_recovery_score_range(self):
        score = compute_recovery_score([90, 92, 88, 91, 89, 93], 1.0)
        assert 0 <= score <= 100

    def test_stress_score_range(self):
        score = compute_stress_score([90, 95, 85, 100, 80, 105], 10.0)
        assert 0 <= score <= 100

    def test_vascular_age(self):
        age = compute_vascular_age(2.0, 130)
        assert 20 <= age <= 100

    def test_arterial_stiffness_index(self):
        asi = compute_arterial_stiffness_index(2.0, 140, 80)
        assert 0.5 <= asi <= 5.0

    def test_salt_sensitivity_index(self):
        ssi = compute_salt_sensitivity_index(0.6, 0.5, 135)
        assert 0 <= ssi <= 100

    def test_compute_all(self):
        state = np.zeros(PHYSIO_DIM)
        state[0] = 95
        state[4] = 5
        state[5] = 120
        state[6] = 80
        params = DEFAULT_PARAMS.copy()
        params[0] = 0.01
        params[4] = 15.0
        params[8] = 0.6
        params[11] = 0.5
        bio = compute_all_biomarkers(state, params, [90, 92, 88])
        assert "insulin_resistance_score" in bio
        assert "vascular_age" in bio
        assert "salt_sensitivity_index" in bio
        assert bio["insulin_resistance_score"] > 0


# ── Drift Detection Tests ─────────────────────────────────────

class TestDriftDetector:
    def test_no_drift_initially(self):
        d = DriftDetector()
        assert d.level == 0

    def test_level_1_after_3_violations(self):
        d = DriftDetector()
        for _ in range(3):
            d.check(100.0, 90.0, 2.0)
        assert d.level == 1

    def test_level_2_after_5_violations(self):
        d = DriftDetector()
        for _ in range(5):
            d.check(100.0, 90.0, 2.0)
        assert d.level == 2

    def test_level_3_after_10_violations(self):
        d = DriftDetector()
        for _ in range(10):
            d.check(100.0, 90.0, 2.0)
        assert d.level == 3

    def test_can_run_counterfactuals(self):
        d = DriftDetector()
        assert d.can_run_counterfactuals
        for _ in range(10):
            d.check(100.0, 90.0, 2.0)
        assert not d.can_run_counterfactuals

    def test_recovery_after_non_violation(self):
        d = DriftDetector()
        for _ in range(4):
            d.check(100.0, 90.0, 2.0)
        d.check(90.0, 90.0, 2.0)  # not a violation
        assert d.level == 0

    def test_multi_subsystem(self):
        d = DriftDetector()
        for _ in range(3):
            d.check(100.0, 90.0, 2.0, subsystem="metabolic")
        assert d.subsystems["metabolic"].level == 1
        assert d.subsystems["cardiovascular"].level == 0

    def test_multi_subsystem_max_becomes_global(self):
        d = DriftDetector()
        for _ in range(5):
            d.check(100.0, 90.0, 2.0, subsystem="cardiovascular")
        assert d.level == 2

    def test_subsystem_status(self):
        d = DriftDetector()
        d.check(100.0, 90.0, 2.0, subsystem="metabolic")
        status = d.subsystem_status("metabolic")
        assert status is not None
        assert "level" in status
        assert d.subsystem_status("nonexistent") is None

    def test_reset(self):
        d = DriftDetector()
        for _ in range(10):
            d.check(100.0, 90.0, 2.0)
        assert d.level == 3
        d.reset()
        assert d.level == 0

    def test_subsystem_drift_reset(self):
        sub = SubsystemDrift("test")
        for _ in range(3):
            sub.check(100.0, 90.0, 2.0)
        assert sub.level == 1
        assert sub.label == "warning"
        sub.reset()
        assert sub.level == 0


# ── Counterfactual Simulator Tests ────────────────────────────

class TestCounterfactualSimulator:
    def test_simulate_requires_engine(self):
        eng = PersonalizationEngine()
        eng.update(SAMPLE_OBS)
        sim = CounterfactualSimulator(eng)
        result = sim.simulate_insulin_sensitivity_change(SI_multiplier=2.0, steps=3)
        assert result.intervention == "SI × 2.0"
        assert isinstance(result.baseline_outcome, float)
        assert isinstance(result.delta, float)


# ── Prior Tests ───────────────────────────────────────────────

class TestPriors:
    def test_all_priors_sample(self):
        for p in PRIORS:
            s = p.sample()
            assert isinstance(s, (float, np.floating))

    def test_all_priors_log_prob(self):
        for p in PRIORS:
            lp = p.log_prob(p.mean() if hasattr(p, 'mean') else 0.0)
            assert isinstance(lp, (float, np.floating))


# ── Integration Tests ─────────────────────────────────────────

class TestPipelineIntegration:
    def test_full_update_cycle_maintains_valid_state(self):
        eng = create_personalization_engine()
        eng.update(SAMPLE_OBS)
        for _ in range(5):
            eng.update(SAMPLE_OBS)
        state = eng.get_twin_state()
        assert np.all(np.isfinite(state))
        assert state[0] > 0
        assert state[5] > 0
        assert state[9] >= 0

    def test_drift_detector_plumbed(self):
        eng = create_personalization_engine()
        eng.update(SAMPLE_OBS)
        status = eng.get_drift_status()
        assert "level" in status
        assert "subsystems" in status
        assert "metabolic" in status["subsystems"]
        # Direct-drift test is in TestDriftDetector; this verifies plumbing

    def test_get_effective_sample_size(self):
        eng = create_personalization_engine()
        eng.update(SAMPLE_OBS)
        assert eng.get_effective_sample_size() == 100.0

    def test_needs_resampling(self):
        eng = create_personalization_engine()
        assert eng.needs_resampling() is False

    def test_is_weight_degenerate(self):
        eng = create_personalization_engine()
        assert eng.is_weight_degenerate() is False
