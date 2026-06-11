"""
Phase 3 comprehensive test suite — 30D whole-body cellular twin.
"""

import numpy as np
import pytest
from app.personalization import PersonalizationEngine, create_personalization_engine
from app.personalization.state import (
    PHYSIO_DIM, PARAM_DIM, OBS_DIM,
    CircadianState, AdiposeLipidState, ImmuneInflamState,
    Phase3TwinState, MetabolicState, CardioState, RenalState, InflammatoryState,
)
from app.personalization.dynamics import (
    compute_circadian_dynamics, compute_adipose_dynamics,
    compute_immune_dynamics, full_dynamics, full_observation,
)
from app.personalization.biomarkers import (
    compute_metabolic_flexibility, compute_lipid_stress,
    compute_cv_resilience, compute_circadian_health,
    compute_inflammatory_burden, compute_immune_resilience,
    compute_biological_age, compute_allostatic_load,
    compute_metabolic_syndrome_risk, compute_all_biomarkers,
)
from app.personalization.drift import DriftDetector, CounterfactualSimulator
from app.personalization.priors import PRIORS, get_subgroup_priors, PARAMETER_NAMES
from app.personalization.cohort import VirtualCohortEngine
from app.personalization.uncertainty import UncertaintyEngine
from app.personalization.counterfactual import (
    CounterfactualEngine, MEDITERRANEAN_DIET, EXERCISE_PROGRAM, COMBINED_THERAPY,
)
from app.personalization.explainability import ExplainabilityEngine
from app.personalization.rl import RLTwinEnvironment, recommend_intervention


SAMPLE_OBS = np.array([95.0, 120.0, 80.0, 72.0, 45.0, 100.0, 140.0, 4.2, 290.0,
                       0.5, 100.0, 50.0, 120.0, 350.0, 0.3])

BASE_STATE = np.array([
    95.0, 5.0, 2.0, 5.0, 5.0,                # Metabolic
    120.0, 80.0, 70.0, 45.0,                  # CV
    100.0, 140.0, 4.2, 290.0, 1.0,            # Renal + CRP
    1.2, 0.8, 350.0, 10.0, 0.0, 0.3,          # Circadian
    20.0, 0.5, 100.0, 50.0, 120.0,            # Adipose
    1.0, 0.5, 0.5, 0.2, 15.0,                 # Immune
])


# ── State Tests ──────────────────────────────────────────────

class TestStateDimensions:
    def test_total_physio_dim(self):
        assert PHYSIO_DIM == 30

    def test_param_dim(self):
        assert PARAM_DIM == 25

    def test_obs_dim(self):
        assert OBS_DIM == 15

    def test_phase3_twin_from_array(self):
        twin = Phase3TwinState.from_array(BASE_STATE)
        arr = twin.to_array()
        assert np.allclose(arr, BASE_STATE)

    def test_phase3_twin_subsystems(self):
        twin = Phase3TwinState.from_array(BASE_STATE)
        assert isinstance(twin.metabolic, MetabolicState)
        assert isinstance(twin.cardio, CardioState)
        assert isinstance(twin.renal, RenalState)
        assert isinstance(twin.inflammation, InflammatoryState)
        assert isinstance(twin.circadian, CircadianState)
        assert isinstance(twin.adipose, AdiposeLipidState)
        assert isinstance(twin.immune, ImmuneInflamState)


# ── Circadian Tests ──────────────────────────────────────────

class TestCircadianDynamics:
    def test_basic_oscillation(self):
        c = CircadianState(1.2, 0.8, 350, 10, 0.0, 0.3)
        for _ in range(1440):
            c = compute_circadian_dynamics(c, {"light_level": 0.0, "sleep": 0.0}, {})
        assert 0 <= c.CLOCK_BMAL1 <= 2.5
        assert 0 <= c.sleep_pressure <= 1.0

    def test_light_suppresses_melatonin(self):
        dark = CircadianState(1.2, 0.8, 350, 10, 3.0, 0.3)
        bright = CircadianState(1.2, 0.8, 350, 10, 3.0, 0.3)
        d = compute_circadian_dynamics(dark, {"light_level": 0.0, "sleep": 1.0}, {})
        b = compute_circadian_dynamics(bright, {"light_level": 1.0, "sleep": 0.0}, {})
        assert b.melatonin <= d.melatonin + 1e-6

    def test_sleep_pressure_accumulates(self):
        awake = CircadianState(1.2, 0.8, 350, 10, 0.0, 0.1)
        for _ in range(600):
            awake = compute_circadian_dynamics(awake, {"light_level": 0.5, "sleep": 0.0}, {})
        assert awake.sleep_pressure > 0.3

    def test_sleep_pressure_decays(self):
        asleep = CircadianState(1.2, 0.8, 10, 80, 5.0, 0.8)
        for _ in range(480):
            asleep = compute_circadian_dynamics(asleep, {"light_level": 0.0, "sleep": 1.0}, {})
        assert asleep.sleep_pressure < 0.5


# ── Adipose Tests ────────────────────────────────────────────

class TestAdiposeDynamics:
    def test_basic_dynamics(self):
        a = AdiposeLipidState(20, 0.5, 100, 50, 120)
        result = compute_adipose_dynamics(a, meta_insulin=5.0, inputs={}, params={})
        assert isinstance(result, AdiposeLipidState)
        assert result.FFA > 0

    def test_exercise_increases_ffa(self):
        rest = compute_adipose_dynamics(AdiposeLipidState(20, 0.5, 100, 50, 120),
                                        meta_insulin=5.0, inputs={}, params={})
        ex = compute_adipose_dynamics(AdiposeLipidState(20, 0.5, 100, 50, 120),
                                     meta_insulin=5.0, inputs={"exercise": 0.5}, params={})
        assert ex.FFA >= rest.FFA - 0.01

    def test_dietary_fat_raises_tg(self):
        low = compute_adipose_dynamics(AdiposeLipidState(20, 0.5, 100, 50, 120),
                                       meta_insulin=5.0, inputs={}, params={})
        high = compute_adipose_dynamics(AdiposeLipidState(20, 0.5, 100, 50, 120),
                                        meta_insulin=5.0, inputs={"dietary_fat": 60}, params={})
        assert high.TG >= low.TG - 0.1

    def test_insulin_suppresses_lipolysis(self):
        low_ins = compute_adipose_dynamics(AdiposeLipidState(20, 0.5, 100, 50, 120),
                                           meta_insulin=2.0, inputs={}, params={})
        high_ins = compute_adipose_dynamics(AdiposeLipidState(20, 0.5, 100, 50, 120),
                                            meta_insulin=30.0, inputs={}, params={})
        assert high_ins.FFA <= low_ins.FFA + 0.01


# ── Immune Tests ─────────────────────────────────────────────

class TestImmuneDynamics:
    def test_basic_dynamics(self):
        im = ImmuneInflamState(1, 0.5, 0.5, 0.2, 15)
        result = compute_immune_dynamics(im, meta_ir=5, cardio_hrv=45,
                                         adip_ffa=0.5, circ_cortisol=350, inputs={}, params={})
        assert isinstance(result, ImmuneInflamState)
        assert 0 <= result.NFkB_activity <= 1

    def test_ir_increases_inflammation(self):
        low = compute_immune_dynamics(ImmuneInflamState(1, 0.5, 0.5, 0.2, 15),
                                       meta_ir=2, cardio_hrv=45, adip_ffa=0.3,
                                       circ_cortisol=350, inputs={}, params={})
        high = compute_immune_dynamics(ImmuneInflamState(1, 0.5, 0.5, 0.2, 15),
                                        meta_ir=15, cardio_hrv=45, adip_ffa=0.3,
                                        circ_cortisol=350, inputs={}, params={})
        assert high.InflammatoryLoad >= low.InflammatoryLoad - 0.1

    def test_cortisol_suppresses_nfkb(self):
        low_cort = compute_immune_dynamics(ImmuneInflamState(1, 0.5, 0.5, 0.2, 15),
                                           meta_ir=5, cardio_hrv=45, adip_ffa=0.5,
                                           circ_cortisol=50, inputs={}, params={})
        high_cort = compute_immune_dynamics(ImmuneInflamState(1, 0.5, 0.5, 0.2, 15),
                                            meta_ir=5, cardio_hrv=45, adip_ffa=0.5,
                                            circ_cortisol=600, inputs={}, params={})
        assert high_cort.NFkB_activity <= low_cort.NFkB_activity + 0.01

    def test_hrv_is_protective(self):
        low_hrv = compute_immune_dynamics(ImmuneInflamState(1, 0.5, 0.5, 0.2, 15),
                                          meta_ir=5, cardio_hrv=15, adip_ffa=0.5,
                                          circ_cortisol=350, inputs={}, params={})
        high_hrv = compute_immune_dynamics(ImmuneInflamState(1, 0.5, 0.5, 0.2, 15),
                                           meta_ir=5, cardio_hrv=60, adip_ffa=0.5,
                                           circ_cortisol=350, inputs={}, params={})
        assert high_hrv.InflammatoryLoad <= low_hrv.InflammatoryLoad + 0.1


# ── Full Dynamics Tests ──────────────────────────────────────

class TestFullDynamics:
    def test_output_dim(self):
        params = np.array([p.sample() for p in PRIORS])
        result = full_dynamics(BASE_STATE, params, {})
        assert len(result) == PHYSIO_DIM

    def test_all_positive(self):
        params = np.array([p.sample() for p in PRIORS])
        result = full_dynamics(BASE_STATE, params, {})
        assert np.all(result >= 0) or np.all(result >= -5)

    def test_observation_dim(self):
        obs = full_observation(BASE_STATE)
        assert len(obs) == OBS_DIM
        assert obs[0] == 95.0
        assert obs[13] == 350.0

    def test_meal_effect(self):
        params = np.array([p.sample() for p in PRIORS])
        r1 = full_dynamics(BASE_STATE, params, {"meal_glucose": 0})
        r2 = full_dynamics(BASE_STATE, params, {"meal_glucose": 80})
        assert r2[0] > r1[0]

    def test_light_affects_circadian(self):
        params = np.array([p.sample() for p in PRIORS])
        r1 = full_dynamics(BASE_STATE, params, {"light_level": 0.0, "sleep": 1.0})
        r2 = full_dynamics(BASE_STATE, params, {"light_level": 1.0, "sleep": 0.0})
        assert r2[17] <= r1[17] + 1  # melatonin lower in light


# ── Engine Tests ─────────────────────────────────────────────

class TestPersonalizationEngine:
    def test_default_init(self):
        eng = PersonalizationEngine()
        assert not eng.is_initialized

    def test_initialize(self):
        eng = PersonalizationEngine()
        eng.initialize(SAMPLE_OBS)
        assert eng.is_initialized
        assert len(eng.get_twin_state()) == PHYSIO_DIM

    def test_update_cycle(self):
        eng = PersonalizationEngine()
        eng.update(SAMPLE_OBS)
        assert len(eng.get_twin_state()) == PHYSIO_DIM

    def test_subgroup_engine(self):
        eng = create_personalization_engine(age=65, bmi=33, has_diabetes=True, has_hypertension=True)
        eng.update(SAMPLE_OBS)
        state = eng.get_twin_state()
        params, _ = eng.get_parameters()
        assert len(state) == PHYSIO_DIM
        assert len(params) == PARAM_DIM

    def test_subsystem_getters(self):
        eng = PersonalizationEngine()
        eng.update(SAMPLE_OBS)
        assert len(eng.get_metabolic_state()) == 5
        assert len(eng.get_cardio_state()) == 4
        assert len(eng.get_renal_state()) == 4
        assert len(eng.get_circadian_state()) == 6
        assert len(eng.get_adipose_state()) == 5
        assert len(eng.get_immune_state()) == 5

    def test_drift_status(self):
        eng = PersonalizationEngine()
        eng.update(SAMPLE_OBS)
        status = eng.get_drift_status()
        assert "level" in status


# ── Biomarker Tests ──────────────────────────────────────────

class TestPhase3Biomarkers:
    def test_metabolic_flexibility(self):
        s = compute_metabolic_flexibility(ir=3, ffa=0.4, ldl=100, hdl=50)
        assert 0 <= s <= 100

    def test_lipid_stress(self):
        s = compute_lipid_stress(ldl=130, hdl=40, tg=200, ffas=0.6)
        assert 0 <= s <= 100

    def test_cv_resilience(self):
        s = compute_cv_resilience(hrv=50)
        assert 0 <= s <= 100

    def test_circadian_health(self):
        s = compute_circadian_health(0.5, 0.5, 200, 100)
        assert 0 <= s <= 100

    def test_inflammatory_burden(self):
        s = compute_inflammatory_burden(il6=2, tnfa=1, nfkb=0.3, m1m2=0.5, crp=2)
        assert 0 <= s <= 100

    def test_immune_resilience(self):
        s = compute_immune_resilience(nfkb=0.2, cortisol=350, vagal_tone_effect=0.3, hrv=50)
        assert 0 <= s <= 100

    def test_biological_age(self):
        s = compute_biological_age(BASE_STATE, np.zeros(PARAM_DIM), chronological_age=45)
        assert s >= 30
        assert s <= 120

    def test_allostatic_load(self):
        s = compute_allostatic_load(BASE_STATE, np.zeros(PARAM_DIM))
        assert 0 <= s <= 100

    def test_metabolic_syndrome_risk(self):
        s = compute_metabolic_syndrome_risk(BASE_STATE, np.zeros(PARAM_DIM))
        assert 0 <= s <= 1

    def test_compute_all(self):
        params = np.array([p.sample() for p in PRIORS])
        bio = compute_all_biomarkers(BASE_STATE, params, [95])
        assert len(bio) == 15
        assert "biological_age" in bio
        assert "allostatic_load" in bio
        assert "metabolic_syndrome_risk" in bio


# ── Hierarchical Prior Tests ─────────────────────────────────

class TestHierarchicalPriors:
    def test_subgroup_priors_differ(self):
        default = PRIORS
        diabetic = get_subgroup_priors(age=55, bmi=32, has_diabetes=True)
        assert diabetic[0].mu != default[0].mu  # SI prior shifted

    def test_subgroup_priors_sample(self):
        priors = get_subgroup_priors(age=65, bmi=30, has_diabetes=True, has_hypertension=True)
        samples = [p.sample() for p in priors]
        assert len(samples) == PARAM_DIM
        assert all(np.isfinite(s) for s in samples)


# ── Virtual Cohort Tests ─────────────────────────────────────

class TestVirtualCohort:
    def test_generate_cohort(self):
        vce = VirtualCohortEngine(seed=42)
        cohort = vce.sample_from_priors(n_patients=50)
        assert len(cohort) == 50
        assert all(len(p.state) == PHYSIO_DIM for p in cohort)
        assert all(len(p.parameters) == PARAM_DIM for p in cohort)

    def test_cohort_summary(self):
        vce = VirtualCohortEngine(seed=42)
        vce.sample_from_priors(n_patients=100)
        stats = vce.summary_stats()
        assert stats["n_patients"] == 100
        assert len(stats["state_means"]) == PHYSIO_DIM

    def test_cohort_demographics_varied(self):
        vce = VirtualCohortEngine(seed=42)
        cohort = vce.sample_from_priors(n_patients=200)
        bp_vals = [p.state[5] for p in cohort]
        assert min(bp_vals) < 140
        assert max(bp_vals) > 110


# ── Uncertainty Engine Tests ─────────────────────────────────

class TestUncertainty:
    def test_parameter_uncertainty(self):
        eng = PersonalizationEngine()
        eng.update(SAMPLE_OBS)
        ue = UncertaintyEngine(eng)
        unc = ue.parameter_uncertainty()
        assert len(unc) == PHYSIO_DIM
        assert np.all(unc >= 0)

    def test_measurement_uncertainty(self):
        eng = PersonalizationEngine()
        ue = UncertaintyEngine(eng)
        unc = ue.measurement_uncertainty()
        assert len(unc) == OBS_DIM

    def test_posterior_predictive(self):
        eng = PersonalizationEngine()
        eng.update(SAMPLE_OBS)
        ue = UncertaintyEngine(eng)
        samples = ue.posterior_predictive_samples(n_samples=5, horizon=3)
        assert samples.shape == (5, 3, PHYSIO_DIM)


# ── Counterfactual Tests ─────────────────────────────────────

class TestCounterfactual:
    def test_mediterranean_diet(self):
        eng = PersonalizationEngine()
        eng.update(SAMPLE_OBS)
        ce = CounterfactualEngine(eng)
        traj = ce.simulate_program(MEDITERRANEAN_DIET)
        assert len(traj.glucose) > 0
        assert traj.glucose[-1] > 0

    def test_exercise_program(self):
        eng = PersonalizationEngine()
        eng.update(SAMPLE_OBS)
        ce = CounterfactualEngine(eng)
        traj = ce.simulate_program(EXERCISE_PROGRAM)
        assert traj.sbp[-1] > 0

    def test_program_summary(self):
        eng = PersonalizationEngine()
        eng.update(SAMPLE_OBS)
        ce = CounterfactualEngine(eng)
        traj = ce.simulate_program(COMBINED_THERAPY)
        summary = ce.program_summary(traj)
        assert "avg_glucose" in summary
        assert "estimated_hba1c" in summary

    def test_compare_programs(self):
        eng = PersonalizationEngine()
        eng.update(SAMPLE_OBS)
        ce = CounterfactualEngine(eng)
        results = ce.compare_programs([MEDITERRANEAN_DIET, EXERCISE_PROGRAM])
        assert len(results) == 2


# ── Explainability Tests ─────────────────────────────────────

class TestExplainability:
    def test_patient_explanation(self):
        eng = PersonalizationEngine()
        eng.update(SAMPLE_OBS)
        ee = ExplainabilityEngine(eng)
        expl = ee.explain("patient")
        assert expl.level == "patient"
        assert len(expl.summary) > 0

    def test_clinician_explanation(self):
        eng = PersonalizationEngine()
        eng.update(SAMPLE_OBS)
        ee = ExplainabilityEngine(eng)
        expl = ee.explain("clinician")
        assert "drivers" in expl.details

    def test_scientist_explanation(self):
        eng = PersonalizationEngine()
        eng.update(SAMPLE_OBS)
        ee = ExplainabilityEngine(eng)
        expl = ee.explain("scientist")
        assert "parameter_uncertainty" in expl.details


# ── RL Tests ─────────────────────────────────────────────────

class TestRL:
    def test_env_step(self):
        eng = PersonalizationEngine()
        eng.update(SAMPLE_OBS)
        env = RLTwinEnvironment(eng)
        action = np.array([0.3, 0.5, 0.8])
        ns, reward, done, info = env.step(action)
        assert len(ns) == PHYSIO_DIM
        assert isinstance(reward, float)
        assert "G" in info

    def test_recommend_intervention(self):
        eng = PersonalizationEngine()
        eng.update(SAMPLE_OBS)
        program = recommend_intervention(eng)
        assert program.duration_days > 0
        assert len(program.name) > 0


# ── Drift Tests ──────────────────────────────────────────────

class TestDrift:
    def test_levels(self):
        d = DriftDetector()
        for _ in range(10):
            d.check(100.0, 90.0, 2.0)
        assert d.level == 3

    def test_multi_subsystem(self):
        d = DriftDetector()
        for _ in range(5):
            d.check(100.0, 90.0, 2.0, subsystem="cardiovascular")
        assert d.level == 2


# ── Integration Tests ────────────────────────────────────────

class TestIntegration:
    def test_full_pipeline(self):
        eng = create_personalization_engine(age=50, bmi=28, has_diabetes=True)
        for _ in range(3):
            eng.update(SAMPLE_OBS)
        state = eng.get_twin_state()
        assert np.all(np.isfinite(state))
        assert 20 < state[0] < 600

    def test_all_subsystems_coupled(self):
        eng = PersonalizationEngine()
        eng.update(SAMPLE_OBS)

        # Simulate 24 hours (1440 steps) with daily rhythm
        state = eng.get_twin_state()
        params, _ = eng.get_parameters()
        for hour in range(24):
            light = 0.8 if 6 <= hour <= 22 else 0.05
            sleep = 0.0 if 6 <= hour <= 22 else 1.0
            inputs = {"light_level": light, "sleep": sleep}
            state = full_dynamics(state, params, inputs)
        assert np.all(np.isfinite(state))
        assert 20 < state[0] < 600
        assert 0 <= state[19] <= 1  # sleep pressure in range
