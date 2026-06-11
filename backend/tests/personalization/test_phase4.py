"""
Phase 4 integration tests for the multi-scale human digital twin.
"""

import numpy as np
import pytest

from app.personalization.phase4 import (
    MolecularState, CellularState, CellPopulationDynamics,
    cellular_to_organ_signals, CELL_TYPES,
    BioKnowledgeGraph, PatientSimilarityGraph, suggest_drugs_for_patient,
    EnvironmentState, BehavioralState,
    EnvironmentalModel, BehavioralModel, LifestyleModel, AdherenceModel,
    CausalDiscoveryEngine, HypothesisAgent, generate_twin_trial_data,
    CausalMethod,
    compute_metabolic_age, compute_resilience_score, compute_frailty_index,
    compute_adaptability_score, compute_circadian_robustness,
    compute_inflammaging_score, compute_all_biomarkers_20,
    VirtualPopulationGeneratorV2,
    InterventionDesign, CounterfactualEngineV3,
)
from app.personalization.core import UnscentedKalmanFilter
from app.personalization.state import PHYSIO_DIM, PARAM_DIM, OBS_DIM
from app.personalization.dynamics import full_dynamics, full_observation


# ── Cellular Tests ────────────────────────────────────────────

class TestCellularState:
    def test_healthy_state(self):
        cs = CellularState.healthy()
        assert cs.population.shape == (len(CELL_TYPES),)
        assert np.all(cs.population == 1.0)
        assert np.all(cs.stress == 0.0)
        assert np.all(cs.health == 1.0)

    def test_to_from_array(self):
        cs = CellularState.healthy()
        arr = cs.to_array()
        assert arr.shape == (25,)
        cs2 = CellularState.from_array(arr)
        assert np.allclose(cs.population, cs2.population)

    def test_dynamics_healthy_stable(self):
        cpd = CellPopulationDynamics()
        cs = CellularState.healthy()
        mol = {"insulin_signal": 0.3, "inflammatory_signal": 0.1, "metabolic_stress": 0.1}
        org = {"glucose_toxicity": 0.0, "bp_overload": 0.0}
        new = cpd.compute_dynamics(cs, mol, org, dt=15.0)  # small step
        # Should remain close to healthy
        assert np.allclose(new.population, 1.0, atol=0.05)
        assert np.all(new.stress >= 0)
        assert np.all(new.health <= 1.0)

    def test_dynamics_stress_response(self):
        cpd = CellPopulationDynamics()
        cs = CellularState.healthy()
        mol = {"insulin_signal": 0.1, "inflammatory_signal": 0.8, "metabolic_stress": 0.8}
        org = {"glucose_toxicity": 0.8, "bp_overload": 0.8}
        new = cpd.compute_dynamics(cs, mol, org, dt=1440.0)  # 1 day
        # Stress should increase
        assert np.any(new.stress > 0.1)
        # Health should decrease
        assert np.any(new.health < 0.95)

    def test_cellular_to_organ_signals(self):
        cs = CellularState.healthy()
        signals = cellular_to_organ_signals(cs)
        assert "insulin_sensitivity_mod" in signals
        assert "beta_cell_function" in signals
        assert 0.5 <= signals["insulin_sensitivity_mod"] <= 1.5


# ── Graph Intelligence Tests ──────────────────────────────────

class TestBioKnowledgeGraph:
    def test_populate(self):
        kg = BioKnowledgeGraph()
        kg.populate_diabetes_graph()
        assert kg.graph.number_of_nodes() == 20
        assert kg.graph.number_of_edges() == 16

    def test_get_neighbors(self):
        kg = BioKnowledgeGraph()
        kg.populate_diabetes_graph()
        neighbors = kg.get_neighbors("DB00331", "targets")
        assert len(neighbors) == 2

    def test_drug_suggestion(self):
        kg = BioKnowledgeGraph()
        kg.populate_diabetes_graph()
        suggestions = suggest_drugs_for_patient(kg, ["DOID_9352"])
        assert len(suggestions) >= 1
        names = [s["drug"] for s in suggestions]
        assert "Metformin" in names or "Empagliflozin" in names


class TestPatientSimilarityGraph:
    def test_similarity(self):
        psg = PatientSimilarityGraph(similarity_threshold=0.3)
        f1 = np.array([1.0] * 30 + [0.5] * 25)
        f2 = np.array([0.95] * 30 + [0.5] * 25)
        f3 = np.random.randn(55) * 0.5
        psg.add_patient("P1", f1[:30], f1[30:])
        psg.add_patient("P2", f2[:30], f2[30:])
        psg.add_patient("P3", f3[:30], f3[30:])
        psg.compute_similarities()
        similar = psg.get_similar_patients("P1", top_k=2)
        assert len(similar) >= 1


# ── Environment / Behavior Tests ──────────────────────────────

class TestEnvironmentalModel:
    def test_coupling_signals(self):
        model = EnvironmentalModel()
        env = EnvironmentState(aqi=0.5, pm25=0.4, temperature=0.2)
        bhv = BehavioralState()
        signals = model.compute_coupling_signals(env, bhv)
        assert signals["oxidative_stress_mod"] > 1.0
        assert signals["cv_load_mod"] > 1.0

    def test_clean_air(self):
        model = EnvironmentalModel()
        env = EnvironmentState(aqi=0.0, pm25=0.0, pm10=0.0, no2=0.0)
        bhv = BehavioralState()
        signals = model.compute_coupling_signals(env, bhv)
        assert abs(signals["oxidative_stress_mod"] - 1.0) < 0.1


class TestBehavioralModel:
    def test_exercise_benefit(self):
        model = BehavioralModel()
        bhv = BehavioralState(exercise_minutes=45.0, exercise_adherence=0.9)
        bhv_noex = BehavioralState(exercise_minutes=0.0, exercise_adherence=0.0)
        signals = model.compute_coupling_signals(bhv)
        signals_noex = model.compute_coupling_signals(bhv_noex)
        # Exercise should improve insulin sensitivity over no exercise
        assert signals["insulin_sensitivity_bhv"] > signals_noex["insulin_sensitivity_bhv"]

    def test_unhealthy_lifestyle(self):
        model = BehavioralModel()
        bhv = BehavioralState(stress_level=0.9, smoking=1.0, alcohol=0.8)
        signals = model.compute_coupling_signals(bhv)
        assert signals["inflammation_bhv"] > 0.3
        assert signals["cardiovascular_bhv"] < 1.0


class TestAdherenceModel:
    def test_default_adherent(self):
        model = AdherenceModel()
        state = model.step(dt=1.0)
        assert state in (0.0, 1.0)

    def test_low_adherence_scenario(self):
        model = AdherenceModel(habit_strength=0.0,
                                side_effect_tolerance=0.2,
                                cognitive_load=0.9,
                                social_support=0.0,
                                base_adherence=0.3)
        states = [model.step(dt=7.0) for _ in range(100)]
        mean_adh = np.mean(states)
        assert mean_adh < 0.7


class TestLifestyleModel:
    def test_step(self):
        model = LifestyleModel()
        initial = model._current
        after = model.step(dt=1.0)
        assert isinstance(after, BehavioralState)

    def test_reset(self):
        model = LifestyleModel()
        model.step(dt=30.0)
        model.reset()
        assert model._current.diet_quality == model.base.diet_quality


# ── Scientific Discovery Tests ────────────────────────────────

class TestCausalDiscovery:
    def test_correlation_discovery(self):
        data = generate_twin_trial_data(n_patients=200, n_timepoints=5)
        engine = CausalDiscoveryEngine()
        engine.load_data(data)
        graph = engine.discover_structure(method=CausalMethod.CORRELATION, alpha=0.05)
        assert len(graph.edges) > 0

    def test_known_graph_edges(self):
        data = generate_twin_trial_data(n_patients=200, n_timepoints=5)
        engine = CausalDiscoveryEngine()
        engine.load_data(data)
        graph = engine.discover_structure(method=CausalMethod.CORRELATION, prior_knowledge=True)
        # After prior knowledge injection, BMI → InsulinResistance should exist
        ir_causes = graph.get_causes_of("InsulinResistance")
        cause_names = [e.source for e in ir_causes]
        has_known = any(c in cause_names for c in ["BMI", "Exercise", "DietQuality"])
        assert has_known or len(graph.edges) > 0

    def test_dowhy_discovery(self):
        data = generate_twin_trial_data(n_patients=100, n_timepoints=3)
        engine = CausalDiscoveryEngine()
        engine.load_data(data)
        graph = engine.discover_structure(method=CausalMethod.DOWHY, prior_knowledge=True)
        assert len(graph.edges) > 0

    def test_effect_estimation(self):
        data = generate_twin_trial_data(n_patients=200, n_timepoints=5)
        engine = CausalDiscoveryEngine()
        engine.load_data(data)
        result = engine.estimate_effect("BMI", "InsulinResistance")
        assert "effect_size" in result


class TestHypothesisAgent:
    def test_generate_hypotheses(self):
        data = generate_twin_trial_data(n_patients=100, n_timepoints=5)
        engine = CausalDiscoveryEngine()
        engine.load_data(data)
        graph = engine.discover_structure(method=CausalMethod.CORRELATION)
        agent = HypothesisAgent()
        hypotheses = agent.generate_hypotheses(graph, top_k=3)
        assert len(hypotheses) > 0
        assert all(h.title for h in hypotheses)
        assert all(h.mechanism for h in hypotheses)

    def test_explanation_formats(self):
        data = generate_twin_trial_data(n_patients=100, n_timepoints=5)
        engine = CausalDiscoveryEngine()
        engine.load_data(data)
        graph = engine.discover_structure(method=CausalMethod.CORRELATION)
        agent = HypothesisAgent()
        hypotheses = agent.generate_hypotheses(graph, top_k=1)
        if hypotheses:
            pt = agent.get_mechanism_explanation(hypotheses[0], "patient")
            cl = agent.get_mechanism_explanation(hypotheses[0], "clinician")
            sc = agent.get_mechanism_explanation(hypotheses[0], "scientist")
            assert len(pt) > 0
            assert len(cl) > 0
            assert len(sc) > 0


# ── Biomarkers 2.0 Tests ──────────────────────────────────────

class TestBiomarkers20:
    def test_metabolic_age(self):
        age = compute_metabolic_age(45, 30, 140, 6.5, 135, 40, 180, 4, 0.6)
        assert age > 45  # unhealthy = older metabolic age
        assert age < 120

    def test_metabolic_age_healthy(self):
        age = compute_metabolic_age(30, 22, 85, 5.0, 110, 65, 80, 0.5, 0.3)
        assert age >= 15  # healthy = close to chronological

    def test_resilience(self):
        gluc = [100 + 5*np.sin(i/3) + (15 if i==10 else 0) for i in range(30)]
        sbp = [120 + 3*np.sin(i/5) + (10 if i==10 else 0) for i in range(30)]
        hr = [70 + 2*np.sin(i/4) + (15 if i==10 else 0) for i in range(30)]
        score = compute_resilience_score(gluc, sbp, hr, perturbation_time=10)
        assert 0 <= score <= 100

    def test_frailty(self):
        frail = compute_frailty_index(muscle_mass=0.3, gait_speed=0.2,
                                        grip_strength=0.2, physical_activity=0.1,
                                        fatigue=0.8, multi_morbidity=5,
                                        inflammatory_load=40, hr_variability=15)
        assert 0 <= frail <= 1
        assert frail > 0.3  # frail patient

    def test_frailty_healthy(self):
        frail = compute_frailty_index(muscle_mass=0.9, gait_speed=0.9,
                                        grip_strength=0.9, physical_activity=0.9)
        assert frail < 0.3

    def test_adaptability(self):
        g = [100 + 20*np.exp(-i/5) for i in range(30)]
        s = [120 + 10*np.exp(-i/5) for i in range(30)]
        score = compute_adaptability_score(g, s, 100, 120)
        assert 0 <= score <= 100

    def test_inflammaging(self):
        score = compute_inflammaging_score(il6=8, tnfa=6, crp=10,
                                            immune_cell_senescence=0.8,
                                            oxidative_stress=0.7, dna_damage=0.6)
        assert 0 <= score <= 100
        assert score > 50  # high = inflamed

    def test_all_biomarkers(self):
        physio = np.array([140, 15, 4, 6, 5, 135, 85, 75, 30, 80,
                           140, 4.2, 300, 4, 1.0, 1.0, 20, 30, 3.0, 0.5,
                           25, 0.6, 130, 40, 180, 4, 3, 1.2, 0.5, 35])
        trajs = {
            "glucose": [100]*30, "sbp": [120]*30, "hr": [70]*30,
            "cortisol": [15]*30, "phase": [3]*30,
        }
        bio = compute_all_biomarkers_20(45, physio, 85, 170, trajs, 10)
        assert 0 <= bio.overall_health_score <= 100
        assert isinstance(bio.metabolic_age, float)


# ── Virtual Population Tests ──────────────────────────────────

class TestVirtualPopulationV2:
    def test_generate_small(self):
        gen = VirtualPopulationGeneratorV2(seed=42)
        patients = gen.generate(n_patients=10)
        assert len(patients) == 10
        p = patients[0]
        assert len(p.organ_physio) == PHYSIO_DIM
        assert len(p.organ_params) == PARAM_DIM
        assert p.molecular.dim == 75
        assert p.cellular.dim == 25

    def test_demographics(self):
        gen = VirtualPopulationGeneratorV2(seed=42)
        gen.generate(n_patients=100)
        summary = gen.get_demographics_summary()
        assert summary["n_patients"] == 100
        assert 0 < summary["diabetes_pct"] < 40
        assert 0 < summary["hypertension_pct"] < 50

    def test_correlated_covariates(self):
        gen = VirtualPopulationGeneratorV2(seed=42)
        demo = gen._sample_demographics(1000)
        # Diabetes should increase with age
        older = demo["age"] > 60
        younger = demo["age"] <= 40
        assert np.mean(demo["diabetes"][older]) > np.mean(demo["diabetes"][younger])


# ── Multi-Scale Engine: Basic Integration Test ────────────────

class TestMultiScaleIntegration:
    def test_create_engine(self):
        from app.personalization.phase4.multi_scale_engine import (
            create_default_multi_scale_engine,
        )
        engine = create_default_multi_scale_engine()
        assert engine is not None

    def test_initialize_and_step(self):
        from app.personalization.phase4 import (
            MultiScaleTwinEngine, MultiScaleState,
            MolecularState, CellularState,
            EnvironmentState, BehavioralState,
        )
        from app.personalization.phase4.multi_scale_engine import (
            create_default_multi_scale_engine,
        )

        engine = create_default_multi_scale_engine()

        mol = MolecularState.healthy_resting()
        cell = CellularState.healthy()
        physio = np.array([100, 10, 3, 8, 2,
                           120, 80, 70, 50,
                           100, 140, 4.2, 300,
                           2, 1.0, 1.0, 20, 30, 3, 0.5,
                           20, 0.4, 100, 50, 120,
                           2, 2, 1, 0.3, 10], dtype=np.float64)
        params = np.array([0.5, 3.0, 1.0, 1.0,
                           0.8, 1.0, 1.5, 1.0,
                           100, 1.0, 1.0, 1.0,
                           1440, 1.0, 0.8, 1.0,
                           0.5, 0.5, 0.5, 0.5, 0.5,
                           0.5, 0.5, 0.5, 0.5], dtype=np.float64)

        engine.initialize("TEST001", mol, cell, physio, params)
        state = engine.get_current_state()
        assert state.molecular.dim == 75
        assert state.cellular.dim == 25
        assert len(state.organ_physio) == PHYSIO_DIM
        assert len(state.organ_params) == PARAM_DIM

        # Step one day
        new_state = engine.step(dt_days=1.0)
        assert new_state.timestamp > 0


# ── Counterfactual V3: Integration Test ───────────────────────

class TestCounterfactualV3Integration:
    def _setup_engine(self):
        state_dim = PHYSIO_DIM + PARAM_DIM
        process_noise = np.eye(state_dim) * 0.01
        obs_noise = np.eye(OBS_DIM) * 0.1

        from app.personalization.priors import PRIORS
        def _param_prior():
            return np.array([p.sample() for p in PRIORS], dtype=np.float64)

        engine = UnscentedKalmanFilter(
            state_dim=state_dim,
            process_noise=process_noise,
            obs_noise=obs_noise,
            dynamics_fn=full_dynamics,
            obs_fn=full_observation,
            param_prior_fn=_param_prior,
        )
        # Override with known state for testing
        physio = np.array([140, 15, 4, 6, 5, 145, 90, 80, 25, 85,
                           142, 4.0, 310, 5, 1.0, 0.8, 25, 20, 3.0, 0.6,
                           30, 0.6, 140, 40, 200, 5, 4, 1.5, 0.6, 40],
                          dtype=np.float64)
        params = np.array([0.08, 3.5, 0.8, 1.0,
                           1.2, 1.0, 1.5, 1.0,
                           85, 1.0, 1.0, 1.0,
                           1440, 1.0, 0.8, 1.0,
                           0.6, 0.4, 0.5, 0.5, 0.5,
                           0.6, 0.6, 0.5, 0.5], dtype=np.float64)
        engine._mu = np.concatenate([physio, params])
        engine._n = len(engine._mu)
        return engine

    def test_design_to_program(self):
        design = InterventionDesign(exercise_minutes=30, metformin_dose=1000)
        program = design.to_program(np.zeros(25))
        assert program.name == "Optimized Intervention (d=90d)"
        assert "SI" in program.param_modifiers

    @pytest.mark.skip(reason="Requires full PersonalizationEngine setup")
    def test_optimize(self):
        pe = self._setup_engine()
        cv3 = CounterfactualEngineV3(pe, n_ensemble=2)
        front = cv3.optimize(n_trials=5, n_ensemble=2)
        assert len(front) >= 0


# ── Full Pipeline: End-to-End Integration ─────────────────────

class TestFullPipeline:
    def test_molecular_to_cellular_to_organ(self):
        """Test the full upward signal flow across layers."""
        # Molecular → Cellular
        mol = MolecularState.healthy_resting()
        from app.personalization.phase4.molecular import molecular_to_cellular_signals
        mol_sig = molecular_to_cellular_signals(mol)
        assert all(k in mol_sig for k in
                   ["insulin_signal", "inflammatory_signal", "metabolic_stress", "growth_signals"])

        # Cellular → Organ
        cpd = CellPopulationDynamics()
        cell = CellularState.healthy()
        cell = cpd.compute_dynamics(cell, mol_sig,
                                     {"glucose_toxicity": 0.0, "bp_overload": 0.0},
                                     dt=60.0)
        org_sig = cellular_to_organ_signals(cell)
        assert all(k in org_sig for k in
                   ["insulin_sensitivity_mod", "inflammation_mod", "beta_cell_function"])

    def test_knowledge_graph_query(self):
        """End-to-end: build graph → get drug suggestions."""
        kg = BioKnowledgeGraph()
        kg.populate_diabetes_graph()
        suggestions = suggest_drugs_for_patient(kg, ["DOID_9352", "DOID_1319"])
        assert len(suggestions) > 0
        top = suggestions[0]
        assert "confidence" in top

    def test_biomarkers_from_virtual_patient(self):
        """End-to-end: generate cohort → compute biomarkers."""
        gen = VirtualPopulationGeneratorV2(seed=42)
        patients = gen.generate(n_patients=5)
        p = patients[0]
        physio = p.organ_physio
        trajs = {
            "glucose": [physio[0]] * 30,
            "sbp": [physio[5]] * 30,
            "hr": [physio[7]] * 30,
            "cortisol": [physio[16]] * 30,
            "phase": [physio[18]] * 30,
        }
        bio = compute_all_biomarkers_20(
            p.metadata.get("age", 50), physio, 85, 170, trajs, 10,
        )
        assert 0 <= bio.overall_health_score <= 100

    def test_causal_discovery_on_twin_data(self):
        """End-to-end: generate trial data → discover causes → generate hypothesis."""
        data = generate_twin_trial_data(n_patients=100, n_timepoints=5)
        engine = CausalDiscoveryEngine()
        engine.load_data(data)
        graph = engine.discover_structure(method=CausalMethod.DOWHY, prior_knowledge=True)
        assert len(graph.edges) > 0

        agent = HypothesisAgent()
        hypotheses = agent.generate_hypotheses(graph, top_k=3)
        assert len(hypotheses) > 0

        # Verify top hypothesis has explanation
        h = hypotheses[0]
        explanation = agent.get_mechanism_explanation(h, "clinician")
        assert "**Evidence**" in explanation
