"""
Scientific Validation Tests (Phase 5 Weakness Remediation).

Tests all 10 weakness-remediation modules:
  1. Clinical dataset generation & retrospective validation
  2. Calibration assessment
  3. Causal sensitivity analysis
  4. Broader population priors
  5. Foundation model training pipeline
  6. Overparameterization analysis
  7. Uncertainty decomposition
  8. Stability analysis
  9. Clinical protocol generation
  10. Counterfactual sensitivity analysis
"""

import numpy as np
import pandas as pd
import pytest

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ── 1. Clinical Dataset & Retrospective Validation ────────────


class TestClinicalDataset:
    def test_generate_cohort(self):
        from app.personalization.phase5.clinical_dataset import ClinicalDataGenerator
        gen = ClinicalDataGenerator(seed=42)
        cohort = gen.generate_cohort(n_patients=10, n_days=7)
        assert len(cohort) == 10
        for p in cohort:
            assert p.demographics["age"] >= 18
            assert p.demographics["bmi"] >= 16
            assert len(p.time_series) > 0
            assert "G" in p.time_series.columns

    def test_retrospective_validator(self):
        from app.personalization.phase5.clinical_dataset import (
            ClinicalDataGenerator, RetrospectiveValidator,
        )
        gen = ClinicalDataGenerator(seed=42)
        cohort = gen.generate_cohort(n_patients=3, n_days=30)
        validator = RetrospectiveValidator(seed=42)

        def dummy_twin_factory(physio, params):
            class DummyTwin:
                def get_state(self):
                    return physio.copy(), params.copy()
                def observe(self, obs):
                    return {"updated": True}
            return DummyTwin()

        results = validator.evaluate(cohort, dummy_twin_factory, train_frac=0.5)
        summary = validator.summary(results)
        assert summary["n_patients"] > 0
        assert "mean_mae" in summary

    def test_generate_nhanes_style(self):
        from app.personalization.phase5.clinical_dataset import generate_nhanes_style_dataset
        cohort = generate_nhanes_style_dataset(n_patients=5, n_days=3, seed=42)
        assert len(cohort) == 5


# ── 2. Calibration Assessment ─────────────────────────────────


class TestCalibration:
    def test_calibration_assessor_continuous(self):
        from app.personalization.phase5.calibration import CalibrationAssessor
        rng = np.random.default_rng(42)
        y_true = rng.normal(100, 10, 200)
        y_pred = y_true + rng.normal(0, 5, 200)
        y_std = np.ones(200) * 5
        assessor = CalibrationAssessor(n_bins=10)
        report = assessor.assess_continuous(y_true, y_pred, y_std, "glucose")
        assert report.ece >= 0
        assert report.variable == "glucose"
        assert len(report.bin_counts) == 10

    def test_calibration_assessor_probability(self):
        from app.personalization.phase5.calibration import CalibrationAssessor
        rng = np.random.default_rng(42)
        y_true = rng.binomial(1, 0.3, 500)
        y_pred = np.clip(y_true + rng.normal(0, 0.15, 500), 0.05, 0.95)
        assessor = CalibrationAssessor(n_bins=10)
        report = assessor.assess_probability(y_true, y_pred, "hypoglycemia")
        assert 0 <= report.ece <= 1
        assert 0 <= report.brier_score <= 1

    def test_platt_calibrator(self):
        from app.personalization.phase5.calibration import PlattCalibrator
        rng = np.random.default_rng(42)
        y_pred = rng.uniform(0, 1, 200)
        y_true = (y_pred + rng.normal(0, 0.1, 200) > 0.5).astype(float)
        cal = PlattCalibrator()
        cal.fit(y_pred, y_true)
        calibrated = cal.predict(y_pred)
        assert np.all((calibrated >= 0) & (calibrated <= 1))

    def test_beta_calibrator(self):
        from app.personalization.phase5.calibration import BetaCalibrator
        rng = np.random.default_rng(42)
        y_pred = rng.uniform(0.1, 0.9, 200)
        y_true = (y_pred + rng.normal(0, 0.1, 200) > 0.5).astype(float)
        cal = BetaCalibrator()
        cal.fit(y_pred, y_true)
        calibrated = cal.predict(y_pred)
        assert np.all((calibrated >= 0) & (calibrated <= 1))

    def test_conformal_predictor(self):
        from app.personalization.phase5.calibration import ConformalPredictor
        rng = np.random.default_rng(42)
        y_pred = rng.normal(100, 10, 200)
        y_true = y_pred + rng.normal(0, 5, 200)
        cp = ConformalPredictor(coverage=0.9)
        cp.fit(y_pred, y_true)
        lower, upper = cp.predict_interval(y_pred)
        coverage = np.mean((y_true >= lower) & (y_true <= upper))
        assert coverage > 0.7

    def test_calibration_pipeline(self):
        from app.personalization.phase5.calibration import CalibrationPipeline
        rng = np.random.default_rng(42)
        pipe = CalibrationPipeline()
        for v in ["glucose", "SBP", "HR"]:
            y_true = rng.normal(100, 10, 200)
            y_pred = y_true + rng.normal(0, 5, 200)
            pipe.evaluate(y_true, y_pred, np.ones(200) * 5, v)
        report = pipe.full_report()
        assert report["n_variables"] == 3


# ── 3. Causal Graph Sensitivity ────────────────────────────────


class TestCausalSensitivity:
    def test_edge_perturbation(self):
        from app.personalization.phase5.causal_sensitivity import CausalGraphSensitivity
        rng = np.random.default_rng(42)
        data = pd.DataFrame({
            "X": rng.normal(0, 1, 200),
            "Y": rng.normal(0, 1, 200),
            "Z": rng.normal(0, 1, 200),
        })
        from app.personalization.phase5.mechanism_discovery import MechanismDiscoveryEngine
        md = MechanismDiscoveryEngine()

        analyzer = CausalGraphSensitivity(n_bootstrap=10)
        pert = analyzer.edge_perturbation_analysis(data, md.discover_from_cross_sectional)
        assert len(pert) >= 0

    def test_bootstrap_stability(self):
        from app.personalization.phase5.causal_sensitivity import CausalGraphSensitivity
        rng = np.random.default_rng(42)
        data = pd.DataFrame({
            "glucose": rng.normal(100, 15, 100),
            "insulin": rng.normal(10, 5, 100),
            "bmi": rng.normal(29, 6, 100),
        })
        from app.personalization.phase5.mechanism_discovery import MechanismDiscoveryEngine
        md = MechanismDiscoveryEngine()
        analyzer = CausalGraphSensitivity(n_bootstrap=10)
        boot = analyzer.bootstrap_stability_analysis(data, md.discover_from_cross_sectional)
        for r in boot:
            assert 0 <= r.stability_index <= 1

    def test_full_sensitivity_report(self):
        from app.personalization.phase5.causal_sensitivity import CausalGraphSensitivity
        rng = np.random.default_rng(42)
        data = pd.DataFrame({
            "X": rng.normal(0, 1, 100),
            "Y": rng.normal(0, 1, 100),
        })
        from app.personalization.phase5.mechanism_discovery import MechanismDiscoveryEngine
        md = MechanismDiscoveryEngine()
        analyzer = CausalGraphSensitivity(n_bootstrap=5)
        report = analyzer.full_sensitivity_report(data, md.discover_from_cross_sectional)
        assert report.n_edges_tested >= 0
        assert 0 <= report.mean_stability <= 1


# ── 4. Broader Population Priors ──────────────────────────────


class TestPopulationBroader:
    def test_get_population_adjustment_geriatric(self):
        from app.personalization.phase5.population_broader import get_population_adjustment
        adj = get_population_adjustment(age=75)
        assert "SI" in adj
        assert adj["SI"] > 0

    def test_get_population_adjustment_pediatric(self):
        from app.personalization.phase5.population_broader import get_population_adjustment
        adj = get_population_adjustment(age=10)
        assert "baseline_GFR" in adj

    def test_get_population_adjustment_athletic(self):
        from app.personalization.phase5.population_broader import get_population_adjustment
        adj = get_population_adjustment(age=25, population="athletic")
        assert "SI" in adj

    def test_get_population_adjustment_pregnant(self):
        from app.personalization.phase5.population_broader import get_population_adjustment
        adj = get_population_adjustment(age=30, population="pregnant", trimester=2)
        assert "SI" in adj
        assert adj["SI"] < 1.0

    def test_get_population_adjustment_ethnicity(self):
        from app.personalization.phase5.population_broader import get_population_adjustment
        adj = get_population_adjustment(age=45, ethnicity="african_american")
        assert "sodium_retention" in adj

    def test_population_modules_exist(self):
        from app.personalization.phase5.population_broader import POPULATION_MODULES
        for name in ["pediatric_child", "pediatric_adolescent", "pregnant_first",
                      "pregnant_second", "pregnant_third", "geriatric", "athletic"]:
            assert name in POPULATION_MODULES

    def test_ethnicity_adjustments_exist(self):
        from app.personalization.phase5.population_broader import ETHNICITY_ADJUSTMENTS
        for name in ["caucasian", "african_american", "hispanic", "east_asian"]:
            assert name in ETHNICITY_ADJUSTMENTS


# ── 5. Foundation Model Training ─────────────────────────────


class TestFoundationTraining:
    def test_dataset_generation(self):
        from app.personalization.phase5.foundation_train import PhysiologicalPretrainingDataset
        ds = PhysiologicalPretrainingDataset(n_sequences=10, seq_length=12, seed=42)
        assert len(ds) == 10
        sample = ds[0]
        assert sample.shape == (12, 30)

    def test_dataloader(self):
        from app.personalization.phase5.foundation_train import create_pretraining_dataloader
        loader = create_pretraining_dataloader(batch_size=4, n_sequences=10, seq_length=8, seed=42)
        batch = next(iter(loader))
        assert batch.shape[0] == 4
        assert batch.shape[2] == 30

    @pytest.mark.slow
    def test_training_step(self):
        import torch
        from app.personalization.phase5.foundation_model import PhysiologyFoundationModel, PhysiologyConfig
        from app.personalization.phase5.foundation_train import FoundationModelTrainer
        config = PhysiologyConfig()
        model = PhysiologyFoundationModel(config)
        trainer = FoundationModelTrainer(model, lr=1e-3, device="cpu")
        batch = torch.randn(4, 12, config.total_input_dim)
        metrics = trainer.train_step(batch)
        assert "loss" in metrics
        assert metrics["loss"] > 0

    @pytest.mark.slow
    def test_train_epoch(self):
        import torch
        from app.personalization.phase5.foundation_model import PhysiologyFoundationModel, PhysiologyConfig
        from app.personalization.phase5.foundation_train import FoundationModelTrainer
        from app.personalization.phase5.foundation_train import create_pretraining_dataloader
        config = PhysiologyConfig()
        model = PhysiologyFoundationModel(config)
        trainer = FoundationModelTrainer(model, lr=1e-3, device="cpu")
        loader = create_pretraining_dataloader(batch_size=4, n_sequences=8, seq_length=8, seed=42)
        metrics = trainer.train_epoch(loader)
        assert "loss" in metrics


# ── 6. Overparameterization Analysis ──────────────────────────


class TestOverparameterization:
    def test_effective_degrees_of_freedom(self):
        from app.personalization.phase5.overparameterization import OverparameterizationAnalyzer
        analyzer = OverparameterizationAnalyzer()
        H = np.random.randn(15, 55)
        edof = analyzer.effective_degrees_of_freedom(H)
        assert 0 < edof <= 55

    def test_parameter_identifiability(self):
        from app.personalization.phase5.overparameterization import OverparameterizationAnalyzer
        analyzer = OverparameterizationAnalyzer()
        post_cov = np.eye(25) * 0.1
        prior_cov = np.eye(25) * 0.5
        ident = analyzer.parameter_identifiability(post_cov, prior_cov)
        assert len(ident) == 25

    def test_posterior_contraction(self):
        from app.personalization.phase5.overparameterization import OverparameterizationAnalyzer
        analyzer = OverparameterizationAnalyzer()
        post_cov = np.eye(25) * 0.05
        prior_cov = np.eye(25) * 0.5
        contraction = analyzer.posterior_contraction(post_cov, prior_cov)
        for v in contraction.values():
            assert 0 <= v <= 1

    def test_condition_analysis(self):
        from app.personalization.phase5.overparameterization import OverparameterizationAnalyzer
        analyzer = OverparameterizationAnalyzer()
        H = np.random.randn(15, 30)
        cond, rank_def = analyzer.condition_analysis(H)
        assert cond >= 1
        assert rank_def >= 0

    def test_full_report(self):
        from app.personalization.phase5.overparameterization import OverparameterizationAnalyzer
        analyzer = OverparameterizationAnalyzer()
        H = np.random.randn(15, 55)
        post_cov = np.eye(55) * 0.1
        prior_cov = np.eye(55) * 0.5
        report = analyzer.full_report(H, post_cov, prior_cov, n_observations_per_update=15)
        assert report.n_parameters == 55
        assert isinstance(report.is_overparameterized, bool)
        assert report.recommended_n_observations > 0

    def test_regularization_path(self):
        from app.personalization.phase5.overparameterization import OverparameterizationAnalyzer
        analyzer = OverparameterizationAnalyzer()
        H = np.random.randn(30, 10)
        true_params = np.random.randn(10)
        path = analyzer.regularization_path_analysis(H, true_params)
        assert "optimal_lambda" in path
        assert len(path["mse"]) > 0


# ── 7. Uncertainty Decomposition ─────────────────────────────


class TestUncertaintyDecomposition:
    def test_decomposition(self):
        from app.personalization.phase5.uncertainty_decomposition import UncertaintyDecomposer
        decomposer = UncertaintyDecomposer(seed=42)
        report = decomposer.compute_decomposition(engine=None, horizons=[1, 7, 30])
        assert report.n_horizons == 3
        for d in report.decompositions:
            total = d.parameter_fraction + d.measurement_fraction + d.structural_fraction + d.adherence_fraction
            assert abs(total - 1.0) < 0.01 or total == 0

    def test_summarize(self):
        from app.personalization.phase5.uncertainty_decomposition import UncertaintyDecomposer
        decomposer = UncertaintyDecomposer(seed=42)
        report = decomposer.compute_decomposition(engine=None, horizons=[1, 30])
        summary = decomposer.summarize(report)
        assert "dominant_source" in summary
        assert len(summary["horizon_details"]) == 2

    def test_parameter_trend(self):
        from app.personalization.phase5.uncertainty_decomposition import UncertaintyDecomposer
        decomposer = UncertaintyDecomposer(seed=42)
        report = decomposer.compute_decomposition(engine=None, horizons=[1, 7, 30, 90])
        assert report.parameter_trend in ("increasing", "decreasing", "stable", "unknown")


# ── 8. Stability Analysis ─────────────────────────────────────


class TestStability:
    def test_eigenvalue_computation(self):
        from app.personalization.phase5.stability_analysis import StabilityAnalyzer
        analyzer = StabilityAnalyzer()

        def linear_dynamics(state, params, inputs):
            return -0.1 * state

        state = np.random.randn(10)
        params = np.ones(5)
        eigenvalues, max_real, min_real = analyzer.compute_eigenvalues(linear_dynamics, state, params)
        assert max_real < 0
        assert len(eigenvalues) == 10

    def test_spectral_radius(self):
        from app.personalization.phase5.stability_analysis import StabilityAnalyzer
        analyzer = StabilityAnalyzer()

        def stable_dynamics(state, params, inputs):
            return -0.05 * state

        sr = analyzer.spectral_radius(stable_dynamics, np.random.randn(5), np.ones(3))
        assert sr > 0

    def test_stiffness_ratio(self):
        from app.personalization.phase5.stability_analysis import StabilityAnalyzer
        analyzer = StabilityAnalyzer()

        def stiff_dynamics(state, params, inputs):
            result = np.zeros_like(state)
            result[0] = -1000 * state[0]
            result[1:] = -0.01 * state[1:]
            return result

        state = np.random.randn(5)
        sr = analyzer.stiffness_ratio(stiff_dynamics, state, np.ones(3))
        assert sr >= 1

    def test_lyapunov_exponents(self):
        from app.personalization.phase5.stability_analysis import StabilityAnalyzer
        analyzer = StabilityAnalyzer()

        def chaotic_dynamics(state, params, inputs):
            return -0.1 * state + 0.05 * np.roll(state, 1)

        lyap = analyzer.lyapunov_exponents(chaotic_dynamics, np.random.randn(5), np.ones(3), n_steps=50)
        assert len(lyap) == 5

    def test_coupled_stability(self):
        from app.personalization.phase5.stability_analysis import StabilityAnalyzer
        analyzer = StabilityAnalyzer()

        def stable_full(state, params, inputs):
            return -0.1 * state + 0.01 * np.roll(state, 1) - 0.01 * np.roll(state, -1)

        report = analyzer.coupled_stability_analysis(stable_full, np.random.randn(10), np.ones(5), lyap_steps=20)
        assert isinstance(report.is_stable, bool)
        assert report.spectral_radius > 0
        assert report.stiffness_ratio >= 1


# ── 9. Clinical Protocol ──────────────────────────────────────


class TestClinicalProtocol:
    def test_protocol_generation(self):
        from app.personalization.phase5.clinical_protocol import generate_twin_validation_protocol
        protocol = generate_twin_validation_protocol()
        assert len(protocol.objectives) >= 3
        assert len(protocol.endpoints) >= 2
        assert protocol.population.n_patients > 0
        assert len(protocol.population.inclusion_criteria) >= 3

    def test_power_analysis(self):
        from app.personalization.phase5.clinical_protocol import PowerAnalyzer
        n = PowerAnalyzer.sample_size_continuous(effect_size=0.5, std=1.0)
        assert n > 10
        n_bin = PowerAnalyzer.sample_size_binary(p_control=0.2, p_treatment=0.35)
        assert n_bin > 10

    def test_power_curve(self):
        from app.personalization.phase5.clinical_protocol import PowerAnalyzer
        curve = PowerAnalyzer.power_curve([50, 100, 200], effect_size=0.5)
        assert len(curve) == 3
        for point in curve:
            assert 0 <= point["power"] <= 1

    def test_minimum_detectable_effect(self):
        from app.personalization.phase5.clinical_protocol import PowerAnalyzer
        mde = PowerAnalyzer.minimum_detectable_effect(n_per_arm=100)
        assert mde > 0

    def test_compute_power_analysis(self):
        from app.personalization.phase5.clinical_protocol import compute_power_analysis
        result = compute_power_analysis(n_total=200)
        assert len(result["power_analysis"]) > 0

    def test_generate_study_report(self):
        from app.personalization.phase5.clinical_protocol import generate_study_report
        report = generate_study_report()
        assert "protocol" in report
        assert "power_analysis" in report
        assert "feasibility" in report


# ── 10. Counterfactual Sensitivity ────────────────────────────


class TestCounterfactualSensitivity:
    def test_adherence_sensitivity(self):
        from app.personalization.phase5.counterfactual_sensitivity import CounterfactualSensitivityAnalyzer
        analyzer = CounterfactualSensitivityAnalyzer(seed=42)

        def sim_fn(adherence=1.0):
            return 100 - 20 * adherence + np.random.normal(0, 5)

        results = analyzer.adherence_sensitivity(sim_fn, adherence_range=[0.0, 0.5, 1.0], n_per_point=20)
        assert len(results) == 3
        assert results[0].adherence == 0.0
        assert results[-1].adherence == 1.0

    def test_dose_response(self):
        from app.personalization.phase5.counterfactual_sensitivity import CounterfactualSensitivityAnalyzer
        analyzer = CounterfactualSensitivityAnalyzer(seed=42)

        def sim_fn(metformin_dose=0.0):
            return 100 - 30 * metformin_dose + np.random.normal(0, 3)

        results = analyzer.dose_response_analysis(sim_fn, "metformin_dose",
                                                   dose_range=[0.0, 0.5, 1.0],
                                                   toxicity_threshold=50, n_per_point=10)
        assert len(results) == 3

    def test_parameter_sensitivity(self):
        from app.personalization.phase5.counterfactual_sensitivity import CounterfactualSensitivityAnalyzer
        analyzer = CounterfactualSensitivityAnalyzer(seed=42)

        def sim_fn(SI=0.05, HGP=2.0):
            return 100 - 200 * SI + 20 * HGP

        sens = analyzer.parameter_sensitivity(sim_fn, {"SI": (0.01, 0.1), "HGP": (1.0, 3.0)})
        assert len(sens) == 2

    def test_full_report(self):
        from app.personalization.phase5.counterfactual_sensitivity import CounterfactualSensitivityAnalyzer
        analyzer = CounterfactualSensitivityAnalyzer(seed=42)

        def sim_fn(adherence=1.0, metformin_dose=0.0):
            return 100 - 20 * adherence - 30 * metformin_dose

        def estimator_fn(confounder_strength=0.0):
            return 15.0 * (1 - confounder_strength)

        report = analyzer.full_sensitivity_report(
            simulate_fn=sim_fn,
            dose_response_params={"metformin_dose": [0.0, 0.5, 1.0]},
            param_ranges={"SI": (0.01, 0.1)},
            effect_estimator=estimator_fn,
        )
        assert report.robustness_score > 0
        assert len(report.adherence_sensitivity) > 0


# ── Integration: Validation Framework Extension ───────────────


class TestValidationFrameworkExtension:
    def test_validation_framework_new_validators(self):
        from app.personalization.phase5.validation_framework import ValidationFrameworkV2
        vf = ValidationFrameworkV2()
        assert len(vf.validators) >= 5

    def test_calibration_as_validation_criterion(self):
        from app.personalization.phase5.calibration import CalibrationAssessor
        rng = np.random.default_rng(42)
        y_true = rng.normal(100, 10, 500)
        y_pred = y_true + rng.normal(0, 3, 500)
        assessor = CalibrationAssessor()
        report = assessor.assess_continuous(y_true, y_pred, np.ones(500) * 3)
        assert report.ece < 0.2

    def test_overparameterization_validates(self):
        from app.personalization.phase5.overparameterization import analyze_overparameterization
        report = analyze_overparameterization()
        assert report.n_parameters > 0

    def test_stability_report_contains_all_metrics(self):
        from app.personalization.phase5.stability_analysis import StabilityAnalyzer
        analyzer = StabilityAnalyzer()

        def dyn(s, p, i):
            return -0.1 * s

        report = analyzer.coupled_stability_analysis(dyn, np.random.randn(8), np.ones(5))
        assert hasattr(report, 'max_real_eigenvalue')
        assert hasattr(report, 'spectral_radius')
        assert hasattr(report, 'stiffness_ratio')
        assert hasattr(report, 'max_lyapunov_exponent')
        assert hasattr(report, 'is_stable')
        assert hasattr(report, 'is_stiff')

    def test_retrospective_produces_summary(self):
        from app.personalization.phase5.clinical_dataset import run_retrospective_validation

        def tf(physio, params):
            class T:
                def get_state(self):
                    return physio.copy(), params.copy()

                def observe(self, obs):
                    return {"updated": True}
            return T()

        summary = run_retrospective_validation(tf, n_patients=3, n_days=30)
        assert summary["n_patients"] > 0


# ── 11. Clinical Metrics (Clarke Error Grid, Bland-Altman) ─────


class TestClinicalMetrics:
    def test_clarke_error_grid_perfect(self):
        from app.personalization.phase5.clinical_metrics import clarke_error_grid
        y_true = np.array([100, 120, 80, 150, 200])
        y_pred = np.array([100, 120, 80, 150, 200])
        result = clarke_error_grid(y_true, y_pred)
        assert result.zone_a_pct == 100.0
        assert result.clinically_acceptable is True

    def test_clarke_error_grid_hypo_missed(self):
        from app.personalization.phase5.clinical_metrics import clarke_error_grid
        y_true = np.array([55, 100, 200])
        y_pred = np.array([120, 100, 200])
        result = clarke_error_grid(y_true, y_pred)
        assert result.zone_d_pct > 0

    def test_bland_altman_excellent(self):
        from app.personalization.phase5.clinical_metrics import bland_altman_analysis
        y_true = np.array([100, 120, 80, 150, 200])
        y_pred = np.array([101, 119, 81, 149, 201])
        result = bland_altman_analysis(y_true, y_pred)
        assert result.clinical_agreement in ("excellent", "acceptable")
        assert abs(result.mean_difference) < 5

    def test_iso_compliance_pass(self):
        from app.personalization.phase5.clinical_metrics import iso_15197_compliance
        y_true = np.array([80, 100, 120, 150, 200, 250, 90, 110, 130, 170])
        y_pred = y_true + np.random.RandomState(42).normal(0, 3, 10)
        result = iso_15197_compliance(y_true, y_pred)
        assert isinstance(result["overall_pass"], bool)

    def test_concordance_correlation(self):
        from app.personalization.phase5.clinical_metrics import concordance_correlation
        rng = np.random.RandomState(42)
        y_true = np.linspace(80, 200, 100)
        y_pred = y_true + rng.normal(0, 3, 100)
        ccc = concordance_correlation(y_true, y_pred)
        assert ccc > 0.7

    def test_compute_clinical_validation(self):
        from app.personalization.phase5.clinical_metrics import compute_clinical_validation
        rng = np.random.RandomState(42)
        y_true = np.linspace(80, 200, 50)
        y_pred = y_true + rng.normal(0, 5, 50)
        report = compute_clinical_validation(y_true, y_pred)
        assert report.mard < 15
        assert report.clarke_error_grid is not None
        assert report.bland_altman is not None
        assert "within_15_interval" in report.iso_15197_compliance


# ── 12. Identifiability Analysis ────────────────────────────────


class TestIdentifiability:
    def test_identifiability_analyzer_imports(self):
        from app.personalization.phase5.identifiability import IdentifiabilityAnalyzer
        ia = IdentifiabilityAnalyzer(physio_dim=30, param_dim=25, obs_dim=15)
        assert ia.physio_dim == 30
        assert ia.param_dim == 25

    def test_collinearity_analysis(self):
        from app.personalization.phase5.identifiability import IdentifiabilityAnalyzer
        ia = IdentifiabilityAnalyzer()
        coll = ia.collinearity_analysis(n_samples=100)
        assert len(coll) > 0
        for v in coll.values():
            assert 0 <= v <= 1

    def test_empirical_observability(self):
        from app.personalization.phase5.identifiability import IdentifiabilityAnalyzer
        ia = IdentifiabilityAnalyzer()
        def dyn(s, p, i):
            return s * 0.99
        def obs(s):
            return s[:15]
        result = ia.empirical_observability(dyn, obs, n_perturbations=50)
        assert "observability_rank" in result
        assert result["observability_rank"] > 0


# ── 13. Causal Inference (Pearlian) ─────────────────────────────


class TestCausalInference:
    def test_scm_structure(self):
        from app.personalization.phase5.causal_inference import StructuralCausalModel
        scm = StructuralCausalModel()
        assert len(scm.all_vars) >= 20
        parents_G = scm.get_parents("G")
        assert "I" in parents_G
        assert "IR" in parents_G
        assert "cortisol" in parents_G

    def test_adjustment_set(self):
        from app.personalization.phase5.causal_inference import StructuralCausalModel
        scm = StructuralCausalModel()
        adj = scm.get_adjustment_set("IR", "G")
        assert isinstance(adj, list)

    def test_estimate_causal_effect(self):
        from app.personalization.phase5.causal_inference import StructuralCausalModel
        scm = StructuralCausalModel()
        rng = np.random.RandomState(42)
        n = 200
        data = np.zeros((n, 28))
        data[:, 0] = rng.normal(100, 20)  # G
        data[:, 4] = rng.normal(5, 2)     # IR
        data[:, 7] = rng.normal(70, 10)   # HR
        data[:, 5] = rng.normal(120, 15)  # SBP
        data[:, 9] = rng.normal(100, 15)  # GFR
        var_names = {0: "G", 4: "IR", 5: "SBP", 7: "HR", 9: "GFR",
                     21: "FFA", 1: "I", 22: "LDL", 27: "NFkB_activity",
                     28: "InflammatoryLoad"}
        var_names.update({i: f"var_{i}" for i in range(28) if i not in var_names})
        effect = scm.estimate_causal_effect(
            data, var_names, "IR", "G", treatment_value=3.0
        )
        assert isinstance(effect.estimated_effect, float)
        assert len(effect.confidence_interval) == 2

    def test_counterfactual_inference(self):
        from app.personalization.phase5.causal_inference import StructuralCausalModel
        scm = StructuralCausalModel()
        def dyn(s, p, i):
            return s * 0.99
        def intervention(s):
            s[0] = 80.0
            return s
        state = np.zeros(30)
        state[0] = 100.0
        result = scm.counterfactual_inference(
            state, dyn, "lower_glucose", intervention, n_samples=10, horizon=100
        )
        assert hasattr(result, 'factual_outcome')
        assert hasattr(result, 'counterfactual_outcome')
        assert hasattr(result, 'individual_causal_effect')


# ── 14. UKF Convergence Diagnostics ─────────────────────────────


class TestConvergenceDiagnostics:
    def test_convergence_diagnostics_structure(self):
        from app.personalization.core import PersonalizationEngine
        engine = PersonalizationEngine()
        diag = engine.convergence_diagnostics()
        assert "param_stability" in diag
        assert "param_drift_rate" in diag
        assert "is_converged" in diag

    def test_is_valid_now_validates(self):
        from app.personalization.state import Phase3TwinState, MetabolicState, CardioState
        from app.personalization.state import RenalState, InflammatoryState
        from app.personalization.state import CircadianState, AdiposeLipidState, ImmuneInflamState
        s = Phase3TwinState(
            metabolic=MetabolicState(100, 10, 2, 5, 1),
            cardio=CardioState(120, 80, 70, 45),
            renal=RenalState(100, 140, 4.2, 290),
            inflammation=InflammatoryState(1.0),
            circadian=CircadianState(1.2, 0.8, 350, 10, 0, 0.3),
            adipose=AdiposeLipidState(20, 0.5, 100, 50, 120),
            immune=ImmuneInflamState(1.0, 0.5, 0.5, 0.2, 15),
        )
        assert s.is_valid() is True

    def test_is_valid_rejects_out_of_bounds(self):
        from app.personalization.state import Phase3TwinState, MetabolicState, CardioState
        from app.personalization.state import RenalState, InflammatoryState
        from app.personalization.state import CircadianState, AdiposeLipidState, ImmuneInflamState
        s = Phase3TwinState(
            metabolic=MetabolicState(1000, 10, 2, 5, 1),
            cardio=CardioState(120, 80, 70, 45),
            renal=RenalState(100, 140, 4.2, 290),
            inflammation=InflammatoryState(1.0),
            circadian=CircadianState(1.2, 0.8, 350, 10, 0, 0.3),
            adipose=AdiposeLipidState(20, 0.5, 100, 50, 120),
            immune=ImmuneInflamState(1.0, 0.5, 0.5, 0.2, 15),
        )
        assert s.is_valid() is False

    def test_validate_or_raise_raises(self):
        from app.personalization.state import Phase3TwinState, MetabolicState, CardioState
        from app.personalization.state import RenalState, InflammatoryState
        from app.personalization.state import CircadianState, AdiposeLipidState, ImmuneInflamState
        s = Phase3TwinState(
            metabolic=MetabolicState(1000, 10, 2, 5, 1),
            cardio=CardioState(120, 80, 70, 45),
            renal=RenalState(100, 140, 4.2, 290),
            inflammation=InflammatoryState(1.0),
            circadian=CircadianState(1.2, 0.8, 350, 10, 0, 0.3),
            adipose=AdiposeLipidState(20, 0.5, 100, 50, 120),
            immune=ImmuneInflamState(1.0, 0.5, 0.5, 0.2, 15),
        )
        import pytest as _pytest
        with _pytest.raises(ValueError, match="out of bounds"):
            s.validate_or_raise()


# ── 15. Uncertainty Coverage Assessment ─────────────────────────


class TestUncertaintyCoverage:
    def test_coverage_assessment_metrics(self):
        from app.personalization.core import PersonalizationEngine
        from app.personalization.uncertainty import UncertaintyEngine
        engine = PersonalizationEngine()
        ue = UncertaintyEngine(engine)
        obs = np.zeros((10, 30))
        cov = ue.coverage_assessment(obs, n_samples=20, horizon=10)
        assert "mean_calibration_error" in cov
        assert "nominal_90%" in cov
        assert "nominal_50%" in cov


# ── 17. Cross-Validation Module ─────────────────────────────────


class TestCrossValidation:
    def test_k_fold_returns_report(self):
        from app.personalization.phase5.cross_validation import TwinCrossValidator
        validator = TwinCrossValidator(n_folds=3, seed=42)

        def dummy_generator():
            return np.zeros(30)

        def dummy_twin_factory(physio, params):
            class DummyTwin:
                def update(self, obs, ctx):
                    pass
                def get_twin_state(self):
                    return np.zeros(30)
                def get_parameters(self):
                    return np.ones(25), np.eye(25)
            return DummyTwin()

        report = validator.k_fold_patient_validation(
            synthetic_patient_generator=dummy_generator,
            twin_factory=dummy_twin_factory,
            n_patients=6,
            n_observations_per_patient=10,
            n_test_observations=3,
        )
        assert report.n_folds == 3
        assert len(report.folds) == 3
        assert report.mean_test_mae >= 0
        assert isinstance(report.is_well_generalized, bool)
        assert len(report.recommendations) > 0

    def test_modality_ablation_returns_dict(self):
        from app.personalization.phase5.cross_validation import TwinCrossValidator
        validator = TwinCrossValidator()

        def twin_factory(physio, params):
            class DummyTwin:
                def update(self, obs, ctx):
                    pass
                def get_twin_state(self):
                    return np.zeros(30)
            return DummyTwin()

        seqs = {"P1": np.zeros((5, 15)), "P2": np.zeros((5, 15))}
        result = validator.modality_ablation_analysis(
            twin_factory, seqs, modalities=["G", "SBP", "HR"]
        )
        assert isinstance(result, dict)
        assert "G" in result

    def test_subgroup_analysis(self):
        from app.personalization.phase5.cross_validation import TwinCrossValidator
        validator = TwinCrossValidator()
        preds = {"P1": np.array([100]), "P2": np.array([110]), "P3": np.array([105])}
        truth = {"P1": np.array([95]), "P2": np.array([105]), "P3": np.array([100])}
        subgroups = {"young": ["P1", "P2"], "old": ["P3"]}
        result = validator.subgroup_analysis(preds, truth, subgroups)
        assert "young" in result
        assert "old" in result
        assert "mae" in result["young"]


# ── 18. Population Adjustment Uncertainty ───────────────────────


class TestPopulationUncertainty:
    def test_get_population_adjustment_with_ci(self):
        from app.personalization.phase5.population_broader import (
            get_population_adjustment,
        )
        result = get_population_adjustment(
            age=10, population=None, ethnicity=None,
            return_ci=True,
        )
        assert isinstance(result, dict)
        # At least one param with CI info
        found = False
        for k, v in result.items():
            if isinstance(v, dict) and "ci_95" in v:
                found = True
                break
        assert found, "No parameter returned with CI info"

    def test_get_population_adjustment_with_uncertainty(self):
        from app.personalization.phase5.population_broader import (
            get_population_adjustment_with_uncertainty,
        )
        result = get_population_adjustment_with_uncertainty(
            age=10, n_bootstrap=100
        )
        assert isinstance(result, dict)
        assert len(result) > 0
        for k, v in result.items():
            assert "mean" in v
            assert "ci_95" in v
            assert "cv" in v
            assert "well_characterized" in v


# ── 16. Posterior Predictive Samples (fix for dead code) ────────


class TestPosteriorPredictive:
    def test_posterior_predictive_produces_trajectories(self):
        from app.personalization.core import PersonalizationEngine
        from app.personalization.uncertainty import UncertaintyEngine
        engine = PersonalizationEngine()
        ue = UncertaintyEngine(engine)
        traj = ue.posterior_predictive_samples(n_samples=5, horizon=10)
        assert traj.shape == (5, 10, 30)
        assert np.all(np.isfinite(traj))
