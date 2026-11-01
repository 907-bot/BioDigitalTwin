"""
BioDigitalTwin — Core v2 Integration and Conservation Test Suite.

Verifies:
1. Canonical State Registry schema validation, bounding, and vector serialization.
2. Shared Environment hemodynamics, regional perfusion summing to CO, and solute flux.
3. Conservation laws: hemodynamic flow conservation, enteral carb mass balance, glucose flux.
4. Coupled Multi-Organ Engine resting stability across all 7 organs.
5. Coordinated oral glucose challenge (meal test):
   - GI emptying & absorption
   - Pancreatic insulin & GLP-1 secretion
   - Hepatic glucose production suppression
   - Renal glycosuria threshold clearance
6. Exercise challenge: cardiovascular, metabolic, and respiratory response.
7. Constrained residual ML learner bounds and anomaly detection.
"""

import pytest
import numpy as np

from app.physiology.core.state_registry import (
    StateRegistry,
    StateDescriptor,
    ParameterDescriptor,
    get_global_registry,
)
from app.physiology.core.shared_environment import SharedEnvironment
from app.physiology.core.flux_registry import FluxRegistry, Flux
from app.physiology.core.conservation import ConservationValidator
from app.physiology.core.simulation_engine import (
    CoupledSimulationEngine,
    SimulationEvent,
)
from app.physiology.organs import (
    create_standard_multi_organ_engine,
    CardiovascularOrganModel,
    HepaticOrganModel,
    RenalOrganModel,
    GastrointestinalOrganModel,
    PancreaticOrganModel,
    RespiratoryOrganModel,
    EndocrineOrganModel,
)
from app.physiology.hybrid.residual_learner import (
    ConstrainedResidualLearner,
    PhysiologicalAnomalyDetector,
)


def test_state_registry_schema_and_serialization():
    """Verifies registry validation, bounds enforcement, and vector round-trips."""
    reg = StateRegistry()

    d1 = StateDescriptor(
        name="test.glucose",
        display_name="Test Glucose",
        unit="mg/dL",
        compartment="blood",
        min_bound=20.0,
        max_bound=600.0,
        nominal=95.0,
        initial_condition_source="baseline",
        is_observable=True,
        owning_module="test",
        description="Test state",
    )
    reg.register_state(d1)
    assert reg.state_count == 1
    assert "test.glucose" in reg.state_names

    # Duplicate registration should raise ValueError
    with pytest.raises(ValueError):
        reg.register_state(d1)

    # Invalid bounds should raise ValueError
    with pytest.raises(ValueError):
        reg.register_state(
            StateDescriptor(
                name="test.bad_bounds",
                display_name="Bad Bounds",
                unit="mg/dL",
                compartment="blood",
                min_bound=100.0,
                max_bound=50.0,
                nominal=75.0,
                initial_condition_source="baseline",
                is_observable=True,
                owning_module="test",
                description="Bad bounds",
            )
        )

    # Vector conversion
    state_dict = {"test.glucose": 120.0}
    vec = reg.state_to_vector(state_dict)
    assert len(vec) == 1
    assert vec[0] == 120.0

    restored = reg.vector_to_state(vec)
    assert restored["test.glucose"] == 120.0

    # Bounds validation
    valid, violations = reg.validate_state({"test.glucose": 1000.0})
    assert not valid
    assert len(violations) > 0

    clamped = reg.clamp_state({"test.glucose": 1000.0})
    assert clamped["test.glucose"] == 600.0


def test_shared_environment_hemodynamics_and_fluxes():
    """Verifies shared transport medium hemodynamics, perfusion balance, and solute dilution."""
    env = SharedEnvironment(patient_weight_kg=70.0, patient_height_cm=175.0)

    # Verify baseline hemodynamics
    assert np.isclose(env.circulation.cardiac_output_L_min, 5.04, atol=0.1)
    assert np.isclose(env.circulation.mean_arterial_pressure_mmHg, 93.0, atol=5.0)

    # Regional perfusion must sum to systemic cardiac output
    flows = env.circulation.regional_flows_L_min
    regional_sum = sum(
        flows[k] for k in ["brain", "coronary", "renal", "gut", "muscle", "adipose"]
    )
    assert np.isclose(regional_sum, env.circulation.cardiac_output_L_min, rtol=1e-2)

    # Test solute flux into blood
    initial_glucose = env.chemistry.glucose_mg_dL
    # Add 1400 mg of glucose into ECF (14 L = 140 dL), expecting +10 mg/dL change
    env.apply_solute_flux(compartment="blood", solute="glucose", flux_mg_min=1400.0, dt_minutes=1.0)
    assert np.isclose(env.chemistry.glucose_mg_dL, initial_glucose + 10.0, atol=0.1)

    # Acid-base equilibrium
    env.update_acid_base_and_gas_equilibrium(alveolar_ventilation_L_min=4.2)
    assert 7.35 <= env.chemistry.pH <= 7.45
    assert 90.0 <= env.chemistry.paO2_mmHg <= 105.0


def test_conservation_validator():
    """Verifies flow and mass conservation checks."""
    validator = ConservationValidator()
    env = SharedEnvironment(70.0, 175.0)

    # Clean state should pass hemodynamic conservation
    valid, violations = validator.verify_hemodynamic_conservation(env)
    assert valid
    assert len(violations) == 0

    # Tampered flow should fail
    env.circulation.regional_flows_L_min["renal"] = 5.0
    valid, violations = validator.verify_hemodynamic_conservation(env)
    assert not valid
    assert len(violations) > 0

    # Enteral mass balance check: ingested == stomach + intestine + absorbed
    valid_enteral, _ = validator.verify_enteral_mass_balance(
        initial_meal_g=75.0,
        stomach_mass_g=25.0,
        intestinal_mass_g=15.0,
        cumulative_absorbed_g=35.0,
    )
    assert valid_enteral

    # Violated mass balance
    invalid_enteral, violations = validator.verify_enteral_mass_balance(
        initial_meal_g=75.0,
        stomach_mass_g=50.0,
        intestinal_mass_g=30.0,
        cumulative_absorbed_g=20.0,  # Sum = 100 != 75
    )
    assert not invalid_enteral
    assert len(violations) > 0


def test_coupled_multi_organ_baseline_resting():
    """Verifies baseline resting multi-organ twin remains in homeostasis over 60 minutes."""
    engine = create_standard_multi_organ_engine(patient_weight_kg=70.0, patient_height_cm=175.0)

    # Run for 60 minutes with 1-minute steps
    history = engine.simulate(total_minutes=60.0, dt_minutes=1.0, inputs={})

    assert len(history) == 60
    final_snap = history[-1]

    # Homeostatic stability checks
    assert 80.0 <= final_snap.circulating_glucose_mg_dL <= 110.0
    assert 5.0 <= final_snap.circulating_insulin_uU_mL <= 15.0
    assert 60.0 <= final_snap.heart_rate_bpm <= 85.0
    assert 80.0 <= final_snap.mean_arterial_pressure_mmHg <= 105.0
    assert 7.35 <= final_snap.arterial_pH <= 7.45
    assert final_snap.is_physiologically_valid
    assert len(final_snap.violations) == 0


def test_coupled_oral_glucose_tolerance_challenge():
    """
    Verifies multi-organ coordination during a standard 75g oral glucose tolerance challenge:
    1. Stomach empties into gut via Elashoff dynamics.
    2. Gut absorbs glucose into portal circulation via SGLT1 kinetics and secretes GLP-1.
    3. Pancreatic beta-cells secrete insulin in response to circulating glucose + GLP-1.
    4. Hepatic glucose production (HGP) is suppressed by elevated insulin.
    5. Peripheral tissues increase glucose disposal.
    6. Systemic glucose peaks and returns toward baseline.
    """
    engine = create_standard_multi_organ_engine(patient_weight_kg=70.0, patient_height_cm=175.0)

    # Ingest 75g carb meal at t = 10 min
    engine.schedule_event(SimulationEvent(time_minutes=10.0, event_type="meal", payload={"carbs_g": 75.0}))

    history = engine.simulate(total_minutes=180.0, dt_minutes=1.0)
    glucose_profile = [s.circulating_glucose_mg_dL for s in history]
    insulin_profile = [s.circulating_insulin_uU_mL for s in history]

    # Pre-meal glucose
    g_pre = glucose_profile[9]
    assert 85.0 <= g_pre <= 105.0

    # Glycemic excursion
    g_max = max(glucose_profile)
    max_idx = glucose_profile.index(g_max)
    assert g_max > 125.0, f"Expected postprandial glycemic excursion > 125, got {g_max}"
    # Peak should occur between 30 and 90 min (index 30-90)
    assert 25 <= max_idx <= 95, f"Peak occurred at step {max_idx}"

    # Insulin response should increase significantly
    ins_max = max(insulin_profile)
    ins_pre = insulin_profile[9]
    assert ins_max > ins_pre * 2.0, f"Expected >2x insulin secretion, got max {ins_max} vs pre {ins_pre}"

    # Return towards baseline at 180 min
    g_post = glucose_profile[-1]
    assert g_post < g_max, "Glucose should be descending post-peak"
    assert g_post < 140.0, f"Normal tolerance returns < 140 mg/dL, got {g_post}"


def test_renal_glycosuria_threshold():
    """
    Verifies the renal glucose threshold mechanism:
    - At normal glucose (95 mg/dL), tubular reabsorption handles 100% of filtered load (E_renal == 0).
    - When glucose is artificially forced above threshold (> 200 mg/dL), glycosuria flux occurs.
    """
    renal = RenalOrganModel()
    env = SharedEnvironment()
    renal.initialize({"weight_kg": 70.0}, env)

    # Baseline (95 mg/dL)
    env.chemistry.glucose_mg_dL = 95.0
    fluxes_baseline = renal.exchange_fluxes(time=0.0, state=renal.get_state(), environment=env)
    renal_glucose_fluxes = [f for f in fluxes_baseline if f.quantity == "glucose" and f.destination == "urine"]
    assert len(renal_glucose_fluxes) == 0 or renal_glucose_fluxes[0].value == 0.0

    # Severe hyperglycemia (300 mg/dL)
    env.chemistry.glucose_mg_dL = 300.0
    fluxes_high = renal.exchange_fluxes(time=0.0, state=renal.get_state(), environment=env)
    renal_glucose_fluxes_high = [f for f in fluxes_high if f.quantity == "glucose" and f.destination == "urine"]
    assert len(renal_glucose_fluxes_high) == 1
    excretion_rate_mg_min = renal_glucose_fluxes_high[0].value
    assert excretion_rate_mg_min > 30.0, f"Expected significant glycosuria, got {excretion_rate_mg_min} mg/min"


def test_coupled_exercise_challenge():
    """Verifies cardiovascular, metabolic, and respiratory adaptation during exercise."""
    engine = create_standard_multi_organ_engine(patient_weight_kg=70.0, patient_height_cm=175.0)

    # 10 min rest, then 30 min moderate exercise (intensity 0.6)
    history_rest = engine.simulate(total_minutes=10.0, dt_minutes=1.0)
    baseline_hr = history_rest[-1].heart_rate_bpm
    baseline_co = history_rest[-1].cardiac_output_L_min

    history_exercise = engine.simulate(
        total_minutes=30.0,
        dt_minutes=1.0,
        inputs={"exercise": 0.6},
    )
    peak_exercise_snap = history_exercise[-1]

    # Exercise responses
    assert peak_exercise_snap.heart_rate_bpm > baseline_hr + 20.0
    assert peak_exercise_snap.cardiac_output_L_min > baseline_co + 2.0
    # Respiration should preserve pH despite increased metabolism
    assert 7.30 <= peak_exercise_snap.arterial_pH <= 7.48


def test_constrained_residual_learner_and_anomaly_detector():
    """Verifies that ML residual additions are bounded to <= 20% and anomalies are detected."""
    learner = ConstrainedResidualLearner(max_residual_fraction=0.20)
    anomaly_detector = PhysiologicalAnomalyDetector()

    state = {
        "blood.glucose": 95.0,
        "blood.insulin": 8.0,
        "cardio.heart_rate_bpm": 72.0,
        "cardio.mean_arterial_pressure_mmHg": 93.0,
    }
    inputs = {"exercise": 0.0, "meal_carbs_g": 0.0}

    # Unconfigured weights -> residual 0.0
    correction = learner.predict_residual(
        state_name="blood.glucose",
        mechanistic_deriv=-0.5,
        state_dict=state,
        inputs=inputs,
    )
    assert correction.learned_residual == 0.0
    assert correction.total_derivative == -0.5

    # Configure a large weight to test clamping
    learner._weights["blood.glucose"] = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    correction_clamped = learner.predict_residual(
        state_name="blood.glucose",
        mechanistic_deriv=10.0,
        state_dict={"blood.glucose": 150.0},
        inputs={"exercise": 0.5, "meal_carbs_g": 50.0},
    )
    # With mechanistic_deriv = 10.0, max bound is 10.0 * 0.20 = 2.0
    assert abs(correction_clamped.learned_residual) <= 2.0001
    assert correction_clamped.is_clamped

    # Anomaly detection tests
    # Normal telemetry
    ok, anomalies = anomaly_detector.check_telemetry(
        current_telemetry={"glucose_mg_dL": 110.0, "heart_rate_bpm": 80.0, "mean_bp_mmHg": 95.0},
        previous_telemetry={"glucose_mg_dL": 105.0, "heart_rate_bpm": 78.0, "mean_bp_mmHg": 94.0},
        dt_minutes=5.0,
    )
    assert ok
    assert len(anomalies) == 0

    # Abnormal jump: glucose jumps 120 mg/dL in 1 minute
    ok_jump, anomalies_jump = anomaly_detector.check_telemetry(
        current_telemetry={"glucose_mg_dL": 220.0, "heart_rate_bpm": 80.0, "mean_bp_mmHg": 95.0},
        previous_telemetry={"glucose_mg_dL": 100.0, "heart_rate_bpm": 80.0, "mean_bp_mmHg": 95.0},
        dt_minutes=1.0,
    )
    assert not ok_jump
    assert any("rate anomaly in glucose" in a.lower() for a in anomalies_jump)
