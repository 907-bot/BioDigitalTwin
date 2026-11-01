"""
Tests for Phase 17 Personal Digital Twin Enhancements:
1. Real Personal Metrics Intake (/personalization/v2/seed-real-twin)
2. Musculoskeletal 36-State Model & Parameters
3. Workout & Progressive Overload Simulator (/personalization/v2/workout/simulate)
4. OTC Cold/Fever Drug Registry & Mismatch Simulator (/phase10/otc-simulation)
5. Cellular & Sub-Organ Level Simulator (/personalization/v2/cellular/simulate)
6. Samsung Health Watch Stream & Twin Sync (/phase17/samsung-health/sync-to-twin)
"""

import pytest
import numpy as np
from app.personalization.priors import STATE_NAMES, PARAMETER_NAMES
from app.personalization.state import PHYSIO_DIM, PARAM_DIM, Phase3TwinState
from app.personalization.core import create_personalization_engine
from app.dynamics.workout_simulator import simulate_workout_trajectory
from app.pkpd.otc_simulator import simulate_otc_mismatch
from app.personalization.cellular_simulator import simulate_cellular_suborgan
from app.pkpd.registry import get_drug


def test_musculoskeletal_state_dimensions():
    assert PHYSIO_DIM == 36
    assert PARAM_DIM == 29
    assert len(STATE_NAMES) == 36
    assert "muscle_mass" in STATE_NAMES
    assert "bone_mineral_density" in STATE_NAMES
    assert "mTOR_activation_rate" in PARAMETER_NAMES


def test_seed_from_personal_metrics():
    eng = create_personalization_engine(age=28, sex="male", bmi=24.0)
    state_arr = eng.seed_from_personal_metrics(
        height_cm=178,
        weight_kg=76,
        resting_hr=64,
        resting_sbp=118,
        resting_dbp=76,
        body_fat_pct=15.5,
        skeletal_muscle_mass_kg=35.2,
        bone_mineral_density_g_cm2=1.25,
        vo2_max=48.0,
        fasting_glucose=90.0,
        crp=0.8,
        serum_creatinine=0.88,
    )
    assert len(state_arr) == 36
    assert state_arr[7] == 64.0  # HR
    assert state_arr[30] == 35.2 # muscle_mass
    assert state_arr[33] == 1.25 # BMD


def test_workout_simulation():
    res = simulate_workout_trajectory(
        initial_muscle_mass=35.2,
        initial_strength=50.0,
        initial_bmd=1.25,
        initial_resting_hr=64.0,
        workout_type="hypertrophy",
        intensity_pct=80.0,
        duration_mins=60.0,
        sets=4,
        reps=10,
        frequency_per_week=4,
        protein_intake_g_day=160.0,
        horizon_days=30,
    )
    assert res["summary"]["muscle_mass_delta_kg"] > 0
    assert res["summary"]["strength_delta"] > 0
    assert len(res["trajectory"]) == 30


def test_otc_drug_registry_and_mismatch():
    paracetamol = get_drug("paracetamol")
    assert paracetamol.drug_class == "otc_antipyretic"
    assert paracetamol.typical_dose_mg == 500

    # Mis-match: Paracetamol given for cold without fever
    res1 = simulate_otc_mismatch(
        patient_condition="viral_cold_no_fever",
        administered_drug="paracetamol",
        dose_mg=500,
    )
    assert any("MISMATCH" in w for w in res1["warnings"])

    # Overdose check
    res2 = simulate_otc_mismatch(
        patient_condition="fever_39C",
        administered_drug="paracetamol",
        dose_mg=2000,
        repeat_doses_per_day=3, # 6000mg/day
    )
    assert any("CRITICAL" in w for w in res2["warnings"])


def test_cellular_simulation():
    res = simulate_cellular_suborgan(
        simulation_type="fever_thermoregulation",
        pyrogen_load=1.2,
        drug_dose_mg=1000,
        drug_name="paracetamol",
        duration_hours=24,
    )
    assert "summary" in res
    assert res["summary"]["peak_core_temp_C"] > 37.0
    assert len(res["trajectory"]) == 24
