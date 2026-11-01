"""
BioDigitalTwin — Core v2 API Endpoints Test Suite.

Verifies:
1. GET  /physiology/v2/state-registry
2. POST /physiology/v2/simulation/run
3. POST /physiology/v2/challenge/meal
4. POST /physiology/v2/challenge/exercise
5. GET  /physiology/v2/conservation/audit
6. POST /physiology/v2/anomaly/detect
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_v2_state_registry_endpoint():
    resp = client.get("/physiology/v2/state-registry")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["total_states"] > 0
    assert data["total_parameters"] > 0
    assert "blood.glucose" in data["states"]
    glucose_meta = data["states"]["blood.glucose"]
    assert glucose_meta["unit"] == "mg/dL"
    assert glucose_meta["min_bound"] < glucose_meta["max_bound"]


def test_v2_simulation_run_endpoint():
    payload = {
        "patient_id": "test_patient_1",
        "patient_weight_kg": 70.0,
        "patient_height_cm": 175.0,
        "duration_minutes": 15.0,
        "dt_minutes": 1.0,
        "solver": "rk4",
        "inputs": {"exercise": 0.0},
        "events": [
            {
                "time_minutes": 5.0,
                "event_type": "meal",
                "payload": {"carbs_g": 50.0},
            }
        ],
    }
    resp = client.post("/physiology/v2/simulation/run", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["total_steps"] == 15
    assert len(data["trajectory"]) == 15
    final_snap = data["final_snapshot"]
    assert "glucose_mg_dL" in final_snap
    assert "heart_rate_bpm" in final_snap
    assert data["conservation_audit"]["is_conserved"]


def test_v2_meal_challenge_endpoint():
    payload = {
        "patient_id": "ogtt_test_patient",
        "carbs_g": 75.0,
        "duration_minutes": 120.0,
        "dt_minutes": 1.0,
        "patient_weight_kg": 70.0,
    }
    resp = client.post("/physiology/v2/challenge/meal", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["challenge_type"] == "oral_glucose_tolerance_test"
    assert data["baseline_glucose_mg_dL"] < data["peak_glucose_mg_dL"]
    assert data["peak_glucose_mg_dL"] > 125.0
    assert data["peak_insulin_uU_mL"] > 20.0
    assert data["cumulative_absorbed_g"] > 30.0
    assert len(data["trajectory"]) == 120


def test_v2_exercise_challenge_endpoint():
    payload = {
        "patient_id": "exercise_test_patient",
        "exercise_intensity": 0.6,
        "exercise_duration_minutes": 20.0,
        "total_observation_minutes": 40.0,
        "dt_minutes": 1.0,
        "patient_weight_kg": 70.0,
    }
    resp = client.post("/physiology/v2/challenge/exercise", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["challenge_type"] == "exercise_stress_challenge"
    assert data["peak_hr_bpm"] > data["baseline_hr_bpm"] + 15.0
    assert data["peak_cardiac_output_L_min"] > 5.5
    assert len(data["trajectory"]) == 40


def test_v2_conservation_audit_endpoint():
    resp = client.get("/physiology/v2/conservation/audit?patient_id=test_patient_1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["is_conserved"]
    assert len(data["violations"]) == 0
    assert "renal" in data["regional_flows_L_min"]
    assert data["cardiac_output_L_min"] > 0.0


def test_v2_anomaly_detect_endpoint():
    # 1. Normal telemetry
    normal_payload = {
        "current_telemetry": {"glucose_mg_dL": 105.0, "heart_rate_bpm": 74.0, "mean_bp_mmHg": 92.0},
        "previous_telemetry": {"glucose_mg_dL": 100.0, "heart_rate_bpm": 72.0, "mean_bp_mmHg": 90.0},
        "dt_minutes": 2.0,
    }
    resp_norm = client.post("/physiology/v2/anomaly/detect", json=normal_payload)
    assert resp_norm.status_code == 200
    data_norm = resp_norm.json()
    assert data_norm["is_physiologically_plausible"]
    assert len(data_norm["detected_anomalies"]) == 0

    # 2. Impossible telemetry rate jump (CGM jump of 80 mg/dL in 1 minute)
    abnormal_payload = {
        "current_telemetry": {"glucose_mg_dL": 180.0, "heart_rate_bpm": 74.0, "mean_bp_mmHg": 92.0},
        "previous_telemetry": {"glucose_mg_dL": 100.0, "heart_rate_bpm": 72.0, "mean_bp_mmHg": 90.0},
        "dt_minutes": 1.0,
    }
    resp_abn = client.post("/physiology/v2/anomaly/detect", json=abnormal_payload)
    assert resp_abn.status_code == 200
    data_abn = resp_abn.json()
    assert not data_abn["is_physiologically_plausible"]
    assert len(data_abn["detected_anomalies"]) > 0
