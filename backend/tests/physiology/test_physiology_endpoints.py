"""
Tests for Physiology Multi-Organ World Model API Endpoints.
Verifies GET /physiology/organs/status, POST /physiology/simulation/step,
POST /physiology/challenge/meal, POST /physiology/challenge/exercise,
and GET /physiology/vvuq/report.
"""

import pytest
from starlette.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_organ_status_endpoint():
    resp = client.get("/physiology/organs/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["organ_count"] == 4
    assert "lungs" in data["organs"]
    assert "liver" in data["organs"]
    assert "gastrointestinal" in data["organs"]
    assert "autonomic_endocrine" in data["organs"]
    assert data["organs"]["lungs"]["status"] == "valid"


def test_simulation_step_endpoint():
    req_payload = {
        "patient_id": "test_patient_001",
        "patient_weight_kg": 70.0,
        "dt_minutes": 1.0,
        "method": "rk4",
        "exercise": 0.2,
        "carbs_intake_g": 0.0,
    }
    resp = client.post("/physiology/simulation/step", json=req_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["is_physiologically_valid"] is True
    snap = data["snapshot"]
    assert "hemodynamics" in snap
    assert "respiratory_and_blood_gases" in snap
    assert "metabolic_and_digestion" in snap
    assert "autonomic_and_neuroendocrine" in snap
    assert 60 <= snap["hemodynamics"]["heart_rate_bpm"] <= 140
    assert 7.20 <= snap["respiratory_and_blood_gases"]["arterial_pH"] <= 7.60


def test_meal_challenge_endpoint():
    req_payload = {
        "patient_id": "test_patient_002",
        "carbs_g": 75.0,
        "duration_minutes": 60,
        "step_minutes": 1.0,
    }
    resp = client.post("/physiology/challenge/meal", json=req_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["carbs_administered_g"] == 75.0
    assert len(data["trajectory"]) > 0
    assert data["summary"]["peak_glucose_mg_dL"] > data["summary"]["baseline_glucose_mg_dL"]


def test_exercise_challenge_endpoint():
    req_payload = {
        "patient_id": "test_patient_003",
        "exercise_intensity": 0.5,
        "duration_minutes": 20,
        "step_minutes": 1.0,
    }
    resp = client.post("/physiology/challenge/exercise", json=req_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["summary"]["exercise_peak_hr"] > data["summary"]["baseline_hr"]
    assert data["summary"]["exercise_peak_co"] > data["summary"]["baseline_co"]
    assert data["summary"]["min_saO2"] >= 90.0


def test_vvuq_report_endpoint():
    resp = client.get("/physiology/vvuq/report")
    assert resp.status_code == 200
    data = resp.json()
    assert "vvuq_standard" in data
    assert data["whole_body_stability"] == "Passed"
    assert "lungs" in data["organ_evaluations"]
    assert "liver" in data["organ_evaluations"]

