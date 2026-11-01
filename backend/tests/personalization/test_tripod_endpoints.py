"""
backend/tests/personalization/test_tripod_endpoints.py
Unit tests for the TRIPOD+AI Identifiability and Baseline Benchmark API endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_identifiability_audit_endpoint():
    resp = client.get("/personalization/identifiability/audit")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "restricted_4_param" in data
    assert "overparameterized_augmented" in data

    # 4-param subset must be practically identifiable
    r4 = data["restricted_4_param"]
    assert r4["is_practically_identifiable"] is True
    assert r4["fim_rank"] == 4
    assert r4["mean_recovery_error_pct"] < 10.0

    # 12-param model must fail with severe rank deficiency
    r_full = data["overparameterized_augmented"]
    assert r_full["is_practically_identifiable"] is False
    assert r_full["fim_rank"] < len(r_full["param_names"])


def test_baselines_benchmark_endpoint():
    resp = client.get("/personalization/baselines/benchmark?days=5")
    assert resp.status_code == 200
    data = resp.json()
    assert "benchmark_results" in data
    results = data["benchmark_results"]
    assert len(results) == 6  # 3 models * 2 horizons

    models = {r["model_name"] for r in results}
    assert "Persistence" in models
    assert "Autoregressive_AR3" in models
    assert "DigitalTwin_Identifiable4" in models
