"""
Test 03: End-to-end patient workflow via the running API.

This script exercises the full patient journey through the deployed
API server:
  1. Health check
  2. OpenAPI spec discovery
  3. List available engines
  4. Initialize a patient twin
  5. Update with observations
  6. Get the personalized state and biomarkers
  7. Run counterfactual evaluation
  8. Get drift, uncertainty, and recommendations

Run:
    python tests/real_world/test_03_e2e_workflow.py
"""
import sys
import json
import numpy as np
import requests

BASE_URL = "http://localhost:8000"
TIMEOUT = 60

PASS = "\033[0;32m✓\033[0m"
FAIL = "\033[0;31m✗\033[0m"
errors = []


def check(name, condition, detail=""):
    if condition:
        print(f"  {PASS} {name}")
    else:
        print(f"  {FAIL} {name}: {detail}")
        errors.append(name)


def req(method, path, **kwargs):
    kwargs.setdefault("timeout", TIMEOUT)
    return requests.request(method, f"{BASE_URL}{path}", **kwargs)


def test_health():
    print("\n[1/8] Health check")
    r = req("GET", "/health")
    check("Server returns 200", r.status_code == 200, f"got {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        check("Health status healthy", data.get("status") in ("healthy", "ok", "up"),
              f"got {data.get('status')}")


def test_openapi():
    print("\n[2/8] OpenAPI spec discovery")
    r = req("GET", "/openapi.json")
    check("OpenAPI returns 200", r.status_code == 200, f"got {r.status_code}")
    if r.status_code == 200:
        spec = r.json()
        paths = spec.get("paths", {})
        check("Has at least 10 endpoints", len(paths) >= 10, f"got {len(paths)}")
        # Find personalization routes
        pers = [p for p in paths if "/personalization/" in p]
        check("Has personalization routes", len(pers) >= 5, f"got {len(pers)}")


def test_list_engines():
    print("\n[3/8] List available engines")
    r = req("GET", "/personalization/v2/engines")
    if r.status_code == 200:
        data = r.json()
        engines = data.get("engines", data) if isinstance(data, dict) else data
        check("Engine list returned", engines is not None)
        if isinstance(engines, list) and engines:
            print(f"  Available engines: {engines[:5]}")
    else:
        print(f"  (status {r.status_code})")


def test_initialize_and_update():
    print("\n[4/8] Initialize + update with synthetic observations")
    # Generate synthetic observations: 15-dim obs vector
    rng = np.random.RandomState(42)
    obs_history = []
    state = np.array([180.0, 8.0, 0.5, 0.3, 50.0, 120.0, 80.0, 70.0,
                      7.45, 40.0, 100.0, 5.0, 1.0, 0.5, 0.2])
    for t in range(40):
        # Slowly drift glucose down
        if t in (5, 25):
            state[0] += 50  # meal
        state[0] -= 1.0
        if t in (10, 30):
            state[1] += 4  # insulin bolus
        state[1] = max(0, state[1] - 0.5)
        obs = state + rng.normal(0, 0.5, size=15)
        obs[0] = max(40, min(400, obs[0]))
        obs_history.append(obs.tolist())

    # Initialize
    r = req("POST", "/personalization/v2/initialize", json={
        "patient_id": "TEST_001",
        "engine": "dual",
        "initial_observation": obs_history[0],
    })
    if r.status_code not in (200, 201):
        print(f"  (initialize status {r.status_code}: {r.text[:200] if r.text else ''})")
    else:
        check("Initialize succeeded", True)

    # Update
    update_count = 0
    for t in range(1, len(obs_history)):
        r = req("POST", "/personalization/v2/update", json={
            "patient_id": "TEST_001",
            "observation": obs_history[t],
        })
        if r.status_code in (200, 201):
            update_count += 1
        elif t < 3:
            print(f"  Update {t} failed: {r.status_code}")
    check("At least 10 updates succeeded", update_count >= 10,
          f"got {update_count}/{len(obs_history)-1}")


def test_get_state_and_biomarkers():
    print("\n[5/8] Get state and biomarkers")
    r = req("GET", "/personalization/v2/state/TEST_001")
    check("State endpoint returns 200", r.status_code == 200, f"got {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        check("Has state field", "state" in data or "twin_state" in data or "estimated_state" in data)
        keys = list(data.keys())[:5]
        print(f"  Response keys: {keys}")

    r = req("GET", "/personalization/v2/TEST_001/biomarkers")
    check("Biomarkers endpoint returns 200", r.status_code == 200, f"got {r.status_code}")


def test_counterfactual():
    print("\n[6/8] Counterfactual evaluation")
    r = req("POST", "/personalization/v2/TEST_001/counterfactual", json={
        "intervention_type": "insulin_bolus",
        "magnitude": 8.0,
        "duration_hours": 2.0,
    })
    if r.status_code == 200:
        data = r.json()
        check("CF response has program/trajectory",
              any(k in data for k in ("program", "ate_glucose", "effect",
                                       "predicted_effect", "ate",
                                       "glucose_trajectory", "final_glucose")),
              f"keys: {list(data.keys())}")
    else:
        print(f"  (status {r.status_code}: {r.text[:200] if r.text else ''})")


def test_drift_uncertainty_recommend():
    print("\n[7/8] Drift, uncertainty, recommendations")
    for path in ("/personalization/v2/TEST_001/drift",
                 "/personalization/v2/TEST_001/uncertainty",
                 "/personalization/v2/TEST_001/recommend"):
        r = req("GET", path)
        check(f"GET {path.split('/')[-1]} returns 200", r.status_code == 200, f"got {r.status_code}")


def test_cleanup():
    print("\n[8/8] Cleanup")
    r = req("DELETE", "/personalization/v2/TEST_001")
    if r.status_code in (200, 204, 404):
        check("Patient deleted (or not present)", True)
    else:
        print(f"  (status {r.status_code})")


def main():
    print("=" * 60)
    print("  TEST 03: End-to-end API workflow")
    print(f"  Target: {BASE_URL}")
    print("=" * 60)
    try:
        r = req("GET", "/health", timeout=3)
        if r.status_code != 200:
            print(f"\n  {FAIL} Server not responding at {BASE_URL}")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print(f"\n  {FAIL} Cannot connect to {BASE_URL}")
        print("  Start it with: ./tests/real_world/start_server.sh")
        sys.exit(1)

    test_health()
    test_openapi()
    test_list_engines()
    test_initialize_and_update()
    test_get_state_and_biomarkers()
    test_counterfactual()
    test_drift_uncertainty_recommend()
    test_cleanup()

    print()
    print("=" * 60)
    if errors:
        print(f"  {len(errors)} E2E STEPS FAILED")
        for e in errors:
            print(f"    - {e}")
    else:
        print("  ALL E2E STEPS PASSED")
    print("=" * 60)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
