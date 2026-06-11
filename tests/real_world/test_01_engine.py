"""
Test 01: Core engine smoke test.

Validates that the digital twin engine produces sensible outputs on
synthetic patient data. This is the foundational test that all other
modules depend on.

Run:
    PYTHONPATH=backend:. python tests/real_world/test_01_engine.py
"""
import sys
import time
import numpy as np

sys.path.insert(0, 'backend')

from app.personalization.dynamics import DEFAULT_PARAMS, full_dynamics, full_observation
from app.personalization.core import PersonalizationEngine
from app.personalization.dual_engine import create_dual_engine
from app.personalization.do_calculus import DoCalculusCounterfactual, InterventionSpec, InterventionType
from app.personalization.safety import (
    HypoglycemiaEarlyWarning, SafetyGuardrails, OODDetector,
    DriftAttributor, AdversarialDetector,
)
from app.personalization.state import PHYSIO_DIM, OBS_DIM


PASS = "\033[0;32m✓\033[0m"
FAIL = "\033[0;31m✗\033[0m"
errors = []


def check(name, condition, detail=""):
    if condition:
        print(f"  {PASS} {name}")
    else:
        print(f"  {FAIL} {name}: {detail}")
        errors.append(name)


def make_synthetic_patient(seed=42, n_steps=100, with_meals=True):
    """Create a synthetic T1DM patient with realistic dynamics."""
    rng = np.random.RandomState(seed)
    params = DEFAULT_PARAMS.copy()
    params[0] = rng.lognormal(-4.0, 0.3)  # SI
    params[2] = 0.0001  # very low beta (T1DM)

    state = np.zeros(PHYSIO_DIM)
    state[0] = rng.normal(180, 30)  # elevated glucose
    state[1] = 0.013 * max(0, state[0] - 80)
    state[5] = 120
    state[6] = 80
    state[7] = 70
    state[21] = 0.5

    states = []
    obs = []
    s = state.copy()
    for t in range(n_steps):
        inputs = {}
        if with_meals and t in (10, 58, 106):
            inputs["carbs_grams"] = 50.0
            inputs["insulin_dose"] = 6.0
        s = full_dynamics(s, params, inputs)
        s[0] = max(20.0, min(600.0, s[0]))
        s[1] = max(0.0, min(500.0, s[1]))
        states.append(s.copy())
        obs.append(full_observation(s))
    return np.array(states), np.array(obs), params


def test_dynamics():
    print("\n[1/6] Dynamics function")
    rng = np.random.RandomState(42)
    state = np.zeros(PHYSIO_DIM)
    state[0] = 100
    state[1] = 5
    state[5] = 120
    state[6] = 80
    state[7] = 70
    state[21] = 0.5
    new_state = full_dynamics(state, DEFAULT_PARAMS, {})
    check("Returns array", isinstance(new_state, np.ndarray))
    check("Glucose stays positive", new_state[0] > 0, f"got {new_state[0]:.1f}")
    check("Glucose stays bounded", new_state[0] < 600, f"got {new_state[0]:.1f}")
    check("All state components finite", np.all(np.isfinite(new_state)),
          f"non-finite at {[i for i, v in enumerate(new_state) if not np.isfinite(v)]}")


def test_observation():
    print("\n[2/6] Observation function")
    state = np.zeros(PHYSIO_DIM)
    state[0] = 100
    state[1] = 5
    state[5] = 120
    state[6] = 80
    state[7] = 70
    state[21] = 0.5
    obs = full_observation(state)
    check("Returns 15-dim observation", len(obs) == 15,
          f"got len {len(obs)}")
    check("Glucose in observation[0]", abs(obs[0] - 100) < 1,
          f"got {obs[0]:.1f}")


def test_personalization_engine():
    print("\n[3/6] PersonalizationEngine (legacy augmented UKF)")
    states, obs, true_params = make_synthetic_patient(seed=42, n_steps=80)
    engine = PersonalizationEngine()
    engine.initialize(obs[0])
    t0 = time.time()
    for t in range(1, 80):
        engine.update(obs[t])
    elapsed = time.time() - t0

    mu = engine.get_twin_state()
    cov = engine.get_twin_state_covariance()
    check("Engine produced state", mu is not None)
    check("State is 30-dim", len(mu) == 30)
    check("Covariance is finite", np.all(np.isfinite(cov)))
    check("Glucose estimate reasonable", 50 < mu[0] < 300,
          f"got {mu[0]:.1f}")
    check("Engine runs in reasonable time", elapsed < 30,
          f"took {elapsed:.1f}s")


def test_dual_engine():
    print("\n[4/6] DualEstimationEngine (physio UKF + MAP params)")
    states, obs, true_params = make_synthetic_patient(seed=42, n_steps=80)
    engine = create_dual_engine()
    engine.initialize(obs[0])
    t0 = time.time()
    for t in range(1, 80):
        engine.update(obs[t])
    elapsed = time.time() - t0

    estimated, std = engine.get_estimated_params()
    check("Estimated params available", estimated is not None)
    check("7 identifiable params", len(estimated) == 7,
          f"got {len(estimated)}")
    check("All estimated params finite", np.all(np.isfinite(estimated)))
    check("All std positive", np.all(std > 0))
    check("Engine runs in reasonable time", elapsed < 60,
          f"took {elapsed:.1f}s")

    # Test prediction
    pred_mean, pred_std = engine.predict(n_steps=1)
    check("Prediction returns 15 values", len(pred_mean) == 15)
    check("Prediction std positive", np.all(pred_std > 0))
    check("Glucose prediction finite", np.isfinite(pred_mean[0]))


def test_do_calculus():
    print("\n[5/6] Do-calculus counterfactual")
    rng = np.random.RandomState(42)
    state = np.zeros(PHYSIO_DIM)
    state[0] = 200  # high glucose
    state[1] = 5
    state[5] = 120
    state[6] = 80
    state[7] = 70

    dc = DoCalculusCounterfactual()
    intervention = InterventionSpec(
        intervention_type=InterventionType.INSULIN_BOLUS,
        magnitude=10.0,
        duration_steps=6,
    )
    result = dc.evaluate_intervention(state, DEFAULT_PARAMS, intervention, n_total_steps=24)
    check("ATE glucose negative (insulin lowers glucose)", result.ate_glucose < 0,
          f"got {result.ate_glucose:.1f}")
    check("ATE magnitude > 10 mg/dL", abs(result.ate_glucose) > 10,
          f"got {result.ate_glucose:.1f}")
    check("Returns baseline states", result.baseline_states.shape == (24, 30))
    check("Returns cf states", result.cf_states.shape == (24, 30))

    # Refutation test
    ref = dc.refutation_test(state, DEFAULT_PARAMS, intervention, n_total_steps=12, test_type="placebo")
    check("Refutation test runs", "factual_ate_glucose" in ref)


def test_safety_layer():
    print("\n[6/6] Safety layer (OOD, Hypo, Guardrails, Adversarial)")
    # OOD
    ood = OODDetector(percentile=0.95)
    rng = np.random.RandomState(42)
    normal = rng.normal(120, 10, size=(50, 15))
    ood.fit(normal)
    r1 = ood.predict(np.array([120.0] * 15))
    r2 = ood.predict(np.array([400.0] * 15))
    check("In-dist: not OOD", not r1.is_ood)
    check("Out-of-dist: OOD detected", r2.is_ood)
    check("Distance increases with deviation", r2.distance > r1.distance)

    # Hypoglycemia prediction
    hypo = HypoglycemiaEarlyWarning(threshold_mg_dL=70, alert_probability=0.30)
    alert1 = hypo.evaluate(120, 10, 6)
    alert2 = hypo.evaluate(60, 5, 6)
    check("Normal glucose: no alert", not alert1.predicted)
    check("Low glucose: alert raised", alert2.predicted)
    check("Alert probability high", alert2.probability > 0.90)

    # Safety guardrails
    guard = SafetyGuardrails(ood_detector=ood)
    twin_state = np.zeros(30)
    twin_state[0] = 120
    twin_cov = np.eye(30) * 0.5
    obs = np.array([120.0] * 15)
    v1 = guard.evaluate(twin_state, twin_cov, obs, drift_level=0)
    twin_state[0] = 800
    v2 = guard.evaluate(twin_state, twin_cov, obs, drift_level=0)
    check("Normal state: safe", v1.safe)
    check("Extreme glucose: abstention", v2.abstention_required)

    # Adversarial
    adv = AdversarialDetector()
    a1 = adv.update(np.array([120.0] * 15))
    a2 = adv.update(np.array([600.0] * 15))
    check("Normal: no anomaly", len(a1) == 0)
    check("Out-of-range: detected", any("out_of_range" in a for a in a2))

    # Drift attribution
    attr = DriftAttributor(slack=0.5, threshold=10.0)
    for _ in range(60):
        r = attr.update("metabolic", np.random.normal(1.0, 0.3))
    check("Drift attribution runs", r is not None)
    check("CUSUM accumulates", r.cusum > 0)


def main():
    print("=" * 60)
    print("  TEST 01: Core Engine Smoke Test")
    print("=" * 60)
    test_dynamics()
    test_observation()
    test_personalization_engine()
    test_dual_engine()
    test_do_calculus()
    test_safety_layer()
    print()
    print("=" * 60)
    if errors:
        print(f"  {len(errors)} FAILED: {errors}")
        sys.exit(1)
    else:
        print("  ALL ENGINE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
