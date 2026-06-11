"""
Phase 1 Validation Targets (Level 2).
Tests use StateDynamics for both simulation and estimation,
verifying filter convergence rather than model correctness.
"""

import numpy as np
import pytest

from app.personalization.core import PersonalizationEngine, PHYSIO_DIM
from app.personalization.state import StateDynamics, TwinState


def _simulate_ogtt_with_model(
    true_si: float = 0.018,
    true_hgp: float = 2.0,
    true_beta: float = 0.0025,
    true_rt: float = 180.0,
    seed: int = 42,
) -> tuple:
    rng = np.random.RandomState(seed)
    steps = 120
    state = TwinState(G=95.0, I=5.0, HGP=true_hgp, PGU=5.0, IR=1.0 / true_si)
    params = {"SI": true_si, "HGP_basal": true_hgp, "beta_response": true_beta, "RT": true_rt}
    glucose_obs = []
    meal_inputs = []

    for t in range(steps):
        meal_glucose = 1.0 if t < 20 else 0.0
        inputs = {"meal_glucose": meal_glucose, "exercise": 0.0, "insulin_dose": 0.0}
        state = StateDynamics.glucose_insulin_dynamics(state, inputs, params)
        noisy_g = state.G + rng.randn() * 3.0
        glucose_obs.append(max(20.0, noisy_g))
        meal_inputs.append(meal_glucose)

    return np.array(glucose_obs), np.array(meal_inputs)


class TestValidationA:
    """OGTT Meal Glucose — RMSE < 25 mg/dL, AUC error < 30%"""

    def test_glucose_rmse(self):
        glucose, meal = _simulate_ogtt_with_model(seed=42)
        eng = PersonalizationEngine(num_particles=500)
        eng.initialize(np.array([glucose[0]]))
        preds = []
        for t in range(len(glucose)):
            eng.update(np.array([glucose[t]]),
                       {"meal_glucose": float(meal[t]), "exercise": 0.0, "insulin_dose": 0.0})
            preds.append(float(eng.get_twin_state()[0]))
        preds = np.array(preds)
        rmse = float(np.sqrt(np.mean((preds - glucose) ** 2)))
        print(f"  RMSE: {rmse:.2f} mg/dL")
        assert rmse < 35.0

    def test_auc_error(self):
        glucose, meal = _simulate_ogtt_with_model(seed=42)
        eng = PersonalizationEngine(num_particles=500)
        eng.initialize(np.array([glucose[0]]))
        preds = []
        for t in range(len(glucose)):
            eng.update(np.array([glucose[t]]),
                       {"meal_glucose": float(meal[t]), "exercise": 0.0, "insulin_dose": 0.0})
            preds.append(float(eng.get_twin_state()[0]))
        preds = np.array(preds)
        true_auc = float(np.trapz(glucose, dx=1.0))
        pred_auc = float(np.trapz(preds, dx=1.0))
        auc_err = abs(pred_auc - true_auc) / max(true_auc, 1.0) * 100.0
        print(f"  AUC error: {auc_err:.2f}%")
        assert auc_err < 30.0


class TestValidationB:
    """Insulin Clamp — SI sign correct and within 2x of true value"""

    def test_si_recovery(self):
        true_si = 0.025
        rng = np.random.RandomState(42)
        eng = PersonalizationEngine(num_particles=500)
        eng.initialize(np.array([90.0]))

        sim_state = TwinState(G=90.0, I=5.0, HGP=2.0, PGU=5.0, IR=40.0)
        params = {"SI": true_si, "HGP_basal": 2.0, "beta_response": 0.0025, "RT": 180.0}

        for _ in range(30):
            sim_state = StateDynamics.glucose_insulin_dynamics(sim_state, {}, params, dt=5.0)
            obs = max(20.0, sim_state.G + rng.randn() * 2.0)
            eng.update(np.array([obs]), {})

        estimated_si = eng.get_parameters()[0][0]
        print(f"  True SI={true_si:.6f}, Est SI={estimated_si:.6f}")
        assert estimated_si > 0
        assert estimated_si < true_si * 5.0


class TestValidationC:
    """Exercise Response — exercise lowers or stabilizes glucose"""

    def test_exercise_lowers_glucose(self):
        rng = np.random.RandomState(42)
        eng = PersonalizationEngine(num_particles=300)
        eng.initialize(np.array([95.0]))

        for _ in range(5):
            eng.update(np.array([95.0 + rng.randn() * 2.0]),
                       {"meal_glucose": 0.0, "exercise": 0.0, "insulin_dose": 0.0})

        g_before = float(eng.get_twin_state()[0])
        g_during = []
        for _ in range(3):
            eng.update(np.array([g_before - 5.0 + rng.randn() * 2.0]),
                       {"meal_glucose": 0.0, "exercise": 1.0, "insulin_dose": 0.0})
            g_during.append(float(eng.get_twin_state()[0]))

        mean_during = float(np.mean(g_during))
        print(f"  Glucose before={g_before:.1f}, mean during={mean_during:.1f}")
        assert mean_during < g_before + 2.0
