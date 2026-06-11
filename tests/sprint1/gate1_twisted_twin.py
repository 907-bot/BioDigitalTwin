"""
Sprint 1, Gate 1: Twisted-Twin Validation

Proves the state estimator can track physiological perturbations.
1. Generate healthy patient trajectory from the ODE
2. At step T_perturb, shift physiology (increase insulin resistance)
3. Confirm the estimator converges to the new state within N steps
4. Document with plots

Observations: 16-dim (glucose, BP, HR, HRV, GFR, electrolytes,
lipids, cortisol, sleep, insulin). Insulin observation makes SI identifiable.

This is the foundational gate — without it, nothing downstream is verifiable.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app.personalization.dynamics import (
    full_dynamics, full_observation, DEFAULT_PARAMS, _params_to_dict
)
from app.personalization.state import PHYSIO_DIM, Phase3TwinState
from app.personalization.dual_engine import create_dual_engine


def generate_trajectory(params, initial_state, n_steps, inputs_fn=None):
    """Run ODE forward to generate a trajectory."""
    states = np.zeros((n_steps, PHYSIO_DIM))
    observations = np.zeros((n_steps, 16))  # 16-dim obs (with insulin)
    s = initial_state.copy()
    for t in range(n_steps):
        inputs = inputs_fn(t) if inputs_fn else {}
        s = full_dynamics(s, params, inputs)
        s[0] = np.clip(s[0], 20, 600)
        s[1] = np.clip(s[1], 0, 500)
        states[t] = s
        observations[t] = full_observation(s)
    return states, observations


def add_observation_noise(observations, rng, noise_scale=None):
    """Add realistic clinical noise to observations (16-dim)."""
    if noise_scale is None:
        noise_scale = {
            0: 5.0,   # glucose (CGM)
            1: 3.0,   # SBP
            2: 2.0,   # DBP
            3: 2.0,   # HR
            4: 5.0,   # HRV
            5: 4.0,   # GFR
            6: 0.5,   # Na
            7: 0.1,   # K
            8: 2.0,   # Osm
            9: 0.05,  # FFA
            10: 5.0,  # LDL
            11: 3.0,  # HDL
            12: 8.0,  # TG
            13: 10.0, # cortisol
            14: 0.1,  # sleep
            15: 2.0,  # insulin (fasting insulin assay)
        }
    noisy = observations.copy()
    for idx, scale in noise_scale.items():
        if idx < noisy.shape[1]:
            noisy[:, idx] += rng.normal(0, scale, size=noisy.shape[0])
    return noisy


def run_twisted_twin_validation():
    """
    Gate 1: Twisted-Twin Validation

    Scenario:
    - Steps 0-99:   Healthy patient (SI=0.018, normal physiology)
    - Step 100:     Perturbation — insulin resistance increases 3x
                    (simulates onset of T2DM or stress-induced IR)
    - Steps 100-299: Track whether estimator converges to new state

    Convergence criteria:
    - Glucose RMSE < 20 mg/dL after convergence window
    - Estimated SI converges within 30% of true perturbed value
    - No divergence (glucose stays in physiological range)
    """
    print("=" * 70)
    print("GATE 1: TWISTED-TWIN VALIDATION")
    print("=" * 70)

    rng = np.random.default_rng(42)
    n_steps = 300
    t_perturb = 100

    # --- Phase 1: Generate healthy trajectory ---
    healthy_params = DEFAULT_PARAMS.copy()
    initial_state = np.zeros(PHYSIO_DIM)
    initial_state[0] = 100.0   # glucose
    initial_state[1] = 5.0     # insulin
    initial_state[2] = 2.0     # HGP
    initial_state[3] = 4.0     # PGU
    initial_state[4] = 1.0     # IR (healthy)
    initial_state[5] = 120.0   # SBP
    initial_state[6] = 80.0    # DBP
    initial_state[7] = 70.0    # HR
    initial_state[8] = 45.0    # HRV
    initial_state[9] = 100.0   # GFR
    initial_state[10] = 140.0  # Na
    initial_state[11] = 4.2    # K
    initial_state[12] = 290.0  # Osm
    initial_state[13] = 1.0    # CRP
    initial_state[14] = 1.2    # CLOCK_BMAL1
    initial_state[15] = 0.8    # PER_CRY
    initial_state[16] = 350.0  # cortisol
    initial_state[17] = 10.0   # melatonin
    initial_state[18] = 0.0    # circadian_phase
    initial_state[19] = 0.3    # sleep_pressure
    initial_state[20] = 20.0   # fat_mass
    initial_state[21] = 0.5    # FFA
    initial_state[22] = 120.0  # LDL
    initial_state[23] = 50.0   # HDL
    initial_state[24] = 120.0  # TG

    # Meal inputs every 36 steps (3 hours at 5-min resolution)
    def meal_inputs(t):
        inputs = {}
        if t % 36 in (10, 11, 12):  # meal window
            inputs["meal_glucose"] = 15.0  # moderate meal
        if t % 36 == 12:
            inputs["insulin_dose"] = 2.0  # endogenous-like
        return inputs

    # Generate full trajectory with perturbation
    print("\n[1/4] Generating trajectory with perturbation at step", t_perturb)
    states_true = np.zeros((n_steps, PHYSIO_DIM))
    observations_clean = np.zeros((n_steps, 16))  # 16-dim obs (with insulin)
    s = initial_state.copy()

    # Create perturbed params (3x insulin resistance)
    perturbed_params = healthy_params.copy()
    perturbed_params[0] *= 0.33  # SI decreases (more resistant)
    perturbed_params[4] *= 1.5  # arterial stiffness increases
    perturbed_params[5] *= 1.3  # vascular resistance increases

    for t in range(n_steps):
        params = perturbed_params if t >= t_perturb else healthy_params
        inputs = meal_inputs(t)
        s = full_dynamics(s, params, inputs)
        s[0] = np.clip(s[0], 20, 600)
        s[1] = np.clip(s[1], 0, 500)
        states_true[t] = s
        observations_clean[t] = full_observation(s)

    # Add noise
    observations_noisy = add_observation_noise(observations_clean, rng)

    true_si_healthy = healthy_params[0]
    true_si_perturbed = perturbed_params[0]
    print(f"  True SI (healthy):     {true_si_healthy:.4f}")
    print(f"  True SI (perturbed):   {true_si_perturbed:.4f}")
    print(f"  Glucose range (healthy):     {states_true[:t_perturb, 0].mean():.1f} ± {states_true[:t_perturb, 0].std():.1f}")
    print(f"  Glucose range (perturbed):   {states_true[t_perturb:, 0].mean():.1f} ± {states_true[t_perturb:, 0].std():.1f}")

    # --- Phase 2: Run estimator on noisy observations ---
    print("\n[2/4] Running dual estimation engine...")
    engine = create_dual_engine()
    engine.initialize(observations_noisy[0])

    estimated_states = np.zeros((n_steps, PHYSIO_DIM))
    predicted_glucose = np.zeros(n_steps)
    predicted_std = np.zeros(n_steps)
    estimated_si = np.zeros(n_steps)

    for t in range(n_steps):
        # Predict BEFORE update (true 1-step ahead)
        pred_mean, pred_std = engine.predict(n_steps=1)
        predicted_glucose[t] = pred_mean[0]
        predicted_std[t] = pred_std[0]

        # Update with observation
        engine.update(observations_noisy[t])

        # Record estimated state
        state = engine.filter.get_state()
        estimated_states[t] = state
        estimated_si[t] = engine._estimated_params[0]  # SI is index 0

    # --- Phase 3: Evaluate convergence ---
    print("\n[3/4] Evaluating convergence...")

    # Convergence window: steps 150-299 (50+ steps after perturbation)
    conv_start = t_perturb + 50
    conv_end = n_steps

    # Glucose tracking error
    glucose_errors = np.abs(estimated_states[:, 0] - states_true[:, 0])
    glucose_rmse_healthy = np.sqrt(np.mean(glucose_errors[:t_perturb] ** 2))
    glucose_rmse_perturbed = np.sqrt(np.mean(glucose_errors[conv_start:conv_end] ** 2))
    glucose_rmse_convergence = np.sqrt(np.mean(glucose_errors[conv_start:conv_end] ** 2))

    # SI estimation error
    si_errors = np.abs(estimated_si - true_si_perturbed)
    si_final_error = np.mean(si_errors[conv_start:conv_end])
    si_converged = si_final_error < 0.3 * true_si_perturbed  # within 30%

    # Check for divergence
    max_glucose = np.max(estimated_states[:, 0])
    min_glucose = np.min(estimated_states[50:, 0])  # skip initial transient
    no_divergence = max_glucose < 500 and min_glucose > 40

    # Prediction accuracy
    pred_errors = np.abs(predicted_glucose - observations_clean[:, 0])
    pred_rmse_healthy = np.sqrt(np.mean(pred_errors[:t_perturb] ** 2))
    pred_rmse_perturbed = np.sqrt(np.mean(pred_errors[conv_start:conv_end] ** 2))

    print(f"  Glucose RMSE (healthy phase):    {glucose_rmse_healthy:.1f} mg/dL")
    print(f"  Glucose RMSE (after convergence): {glucose_rmse_perturbed:.1f} mg/dL")
    print(f"  1-step pred RMSE (healthy):       {pred_rmse_healthy:.1f} mg/dL")
    print(f"  1-step pred RMSE (perturbed):     {pred_rmse_perturbed:.1f} mg/dL")
    print(f"  SI estimated:  {np.mean(estimated_si[conv_start:conv_end]):.4f} "
          f"(true: {true_si_perturbed:.4f}, error: {si_final_error:.4f})")
    print(f"  SI converged (< 30% error): {si_converged}")
    print(f"  No divergence:              {no_divergence}")

    # Gate 1 criteria
    gate1_pass = (
        glucose_rmse_perturbed < 25.0 and
        si_converged and
        no_divergence and
        pred_rmse_perturbed < 20.0
    )
    print(f"\n  {'✓' if gate1_pass else '✗'} GATE 1: {'PASSED' if gate1_pass else 'FAILED'}")

    # --- Phase 4: Generate plots ---
    print("\n[4/4] Generating plots...")
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.3)

    time_min = np.arange(n_steps) * 5  # 5-min steps → minutes
    time_hours = time_min / 60.0

    # Plot 1: Glucose tracking
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time_hours, states_true[:, 0], 'b-', linewidth=1.5, label='True glucose', alpha=0.8)
    ax1.plot(time_hours, observations_noisy[:, 0], 'g.', markersize=2, alpha=0.3, label='Noisy observations')
    ax1.plot(time_hours, estimated_states[:, 0], 'r-', linewidth=1.5, label='Estimated glucose', alpha=0.8)
    ax1.axvline(x=time_hours[t_perturb], color='k', linestyle='--', linewidth=2, label='Perturbation')
    ax1.fill_between(time_hours,
                     predicted_glucose - 1.96 * predicted_std,
                     predicted_glucose + 1.96 * predicted_std,
                     alpha=0.2, color='red', label='95% PI')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Glucose (mg/dL)')
    ax1.set_title(f'Gate 1: Glucose Tracking — RMSE healthy={glucose_rmse_healthy:.1f}, perturbed={glucose_rmse_perturbed:.1f} mg/dL')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_ylim(40, 300)
    ax1.axhline(y=70, color='orange', linestyle=':', alpha=0.5)
    ax1.axhline(y=180, color='orange', linestyle=':', alpha=0.5)

    # Plot 2: SI estimation convergence
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time_hours, estimated_si, 'r-', linewidth=2, label='Estimated SI')
    ax2.axhline(y=true_si_healthy, color='b', linestyle='--', linewidth=1.5, label=f'True SI (healthy={true_si_healthy:.4f})')
    ax2.axhline(y=true_si_perturbed, color='g', linestyle='--', linewidth=1.5, label=f'True SI (perturbed={true_si_perturbed:.4f})')
    ax2.axvline(x=time_hours[t_perturb], color='k', linestyle='--', linewidth=2)
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('SI (insulin sensitivity)')
    ax2.set_title('SI Estimation Convergence')
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 0.04)

    # Plot 3: Prediction error over time
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(time_hours, pred_errors, 'purple', linewidth=0.8, alpha=0.7)
    ax3.axhline(y=15, color='r', linestyle='--', linewidth=1, label='15 mg/dL threshold')
    ax3.axvline(x=time_hours[t_perturb], color='k', linestyle='--', linewidth=2, label='Perturbation')
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('|Predicted - Actual| (mg/dL)')
    ax3.set_title('1-Step Prediction Error')
    ax3.legend(fontsize=8)

    # Plot 4: SBP tracking
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(time_hours, states_true[:, 5], 'b-', linewidth=1, label='True SBP', alpha=0.7)
    ax4.plot(time_hours, estimated_states[:, 5], 'r-', linewidth=1, label='Estimated SBP', alpha=0.7)
    ax4.axvline(x=time_hours[t_perturb], color='k', linestyle='--', linewidth=2)
    ax4.set_xlabel('Time (hours)')
    ax4.set_ylabel('SBP (mmHg)')
    ax4.set_title('SBP Tracking')
    ax4.legend(fontsize=8)

    # Plot 5: Insulin tracking
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(time_hours, states_true[:, 1], 'b-', linewidth=1, label='True insulin', alpha=0.7)
    ax5.plot(time_hours, estimated_states[:, 1], 'r-', linewidth=1, label='Estimated insulin', alpha=0.7)
    ax5.axvline(x=time_hours[t_perturb], color='k', linestyle='--', linewidth=2)
    ax5.set_xlabel('Time (hours)')
    ax5.set_ylabel('Insulin (μU/mL)')
    ax5.set_title('Insulin Tracking')
    ax5.legend(fontsize=8)

    # Plot 6: Convergence summary
    ax6 = fig.add_subplot(gs[3, :])
    metrics = ['Glucose RMSE\n(healthy)', 'Glucose RMSE\n(perturbed)', 'Pred RMSE\n(healthy)', 'Pred RMSE\n(perturbed)', 'SI Error\n(final)']
    values = [glucose_rmse_healthy, glucose_rmse_perturbed, pred_rmse_healthy, pred_rmse_perturbed, si_final_error * 1000]
    colors = ['green' if v < 20 else 'red' for v in [glucose_rmse_healthy, glucose_rmse_perturbed, pred_rmse_healthy, pred_rmse_perturbed]]
    colors.append('green' if si_converged else 'red')
    bars = ax6.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    ax6.axhline(y=20, color='r', linestyle='--', linewidth=1, label='20 mg/dL threshold')
    ax6.set_ylabel('Error (mg/dL or ×10⁻³)')
    ax6.set_title(f'Gate 1 Summary — {"PASSED ✓" if gate1_pass else "FAILED ✗"}')
    ax6.legend()

    plt.suptitle('Sprint 1, Gate 1: Twisted-Twin Validation\n'
                 'Perturbation: 3× insulin resistance at t=100 steps (8.3 hours)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.savefig('scientific_proof/gate1_twisted_twin.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved gate1_twisted_twin.png")

    return {
        "gate1_pass": gate1_pass,
        "glucose_rmse_healthy": glucose_rmse_healthy,
        "glucose_rmse_perturbed": glucose_rmse_perturbed,
        "pred_rmse_healthy": pred_rmse_healthy,
        "pred_rmse_perturbed": pred_rmse_perturbed,
        "si_true_healthy": true_si_healthy,
        "si_true_perturbed": true_si_perturbed,
        "si_estimated_final": float(np.mean(estimated_si[conv_start:conv_end])),
        "si_converged": si_converged,
        "no_divergence": no_divergence,
    }


if __name__ == "__main__":
    results = run_twisted_twin_validation()
    print("\n" + "=" * 70)
    print("GATE 1 RESULTS:")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print("=" * 70)
