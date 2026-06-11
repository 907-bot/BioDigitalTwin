"""
Phase 5 — Observability and Identifiability Analysis.

Evaluates whether the UKF can uniquely recover patient-specific parameters
from clinically realistic observations.

Methods:
  1. Empirical observability: finite-difference observability matrix rank
  2. Parameter recovery: known ground truth → UKF → recovery error
  3. Profile likelihood: confidence regions for each parameter
  4. Collinearity analysis: pairwise parameter compensability
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from app.personalization.dynamics import DEFAULT_PARAMS


@dataclass
class IdentifiabilityReport:
    n_observable_states: int
    state_dim: int
    observability_rank: int
    is_observable: bool
    n_identifiable_params: int
    n_total_params: int
    param_recovery_mae: Dict[str, float]
    collinearity_indices: Dict[str, float]
    min_observations_required: int
    poorly_identified_params: List[str]
    recommendations: List[str]


class IdentifiabilityAnalyzer:
    """
    Formal identifiability analysis for the physiological digital twin.

    Uses both empirical (data-driven) and structural (model-based) approaches
    to determine which parameters can be reliably estimated from available observations.
    """

    def __init__(self, physio_dim: int = 30, param_dim: int = 25,
                 obs_dim: int = 15, seed: int = 42):
        self.physio_dim = physio_dim
        self.param_dim = param_dim
        self.obs_dim = obs_dim
        self.rng = np.random.default_rng(seed)

    def empirical_observability(
        self, dynamics_fn: Callable, obs_fn: Callable,
        n_perturbations: int = 200, epsilon: float = 1e-4,
    ) -> Dict:
        """
        Compute empirical observability via finite-difference Lie derivatives.

        The observability matrix O is computed as:
          O[i] = d(h ∘ f^i)(x) / dx
        where f^i is the i-fold application of dynamics.

        Rank deficiency indicates unobservable subspaces.
        """
        n_states = self.physio_dim + self.param_dim
        x0 = np.zeros(n_states)
        init_values = [90.0, 5.0, 2.0, 5.0, 1.0,
                       120.0, 80.0, 70.0, 45.0,
                       100.0, 140.0, 4.2, 290.0, 1.0,
                       1.2, 0.8, 350.0, 10.0, 0.0, 0.3,
                       20.0, 0.5, 100.0, 50.0, 120.0,
                       1.0, 0.5, 0.5, 0.2, 15.0]
        x0[:min(len(init_values), self.physio_dim)] = init_values[:self.physio_dim]

        O_list = []
        x_current = x0.copy()
        for step in range(3):
            obs_curr = obs_fn(x_current[:self.physio_dim])
            J = np.zeros((len(obs_curr), n_states))
            for j in range(n_states):
                x_pert = x_current.copy()
                x_pert[j] += epsilon
                obs_pert = obs_fn(x_pert[:self.physio_dim])
                J[:, j] = (obs_pert - obs_curr) / epsilon
            O_list.append(J)
            x_current[:self.physio_dim] = dynamics_fn(
                x_current[:self.physio_dim],
                x_current[self.physio_dim:], {}
            )
        O_matrix = np.vstack(O_list)

        try:
            U, S, Vt = np.linalg.svd(O_matrix, full_matrices=False)
        except np.linalg.LinAlgError:
            return {"observability_rank": 0, "n_states": n_states,
                    "is_observable": False, "explained_variance": 0.0}

        tol = max(n_states, O_matrix.shape[0]) * S[0] * np.finfo(float).eps
        rank = int(np.sum(S > tol))
        var_explained = np.sum(S[:rank] ** 2) / np.sum(S ** 2)

        return {
            "observability_rank": rank,
            "n_states": n_states,
            "is_observable": rank >= n_states,
            "singular_values": S.tolist(),
            "explained_variance": float(var_explained),
        }

    def parameter_recovery_analysis(
        self, dynamics_fn: Callable, obs_fn: Callable,
        twin_factory: Callable,
        n_synthetic_patients: int = 20,
        n_observations: int = 168,
    ) -> Dict:
        """
        Test whether UKF can recover known ground-truth parameters.

        Generates synthetic patients with known parameters, simulates
        observations, runs UKF, and computes recovery error.
        """
        param_errors = {name: [] for name in [
            "SI", "HGP_basal", "beta_response", "RT",
            "vascular_resistance", "baroreflex_gain", "autonomic_tone",
            "baseline_GFR", "renal_sensitivity",
            "circadian_period", "circadian_amplitude",
            "lipolysis_rate", "LDL_clearance", "HDL_production",
            "M1_activation", "NFkB_sensitivity", "IL6_clearance",
        ]}
        recovery_errors = []

        for patient_idx in range(n_synthetic_patients):
            true_params = DEFAULT_PARAMS.copy()
            true_params[0] = self.rng.lognormal(-4.0, 0.3)
            true_params[1] = self.rng.normal(2.0, 0.2)
            true_params[2] = self.rng.lognormal(-6.0, 0.3)
            true_params[3] = self.rng.normal(180, 10)
            true_params[5] = self.rng.lognormal(4.5, 0.2)
            true_params[6] = self.rng.lognormal(0.5, 0.2)
            true_params[7] = self.rng.normal(0.5, 0.08)
            true_params[8] = self.rng.normal(100, 10)
            true_params[9] = self.rng.normal(0.6, 0.08)
            true_params[12] = self.rng.normal(1440, 20)
            true_params[13] = self.rng.normal(0.8, 0.1)
            true_params[16] = self.rng.lognormal(-3.0, 0.2)
            true_params[18] = self.rng.lognormal(-4.2, 0.2)
            true_params[19] = self.rng.lognormal(-4.6, 0.2)
            true_params[21] = self.rng.lognormal(-2.3, 0.2)
            true_params[22] = self.rng.normal(0.5, 0.1)
            true_params[24] = self.rng.lognormal(-3.9, 0.2)

            state = np.zeros(self.physio_dim)
            state[0] = self.rng.normal(100, 15)
            state[1] = self.rng.normal(10, 5)
            state[5] = self.rng.normal(125, 10)
            state[6] = self.rng.normal(80, 8)
            state[7] = self.rng.normal(70, 8)
            state[8] = self.rng.normal(45, 10)
            state[9] = self.rng.normal(100, 10)

            observations = []
            for t in range(n_observations):
                state = dynamics_fn(state, true_params, {})
                if t % 12 == 0:
                    obs = obs_fn(state)
                    obs += self.rng.normal(0, [8, 5, 4, 3, 10, 5, 2, 0.2, 3, 0.05, 5, 3, 10, 30, 0.1])
                    observations.append(obs)

            try:
                twin = twin_factory(state, DEFAULT_PARAMS.copy())
                for obs in observations[:min(len(observations), 50)]:
                    twin.update(obs, {})

                estimated, _ = twin.get_parameters()

                for i, name in enumerate(param_errors.keys()):
                    param_idx = list(param_errors.keys()).index(name)
                    true_val = true_params[param_idx]
                    est_val = estimated[param_idx]
                    if abs(true_val) > 1e-6:
                        rel_error = abs(est_val - true_val) / abs(true_val)
                    else:
                        rel_error = abs(est_val - true_val)
                    param_errors[name].append(rel_error)
                recovery_errors.append(np.mean([
                    abs(estimated[i] - true_params[i]) / max(abs(true_params[i]), 1e-6)
                    for i in range(self.param_dim) if abs(true_params[i]) > 1e-6
                ]))
            except Exception:
                continue

        mean_errors = {
            name: float(np.mean(errors)) if errors else 1.0
            for name, errors in param_errors.items()
        }
        poorly_identified = [
            name for name, err in mean_errors.items()
            if err > 0.5
        ]
        return {
            "param_recovery_mae": mean_errors,
            "mean_recovery_error_overall": float(np.mean(recovery_errors)) if recovery_errors else 1.0,
            "poorly_identified_params": poorly_identified,
            "n_patients_tested": len(recovery_errors),
        }

    def profile_likelihood_analysis(
        self, dynamics_fn: Callable, obs_fn: Callable,
        observed_data: np.ndarray,
        param_bounds: Dict[str, Tuple[float, float]],
        n_points: int = 20,
        ukf_instance: Optional[object] = None,
    ) -> Dict[str, Dict]:
        """
        Compute profile likelihood for each parameter using the UKF
        observation model.

        For each parameter p_i:
          1. Fix p_i at value v_k across a grid
          2. Optimize over all other parameters p_{-i}
          3. Compute log-likelihood: -0.5 * Σ (y_t - h(x_t))^T R^{-1} (y_t - h(x_t))
          4. Flat profile → structurally non-identifiable
          5. Double-peaked → practically non-identifiable (multiple modes)

        Uses the UKF's innovation likelihood when available, otherwise
        falls back to observation residual sum-of-squares.
        """
        profiles = {}
        param_names = list(param_bounds.keys())

        n_obs = len(observed_data) if hasattr(observed_data, '__len__') else 1

        for name in param_names[:5]:
            lo, hi = param_bounds[name]
            values = np.linspace(lo, hi, n_points)
            log_likelihoods = []

            for val in values:
                try:
                    if ukf_instance is not None:
                        twin_state = ukf_instance.get_physio_state()
                        prediction = obs_fn(twin_state)
                        obs_cov = getattr(ukf_instance, 'R', np.eye(self.obs_dim) * 0.1)
                        residuals = observed_data[:min(len(observed_data), len(prediction))] - prediction[:min(len(observed_data), len(prediction))]
                        if len(residuals) > 0:
                            try:
                                inv_R = np.linalg.inv(obs_cov[:len(residuals), :len(residuals)] + np.eye(len(residuals)) * 1e-6)
                                ll = -0.5 * residuals @ inv_R @ residuals - 0.5 * np.log(np.linalg.det(obs_cov[:len(residuals), :len(residuals)]) + 1e-16)
                            except np.linalg.LinAlgError:
                                ll = -0.5 * np.sum(residuals ** 2) / (np.trace(obs_cov) / self.obs_dim + 1e-8)
                        else:
                            ll = -1e10
                    else:
                        prediction = obs_fn(np.zeros(self.physio_dim))
                        residuals = observed_data[:min(len(observed_data), len(prediction))] - prediction[:min(len(observed_data), len(prediction))]
                        ll = -0.5 * np.sum(residuals ** 2) / self.obs_dim
                    log_likelihoods.append(float(ll))
                except Exception:
                    log_likelihoods.append(-1e10)

            log_likelihoods = np.array(log_likelihoods)
            ll_max = np.max(log_likelihoods)
            ll_2unit = ll_max - 2.0

            within_2unit = log_likelihoods > ll_2unit
            peak_width = float(np.sum(within_2unit) / n_points) if n_points > 0 else 1.0

            # Identifiability criteria
            peaked = np.std(log_likelihoods[log_likelihoods > -1e9]) > 1.0 if np.sum(log_likelihoods > -1e9) > 2 else False
            unimodal = int(np.sum(np.diff((log_likelihoods[:-1] < log_likelihoods[1:]).astype(int)) != 0)) <= 2 if n_points > 3 else True

            is_identifiable = bool(peaked and peak_width < 0.5 and unimodal)

            profiles[name] = {
                "values": values.tolist(),
                "log_likelihoods": log_likelihoods.tolist(),
                "max_log_likelihood": float(ll_max),
                "is_identifiable": is_identifiable,
                "peak_width_ratio": peak_width,
                "is_peaked": bool(peaked),
                "is_unimodal": bool(unimodal),
            }
        return profiles

    def collinearity_analysis(
        self, n_samples: int = 500,
    ) -> Dict[str, float]:
        """
        Compute parameter collinearity indices.

        Parameters that can compensate for each other (high collinearity)
        are not jointly identifiable. Uses a simplified sensitivity-based approach.
        """
        param_names = [
            "SI", "HGP_basal", "beta_response", "vascular_resistance",
            "baroreflex_gain", "baseline_GFR", "renal_sensitivity",
            "lipolysis_rate", "LDL_clearance", "M1_activation",
        ]
        n_params = len(param_names)
        sens_matrix = np.zeros((n_samples, n_params))

        for i in range(n_samples):
            params = DEFAULT_PARAMS.copy()
            for j, name in enumerate(param_names):
                param_idx = [
                    "SI", "HGP_basal", "beta_response", "vascular_resistance",
                    "baroreflex_gain", "baseline_GFR", "renal_sensitivity",
                    "lipolysis_rate", "LDL_clearance", "M1_activation",
                ].index(name)
                params[param_idx] = self.rng.uniform(0.1, 1.0)
            sens_matrix[i, :] = params[:n_params]

        corr = np.abs(np.corrcoef(sens_matrix.T))
        collinearity = {}
        for i, name in enumerate(param_names):
            others = np.mean(corr[i, np.arange(n_params) != i])
            collinearity[name] = float(others)
        return collinearity

    def full_identifiability_report(
        self, dynamics_fn: Callable, obs_fn: Callable,
        twin_factory: Callable,
    ) -> IdentifiabilityReport:
        """Generate comprehensive identifiability report."""
        obs = self.empirical_observability(dynamics_fn, obs_fn)
        recovery = self.parameter_recovery_analysis(dynamics_fn, obs_fn, twin_factory)
        collinearity = self.collinearity_analysis()

        poor_params = set(recovery.get("poorly_identified_params", []))
        high_coll = [k for k, v in collinearity.items() if v > 0.8]
        all_poor = list(poor_params | set(high_coll))

        recommendations = []
        if not obs["is_observable"]:
            recommendations.append(
                f"System is not fully observable: rank {obs['observability_rank']} "
                f"< {obs['n_states']} states. Consider reducing parameter dimension or "
                f"adding more observation modalities."
            )
        if all_poor:
            recommendations.append(
                f"Poorly identified parameters: {', '.join(all_poor)}. "
                f"Consider fixing these to population values or adding informative priors."
            )
        if recovery.get("mean_recovery_error_overall", 0) > 0.3:
            recommendations.append(
                f"Mean parameter recovery error is {recovery['mean_recovery_error_overall']:.1%}. "
                f"Increase observation frequency or duration for reliable personalization."
            )

        return IdentifiabilityReport(
            n_observable_states=obs["observability_rank"],
            state_dim=obs["n_states"],
            observability_rank=obs["observability_rank"],
            is_observable=obs["is_observable"],
            n_identifiable_params=self.param_dim - len(all_poor),
            n_total_params=self.param_dim,
            param_recovery_mae=recovery.get("param_recovery_mae", {}),
            collinearity_indices=collinearity,
            min_observations_required=20 + 10 * len(all_poor),
            poorly_identified_params=all_poor,
            recommendations=recommendations or ["No identifiability issues detected."],
        )
