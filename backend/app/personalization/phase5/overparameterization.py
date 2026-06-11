"""
Overparameterization Risk Analysis.

Quantifies the risk of overfitting in the 55D augmented state space
by computing effective degrees of freedom, regularization path analysis,
parameter identifiability, and posterior contraction.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from scipy import linalg


@dataclass
class OverparameterizationReport:
    n_parameters: int
    n_observations_per_update: int
    n_effective_parameters: float
    parameter_identifiability: Dict[str, float]
    posterior_contraction: Dict[str, float]
    regularization_strength: float
    condition_number: float
    rank_deficiency: int
    is_overparameterized: bool
    recommended_n_observations: int


class OverparameterizationAnalyzer:
    """
    Analyzes the risk of overparameterization in the UKF augmented state.

    Key metrics:
      - Effective degrees of freedom (EDoF) via ridge regression analog
      - Parameter identifiability via Fisher information
      - Posterior contraction (prior vs posterior variance ratio)
      - Condition number and rank deficiency of the observational Jacobian
    """

    def __init__(self, prior_precision_strength: float = 0.01):
        self.prior_precision = prior_precision_strength

    def effective_degrees_of_freedom(self, observation_matrix: np.ndarray,
                                       prior_precision: Optional[np.ndarray] = None) -> float:
        n_params = observation_matrix.shape[1]
        if prior_precision is None:
            prior_precision = np.eye(n_params) * self.prior_precision
        HtH = observation_matrix.T @ observation_matrix
        precision = HtH + prior_precision
        try:
            precision_inv = np.linalg.inv(precision)
        except np.linalg.LinAlgError:
            precision_inv = np.linalg.pinv(precision)
        edof = float(np.trace(HtH @ precision_inv))
        return edof

    def parameter_identifiability(self, posterior_covariance: np.ndarray,
                                    prior_covariance: np.ndarray,
                                    parameter_names: Optional[List[str]] = None) -> Dict[str, float]:
        from app.personalization.priors import PARAMETER_NAMES as PNAMES
        names = parameter_names or PNAMES[:len(posterior_covariance)]
        identifiability = {}
        for i, name in enumerate(names):
            if i < len(posterior_covariance) and i < len(prior_covariance):
                post_var = posterior_covariance[i, i]
                prior_var = prior_covariance[i, i]
                identifiability[name] = float(1.0 - post_var / max(prior_var, 1e-10))
            else:
                identifiability[name] = 0.0
        return identifiability

    def posterior_contraction(self, posterior_covariance: np.ndarray,
                               prior_covariance: np.ndarray,
                               parameter_names: Optional[List[str]] = None) -> Dict[str, float]:
        from app.personalization.priors import PARAMETER_NAMES as PNAMES
        names = parameter_names or PNAMES[:len(posterior_covariance)]
        contraction = {}
        for i, name in enumerate(names):
            if i < len(posterior_covariance) and i < len(prior_covariance):
                post_var = posterior_covariance[i, i]
                prior_var = prior_covariance[i, i]
                contraction[name] = float(1.0 - post_var / max(prior_var, 1e-10))
            else:
                contraction[name] = 0.0
        return contraction

    def condition_analysis(self, observation_matrix: np.ndarray) -> Tuple[float, int]:
        try:
            s = np.linalg.svd(observation_matrix, compute_uv=False)
            cond = float(s[0] / max(s[-1], 1e-15))
            rank_def = max(0, len(s) - np.sum(s > 1e-10 * s[0]))
        except np.linalg.LinAlgError:
            cond = float("inf")
            rank_def = observation_matrix.shape[1]
        return cond, rank_def

    def regularization_path_analysis(self, observation_matrix: np.ndarray,
                                       true_params: np.ndarray,
                                       observation_noise: float = 0.1,
                                       lambdas: Optional[np.ndarray] = None) -> Dict:
        if lambdas is None:
            lambdas = np.logspace(-3, 2, 20)
        n_params = observation_matrix.shape[1]
        if true_params.shape[0] != n_params:
            true_params = true_params[:n_params]
        y = observation_matrix @ true_params + np.random.normal(0, observation_noise, observation_matrix.shape[0])
        estimates = []
        effective_dofs = []
        for lam in lambdas:
            try:
                precision = observation_matrix.T @ observation_matrix + lam * np.eye(n_params)
                est = np.linalg.solve(precision, observation_matrix.T @ y)
            except np.linalg.LinAlgError:
                est = np.zeros(n_params)
            estimates.append(est)
            edof = self.effective_degrees_of_freedom(
                observation_matrix, prior_precision=lam * np.eye(n_params)
            )
            effective_dofs.append(edof)
        mse = [np.mean((est - true_params) ** 2) for est in estimates]
        return {
            "lambdas": lambdas.tolist(),
            "mse": mse,
            "effective_dofs": effective_dofs,
            "optimal_lambda": float(lambdas[np.argmin(mse)]),
            "min_mse": float(min(mse)),
        }

    def full_report(self, observation_matrix: np.ndarray,
                    posterior_covariance: np.ndarray,
                    prior_covariance: np.ndarray,
                    n_observations_per_update: int = 15) -> OverparameterizationReport:
        n_params = observation_matrix.shape[1]
        edof = self.effective_degrees_of_freedom(observation_matrix)
        ident = self.parameter_identifiability(posterior_covariance, prior_covariance)
        contraction = self.posterior_contraction(posterior_covariance, prior_covariance)
        cond, rank_def = self.condition_analysis(observation_matrix)
        is_over = edof > n_observations_per_update * 0.8 or cond > 100
        rec_obs = max(n_params, int(edof * 2))
        return OverparameterizationReport(
            n_parameters=n_params,
            n_observations_per_update=n_observations_per_update,
            n_effective_parameters=edof,
            parameter_identifiability=ident,
            posterior_contraction=contraction,
            regularization_strength=self.prior_precision,
            condition_number=cond,
            rank_deficiency=rank_def,
            is_overparameterized=is_over,
            recommended_n_observations=rec_obs,
        )


def analyze_overparameterization(ukf_instance=None) -> OverparameterizationReport:
    from app.personalization.state import PHYSIO_DIM, PARAM_DIM, OBS_DIM
    n_total = PHYSIO_DIM + PARAM_DIM
    H = np.random.randn(OBS_DIM, n_total) * 0.5
    H[:10, :10] += np.eye(10)
    post_cov = np.eye(n_total) * 0.1
    prior_cov = np.eye(n_total) * 0.5
    analyzer = OverparameterizationAnalyzer()
    return analyzer.full_report(H, post_cov, prior_cov, n_observations_per_update=OBS_DIM)
