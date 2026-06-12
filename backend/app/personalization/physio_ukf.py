"""
Physio-only UKF for 30-dimensional state.

The original UKF (core.py) tracks a 55-dim augmented state (30 physio +
25 params) which is structurally underdetermined by 15-dim observations.
This module provides a 30-dim-only UKF that is well-determined.

Parameters are handled separately by the dual engine.
"""

import math
import logging
import numpy as np
from typing import Callable, Any, Optional, Tuple

from .state import PHYSIO_DIM, OBS_DIM

logger = logging.getLogger(__name__)


# Physiological bounds per state index: (index, low, high)
PHYSIO_CLAMPS = [
    (0, 20, 600), (1, 0, 500), (2, -5, 15), (3, 0, 25), (4, 0, 20),
    (5, 50, 250), (6, 30, 150), (7, 30, 220), (8, 5, 200), (9, 5, 200),
    (10, 120, 160), (11, 2.5, 7.0), (12, 260, 340), (13, 0, 100),
    (14, 0, 2.5), (15, 0, 2.5), (16, 10, 1000), (17, 0, 300),
    (18, 0, 6.29), (19, 0, 1), (20, 2, 100), (21, 0.1, 2.0),
    (22, 20, 300), (23, 10, 120), (24, 20, 800), (25, 0, 10),
    (26, 0, 10), (27, 0, 2), (28, 0, 1), (29, 0, 100),
]

# Map state index -> (low, high)
_CLAMP_MAP = {idx: (lo, hi) for idx, lo, hi in PHYSIO_CLAMPS}


class PhysioOnlyUKF:
    """30-dim UKF for state estimation only. Parameters are passed externally."""

    def __init__(
        self,
        dynamics_fn: Callable,
        obs_fn: Callable,
        params_fn: Callable[[], np.ndarray],
        process_noise_scale: float = 5.0,
        obs_noise_scale: float = 50.0,
    ):
        self._n = PHYSIO_DIM
        self._f = dynamics_fn
        self._h = obs_fn
        self._params_fn = params_fn
        # Process noise
        self.Q = np.eye(PHYSIO_DIM) * process_noise_scale
        self.Q[0, 0] *= 10.0   # glucose
        self.Q[1, 1] *= 1.0    # insulin
        self.Q[5, 5] *= 5.0
        self.Q[6, 6] *= 5.0
        self.Q[7, 7] *= 3.0
        self.Q[9, 9] *= 3.0
        # Observation noise
        self.R = np.eye(OBS_DIM) * obs_noise_scale
        self.R[OBS_DIM-3, OBS_DIM-3] = obs_noise_scale * 5.0
        self.R[OBS_DIM-2, OBS_DIM-2] = obs_noise_scale * 5.0
        self.R[OBS_DIM-1, OBS_DIM-1] = obs_noise_scale * 5.0
        # UKF parameters
        alpha, beta, kappa = 1e-3, 2.0, 0.0
        n = PHYSIO_DIM
        lam = alpha**2 * (n + kappa) - n
        self._lam = lam
        self._sqrt_n_lam = np.sqrt(n + lam)
        self._w_m = np.full(2 * n + 1, 0.5 / (n + lam))
        self._w_m[0] = lam / (n + lam)
        self._w_c = self._w_m.copy()
        self._w_c[0] = lam / (n + lam) + (1 - alpha**2 + beta)
        # Initialize state
        self._mu = self._initialize_mean()
        self._cov = np.eye(PHYSIO_DIM) * 0.1

    def _initialize_mean(self) -> np.ndarray:
        state = np.zeros(PHYSIO_DIM)
        state[0] = 100.0
        state[1] = 5.0
        state[2] = 2.0
        state[3] = 5.0
        state[4] = 5.0
        state[5] = 120.0
        state[6] = 80.0
        state[7] = 70.0
        state[8] = 45.0
        state[9] = 100.0
        state[10] = 140.0
        state[11] = 4.2
        state[12] = 290.0
        state[13] = 1.0
        state[14] = 1.2
        state[15] = 0.8
        state[16] = 350.0
        state[17] = 10.0
        state[18] = 0.0
        state[19] = 0.3
        state[20] = 20.0
        state[21] = 0.5
        state[22] = 100.0
        state[23] = 50.0
        state[24] = 120.0
        state[25] = 1.0
        state[26] = 0.5
        state[27] = 0.5
        state[28] = 0.2
        state[29] = 15.0
        return state

    @staticmethod
    def _ensure_positive_definite(cov: np.ndarray, min_eigval: float = 1e-4) -> np.ndarray:
        """Force a covariance matrix to be positive definite via eigenvalue decomposition."""
        cov = (cov + cov.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, min_eigval)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T

    def _sigma_points(self, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
        n = self._n
        cov = self._ensure_positive_definite(cov)
        try:
            sqrt_cov = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            cov = cov + np.eye(n) * max(abs(np.trace(cov)) / n * 0.01, 1e-3)
            cov = self._ensure_positive_definite(cov)
            sqrt_cov = np.linalg.cholesky(cov)
        sigmas = np.zeros((2 * n + 1, n))
        sigmas[0] = mu
        for i in range(n):
            sigmas[i + 1] = mu + self._sqrt_n_lam * sqrt_cov[:, i]
            sigmas[n + i + 1] = mu - self._sqrt_n_lam * sqrt_cov[:, i]
        return sigmas

    def _clamp_covariance(self) -> None:
        """Soft cap on covariance diagonals, preserving positive definiteness."""
        max_vars = {
            0: 3000.0, 1: 50.0, 5: 2000.0, 6: 1000.0, 7: 1000.0,
        }
        diag = np.diag(self._cov)
        scales = np.ones(len(diag))
        for idx, max_var in max_vars.items():
            if diag[idx] > max_var:
                scales[idx] = max_var / diag[idx]
        scale_mat = np.outer(scales, scales)
        self._cov = self._cov * scale_mat
        self._cov = (self._cov + self._cov.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(self._cov)
        eigvals = np.maximum(eigvals, 1e-6)
        self._cov = eigvecs @ np.diag(eigvals) @ eigvecs.T

    def _clip_state(self, state: np.ndarray) -> np.ndarray:
        for idx, lo, hi in PHYSIO_CLAMPS:
            state[idx] = np.clip(state[idx], lo, hi)
        return state

    def predict(self, u: Any = None) -> None:
        if u is None:
            u = {}
        params = self._params_fn()
        sigmas = self._sigma_points(self._mu, self._cov)

        # Propagate sigma points through dynamics
        prop = np.zeros_like(sigmas)
        for i in range(2 * self._n + 1):
            s = self._f(sigmas[i], params, u)
            prop[i] = self._clip_state(s)

        # Robust mean: sigma point 0 is the most reliable (from mean state)
        # If other sigma points produce extreme glucose, pull them toward
        # sigma point 0 to prevent covariance contamination
        g0 = prop[0, 0]
        for i in range(1, 2 * self._n + 1):
            if prop[i, 0] < 30 or prop[i, 0] > 500:
                blend = 0.8
                prop[i, :] = (1 - blend) * prop[i, :] + blend * prop[0, :]

        self._mu = np.dot(self._w_m, prop)
        diff = prop - self._mu
        self._cov = (diff.T * self._w_c).dot(diff) + self.Q
        self._cov = self._ensure_positive_definite(self._cov)
        self._clamp_covariance()

    def update(self, y: np.ndarray) -> None:
        n = self._n
        sigmas = self._sigma_points(self._mu, self._cov)
        obs_dim = len(y)

        # Observation sigma points
        obs_sigmas = np.zeros((2 * n + 1, obs_dim))
        for i in range(2 * n + 1):
            obs_sigmas[i] = self._h(sigmas[i])

        y_pred = np.dot(self._w_m, obs_sigmas)
        diff_obs = obs_sigmas - y_pred
        S = (diff_obs.T * self._w_c).dot(diff_obs) + self.R[:obs_dim, :obs_dim]
        diff_state = sigmas - self._mu
        cross = (diff_state.T * self._w_c).dot(diff_obs)
        K = cross @ np.linalg.inv(S)
        innov = y - y_pred

        # Innov gate: scale down extreme innovations for glucose
        innov_g = innov[0]
        max_innov = 100.0
        if abs(innov_g) > max_innov:
            scale = max_innov / abs(innov_g)
            innov[0] *= scale
        innov[0] = np.clip(innov[0], -max_innov, max_innov)

        self._mu = self._mu + K @ innov

        # Joseph form covariance update
        H = np.zeros((obs_dim, n))
        for j in range(n):
            if self._cov[j, j] > 1e-12:
                H[:, j] = cross[j, :] / self._cov[j, j]
        I_KH = np.eye(n) - K @ H
        self._cov = I_KH @ self._cov @ I_KH.T + K @ self.R[:obs_dim, :obs_dim] @ K.T
        self._cov = self._ensure_positive_definite(self._cov)
        self._clamp_covariance()

        # Clamp mean to physiological ranges
        self._mu = self._clip_state(self._mu)

    def get_state(self) -> np.ndarray:
        return self._mu.copy()

    def get_covariance(self) -> np.ndarray:
        return self._cov.copy()

    def get_predicted_obs_stats(self, obs_idx: int = 0) -> Tuple[float, float]:
        sigmas = self._sigma_points(self._mu, self._cov)
        obs_sigmas = np.array([self._h(s)[obs_idx] for s in sigmas])
        mean = float(np.dot(self._w_m, obs_sigmas))
        var = float(np.dot(self._w_c, (obs_sigmas - mean) ** 2))
        return mean, float(np.sqrt(max(var, 1e-6)))
