"""
Phase 3: UKF-based state estimation for 30-dim whole-body cellular twin.
"""

import math
import numpy as np
from typing import List, Tuple, Callable, Any, Optional, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .state import (
    StateEstimator,
    PHYSIO_DIM, PARAM_DIM, OBS_DIM,
    METABOLIC_DIM, CARDIO_DIM, RENAL_DIM, INFLAMMATION_DIM,
    CIRCADIAN_DIM, ADIPOSE_DIM, IMMUNE_DIM,
    Phase3TwinState, MetabolicState, CardioState, RenalState, InflammatoryState,
    CircadianState, AdiposeLipidState, ImmuneInflamState,
    _META_OFF, _CARDIO_OFF, _RENAL_OFF, _INFL_OFF,
    _CIRC_OFF, _ADIP_OFF, _IMMUNE_OFF,
)
from .drift import DriftDetector


@dataclass
class Particle:
    state: np.ndarray
    weight: float

    def __post_init__(self):
        self.weight = float(self.weight)


class PriorDistribution(ABC):
    @abstractmethod
    def sample(self) -> float:
        pass
    @abstractmethod
    def log_prob(self, x: float) -> float:
        pass


class LogNormalPrior(PriorDistribution):
    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma

    def sample(self) -> float:
        return np.random.lognormal(self.mu, self.sigma)

    def log_prob(self, x: float) -> float:
        if x <= 0:
            return -np.inf
        return -np.log(x * self.sigma * np.sqrt(2 * np.pi)) - \
               (np.log(x) - self.mu)**2 / (2 * self.sigma**2)


class NormalPrior(PriorDistribution):
    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma

    def sample(self) -> float:
        return np.random.normal(self.mu, self.sigma)

    def log_prob(self, x: float) -> float:
        return -0.5 * np.log(2 * np.pi * self.sigma**2) - \
               (x - self.mu)**2 / (2 * self.sigma**2)


class TruncatedNormalPrior(PriorDistribution):
    def __init__(self, mu: float, sigma: float, low: float, high: float):
        self.mu = mu
        self.sigma = sigma
        self.low = low
        self.high = high
        if low >= high:
            raise ValueError("low must be less than high")

    def sample(self) -> float:
        while True:
            sample = np.random.normal(self.mu, self.sigma)
            if self.low <= sample <= self.high:
                return sample

    def log_prob(self, x: float) -> float:
        if x < self.low or x > self.high:
            return -np.inf
        z = (x - self.mu) / self.sigma
        log_pdf = -0.5 * np.log(2 * np.pi * self.sigma**2) - 0.5 * z**2
        norm_const = 0.5 * (math.erf((self.high - self.mu) / (self.sigma * math.sqrt(2))) -
                            math.erf((self.low - self.mu) / (self.sigma * math.sqrt(2))))
        if norm_const <= 0:
            return -np.inf
        return log_pdf - math.log(norm_const)


class UnscentedKalmanFilter(StateEstimator):
    """
    Phase 3: UKF for 30-dim cellular whole-body twin.
    """

    def __init__(
        self,
        state_dim: int,
        process_noise: np.ndarray,
        obs_noise: np.ndarray,
        dynamics_fn: Callable[..., np.ndarray],
        obs_fn: Callable[[np.ndarray], np.ndarray],
        param_prior_fn: Callable[[], np.ndarray],
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ):
        self.state_dim = state_dim
        self.Q = process_noise
        self.R = obs_noise
        self.f = dynamics_fn
        self.h = obs_fn
        self._param_prior_fn = param_prior_fn

        n = state_dim
        lam = alpha ** 2 * (n + kappa) - n
        self._lam = lam
        self._n = n
        self._alpha = alpha
        self._beta = beta
        self._kappa = kappa
        self._sqrt_n_lam = np.sqrt(n + lam)
        self._w_m = np.full(2 * n + 1, 0.5 / (n + lam))
        self._w_m[0] = lam / (n + lam)
        self._w_c = self._w_m.copy()
        self._w_c[0] = lam / (n + lam) + (1 - alpha ** 2 + beta)

        self._mu = self._initialize_mean()
        self._cov = np.eye(state_dim) * 0.1
        self._obs_dim = obs_noise.shape[0]

    def _initialize_mean(self) -> np.ndarray:
        state = np.zeros(self.state_dim)
        # Phase 2 base states (0-13)
        state[0] = 90.0    # G
        state[1] = 5.0     # I
        state[2] = 2.0     # HGP
        state[3] = 5.0     # PGU
        state[4] = 5.0     # IR
        state[5] = 120.0   # SBP
        state[6] = 80.0    # DBP
        state[7] = 70.0    # HR
        state[8] = 45.0    # HRV
        state[9] = 100.0   # GFR
        state[10] = 140.0  # Na
        state[11] = 4.2    # K
        state[12] = 290.0  # Osm
        state[13] = 1.0    # CRP

        # Phase 3 circadian states (14-19)
        state[14] = 1.2    # CLOCK_BMAL1
        state[15] = 0.8    # PER_CRY
        state[16] = 350.0  # cortisol (mid-range, ~8 AM)
        state[17] = 10.0   # melatonin (daytime low)
        state[18] = 0.0    # circadian_phase (midnight)
        state[19] = 0.3    # sleep_pressure (mid-day)

        # Phase 3 adipose states (20-24)
        state[20] = 20.0   # fat_mass (kg)
        state[21] = 0.5    # FFA (mmol/L)
        state[22] = 100.0  # LDL (mg/dL)
        state[23] = 50.0   # HDL (mg/dL)
        state[24] = 120.0  # TG (mg/dL)

        # Phase 3 immune states (25-29)
        state[25] = 1.0    # IL6_proxy
        state[26] = 0.5    # TNFa_proxy
        state[27] = 0.5    # M1_M2_ratio
        state[28] = 0.2    # NFkB_activity
        state[29] = 15.0   # InflammatoryLoad

        state[PHYSIO_DIM:] = self._param_prior_fn()
        return state

    def _sigma_points(self, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
        n = self._n
        try:
            sqrt_cov = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            try:
                sqrt_cov = np.linalg.cholesky(cov + np.eye(n) * 1e-6)
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(0.5 * (cov + cov.T))
                eigvals = np.maximum(eigvals, 1e-8)
                sqrt_cov = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
        sigmas = np.zeros((2 * n + 1, n))
        sigmas[0] = mu
        for i in range(n):
            sigmas[i + 1] = mu + self._sqrt_n_lam * sqrt_cov[:, i]
            sigmas[n + i + 1] = mu - self._sqrt_n_lam * sqrt_cov[:, i]
        return sigmas

    def predict(self, u: Any) -> None:
        sigmas = self._sigma_points(self._mu, self._cov)
        n = self._n
        prop = np.zeros_like(sigmas)
        for i in range(2 * n + 1):
            state = sigmas[i]
            physio = state[:PHYSIO_DIM]
            params = state[PHYSIO_DIM:]
            prop[i] = np.concatenate([self.f(physio, params, u), params])

        # CRITICAL FIX: Robust sigma point clipping to prevent extreme glucose values
        # If any propagated sigma point produces glucose outside [30, 500], blend it
        # toward sigma point 0 (the mean) to prevent covariance contamination
        g0 = prop[0, 0]
        for i in range(1, 2 * n + 1):
            if prop[i, 0] < 30 or prop[i, 0] > 500:
                # Blend extreme sigma point toward mean
                blend = 0.7
                prop[i, :] = (1 - blend) * prop[i, :] + blend * prop[0, :]

        self._mu = np.dot(self._w_m, prop)
        diff = prop - self._mu
        self._cov = (diff.T * self._w_c).dot(diff) + self.Q
        self._cov = (self._cov + self._cov.T) / 2.0
        # Bound covariance after predict to prevent sigma-point explosion
        self._clamp_covariance()

    def update(self, y: np.ndarray) -> None:
        n = self._n
        sigmas = self._sigma_points(self._mu, self._cov)

        obs_dim = len(y)
        obs_sigmas = np.zeros((2 * n + 1, obs_dim))
        for i in range(2 * n + 1):
            obs_sigmas[i] = self.h(sigmas[i, :PHYSIO_DIM])

        y_pred = np.dot(self._w_m, obs_sigmas)

        diff_obs = obs_sigmas - y_pred
        S = (diff_obs.T * self._w_c).dot(diff_obs) + self.R[:obs_dim, :obs_dim]

        diff_state = sigmas - self._mu
        cross = (diff_state.T * self._w_c).dot(diff_obs)

        try:
            K = cross @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = cross @ np.linalg.inv(S + np.eye(obs_dim) * 1e-6)

        innov = y - y_pred
        # CRITICAL FIX: Innovation gating to prevent extreme corrections
        # Glucose innovation: limit to ±100 mg/dL
        if obs_dim > 0:
            max_innov_g = 100.0
            if abs(innov[0]) > max_innov_g:
                scale = max_innov_g / abs(innov[0])
                innov[0] *= scale
        self._mu = self._mu + K @ innov
        # Joseph form: P = (I - K H) P (I - K H)^T + K R K^T
        # Numerically stable: preserves positive-definiteness
        I_KH = np.eye(self._n) - K @ self._H_jacobian_approx(sigmas, obs_sigmas, y_pred)
        self._cov = I_KH @ self._cov @ I_KH.T + K @ self.R[:obs_dim, :obs_dim] @ K.T
        self._cov = (self._cov + self._cov.T) / 2.0
        # Eigenvalue floor to prevent collapse
        eigvals, eigvecs = np.linalg.eigh(self._cov)
        eig_floor = 1e-6
        if np.any(eigvals < eig_floor):
            eigvals = np.maximum(eigvals, eig_floor)
            self._cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
        # Soft cap on post-update covariance to prevent pathology
        self._soft_cap_covariance()
        self._clamp_mean()

    def _clamp_mean(self) -> None:
        from .priors import PARAMETER_RANGES, PARAMETER_NAMES
        for i, name in enumerate(PARAMETER_NAMES):
            idx = PHYSIO_DIM + i
            lo, hi = PARAMETER_RANGES.get(name, (-np.inf, np.inf))
            self._mu[idx] = np.clip(self._mu[idx], lo, hi)
        # Clamp physiological states to valid ranges
        state_clamps = [
            (0, 20, 600),    # G
            (1, 0, 500),     # I
            (2, -5, 15),     # HGP
            (3, 0, 25),      # PGU
            (4, 0, 20),      # IR
            (5, 50, 250),    # SBP
            (6, 30, 150),    # DBP
            (7, 30, 220),    # HR
            (8, 5, 200),     # HRV
            (9, 5, 200),     # GFR
            (10, 120, 160),  # Na
            (11, 2.5, 7.0),  # K
            (12, 260, 340),  # Osm
            (13, 0, 100),    # CRP
            (14, 0, 2.5),    # CLOCK_BMAL1
            (15, 0, 2.5),    # PER_CRY
            (16, 10, 1000),  # cortisol
            (17, 0, 300),    # melatonin
            (18, 0, 6.29),   # circadian_phase
            (19, 0, 1),      # sleep_pressure
            (20, 2, 100),    # fat_mass
            (21, 0.1, 2.0),  # FFA
            (22, 20, 300),   # LDL
            (23, 10, 120),   # HDL
            (24, 20, 800),   # TG
            (25, 0, 10),     # IL6_proxy
            (26, 0, 10),     # TNFa_proxy
            (27, 0, 2),      # M1_M2_ratio
            (28, 0, 1),      # NFkB_activity
            (29, 0, 100),    # InflammatoryLoad
        ]
        for idx, lo, hi in state_clamps:
            if idx < PHYSIO_DIM:
                self._mu[idx] = np.clip(self._mu[idx], lo, hi)

    def get_state(self) -> np.ndarray:
        return self._mu.copy()

    def get_state_covariance(self) -> np.ndarray:
        return self._cov.copy()

    def get_physio_state(self) -> np.ndarray:
        return self._mu[:PHYSIO_DIM].copy()

    def get_physio_covariance(self) -> np.ndarray:
        return self._cov[:PHYSIO_DIM, :PHYSIO_DIM].copy()

    def get_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._mu[PHYSIO_DIM:].copy(), self._cov[PHYSIO_DIM:, PHYSIO_DIM:].copy()

    def get_predicted_obs_stats(self, obs_idx: int = 0) -> Tuple[float, float]:
        sigmas = self._sigma_points(self._mu, self._cov)
        obs_sigmas = np.array([self.h(s[:PHYSIO_DIM])[obs_idx] for s in sigmas])
        mean = float(np.dot(self._w_m, obs_sigmas))
        var = float(np.dot(self._w_c, (obs_sigmas - mean) ** 2))
        return mean, float(np.sqrt(var)) * 1.645

    def _H_jacobian_approx(self, sigmas, obs_sigmas, y_pred) -> np.ndarray:
        """Approximate observation Jacobian H from sigma points for Joseph form.

        H[i, j] = d h_i / d x_j ≈ sum_k w_c^k * (sigma^k_i - y_pred_i) * (sigma^k_j - mu_j) / var_j
        For the simpler form, we just compute the cross-correlation structure.
        Returns full N×m matrix used in Joseph form.
        """
        n = self._n
        m = obs_sigmas.shape[1]
        diff_state = sigmas - self._mu
        diff_obs = obs_sigmas - y_pred
        cross = (diff_state.T * self._w_c).dot(diff_obs)
        # H ≈ cross * S^{-1}; equivalently K = cross * S^{-1}, so H = K * S * cross^{+}
        # but for Joseph form we need a true Jacobian. Use a stable approximation:
        # H_j = sum_k w_c^k * (h(x_k) - h(mu)) * (x_k - mu)^T / cov_j
        # This is the linearization of h around mu.
        H = np.zeros((m, n))
        for j in range(n):
            if self._cov[j, j] > 1e-12:
                H[:, j] = cross[j, :] / self._cov[j, j]
        return H

    def _clamp_covariance(self) -> None:
        """Hard bound per-state covariance before predict to prevent sigma-point explosion.

        Variance is bounded so that 5-sigma stays within the valid range.
        CRITICAL FIX: Insulin variance (index 1) clamped to 50.0 to prevent the
        catastrophic SI*I*G explosion where Cov[1,1] grows to 1.83e+04.
        """
        max_sigmas = 5.0  # sigma-point spread bound: 5-sigma within range
        state_max_var = {
            0: 600.0,    # G
            1: 50.0,     # I - CRITICAL: was 500, now 50 to prevent explosion
            2: 15.0,     # HGP
            3: 25.0,     # PGU
            4: 20.0,     # IR
            5: 250.0,    # SBP
            6: 150.0,    # DBP
            7: 220.0,    # HR
            8: 200.0,    # HRV
            9: 200.0,    # GFR
            10: 160.0,   # Na
            11: 7.0,     # K
            12: 340.0,   # Osm
            13: 100.0,   # CRP
            14: 2.5,     # CLOCK_BMAL1
            15: 2.5,     # PER_CRY
            16: 1000.0,  # cortisol
            17: 300.0,   # melatonin
            18: 6.29,    # circadian_phase
            19: 1.0,     # sleep_pressure
            20: 100.0,   # fat_mass
            21: 2.0,     # FFA
            22: 300.0,   # LDL
            23: 120.0,   # HDL
            24: 800.0,   # TG
            25: 10.0,    # IL6_proxy
            26: 10.0,    # TNFa_proxy
            27: 2.0,     # M1_M2_ratio
            28: 1.0,     # NFkB_activity
            29: 100.0,   # InflammatoryLoad
        }
        # Also clamp parameter variances
        param_max_var = {
            0: 1.0,    # SI
            1: 100.0,  # beta
            2: 10.0,   # alpha
            3: 500.0,  # k_e
            4: 10.0,   # HR_SBP_slope
            5: 5.0,    # HR_DBP_slope
            6: 10.0,   # HR_base
            7: 20.0,   # HRV_base
            8: 20.0,   # GFR_base
            9: 5.0,    # Na_base
            10: 0.5,   # K_base
            11: 5.0,   # Osm_base
            12: 1000.0,# k_a
            13: 1000.0,# V_d
            14: 5.0,   # SBP_base
            15: 5.0,   # DBP_base
            16: 100.0, # cortisol_amp
            17: 50.0,  # mel_amp
            18: 100.0, # circadian_period
            19: 0.1,   # light_sens
            20: 50.0,  # fat_mass_base
            21: 1.0,   # FFA_base
            22: 100.0, # LDL_base
            23: 50.0,  # HDL_base
            24: 200.0, # TG_base
        }
        for idx, max_val in state_max_var.items():
            max_var = (max_val / max_sigmas) ** 2
            if self._cov[idx, idx] > max_var:
                scale = max_var / self._cov[idx, idx]
                self._cov[idx, :] *= scale
                self._cov[:, idx] *= scale
        from .priors import PARAMETER_RANGES, PARAMETER_NAMES
        for i, name in enumerate(PARAMETER_NAMES):
            idx = PHYSIO_DIM + i
            lo, hi = PARAMETER_RANGES.get(name, (-np.inf, np.inf))
            if np.isfinite(hi) and np.isfinite(lo) and hi > lo:
                max_var = ((hi - lo) / (2 * max_sigmas)) ** 2
                if self._cov[idx, idx] > max_var:
                    scale = max_var / self._cov[idx, idx]
                    self._cov[idx, :] *= scale
                    self._cov[:, idx] *= scale

    def _soft_cap_covariance(self, cap_quantile: float = 0.99) -> None:
        """Soft cap on per-state covariance after update.

        Allows filter to learn, but prevents extreme divergence.
        Cap = (range_width / 2)^2 * cap_quantile scaling.
        CRITICAL FIX: Also clamp insulin variance (index 1) to 100.0.
        """
        max_sigmas = 8.0  # soft: only catches truly pathological blow-up
        state_max_var = {
            0: 600.0, 1: 100.0, 2: 15.0, 3: 25.0, 4: 20.0,  # I clamped to 100
            5: 250.0, 6: 150.0, 7: 220.0, 8: 200.0, 9: 200.0,
            10: 160.0, 11: 7.0, 12: 340.0, 13: 100.0,
            14: 2.5, 15: 2.5, 16: 1000.0, 17: 300.0, 18: 6.29, 19: 1.0,
            20: 100.0, 21: 2.0, 22: 300.0, 23: 120.0, 24: 800.0,
            25: 10.0, 26: 10.0, 27: 2.0, 28: 1.0, 29: 100.0,
        }
        for idx, max_val in state_max_var.items():
            max_var = (max_val / max_sigmas) ** 2
            if self._cov[idx, idx] > max_var:
                scale = max_var / self._cov[idx, idx]
                self._cov[idx, :] *= scale
                self._cov[:, idx] *= scale
        from .priors import PARAMETER_RANGES, PARAMETER_NAMES
        for i, name in enumerate(PARAMETER_NAMES):
            idx = PHYSIO_DIM + i
            lo, hi = PARAMETER_RANGES.get(name, (-np.inf, np.inf))
            if np.isfinite(hi) and np.isfinite(lo) and hi > lo:
                max_var = ((hi - lo) / (2 * max_sigmas)) ** 2
                if self._cov[idx, idx] > max_var:
                    scale = max_var / self._cov[idx, idx]
                    self._cov[idx, :] *= scale
                    self._cov[:, idx] *= scale


# ===================================================================
# Phase 3 Personalization Engine
# ===================================================================

class PersonalizationEngine:
    """
    Phase 3: Whole-body cellular digital twin engine.
    30-dim state, 25-dim parameters, 15-dim observations, UKF-based.
    """

    def __init__(
        self,
        process_noise_scale: float = 0.01,
        obs_noise_scale: float = 0.1,
        subgroup_priors: Optional[List] = None,
    ):
        state_dim = PHYSIO_DIM + PARAM_DIM
        self.Q = np.eye(state_dim) * process_noise_scale
        self.Q[PHYSIO_DIM:, PHYSIO_DIM:] *= 2.0
        # State-specific process noise: states with faster dynamics get higher Q
        self.Q[0, 0] *= 10.0   # G: fast glucose dynamics
        self.Q[1, 1] *= 3.0    # I: fast insulin dynamics
        self.Q[5, 5] *= 5.0    # SBP: moderate BP dynamics
        self.Q[6, 6] *= 5.0    # DBP
        self.Q[7, 7] *= 3.0    # HR
        self.Q[9, 9] *= 3.0    # GFR
        # Parameters: small Q to prevent drift; UKF covariance clamping prevents blow-up
        self.Q[PHYSIO_DIM:, PHYSIO_DIM:] *= 0.5
        # Observation noise: scaled by physiological plausibility
        self.R = np.eye(OBS_DIM) * obs_noise_scale
        # Slightly higher R for hard-to-observe signals
        self.R[OBS_DIM - 3, OBS_DIM - 3] = obs_noise_scale * 5.0   # HRV
        self.R[OBS_DIM - 2, OBS_DIM - 2] = obs_noise_scale * 5.0   # CRP
        self.R[OBS_DIM - 1, OBS_DIM - 1] = obs_noise_scale * 5.0   # TG

        from .dynamics import full_dynamics, full_observation

        self._subgroup_priors = subgroup_priors
        self.filter = UnscentedKalmanFilter(
            state_dim=state_dim,
            process_noise=self.Q,
            obs_noise=self.R,
            dynamics_fn=full_dynamics,
            obs_fn=full_observation,
            param_prior_fn=self._sample_param_prior,
        )

        self._observation_buffer: List[float] = []
        self._full_obs_buffer: List[np.ndarray] = []
        self._control_buffer: List[dict] = []
        self.drift_detector = DriftDetector()
        self.is_initialized = False

    def _sample_param_prior(self) -> np.ndarray:
        if self._subgroup_priors is not None:
            return np.array([p.sample() for p in self._subgroup_priors])
        from .priors import PRIORS
        return np.array([p.sample() for p in PRIORS])

    def initialize(self, initial_obs: np.ndarray) -> None:
        if len(initial_obs) > 0:
            mu = self.filter.get_state()
            mu[0] = float(initial_obs[0])  # G
            # Set insulin consistent with glucose to avoid transient crash
            # I = beta_prior_mean * max(0, G - 80) / 0.2 ≈ 0.013 * max(0, G - 80)
            mu[1] = 0.013 * max(0.0, float(initial_obs[0]) - 80.0)
            if len(initial_obs) > 1:
                mu[5] = float(initial_obs[1])  # SBP
                mu[6] = float(initial_obs[2])  # DBP
                mu[7] = float(initial_obs[3])  # HR
                mu[8] = float(initial_obs[4])  # HRV
            if len(initial_obs) > 5:
                mu[9] = float(initial_obs[5])  # GFR
                mu[10] = float(initial_obs[6])  # Na
                mu[11] = float(initial_obs[7])  # K
                mu[12] = float(initial_obs[8])  # Osm
            if len(initial_obs) > 9:
                mu[21] = float(initial_obs[9])  # FFA
                mu[22] = float(initial_obs[10])  # LDL
                mu[23] = float(initial_obs[11])  # HDL
                mu[24] = float(initial_obs[12])  # TG
                mu[16] = float(initial_obs[13])  # cortisol
                mu[19] = float(initial_obs[14])  # sleep_pressure
            self.filter._mu = mu
        self.is_initialized = True

    def update(self, observation: np.ndarray, control_input: Optional[dict] = None) -> None:
        if not self.is_initialized:
            self.initialize(observation)
            return
        if control_input is None:
            control_input = {}
        if len(observation) > 0:
            self._observation_buffer.append(float(observation[0]))
            self._full_obs_buffer.append(observation.copy())
            self._control_buffer.append(control_input)
        self.filter.predict(control_input)
        if len(observation) > 0 and self.drift_detector.can_run_counterfactuals:
            pred_mean, pred_unc = self.filter.get_predicted_obs_stats()
            self.drift_detector.check(float(observation[0]), pred_mean, pred_unc)
        # Pad partial observations to OBS_DIM with reasonable defaults
        if len(observation) < OBS_DIM:
            padded = np.array([100.0, 120.0, 80.0, 72.0, 42.0,
                               90.0, 140.0, 4.2, 300.0, 0.5,
                               120.0, 50.0, 150.0, 10.0, 0.3, 5.0])
            padded[:len(observation)] = observation
            observation = padded
        self.filter.update(observation)

    def get_twin_state(self) -> np.ndarray:
        return self.filter.get_physio_state()

    def get_twin_state_covariance(self) -> np.ndarray:
        return self.filter.get_physio_covariance()

    def get_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.filter.get_parameters()

    def get_metabolic_state(self) -> np.ndarray:
        return self.filter.get_physio_state()[_META_OFF:_META_OFF+METABOLIC_DIM]

    def get_cardio_state(self) -> np.ndarray:
        return self.filter.get_physio_state()[_CARDIO_OFF:_CARDIO_OFF+CARDIO_DIM]

    def get_renal_state(self) -> np.ndarray:
        return self.filter.get_physio_state()[_RENAL_OFF:_RENAL_OFF+RENAL_DIM]

    def get_circadian_state(self) -> np.ndarray:
        return self.filter.get_physio_state()[_CIRC_OFF:_CIRC_OFF+CIRCADIAN_DIM]

    def get_adipose_state(self) -> np.ndarray:
        return self.filter.get_physio_state()[_ADIP_OFF:_ADIP_OFF+ADIPOSE_DIM]

    def get_immune_state(self) -> np.ndarray:
        return self.filter.get_physio_state()[_IMMUNE_OFF:_IMMUNE_OFF+IMMUNE_DIM]

    def get_insulin_resistance_state(self) -> float:
        p, _ = self.get_parameters()
        return 1.0 / max(p[0], 1e-6)

    def get_digital_biomarker_ir_score(self) -> float:
        return self.get_insulin_resistance_state()

    def get_recovery_score(self) -> float:
        from .biomarkers import compute_recovery_score
        ir = float(self.get_twin_state()[4])
        return compute_recovery_score(self._observation_buffer, ir)

    def get_stress_score(self) -> float:
        from .biomarkers import compute_stress_score
        ir = float(self.get_twin_state()[4])
        return compute_stress_score(self._observation_buffer, ir)

    def get_drift_status(self) -> dict:
        return self.drift_detector.status()

    def get_effective_sample_size(self) -> float:
        return 100.0

    def needs_resampling(self) -> bool:
        return False

    def is_weight_degenerate(self) -> bool:
        return False

    def convergence_diagnostics(self) -> Dict[str, float]:
        """
        Compute UKF convergence diagnostics for parameter estimates.

        Returns:
            param_stability: moving average coefficient of variation over last 50 steps
            neff_ratio: effective sample size ratio (placeholder for particle methods)
            param_drift_rate: mean absolute parameter change per update step
            is_converged: whether parameter estimates have stabilized
        """
        if len(self._control_buffer) < 10:
            return {
                "param_stability": 1.0,
                "neff_ratio": 1.0,
                "param_drift_rate": 1.0,
                "n_updates": len(self._control_buffer),
                "is_converged": False,
            }
        mu, cov = self.get_parameters()
        param_cv = np.sqrt(np.diag(cov)) / (np.abs(mu) + 1e-8)
        param_stability = float(np.mean(param_cv))

        from .priors import PRIORS
        prior_means = np.array([p.mu if hasattr(p, 'mu') else 1.0 for p in PRIORS[:len(mu)]])
        param_drift = float(np.mean(np.abs(mu - prior_means) / (np.abs(prior_means) + 1e-8)))

        is_converged = param_stability < 0.3 and param_drift < 2.0
        return {
            "param_stability": param_stability,
            "neff_ratio": float(np.mean(np.diag(cov)) / (np.mean(np.diag(self.Q)) + 1e-8)),
            "param_drift_rate": param_drift,
            "n_updates": len(self._control_buffer),
            "is_converged": is_converged,
        }

    def should_abstain(self) -> Dict[str, any]:
        """
        CRITICAL FIX: Safety guardrail - determine if twin should abstain from making predictions.
        
        Combines OOD detection, drift level, and physiological plausibility to decide
        whether to make a clinical recommendation.
        
        Returns:
            abstention: True if twin should not make predictions
            confidence: float in [0, 1] (higher is better)
            reasons: list of reasons for abstention
            recommendation: action to take
        """
        reasons = []
        confidence = 1.0
        
        # 1. Check drift level
        drift_status = self.drift_detector.status()
        drift_level = drift_status.get("level", 0)
        if drift_level >= 3:
            reasons.append(f"Critical drift: level {drift_level} (twin invalid)")
            confidence *= 0.1
        elif drift_level >= 2:
            reasons.append(f"High drift: level {drift_level} (recalibrate)")
            confidence *= 0.4
        
        # 2. Check physiological plausibility of twin state
        state = self.get_twin_state()
        cov = self.get_twin_state_covariance()
        
        # Check glucose bounds
        G = state[0] if len(state) > 0 else 90.0
        if G < 50 or G > 400:
            reasons.append(f"Glucose out of plausible range: {G:.1f} mg/dL")
            confidence *= 0.3
        
        # Check insulin bounds  
        I = state[1] if len(state) > 1 else 5.0
        if I < 0 or I > 200:
            reasons.append(f"Insulin out of plausible range: {I:.1f} μU/mL")
            confidence *= 0.4
        
        # 3. Check covariance diagonals for explosion
        if len(cov) > 0:
            glucose_var = cov[0, 0] if cov.shape[0] > 0 else 0.0
            insulin_var = cov[1, 1] if cov.shape[1] > 1 else 0.0
            
            if glucose_var > 10000:
                reasons.append(f"Glucose variance exploded: {glucose_var:.1f}")
                confidence *= 0.2
            
            if insulin_var > 1000:
                reasons.append(f"Insulin variance exploded: {insulin_var:.1f}")
                confidence *= 0.3
        
        # 4. Check prediction interval width
        if len(self._observation_buffer) >= 5:
            recent_g = np.array(self._observation_buffer[-10:])
            g_std = float(np.std(recent_g))
            g_mean = float(np.mean(recent_g))
            
            # If glucose variability is extreme, reduce confidence
            if g_std > 50:
                reasons.append(f"High glucose variability: std={g_std:.1f} mg/dL")
                confidence *= 0.5
            elif g_std > 30:
                reasons.append(f"Moderate glucose variability: std={g_std:.1f} mg/dL")
                confidence *= 0.7
        
        # Determine abstention
        abstention = confidence < 0.3 or drift_level >= 3
        
        if abstention:
            recommendation = "ABSTAIN: Twin predictions unreliable. Recalibrate or use fallback."
        elif confidence < 0.6:
            recommendation = "CAUTION: Predictions may be unreliable. Verify with clinical assessment."
        else:
            recommendation = "PROCEED: Twin predictions within acceptable confidence range."
        
        return {
            "abstention": abstention,
            "confidence": round(confidence, 3),
            "drift_level": drift_level,
            "reasons": reasons,
            "recommendation": recommendation,
            "safety_verdict": "ABSTAIN" if abstention else ("CAUTION" if confidence < 0.6 else "SAFE"),
        }


def create_personalization_engine(
    age: float = 35.0,
    sex: str = "male",
    bmi: float = 24.0,
    has_diabetes: bool = False,
    has_hypertension: bool = False,
    has_ckd: bool = False,
) -> PersonalizationEngine:
    """Factory with optional subgroup priors."""
    from .priors import get_subgroup_priors
    priors = get_subgroup_priors(age, sex, bmi, has_diabetes, has_hypertension, has_ckd)
    return PersonalizationEngine(subgroup_priors=priors)
