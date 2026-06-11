"""
Multi-Scale Stability Analysis.

Analyzes numerical stability of the coupled multi-scale ODE system.
Computes Jacobian eigenvalues, Lyapunov exponents, stiffness ratios,
and identifies unstable coupling topologies.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Callable


@dataclass
class StabilityReport:
    max_real_eigenvalue: float
    min_real_eigenvalue: float
    spectral_radius: float
    stiffness_ratio: float
    max_lyapunov_exponent: float
    is_stable: bool
    is_stiff: bool
    unstable_modes: List[str]
    n_positive_exponents: int


class StabilityAnalyzer:
    """
    Analyzes the numerical stability of the multi-scale ODE system.

    Uses:
      - Finite-difference Jacobian approximation
      - Power iteration for spectral radius
      - QR decomposition for Lyapunov spectrum
      - Stiffness ratio (max |Re(lambda)| / min |Re(lambda)|)
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def _finite_difference_jacobian(self, dynamics_fn: Callable,
                                     state: np.ndarray, params: np.ndarray,
                                     inputs: Optional[Dict] = None) -> np.ndarray:
        if inputs is None:
            inputs = {}
        n = len(state)
        J = np.zeros((n, n))
        f0 = dynamics_fn(state, params, inputs)
        for i in range(n):
            state_pert = state.copy()
            state_pert[i] += self.eps
            f_pert = dynamics_fn(state_pert, params, inputs)
            J[:, i] = (f_pert - f0) / self.eps
        return J

    def compute_eigenvalues(self, dynamics_fn: Callable,
                             state: np.ndarray, params: np.ndarray,
                             inputs: Optional[Dict] = None) -> Tuple[np.ndarray, float, float]:
        J = self._finite_difference_jacobian(dynamics_fn, state, params, inputs)
        try:
            eigenvalues = np.linalg.eigvals(J)
        except np.linalg.LinAlgError:
            eigenvalues = np.array([0j])
        real_parts = np.real(eigenvalues)
        max_real = float(np.max(real_parts))
        min_real = float(np.min(real_parts))
        return eigenvalues, max_real, min_real

    def spectral_radius(self, dynamics_fn: Callable,
                         state: np.ndarray, params: np.ndarray,
                         inputs: Optional[Dict] = None) -> float:
        J = self._finite_difference_jacobian(dynamics_fn, state, params, inputs)
        try:
            s = np.linalg.svd(J, compute_uv=False)
            return float(s[0])
        except np.linalg.LinAlgError:
            return float("inf")

    def stiffness_ratio(self, dynamics_fn: Callable,
                         state: np.ndarray, params: np.ndarray,
                         inputs: Optional[Dict] = None) -> float:
        eigenvalues, _, _ = self.compute_eigenvalues(dynamics_fn, state, params, inputs)
        real_abs = np.abs(np.real(eigenvalues))
        real_abs = real_abs[real_abs > 1e-10]
        if len(real_abs) < 2:
            return 1.0
        return float(np.max(real_abs) / np.min(real_abs))

    def lyapunov_exponents(self, dynamics_fn: Callable,
                            state: np.ndarray, params: np.ndarray,
                            n_steps: int = 100, dt: float = 0.1,
                            inputs: Optional[Dict] = None) -> np.ndarray:
        n = len(state)
        Q = np.eye(n)
        exponents = np.zeros(n)
        x = state.copy()
        for _ in range(n_steps):
            J = self._finite_difference_jacobian(dynamics_fn, x, params, inputs)
            M = J * dt
            Q_next = M @ Q
            try:
                Q, R = np.linalg.qr(Q_next)
            except np.linalg.LinAlgError:
                break
            exponents += np.log(np.abs(np.diag(R)) + 1e-15)
            x = dynamics_fn(x, params, inputs)
        exponents /= (n_steps * dt)
        return np.sort(exponents)[::-1]

    def analyze_subsystem_stability(self, subsystem_dynamics: Dict[str, Callable],
                                     full_state: np.ndarray, params: np.ndarray,
                                     offsets: Dict[str, int],
                                     dimensions: Dict[str, int]) -> Dict[str, StabilityReport]:
        reports = {}
        for name, dynamics_fn in subsystem_dynamics.items():
            offset = offsets.get(name, 0)
            dim = dimensions.get(name, 5)
            sub_state = full_state[offset:offset + dim]
            sub_params = params.copy()
            eigvals, max_real, min_real = self.compute_eigenvalues(dynamics_fn, sub_state, sub_params)
            sr = self.spectral_radius(dynamics_fn, sub_state, sub_params)
            stiff = self.stiffness_ratio(dynamics_fn, sub_state, sub_params)
            lyap = self.lyapunov_exponents(dynamics_fn, sub_state, sub_params, n_steps=50)
            n_pos = int(np.sum(lyap > 0.01))
            unstable = []
            if max_real > 0.01:
                unstable.append(f"{name}: positive real eigenvalue ({max_real:.4f})")
            if n_pos > 0:
                unstable.append(f"{name}: {n_pos} positive Lyapunov exponents")
            reports[name] = StabilityReport(
                max_real_eigenvalue=max_real,
                min_real_eigenvalue=min_real,
                spectral_radius=float(sr),
                stiffness_ratio=float(stiff),
                max_lyapunov_exponent=float(lyap[0]) if len(lyap) > 0 else 0.0,
                is_stable=max_real < 0.01 and n_pos == 0,
                is_stiff=stiff > 100,
                unstable_modes=unstable,
                n_positive_exponents=n_pos,
            )
        return reports

    def coupled_stability_analysis(self, full_dynamics_fn: Callable,
                                    full_state: np.ndarray, params: np.ndarray,
                                    inputs: Optional[Dict] = None,
                                    lyap_steps: int = 100) -> StabilityReport:
        J = self._finite_difference_jacobian(full_dynamics_fn, full_state, params, inputs)
        eigvals, max_real, min_real = self.compute_eigenvalues(full_dynamics_fn, full_state, params, inputs)
        sr = self.spectral_radius(full_dynamics_fn, full_state, params, inputs)
        stiff = self.stiffness_ratio(full_dynamics_fn, full_state, params, inputs)
        lyap = self.lyapunov_exponents(full_dynamics_fn, full_state, params, n_steps=lyap_steps, inputs=inputs)
        n_pos = int(np.sum(np.array(lyap) > 0.01)) if len(lyap) > 0 else 0
        J_off_diag = J.copy()
        np.fill_diagonal(J_off_diag, 0)
        coupling_strength = float(np.mean(np.abs(J_off_diag)))
        unstable = []
        if max_real > 0.01:
            unstable.append(f"Positive eigenvalue {max_real:.4f}")
        if n_pos > 0:
            unstable.append(f"{n_pos} positive Lyapunov exponents")
        if coupling_strength > 1.0:
            unstable.append(f"Strong coupling ({coupling_strength:.4f}) may cause instability")
        return StabilityReport(
            max_real_eigenvalue=max_real,
            min_real_eigenvalue=min_real,
            spectral_radius=float(sr),
            stiffness_ratio=float(stiff),
            max_lyapunov_exponent=float(lyap[0]) if len(lyap) > 0 else 0.0,
            is_stable=max_real < 0.01 and n_pos == 0 and coupling_strength < 2.0,
            is_stiff=stiff > 100,
            unstable_modes=unstable,
            n_positive_exponents=n_pos,
        )


def analyze_multi_scale_stability(dynamics_fn, state: np.ndarray,
                                   params: np.ndarray) -> Dict:
    analyzer = StabilityAnalyzer()
    full_report = analyzer.coupled_stability_analysis(dynamics_fn, state, params)
    return {
        "max_eigenvalue": full_report.max_real_eigenvalue,
        "spectral_radius": full_report.spectral_radius,
        "stiffness_ratio": full_report.stiffness_ratio,
        "max_lyapunov": full_report.max_lyapunov_exponent,
        "is_stable": full_report.is_stable,
        "is_stiff": full_report.is_stiff,
        "unstable_modes": full_report.unstable_modes,
    }
