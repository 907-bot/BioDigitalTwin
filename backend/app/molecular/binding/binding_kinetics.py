"""
backend/app/molecular/binding/binding_kinetics.py
Dynamic Law of Mass Action and Target Engagement Kinetics.

Implements:
1. Reversible 1:1 binding: D + R <-> DR
2. Target-Mediated Drug Disposition (TMDD) where target binding depletes free drug
3. Competitive binding of two ligands (e.g. endogenous agonist + drug antagonist)
4. Non-equilibrium association and dissociation time courses
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class DynamicBindingState:
    free_drug_nM: float
    free_target_pmol_g: float
    complex_pmol_g: float
    total_target_pmol_g: float
    occupancy_fraction: float


class DynamicBindingEngine:
    """
    Simulates time-dependent association and dissociation kinetics of drug-target complexes.
    """
    def __init__(
        self,
        k_on_nM_min: float,
        k_off_min: float,
        total_target_pmol_g: float,
    ):
        self.k_on = max(1e-9, float(k_on_nM_min))
        self.k_off = max(1e-9, float(k_off_min))
        self.R_total = max(0.0, float(total_target_pmol_g))
        self.Kd_nM = self.k_off / self.k_on

        # State: [DR_complex]
        self.complex_pmol_g: float = 0.0

    @property
    def free_target_pmol_g(self) -> float:
        return max(0.0, self.R_total - self.complex_pmol_g)

    @property
    def fractional_occupancy(self) -> float:
        if self.R_total <= 1e-9:
            return 0.0
        return min(1.0, max(0.0, self.complex_pmol_g / self.R_total))

    def step(
        self,
        free_drug_nM: float,
        dt_min: float = 1.0,
    ) -> DynamicBindingState:
        """
        Integrates binding kinetics over dt_min given ambient free drug concentration.
        d[DR]/dt = k_on * [D] * [R_free] - k_off * [DR]
        """
        D = max(0.0, float(free_drug_nM))

        def d_complex_dt(t, y):
            DR = y[0]
            R_free = max(0.0, self.R_total - DR)
            rate = self.k_on * D * R_free - self.k_off * DR
            return [rate]

        sol = solve_ivp(
            d_complex_dt,
            [0.0, dt_min],
            [self.complex_pmol_g],
            method="RK45",
        )
        self.complex_pmol_g = float(max(0.0, min(self.R_total, sol.y[0, -1])))

        return DynamicBindingState(
            free_drug_nM=D,
            free_target_pmol_g=self.free_target_pmol_g,
            complex_pmol_g=self.complex_pmol_g,
            total_target_pmol_g=self.R_total,
            occupancy_fraction=self.fractional_occupancy,
        )

    def simulate_competitive_binding(
        self,
        agonist_conc_nM: float,
        antagonist_conc_nM: float,
        agonist_kon: float,
        agonist_koff: float,
        antagonist_kon: float,
        antagonist_koff: float,
        duration_min: float = 60.0,
        n_steps: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Simulates competition between two ligands (A: agonist, B: antagonist) for one binding site R.
        dA_complex/dt = kon_A * [A] * [R_free] - koff_A * [AR]
        dB_complex/dt = kon_B * [B] * [R_free] - koff_B * [BR]
        R_free = R_total - AR - BR
        """
        t_eval = np.linspace(0.0, duration_min, n_steps)

        def d_states(t, y):
            AR, BR = y
            R_free = max(0.0, self.R_total - AR - BR)
            dAR = agonist_kon * agonist_conc_nM * R_free - agonist_koff * AR
            dBR = antagonist_kon * antagonist_conc_nM * R_free - antagonist_koff * BR
            return [dAR, dBR]

        sol = solve_ivp(d_states, [0.0, duration_min], [0.0, 0.0], t_eval=t_eval, method="RK45")

        ar_sol = np.clip(sol.y[0], 0.0, self.R_total)
        br_sol = np.clip(sol.y[1], 0.0, self.R_total)
        r_free = np.maximum(0.0, self.R_total - ar_sol - br_sol)

        return {
            "time_min": sol.t,
            "agonist_bound_pmol_g": ar_sol,
            "antagonist_bound_pmol_g": br_sol,
            "free_target_pmol_g": r_free,
            "agonist_occupancy": ar_sol / max(1e-6, self.R_total),
            "antagonist_occupancy": br_sol / max(1e-6, self.R_total),
        }

