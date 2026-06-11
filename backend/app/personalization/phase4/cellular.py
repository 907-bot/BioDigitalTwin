"""
Phase 4: Cellular Twin — Neural ODE cell population dynamics.

Models cell populations (hepatocytes, adipocytes, immune cells, beta cells, cardiomyocytes)
with stress, health, turnover, and inflammation states.

Uses a Neural ODE (torch-based) for continuous-time population dynamics.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# ── Cell Types ────────────────────────────────────────────────

CELL_TYPES = [
    "hepatocytes", "adipocytes", "immune_macrophages",
    "beta_cells", "cardiomyocytes",
]
N_CELL_TYPES = len(CELL_TYPES)


@dataclass
class CellularState:
    """
    Cell-population-level state.

    Each cell type has:
      - population: relative cell count (0.5-1.5, 1.0 = healthy baseline)
      - stress: cellular stress level (0-1)
      - health: cellular health (0-1, 1 = fully functional)
      - turnover: proliferation/apoptosis rate (normalized)
      - inflammation: cell-intrinsic inflammatory state (0-1)
    """
    population: np.ndarray       # (n_cell_types,) — relative cell count
    stress: np.ndarray           # (n_cell_types,) — cellular stress
    health: np.ndarray           # (n_cell_types,) — cellular health
    turnover: np.ndarray         # (n_cell_types,) — net turnover
    inflammation: np.ndarray     # (n_cell_types,) — cell-intrinsic inflammation

    @property
    def dim(self) -> int:
        return 5 * N_CELL_TYPES

    def to_array(self) -> np.ndarray:
        return np.concatenate([
            self.population, self.stress, self.health,
            self.turnover, self.inflammation,
        ])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'CellularState':
        return cls(
            population=arr[0:N_CELL_TYPES],
            stress=arr[N_CELL_TYPES:2*N_CELL_TYPES],
            health=arr[2*N_CELL_TYPES:3*N_CELL_TYPES],
            turnover=arr[3*N_CELL_TYPES:4*N_CELL_TYPES],
            inflammation=arr[4*N_CELL_TYPES:5*N_CELL_TYPES],
        )

    @classmethod
    def healthy(cls) -> 'CellularState':
        return cls(
            population=np.ones(N_CELL_TYPES),
            stress=np.zeros(N_CELL_TYPES),
            health=np.ones(N_CELL_TYPES),
            turnover=np.zeros(N_CELL_TYPES),
            inflammation=np.zeros(N_CELL_TYPES),
        )


CELLULAR_DIM = 5 * N_CELL_TYPES


# ── Neural ODE for Cell Dynamics ─────────────────────────────

class CellODEFunc(nn.Module):
    """
    Neural ODE governing cell population dynamics.

    Input: [cell_state (25d) + molecular_signals (4d) + organ_signals (4d)] = 33d
    Output: time derivative of cell state (25d)
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(33, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, CELLULAR_DIM),
        )

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # state = [cellular (25d), molecular_signals (4d), organ_signals (4d)]
        return self.net(state)


class CellPopulationDynamics:
    """
    Cell population dynamics with mechanistic base model.
    The Neural ODE serves as a learned correction on top.
    """

    def __init__(self):
        self.ode_func = CellODEFunc()
        self.ode_func.eval()

    def compute_dynamics(
        self,
        cell_state: CellularState,
        mol_signals: Dict[str, float],
        organ_signals: Dict[str, float],
        dt: float = 1.0,
    ) -> CellularState:
        """
        Compute cell state update using mechanistic model +
        Neural ODE correction.

        Mechanistic model captures known physiology:
          - Hepatocyte: insulin promotes health, FFA induces stress
          - Adipocyte: hyperplasia in overnutrition, hypertrophy stress
          - Immune: M1 polarization with inflammation
          - Beta cell: glucose toxicity at high glucose
          - Cardiomyocyte: pressure overload stress
        """
        s = cell_state

        # --- Mechanistic dynamics ---
        new_pop = s.population.copy()
        new_stress = s.stress.copy()
        new_health = s.health.copy()
        new_turnover = s.turnover.copy()
        new_inflam = s.inflammation.copy()

        ins_sig = mol_signals.get("insulin_signal", 0.3)
        inflam_sig = mol_signals.get("inflammatory_signal", 0.1)
        met_stress = mol_signals.get("metabolic_stress", 0.1)

        for i, ctype in enumerate(CELL_TYPES):
            if ctype == "hepatocytes":
                # Insulin promotes health; FFA/metabolic stress damages
                d_health = 0.01 * ins_sig - 0.02 * met_stress
                d_stress = 0.005 * met_stress - 0.01 * s.stress[i]
                d_pop = 0.001 * (1.0 - s.population[i])  # gentle homeostasis
                d_turn = d_pop
                d_inflam = 0.005 * inflam_sig - 0.01 * s.inflammation[i]

            elif ctype == "adipocytes":
                # FFA drives hypertrophy stress; insulin promotes storage
                d_health = -0.015 * met_stress + 0.005 * ins_sig
                d_stress = 0.01 * met_stress - 0.008 * s.stress[i]
                d_pop = 0.002 * met_stress  # hyperplasia with overnutrition
                d_turn = d_pop * 2.0
                d_inflam = 0.01 * inflam_sig - 0.005 * s.inflammation[i]

            elif ctype == "immune_macrophages":
                # Inflammation recruits and polarizes macrophages
                d_health = -0.01 * inflam_sig
                d_stress = 0.02 * inflam_sig - 0.01 * s.stress[i]
                d_pop = 0.01 * inflam_sig - 0.005 * (s.population[i] - 1.0)
                d_turn = d_pop
                d_inflam = 0.02 * inflam_sig - 0.01 * s.inflammation[i]

            elif ctype == "beta_cells":
                # Glucose toxicity: high glucose damages beta cells
                gluc_tox = organ_signals.get("glucose_toxicity", 0.0)
                d_health = -0.02 * gluc_tox
                d_stress = 0.02 * gluc_tox - 0.01 * s.stress[i]
                d_pop = -0.005 * gluc_tox  # apoptosis from glucotoxicity
                d_turn = d_pop
                d_inflam = 0.005 * inflam_sig

            elif ctype == "cardiomyocytes":
                # Pressure overload stress
                bp_stress = organ_signals.get("bp_overload", 0.0)
                d_health = -0.01 * bp_stress
                d_stress = 0.015 * bp_stress - 0.01 * s.stress[i]
                d_pop = -0.002 * bp_stress
                d_turn = d_pop
                d_inflam = 0.005 * inflam_sig

            else:
                d_health = d_stress = d_pop = d_turn = d_inflam = 0.0

            new_health[i] = np.clip(s.health[i] + d_health * dt, 0.0, 1.0)
            new_stress[i] = np.clip(s.stress[i] + d_stress * dt, 0.0, 1.0)
            new_pop[i] = np.clip(s.population[i] + d_pop * dt, 0.3, 2.0)
            new_turnover[i] = np.clip(d_turn, -0.1, 0.1)
            new_inflam[i] = np.clip(s.inflammation[i] + d_inflam * dt, 0.0, 1.0)

        return CellularState(
            population=new_pop, stress=new_stress, health=new_health,
            turnover=new_turnover, inflammation=new_inflam,
        )


# ── Cellular → Organ Interface ────────────────────────────────

def cellular_to_organ_signals(
    cell_state: CellularState,
) -> Dict[str, float]:
    """
    Map cell population states to organ-level parameter modifiers.

    Returns:
        insulin_sensitivity_mod: multiplier for SI (0.5-1.5)
        inflammation_mod: additive to CRP / inflammatory load
        beta_cell_function: multiplier for insulin secretion (0-1)
        vascular_health_mod: multiplier for vascular resistance
        metabolic_rate_mod: multiplier for BMR
    """
    # Hepatocyte health → insulin sensitivity
    hep_health = cell_state.health[0]
    ins_sens_mod = 0.5 + 0.5 * hep_health

    # Immune inflammation → CRP
    imm_inflam = cell_state.inflammation[2]
    inflam_mod = imm_inflam * 5.0

    # Beta cell health → insulin secretion
    beta_health = cell_state.health[3]
    beta_fn = max(0.1, beta_health)

    # Adipocyte stress → vascular resistance
    adi_stress = cell_state.stress[1]
    vasc_mod = 1.0 + 0.3 * adi_stress

    # Cardiomyocyte health → cardiac output
    cardio_health = cell_state.health[4]
    bmr_mod = 0.8 + 0.2 * cardio_health

    return {
        "insulin_sensitivity_mod": float(ins_sens_mod),
        "inflammation_mod": float(inflam_mod),
        "beta_cell_function": float(beta_fn),
        "vascular_health_mod": float(vasc_mod),
        "metabolic_rate_mod": float(bmr_mod),
    }
