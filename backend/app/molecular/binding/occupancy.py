"""
backend/app/molecular/binding/occupancy.py
Equilibrium Target Occupancy and Competition Mathematics.

Implements:
1. One-site specific binding (Langmuir isotherm): Occupancy = C_free / (C_free + Kd)
2. Hill-Langmuir cooperative binding: Occupancy = C_free^h / (C_free^h + Kd^h)
3. Competitive displacement (Schild equation):
   Occupancy = C_free / (C_free + Kd * (1 + [I] / Ki))
4. Cheng-Prusoff conversion:
   Ki = IC50 / (1 + [S] / Km)
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional


def compute_equilibrium_occupancy(
    free_conc_nM: float,
    Kd_nM: float,
    hill_coefficient: float = 1.0,
    competing_inhibitor_conc_nM: float = 0.0,
    competing_inhibitor_Ki_nM: float = 1.0,
) -> float:
    """
    Computes fractional target occupancy at thermodynamic equilibrium.
    """
    C = max(0.0, float(free_conc_nM))
    Kd = max(1e-12, float(Kd_nM))
    h = max(0.1, float(hill_coefficient))
    I = max(0.0, float(competing_inhibitor_conc_nM))
    Ki = max(1e-12, float(competing_inhibitor_Ki_nM))

    # Competitive shift multiplier (Schild)
    shift_factor = 1.0 + (I / Ki) if I > 0 else 1.0
    effective_Kd = Kd * shift_factor

    if C <= 1e-12:
        return 0.0

    c_h = C ** h
    kd_h = effective_Kd ** h
    occupancy = c_h / (c_h + kd_h)
    return float(np.clip(occupancy, 0.0, 1.0))


def cheng_prusoff_ki(
    IC50_nM: float,
    substrate_conc_nM: float,
    Km_nM: float,
) -> float:
    """
    Converts functional IC50 of a competitive inhibitor into true thermodynamic Ki.
    Ki = IC50 / (1 + [S] / Km)
    """
    ic50 = max(1e-12, float(IC50_nM))
    s = max(0.0, float(substrate_conc_nM))
    km = max(1e-12, float(Km_nM))
    return float(ic50 / (1.0 + (s / km)))
