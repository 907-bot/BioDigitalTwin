"""
backend/app/molecular/drug_response/pharmacodynamics.py
Concentration-Response and Operational Pharmacodynamic Models.
"""
from __future__ import annotations

import numpy as np


def hill_equation_response(
    conc_nM: float,
    EC50_nM: float,
    Emax: float = 1.0,
    gamma: float = 1.0,
    E0: float = 0.0,
) -> float:
    """
    Standard Sigmoidal Emax (Hill) pharmacodynamic model:
    E = E0 + (Emax - E0) * C^gamma / (EC50^gamma + C^gamma)
    """
    c = max(0.0, float(conc_nM))
    ec50 = max(1e-12, float(EC50_nM))
    g = max(0.1, float(gamma))

    if c <= 1e-12:
        return E0

    c_g = c ** g
    ec_g = ec50 ** g
    resp = E0 + (Emax - E0) * (c_g / (ec_g + c_g))
    return float(resp)
