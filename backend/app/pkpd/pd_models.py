"""
Pharmacodynamic (PD) models.

Implements the standard models used in industry:
  - Linear:           E = E0 + slope * C
  - Log-linear:       E = E0 + slope * log(C)
  - Emax:             E = E0 + (Emax * C) / (EC50 + C)
  - Sigmoid Emax:     E = E0 + (Emax * C^γ) / (EC50^γ + C^γ)   [most common]

Optionally links PK to PD via an effect compartment (ke0).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class PDModel(str, Enum):
    LINEAR = "linear"
    LOG_LINEAR = "log_linear"
    EMAX = "emax"
    SIGMOID_EMAX = "sigmoid_emax"   # default


@dataclass
class PDParams:
    model: PDModel = PDModel.SIGMOID_EMAX
    E0: float = 0.0          # baseline effect
    Emax: float = 1.0        # maximum effect
    EC50: float = 1.0        # concentration at 50% Emax (mg/L)
    gamma: float = 1.0       # Hill coefficient (1 for plain Emax)
    slope: float = 1.0       # for linear / log-linear
    ke0: float = 0.5         # 1/h, effect-compartment rate constant (0 = direct link)
    target_unit: str = ""    # optional: e.g. "mmHg" for SBP reduction


def effect_at_concentration(c: np.ndarray | float, p: PDParams) -> np.ndarray | float:
    """PD response at a given plasma concentration (no effect-compartment delay)."""
    c = np.asarray(c, dtype=float)
    if p.model == PDModel.LINEAR:
        return p.E0 + p.slope * c
    if p.model == PDModel.LOG_LINEAR:
        return p.E0 + p.slope * np.log(np.maximum(c, 1e-9))
    if p.model == PDModel.EMAX:
        return p.E0 + (p.Emax * c) / (p.EC50 + c)
    if p.model == PDModel.SIGMOID_EMAX:
        c_gamma = np.power(np.maximum(c, 1e-12), p.gamma)
        return p.E0 + (p.Emax * c_gamma) / (p.EC50 ** p.gamma + c_gamma)
    raise ValueError(f"unknown PD model: {p.model}")


def simulate_effect_compartment(c: np.ndarray, t: np.ndarray, p: PDParams) -> np.ndarray:
    """
    If ke0 > 0, build a delayed effect compartment:
      dCe/dt = ke0 * (C - Ce)
    Then compute E from Ce.
    """
    if p.ke0 <= 0:
        return np.asarray(effect_at_concentration(c, p), dtype=float)
    ce = np.zeros_like(c)
    ce[0] = c[0]
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        ce[i] = ce[i - 1] + p.ke0 * (c[i - 1] - ce[i - 1]) * dt
    return np.asarray(effect_at_concentration(ce, p), dtype=float)
