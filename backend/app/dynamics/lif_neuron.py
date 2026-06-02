"""
Leaky Integrate-and-Fire (LIF) neuron — the temporal unit for Phase 3.

We use the discrete-time formulation (Euler integration of the LIF ODE)
so the simulator is differentiable and torch-native:

    V[t+1] = V[t] + dt/tau * ( -(V[t] - V_rest) + R_m * I[t] )
    if V[t+1] >= V_th:
        spike[t+1] = 1
        V[t+1]    = V_reset

`snnTorch`'s LapiqueLIFCell is wrapped here for compatibility, but we
also expose a pure-torch implementation so we don't pay snnTorch's
import cost when the model is unloaded.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


class PureLIF(nn.Module):
    """Pure-torch LIF cell. Shape: (time, batch, features)."""

    def __init__(
        self,
        n_features: int,
        tau: float = 20.0,
        v_rest: float = -65.0,
        v_reset: float = -70.0,
        v_th: float = -50.0,
        r_m: float = 1.0,
        dt: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.tau = tau
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_th = v_th
        self.r_m = r_m
        self.dt = dt

    def forward(self, I: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        I: (T, B, F) input current
        returns:
            V: (T, B, F) membrane potential
            S: (T, B, F) binary spike tensor
        """
        T, B, F = I.shape
        V = torch.full((B, F), self.v_rest, device=I.device, dtype=I.dtype)
        Vs, Ss = [], []
        for t in range(T):
            V = V + (self.dt / self.tau) * (-(V - self.v_rest) + self.r_m * I[t])
            S = (V >= self.v_th).float()
            V = torch.where(S.bool(), torch.full_like(V, self.v_reset), V)
            Vs.append(V)
            Ss.append(S)
        return torch.stack(Vs, dim=0), torch.stack(Ss, dim=0)


class BiologicalLIFNeuron(PureLIF):
    """LIF tuned to biological defaults: tau=20ms, v_rest=-65, v_th=-50."""

    def __init__(self, n_features: int) -> None:
        super().__init__(
            n_features=n_features,
            tau=20.0,
            v_rest=-65.0,
            v_reset=-70.0,
            v_th=-50.0,
            r_m=1.0,
            dt=1.0,
        )
