"""
Base contract for physiological organ models in BioDigitalTwin.

Provides a unified interface for all organ systems:
- SI units and physical bounds
- Parameters, states, inputs, outputs
- ODE derivatives and numerical integration
- Boundary fluxes for vascular transport coupling
- Conservation law verification (mass, volume, charge)
- Explicit uncertainty quantification
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


@dataclass
class OrganStateDescriptor:
    name: str
    si_unit: str
    nominal: float
    min_bound: float
    max_bound: float
    description: str


@dataclass
class OrganParamDescriptor:
    name: str
    si_unit: str
    default_value: float
    min_bound: float
    max_bound: float
    description: str


class PhysiologicalOrganModel(ABC):
    """
    Standard interface for all whole-body physiological organ models.
    """

    def __init__(self, name: str):
        self.name = name
        self._state: Dict[str, float] = {}
        self._params: Dict[str, float] = {}
        self._uncertainty: Dict[str, float] = {}
        self.descriptors_state: Dict[str, OrganStateDescriptor] = {}
        self.descriptors_param: Dict[str, OrganParamDescriptor] = {}

    @abstractmethod
    def declare_descriptors(self) -> None:
        """Register state and parameter descriptors, bounds, and SI units."""
        pass

    @abstractmethod
    def initialize(self, patient_parameters: Optional[Dict[str, float]] = None) -> None:
        """Initialize parameters and baseline steady-state values."""
        pass

    def get_state(self) -> Dict[str, float]:
        """Return a copy of the current state variables."""
        return dict(self._state)

    def set_state(self, state: Dict[str, float]) -> None:
        """Set state variables with boundary validation."""
        for k, v in state.items():
            if k in self.descriptors_state:
                desc = self.descriptors_state[k]
                clamped_v = float(np.clip(v, desc.min_bound, desc.max_bound))
                self._state[k] = clamped_v
            else:
                self._state[k] = float(v)

    def get_parameters(self) -> Dict[str, float]:
        """Return a copy of model parameters."""
        return dict(self._params)

    def set_parameters(self, params: Dict[str, float]) -> None:
        """Update model parameters."""
        for k, v in params.items():
            if k in self.descriptors_param:
                desc = self.descriptors_param[k]
                self._params[k] = float(np.clip(v, desc.min_bound, desc.max_bound))
            else:
                self._params[k] = float(v)

    @abstractmethod
    def compute_derivatives(
        self,
        state: Dict[str, float],
        inputs: Dict[str, float],
        transport: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Compute d(State)/dt given current state, clinical inputs, and transport conditions.
        """
        pass

    def step(
        self,
        dt: float,
        inputs: Dict[str, float],
        transport: Dict[str, float],
        method: str = "rk4",
    ) -> Dict[str, float]:
        """
        Advance state forward in time by dt minutes using Euler or Runge-Kutta 4th-order (RK4).
        """
        s0 = self.get_state()

        if method == "euler":
            d_dt = self.compute_derivatives(s0, inputs, transport)
            s_new = {k: s0[k] + d_dt.get(k, 0.0) * dt for k in s0}
        else:
            # RK4 numerical integration
            k1 = self.compute_derivatives(s0, inputs, transport)

            s_k1 = {k: s0[k] + 0.5 * dt * k1.get(k, 0.0) for k in s0}
            k2 = self.compute_derivatives(s_k1, inputs, transport)

            s_k2 = {k: s0[k] + 0.5 * dt * k2.get(k, 0.0) for k in s0}
            k3 = self.compute_derivatives(s_k2, inputs, transport)

            s_k3 = {k: s0[k] + dt * k3.get(k, 0.0) for k in s0}
            k4 = self.compute_derivatives(s_k3, inputs, transport)

            s_new = {
                k: s0[k] + (dt / 6.0) * (
                    k1.get(k, 0.0) + 2.0 * k2.get(k, 0.0) + 2.0 * k3.get(k, 0.0) + k4.get(k, 0.0)
                )
                for k in s0
            }

        self.set_state(s_new)
        return self.get_state()

    @abstractmethod
    def exchange_fluxes(self, transport: Dict[str, float]) -> Dict[str, float]:
        """
        Return substance exchange rates with systemic blood transport (mass/time or moles/time).
        Positive = net secretion/release into blood.
        Negative = net consumption/extraction from blood.
        """
        pass

    def validate_state(self) -> Tuple[bool, List[str]]:
        """Verify whether all state variables remain within plausible biological bounds."""
        violations = []
        for k, v in self._state.items():
            if k in self.descriptors_state:
                desc = self.descriptors_state[k]
                if not (desc.min_bound <= v <= desc.max_bound + 1e-6):
                    violations.append(
                        f"{self.name}.{k}={v:.4f} {desc.si_unit} out of bounds "
                        f"[{desc.min_bound}, {desc.max_bound}]"
                    )
        return (len(violations) == 0, violations)

    def check_conservation_laws(self) -> Tuple[bool, List[str]]:
        """Default hook for checking physical mass and energy conservation."""
        return (True, [])

    def get_uncertainty(self) -> Dict[str, float]:
        """Return 90% confidence half-width for state variables."""
        return {k: self._uncertainty.get(k, 0.05 * abs(self._state.get(k, 1.0))) for k in self._state}

