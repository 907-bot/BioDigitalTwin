"""
BioDigitalTwin — Common Organ Model Interface.

Every physiological organ subsystem implements this contract:
- Explicit state and parameter schemas
- compute_derivatives() returns rates of change for the organ's internal state
- exchange_fluxes() returns mass/energy transfer fluxes into shared circulation
- validate_state() enforces physiological boundaries
- get_observables() exposes measurable clinical biomarkers
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from .state_registry import StateDescriptor, ParameterDescriptor
from .shared_environment import SharedEnvironment
from .flux_registry import Flux


class OrganModel(ABC):
    """
    Standard interface for all independent, testable physiological organ modules.
    """

    def __init__(self, organ_id: str):
        self.organ_id = organ_id
        self.state_schema: Dict[str, StateDescriptor] = {}
        self.parameter_schema: Dict[str, ParameterDescriptor] = {}
        self._state: Dict[str, float] = {}
        self._params: Dict[str, float] = {}

    @abstractmethod
    def initialize(
        self,
        patient_profile: Optional[Dict[str, Any]] = None,
        environment: Optional[SharedEnvironment] = None,
    ) -> None:
        """Initializes internal states and parameters to baseline conditions."""
        pass

    @abstractmethod
    def compute_derivatives(
        self,
        time: float,
        state: Dict[str, float],
        environment: SharedEnvironment,
    ) -> Dict[str, float]:
        """
        Computes rates of change (dState/dt) for all internal states of this organ.
        Must not directly mutate the environment or other organs.
        """
        pass

    @abstractmethod
    def exchange_fluxes(
        self,
        time: float,
        state: Dict[str, float],
        environment: SharedEnvironment,
    ) -> List[Flux]:
        """
        Returns all mass, solute, gas, or heat fluxes exchanged with shared compartments.
        """
        pass

    def validate_state(self, state: Optional[Dict[str, float]] = None) -> Tuple[bool, List[str]]:
        """
        Checks internal state variables against their physiological boundaries.
        """
        s = state if state is not None else self._state
        violations = []
        for name, desc in self.state_schema.items():
            if name not in s:
                violations.append(f"Organ {self.organ_id} missing state: '{name}'")
                continue
            val = s[name]
            if val < desc.min_bound or val > desc.max_bound:
                violations.append(
                    f"Organ {self.organ_id} state '{name}' = {val:.3f} {desc.unit} out of bounds [{desc.min_bound}, {desc.max_bound}]"
                )
        return len(violations) == 0, violations

    def get_observables(
        self,
        state: Optional[Dict[str, float]] = None,
        environment: Optional[SharedEnvironment] = None,
    ) -> Dict[str, float]:
        """
        Exposes clinically observable biomarker values from this organ.
        """
        s = state if state is not None else self._state
        return {
            name: s[name]
            for name, desc in self.state_schema.items()
            if desc.is_observable and name in s
        }

    def get_state(self) -> Dict[str, float]:
        return dict(self._state)

    def set_state(self, new_state: Dict[str, float]) -> None:
        self._state.update(new_state)

    def get_parameters(self) -> Dict[str, float]:
        return dict(self._params)

    def set_parameters(self, new_params: Dict[str, float]) -> None:
        self._params.update(new_params)
