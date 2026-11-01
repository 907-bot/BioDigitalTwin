"""
BioDigitalTwin — Canonical Physiological State Registry.

Provides stable, unambiguous named state and parameter schemas:
- Replaces undocumented integer indices with stable hierarchical names: e.g. "blood.glucose"
- Enforces explicit SI and clinical units
- Validates physiological bounds, compartments, and ownership
- Provides zero-copy or fast vectorized conversions between named states and solver arrays
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StateDescriptor:
    """Metadata specification for a single physiological state variable."""
    name: str                           # e.g., "blood.glucose"
    display_name: str                   # e.g., "Arterial Plasma Glucose"
    unit: str                           # e.g., "mg/dL"
    compartment: str                    # e.g., "blood", "liver", "gut", "kidney"
    min_bound: float                    # e.g., 20.0
    max_bound: float                    # e.g., 600.0
    nominal: float                      # e.g., 95.0
    initial_condition_source: str       # e.g., "fasting_baseline", "population_prior"
    is_observable: bool                 # e.g., True for CGM / lab test
    owning_module: str                  # e.g., "transport", "hepatic", "gastrointestinal"
    description: str                    # Detailed physiological definition


@dataclass(frozen=True)
class ParameterDescriptor:
    """Metadata specification for a physiological parameter."""
    name: str                           # e.g., "liver.glycogen_synthase_vmax"
    display_name: str
    unit: str
    default_value: float
    min_bound: float
    max_bound: float
    provenance: str                     # Medical literature citation / reference
    is_personalized: bool               # True if estimated via MAP/Bayesian assimilation
    description: str


class StateRegistry:
    """
    Central authoritative repository for all physiological states and parameters.
    Guarantees no duplicate keys, verifies units, and manages vector serialization.
    """

    def __init__(self):
        self._states: Dict[str, StateDescriptor] = {}
        self._parameters: Dict[str, ParameterDescriptor] = {}
        self._name_to_index: Dict[str, int] = {}
        self._index_to_name: List[str] = []
        self._frozen: bool = False

    def register_state(self, descriptor: StateDescriptor) -> None:
        if self._frozen:
            raise RuntimeError("Cannot register state on frozen StateRegistry")
        if descriptor.name in self._states:
            raise ValueError(f"Duplicate state identifier: '{descriptor.name}'")
        if descriptor.min_bound >= descriptor.max_bound:
            raise ValueError(f"Invalid bounds for '{descriptor.name}': min {descriptor.min_bound} >= max {descriptor.max_bound}")
        if not (descriptor.min_bound <= descriptor.nominal <= descriptor.max_bound):
            raise ValueError(f"Nominal value {descriptor.nominal} outside bounds for '{descriptor.name}'")

        self._states[descriptor.name] = descriptor
        self._name_to_index[descriptor.name] = len(self._index_to_name)
        self._index_to_name.append(descriptor.name)

    def register_parameter(self, descriptor: ParameterDescriptor) -> None:
        if self._frozen:
            raise RuntimeError("Cannot register parameter on frozen StateRegistry")
        if descriptor.name in self._parameters:
            raise ValueError(f"Duplicate parameter identifier: '{descriptor.name}'")
        if descriptor.min_bound > descriptor.max_bound:
            raise ValueError(f"Invalid parameter bounds for '{descriptor.name}'")

        self._parameters[descriptor.name] = descriptor

    def freeze(self) -> None:
        """Lock registry schema against further modifications."""
        self._frozen = True

    @property
    def state_count(self) -> int:
        return len(self._index_to_name)

    @property
    def parameter_count(self) -> int:
        return len(self._parameters)

    @property
    def state_names(self) -> List[str]:
        return list(self._index_to_name)

    def get_state_descriptor(self, name: str) -> StateDescriptor:
        if name not in self._states:
            raise KeyError(f"State '{name}' not found in registry")
        return self._states[name]

    def get_parameter_descriptor(self, name: str) -> ParameterDescriptor:
        if name not in self._parameters:
            raise KeyError(f"Parameter '{name}' not found in registry")
        return self._parameters[name]

    def get_index(self, name: str) -> int:
        if name not in self._name_to_index:
            raise KeyError(f"State '{name}' not mapped to registry index")
        return self._name_to_index[name]

    def create_nominal_state(self) -> Dict[str, float]:
        """Returns dictionary of all registered states set to nominal values."""
        return {name: desc.nominal for name, desc in self._states.items()}

    def create_nominal_parameters(self) -> Dict[str, float]:
        """Returns dictionary of all registered parameters set to default values."""
        return {name: desc.default_value for name, desc in self._parameters.items()}

    def state_to_vector(self, state_dict: Dict[str, float]) -> np.ndarray:
        """Serializes dictionary of states into a contiguous numpy array."""
        vec = np.zeros(self.state_count, dtype=np.float64)
        for name, idx in self._name_to_index.items():
            vec[idx] = state_dict.get(name, self._states[name].nominal)
        return vec

    def vector_to_state(self, vec: np.ndarray) -> Dict[str, float]:
        """Deserializes contiguous numpy array into a named state dictionary."""
        if len(vec) != self.state_count:
            raise ValueError(f"Vector dimension {len(vec)} does not match registry state count {self.state_count}")
        return {name: float(vec[idx]) for name, idx in self._name_to_index.items()}

    def clamp_state(self, state_dict: Dict[str, float]) -> Dict[str, float]:
        """Clamps all values in state_dict to their physiological min/max bounds."""
        clamped = dict(state_dict)
        for name, val in state_dict.items():
            if name in self._states:
                desc = self._states[name]
                clamped[name] = float(np.clip(val, desc.min_bound, desc.max_bound))
        return clamped

    def validate_state(self, state_dict: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Validates that all registered states are within physiological bounds."""
        violations = []
        for name, desc in self._states.items():
            if name not in state_dict:
                violations.append(f"Missing required state: '{name}'")
                continue
            val = state_dict[name]
            if val < desc.min_bound or val > desc.max_bound:
                violations.append(
                    f"State '{name}' = {val:.3f} {desc.unit} out of bounds [{desc.min_bound}, {desc.max_bound}]"
                )
        return len(violations) == 0, violations

    def export_schema(self) -> Dict[str, Any]:
        """Exports human-readable JSON schema of the physiological model."""
        return {
            "total_states": self.state_count,
            "total_parameters": self.parameter_count,
            "compartments": list(set(s.compartment for s in self._states.values())),
            "states": {
                name: {
                    "display_name": desc.display_name,
                    "unit": desc.unit,
                    "compartment": desc.compartment,
                    "bounds": [desc.min_bound, desc.max_bound],
                    "nominal": desc.nominal,
                    "is_observable": desc.is_observable,
                    "owning_module": desc.owning_module,
                    "description": desc.description,
                }
                for name, desc in self._states.items()
            },
            "parameters": {
                name: {
                    "display_name": desc.display_name,
                    "unit": desc.unit,
                    "default": desc.default_value,
                    "bounds": [desc.min_bound, desc.max_bound],
                    "provenance": desc.provenance,
                    "is_personalized": desc.is_personalized,
                    "description": desc.description,
                }
                for name, desc in self._parameters.items()
            },
        }


# Global registry singleton populated by default models
_GLOBAL_REGISTRY: Optional[StateRegistry] = None


def get_global_registry() -> StateRegistry:
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = StateRegistry()
    return _GLOBAL_REGISTRY
