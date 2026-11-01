"""
BioDigitalTwin — Flux Registry & Shared Flux Contracts.

Formalizes inter-organ mass and energy exchange:
- Source, Destination, Quantity, Rate, Unit, and Physiological Mechanism
- Tracks and aggregates influxes and effluxes for shared compartments
- Audits conservation constraints (e.g. gut appearance == portal absorption)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Flux:
    """An explicit mass or solute exchange between an organ and a compartment."""
    source: str             # e.g. "gastrointestinal", "hepatic", "pancreas", "lungs"
    destination: str        # e.g. "blood", "portal_vein", "urine", "interstitium"
    quantity: str           # e.g. "glucose", "insulin", "glucagon", "sodium", "oxygen", "co2"
    value: float            # Positive rate of transfer (amount / minute)
    unit: str               # e.g. "mg/min", "uU/min", "pg/min", "mEq/min", "mL/min"
    mechanism: str = ""     # e.g. "SGLT1 Carrier Absorption", "Glycogenolysis / Gluconeogenesis"


class FluxRegistry:
    """
    Manages and validates all active fluxes in the coupled simulation network.
    """

    def __init__(self):
        self._fluxes: List[Flux] = []

    def clear(self) -> None:
        self._fluxes.clear()

    def register_flux(self, flux: Flux) -> None:
        if flux.value < 0.0:
            logger.warning(f"Negative flux registered for {flux.quantity} ({flux.source} -> {flux.destination}): {flux.value}. Inverting sign.")
            flux = Flux(
                source=flux.destination,
                destination=flux.source,
                quantity=flux.quantity,
                value=abs(flux.value),
                unit=flux.unit,
                mechanism=f"Reversed: {flux.mechanism}",
            )
        self._fluxes.append(flux)

    def register_fluxes(self, fluxes: List[Flux]) -> None:
        for f in fluxes:
            self.register_flux(f)

    @property
    def all_fluxes(self) -> List[Flux]:
        return list(self._fluxes)

    def get_net_flux_to_compartment(self, compartment: str, quantity: str) -> float:
        """
        Computes net flux into a specific compartment:
        Net = (sum incoming) - (sum outgoing)
        """
        incoming = sum(f.value for f in self._fluxes if f.destination == compartment and f.quantity == quantity)
        outgoing = sum(f.value for f in self._fluxes if f.source == compartment and f.quantity == quantity)
        return float(incoming - outgoing)

    def get_fluxes_by_quantity(self, quantity: str) -> List[Flux]:
        return [f for f in self._fluxes if f.quantity == quantity]

    def summarize_balances(self) -> Dict[str, Dict[str, float]]:
        """
        Summarizes net rates for all active quantities across compartments.
        """
        summary: Dict[str, Dict[str, float]] = {}
        for f in self._fluxes:
            if f.quantity not in summary:
                summary[f.quantity] = {
                    "total_generated": 0.0,
                    "total_consumed": 0.0,
                    "net_balance": 0.0,
                }
            summary[f.quantity]["total_generated"] += f.value
            summary[f.quantity]["net_balance"] += f.value
        return summary
