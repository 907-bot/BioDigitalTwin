"""
BioDigitalTwin — Physiological Organs Module.

Exports standard organ models implementing the OrganModel contract:
- CardiovascularOrganModel
- HepaticOrganModel
- RenalOrganModel
- GastrointestinalOrganModel
- PancreaticOrganModel
- RespiratoryOrganModel
- EndocrineOrganModel
"""

from .cardiovascular import CardiovascularOrganModel
from .hepatic import HepaticOrganModel
from .renal import RenalOrganModel
from .gastrointestinal import GastrointestinalOrganModel
from .pancreas import PancreaticOrganModel
from .respiratory import RespiratoryOrganModel
from .endocrine import EndocrineOrganModel
from ..core.simulation_engine import CoupledSimulationEngine


def create_standard_multi_organ_engine(
    patient_weight_kg: float = 70.0,
    patient_height_cm: float = 175.0,
) -> CoupledSimulationEngine:
    """
    Factory constructing a fully coupled 6-organ physiological digital twin:
    Cardiovascular, Hepatic, Renal, Gastrointestinal, Pancreatic, Respiratory, and Endocrine.
    """
    engine = CoupledSimulationEngine(
        patient_weight_kg=patient_weight_kg,
        patient_height_cm=patient_height_cm,
    )
    engine.register_organ(CardiovascularOrganModel())
    engine.register_organ(HepaticOrganModel())
    engine.register_organ(RenalOrganModel())
    engine.register_organ(GastrointestinalOrganModel())
    engine.register_organ(PancreaticOrganModel())
    engine.register_organ(RespiratoryOrganModel())
    engine.register_organ(EndocrineOrganModel())
    engine.initialize()
    return engine


__all__ = [
    "CardiovascularOrganModel",
    "HepaticOrganModel",
    "RenalOrganModel",
    "GastrointestinalOrganModel",
    "PancreaticOrganModel",
    "RespiratoryOrganModel",
    "EndocrineOrganModel",
    "create_standard_multi_organ_engine",
]
