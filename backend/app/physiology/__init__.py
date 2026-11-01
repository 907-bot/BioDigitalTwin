"""
BioDigitalTwin Physiological World Model.

Scientifically grounded, interconnected, dynamic multi-organ physiological simulation layer.
Implements:
- Standardized PhysiologicalOrganModel contract with SI units and conservation laws
- Shared vascular transport and circulation layer (blood flow, cardiac output, O2/CO2 transport)
- Dedicated Respiratory System (pulmonary ventilation, alveolar gas exchange, acid-base pH)
- Dedicated Hepatic System (glycogenolysis, gluconeogenesis, lipogenesis, urea cycle, CYP clearance)
- Dedicated Gastrointestinal System (gastric emptying, nutrient absorption, GLP-1/GIP incretins)
- Dedicated Autonomic & Endocrine Regulation (SNS/PNS baroreflex, HPA axis, RAAS axis, ADH)
- MultiOrganPhysiologicalTwin coordinator linking organ fluxes to systemic circulation.
"""

from .base import PhysiologicalOrganModel, OrganStateDescriptor, OrganParamDescriptor
from .transport import BloodTransportLayer, HemodynamicState
from .respiratory import RespiratoryOrganModel, RespiratoryState
from .hepatic import HepaticOrganModel, HepaticState
from .gastrointestinal import GastrointestinalOrganModel, GastrointestinalState
from .autonomic_endocrine import AutonomicEndocrineModel, AutonomicEndocrineState
from .multi_scale_twin import MultiOrganPhysiologicalTwin

__all__ = [
    "PhysiologicalOrganModel",
    "OrganStateDescriptor",
    "OrganParamDescriptor",
    "BloodTransportLayer",
    "HemodynamicState",
    "RespiratoryOrganModel",
    "RespiratoryState",
    "HepaticOrganModel",
    "HepaticState",
    "GastrointestinalOrganModel",
    "GastrointestinalState",
    "AutonomicEndocrineModel",
    "AutonomicEndocrineState",
    "MultiOrganPhysiologicalTwin",
]

