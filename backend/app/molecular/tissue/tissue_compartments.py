"""
backend/app/molecular/tissue/tissue_compartments.py
Physiological Tissue Compartment Specifications for Human Organs.

Based on standard ICRP / PK-Sim physiological values for a 70 kg reference human:
- Organ mass (g)
- Blood perfusion flow rate (L/min)
- Vascular volume fraction (V_vasc)
- Interstitial volume fraction (V_int)
- Intracellular volume fraction (V_cell)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
from app.molecular.tissue.cell_types import OrganType


@dataclass
class OrganPhysiologySpec:
    organ: OrganType
    weight_g: float                    # Total organ wet weight [g]
    blood_flow_L_min: float            # Perfusion blood flow [L/min]
    fraction_vascular: float           # Vascular blood volume fraction
    fraction_interstitial: float       # Extracellular interstitial fluid fraction
    fraction_intracellular: float      # Intracellular space fraction
    tissue_pH: float = 7.4             # Physiological interstitial pH


# Reference human physiological organ parameters (70 kg adult, Cardiac Output = 5.6 L/min)
ORGAN_PHYSIOLOGY_ATLAS: Dict[OrganType, OrganPhysiologySpec] = {
    OrganType.KIDNEY: OrganPhysiologySpec(
        organ=OrganType.KIDNEY,
        weight_g=310.0,
        blood_flow_L_min=1.15,         # ~20% of cardiac output
        fraction_vascular=0.16,
        fraction_interstitial=0.20,
        fraction_intracellular=0.64,
        tissue_pH=7.35,
    ),
    OrganType.HEART: OrganPhysiologySpec(
        organ=OrganType.HEART,
        weight_g=330.0,
        blood_flow_L_min=0.25,         # Coronary blood flow ~4.5%
        fraction_vascular=0.26,
        fraction_interstitial=0.14,
        fraction_intracellular=0.60,
        tissue_pH=7.30,
    ),
    OrganType.PANCREAS: OrganPhysiologySpec(
        organ=OrganType.PANCREAS,
        weight_g=100.0,
        blood_flow_L_min=0.06,         # ~1% of cardiac output
        fraction_vascular=0.12,
        fraction_interstitial=0.18,
        fraction_intracellular=0.70,
        tissue_pH=7.40,
    ),
    OrganType.LIVER: OrganPhysiologySpec(
        organ=OrganType.LIVER,
        weight_g=1800.0,
        blood_flow_L_min=1.45,         # Hepatic artery + portal vein ~25%
        fraction_vascular=0.21,
        fraction_interstitial=0.16,
        fraction_intracellular=0.63,
        tissue_pH=7.20,
    ),
    OrganType.LUNG: OrganPhysiologySpec(
        organ=OrganType.LUNG,
        weight_g=1000.0,
        blood_flow_L_min=5.60,         # Receives 100% of cardiac output
        fraction_vascular=0.36,
        fraction_interstitial=0.18,
        fraction_intracellular=0.46,
        tissue_pH=7.40,
    ),
    OrganType.SKELETAL_MUSCLE: OrganPhysiologySpec(
        organ=OrganType.SKELETAL_MUSCLE,
        weight_g=28000.0,              # ~40% of body weight
        blood_flow_L_min=0.75,         # Resting flow (~15% CO), rises 10x in exercise
        fraction_vascular=0.04,
        fraction_interstitial=0.12,
        fraction_intracellular=0.84,
        tissue_pH=7.25,
    ),
    OrganType.ADIPOSE: OrganPhysiologySpec(
        organ=OrganType.ADIPOSE,
        weight_g=14500.0,              # ~20% of body weight
        blood_flow_L_min=0.26,         # Low perfusion
        fraction_vascular=0.03,
        fraction_interstitial=0.13,
        fraction_intracellular=0.84,
        tissue_pH=7.35,
    ),
    OrganType.VASCULATURE: OrganPhysiologySpec(
        organ=OrganType.VASCULATURE,
        weight_g=2500.0,
        blood_flow_L_min=5.60,
        fraction_vascular=0.50,
        fraction_interstitial=0.25,
        fraction_intracellular=0.25,
        tissue_pH=7.40,
    ),
    OrganType.GASTROINTESTINAL: OrganPhysiologySpec(
        organ=OrganType.GASTROINTESTINAL,
        weight_g=1200.0,
        blood_flow_L_min=1.10,         # Splanchnic bed
        fraction_vascular=0.08,
        fraction_interstitial=0.17,
        fraction_intracellular=0.75,
        tissue_pH=6.80,                # Mucosal microclimate
    ),
    OrganType.BRAIN: OrganPhysiologySpec(
        organ=OrganType.BRAIN,
        weight_g=1400.0,
        blood_flow_L_min=0.75,         # ~13% of cardiac output
        fraction_vascular=0.03,
        fraction_interstitial=0.15,
        fraction_intracellular=0.82,
        tissue_pH=7.30,
    ),
}
