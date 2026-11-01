"""
backend/app/molecular/tissue/local_exposure.py
Mechanistic Local Exposure Engine.

Translates systemic blood/plasma pharmacokinetics into micro-compartment concentrations:
- Plasma total & free concentrations
- Tissue total concentration (via perfusion & partition coefficient Kp)
- Interstitial fluid free concentration (C_ISF,free)
- Intracellular cytosolic free concentration (C_cell,free)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
from app.molecular.tissue.cell_types import OrganType, SubcellularLocation
from app.molecular.tissue.tissue_compartments import ORGAN_PHYSIOLOGY_ATLAS, OrganPhysiologySpec


@dataclass
class LocalExposureProfile:
    organ: OrganType
    cell_type_id: str
    subcellular_location: SubcellularLocation
    plasma_total_mg_L: float
    plasma_free_nM: float
    tissue_total_mg_L: float
    interstitial_free_nM: float
    intracellular_free_nM: float
    target_accessible_conc_nM: float
    partition_coefficient_Kp: float
    fu_plasma: float


class LocalExposureCalculator:
    """
    Physiologically based computation of local drug exposure accessible to target molecules.
    """
    @staticmethod
    def calculate(
        organ: OrganType,
        cell_type_id: str,
        subcellular_location: SubcellularLocation,
        plasma_total_mg_L: float,
        molecular_weight_g_mol: float,
        fu_plasma: float = 0.15,
        partition_coefficient_Kp: float = 1.0,
        intracellular_permeability_ratio: float = 0.85,
    ) -> LocalExposureProfile:
        mw = max(10.0, float(molecular_weight_g_mol))
        c_plasma_mg_L = max(0.0, float(plasma_total_mg_L))
        fu = float(max(0.001, min(1.0, fu_plasma)))
        kp = float(max(0.01, partition_coefficient_Kp))

        # 1. Plasma concentrations: mg/L -> nM
        # (mg/L) / (g/mol) = mol / (1000 * L) = mmol/L = 10^6 nM
        # 1 mg/L = (1e-3 g / L) / (MW g/mol) = (1e-3 / MW) mol/L = (1e6 / MW) nM
        plasma_total_nM = (c_plasma_mg_L * 1e6) / mw
        plasma_free_nM = plasma_total_nM * fu

        # 2. Total tissue concentration based on tissue partition coefficient Kp
        tissue_total_mg_L = c_plasma_mg_L * kp

        # 3. Interstitial free concentration (Free drug hypothesis: C_ISF,free = C_plasma,free at steady state)
        # In rapidly perfused organs, interstitial free concentration equals plasma free concentration.
        interstitial_free_nM = plasma_free_nM

        # 4. Intracellular free concentration
        intracellular_free_nM = interstitial_free_nM * float(max(0.05, intracellular_permeability_ratio))

        # 5. Determine target accessible concentration based on subcellular location
        if subcellular_location in [
            SubcellularLocation.PLASMA_MEMBRANE_APICAL,
            SubcellularLocation.PLASMA_MEMBRANE_BASOLATERAL,
            SubcellularLocation.PLASMA_MEMBRANE_NONPOLARIZED,
        ]:
            accessible_nM = interstitial_free_nM
        elif subcellular_location in [
            SubcellularLocation.CYTOPLASM,
            SubcellularLocation.NUCLEUS,
            SubcellularLocation.ENDOPLASMIC_RETICULUM,
            SubcellularLocation.MITOCHONDRIA,
            SubcellularLocation.SARCOPLASMIC_RETICULUM,
        ]:
            accessible_nM = intracellular_free_nM
        else:
            accessible_nM = interstitial_free_nM

        return LocalExposureProfile(
            organ=organ,
            cell_type_id=cell_type_id,
            subcellular_location=subcellular_location,
            plasma_total_mg_L=c_plasma_mg_L,
            plasma_free_nM=plasma_free_nM,
            tissue_total_mg_L=tissue_total_mg_L,
            interstitial_free_nM=interstitial_free_nM,
            intracellular_free_nM=intracellular_free_nM,
            target_accessible_conc_nM=accessible_nM,
            partition_coefficient_Kp=kp,
            fu_plasma=fu,
        )
