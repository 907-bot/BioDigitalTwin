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
    luminal_free_nM: float
    intracellular_free_nM: float
    target_accessible_conc_nM: float
    route_of_target_access: str
    partition_coefficient_Kp: float
    fu_plasma: float


class LocalExposureCalculator:
    """
    Physiologically based computation of local drug exposure accessible to target molecules.
    Accounts for cell polarity, apical luminal filtration vs basolateral interstitial fluid,
    and intracellular active transport accumulation.
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
        apical_luminal_concentration_factor: float = 1.4,
    ) -> LocalExposureProfile:
        mw = max(10.0, float(molecular_weight_g_mol))
        c_plasma_mg_L = max(0.0, float(plasma_total_mg_L))
        fu = float(max(0.001, min(1.0, fu_plasma)))
        kp = float(max(0.01, partition_coefficient_Kp))

        # 1. Plasma concentrations: mg/L -> nM
        plasma_total_nM = (c_plasma_mg_L * 1e6) / mw
        plasma_free_nM = plasma_total_nM * fu

        # 2. Total tissue concentration based on tissue partition coefficient Kp
        tissue_total_mg_L = c_plasma_mg_L * kp

        # 3. Interstitial free concentration (Free drug hypothesis: C_ISF,free = C_plasma,free at steady state)
        interstitial_free_nM = plasma_free_nM

        # 4. Intracellular free concentration (incorporates tissue partition Kp for active transport)
        intracellular_free_nM = interstitial_free_nM * float(max(0.05, intracellular_permeability_ratio)) * kp

        # 5. Luminal free concentration for polarized epithelial cells
        # In renal proximal tubule, glomerular filtration of unbound drug combined with proximal water
        # reabsorption establishes an enriched luminal concentration at the apical brush border
        if organ == OrganType.KIDNEY and "proximal_tubule" in cell_type_id:
            luminal_free_nM = plasma_free_nM * float(max(0.5, apical_luminal_concentration_factor))
        elif organ == OrganType.GASTROINTESTINAL:
            luminal_free_nM = plasma_free_nM * 0.8
        else:
            luminal_free_nM = interstitial_free_nM

        # 6. Determine target-accessible concentration by physiological access route
        if subcellular_location == SubcellularLocation.PLASMA_MEMBRANE_APICAL:
            accessible_nM = luminal_free_nM
            access_route = "apical_luminal_fluid_exposure"
        elif subcellular_location in [
            SubcellularLocation.PLASMA_MEMBRANE_BASOLATERAL,
            SubcellularLocation.PLASMA_MEMBRANE_NONPOLARIZED,
        ]:
            accessible_nM = interstitial_free_nM
            access_route = "peritubular_interstitial_fluid_exposure"
        elif subcellular_location in [
            SubcellularLocation.CYTOPLASM,
            SubcellularLocation.NUCLEUS,
            SubcellularLocation.ENDOPLASMIC_RETICULUM,
            SubcellularLocation.MITOCHONDRIA,
            SubcellularLocation.SARCOPLASMIC_RETICULUM,
        ]:
            accessible_nM = intracellular_free_nM
            access_route = "transcellular_cytosolic_uptake"
        else:
            accessible_nM = interstitial_free_nM
            access_route = "standard_interstitial_exposure"

        return LocalExposureProfile(
            organ=organ,
            cell_type_id=cell_type_id,
            subcellular_location=subcellular_location,
            plasma_total_mg_L=c_plasma_mg_L,
            plasma_free_nM=plasma_free_nM,
            tissue_total_mg_L=tissue_total_mg_L,
            interstitial_free_nM=interstitial_free_nM,
            luminal_free_nM=luminal_free_nM,
            intracellular_free_nM=intracellular_free_nM,
            target_accessible_conc_nM=accessible_nM,
            route_of_target_access=access_route,
            partition_coefficient_Kp=kp,
            fu_plasma=fu,
        )
