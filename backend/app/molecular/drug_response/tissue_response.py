"""
backend/app/molecular/drug_response/tissue_response.py
Tissue-level Integration of Cellular Drug Responses.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List
from app.molecular.tissue.cell_types import OrganType
from app.molecular.signaling.downstream_response import CellularResponse


@dataclass
class TissueResponse:
    organ: OrganType
    tissue_region: str
    predominant_cell_type: str
    tissue_level_metric: str
    baseline_value: float
    modulated_value: float
    unit: str
    clinical_interpretation: str


class TissueResponseIntegrator:
    """
    Scales microscopic cellular phenotypes to macroscopic tissue functional changes.
    """
    @staticmethod
    def integrate(
        organ: OrganType,
        region: str,
        cell_response: CellularResponse,
        target_id: str,
    ) -> TissueResponse:
        tid = target_id.upper()
        fold = cell_response.quantitative_change_fold

        if tid == "SGLT2" and organ == OrganType.KIDNEY:
            # Proximal tubule glucose reabsorptive capacity (TmG)
            baseline = 375.0 # mg/min healthy adult maximum
            modulated = baseline * fold
            return TissueResponse(
                organ=organ,
                tissue_region=region,
                predominant_cell_type=cell_response.cell_type_id,
                tissue_level_metric="maximal_glucose_transport_capacity_TmG",
                baseline_value=baseline,
                modulated_value=round(modulated, 1),
                unit="mg/min",
                clinical_interpretation=f"Tubular reabsorptive capacity TmG reduced from {baseline:.0f} to {modulated:.1f} mg/min, lowering renal threshold.",
            )

        elif tid == "ADRB1" and organ == OrganType.HEART:
            # Ventricular myocardial elastance / peak contractility
            baseline = 1.0 # arbitrary normalized contractility index
            modulated = baseline * fold
            return TissueResponse(
                organ=organ,
                tissue_region=region,
                predominant_cell_type=cell_response.cell_type_id,
                tissue_level_metric="myocardial_contractile_elastance_index",
                baseline_value=baseline,
                modulated_value=round(modulated, 2),
                unit="relative_index",
                clinical_interpretation=f"Myocardial inotropic contractility index adjusted to {modulated:.2f}x of baseline.",
            )

        elif tid in ["AGTR1", "CACNA1C"] and organ == OrganType.VASCULATURE:
            # Systemic arteriolar resistance index
            baseline = 1.0
            modulated = baseline * fold
            return TissueResponse(
                organ=organ,
                tissue_region=region,
                predominant_cell_type=cell_response.cell_type_id,
                tissue_level_metric="arteriolar_vascular_resistance_index",
                baseline_value=baseline,
                modulated_value=round(modulated, 2),
                unit="relative_index",
                clinical_interpretation=f"Arteriolar resistance shifted to {modulated:.2f}x of baseline, promoting systemic vasodilation.",
            )

        elif tid == "GLP1R" and organ == OrganType.PANCREAS:
            baseline = 1.0
            modulated = baseline * fold
            return TissueResponse(
                organ=organ,
                tissue_region=region,
                predominant_cell_type=cell_response.cell_type_id,
                tissue_level_metric="beta_cell_insulin_secretory_reserve",
                baseline_value=baseline,
                modulated_value=round(modulated, 2),
                unit="relative_multiplier",
                clinical_interpretation=f"Endocrine islet insulin secretion sensitivity potentiated by {modulated:.2f}x.",
            )

        elif tid == "HMGCR" and organ == OrganType.LIVER:
            baseline = 100.0 # 100% capacity
            modulated = baseline * fold
            return TissueResponse(
                organ=organ,
                tissue_region=region,
                predominant_cell_type=cell_response.cell_type_id,
                tissue_level_metric="hepatic_mevalonate_synthesis_rate",
                baseline_value=baseline,
                modulated_value=round(modulated, 1),
                unit="percent_of_normal",
                clinical_interpretation=f"Hepatic de novo cholesterol synthesis reduced to {modulated:.1f}% of baseline.",
            )

        # Fallback generic
        return TissueResponse(
            organ=organ,
            tissue_region=region,
            predominant_cell_type=cell_response.cell_type_id,
            tissue_level_metric="tissue_functional_activity_index",
            baseline_value=1.0,
            modulated_value=round(fold, 2),
            unit="relative_index",
            clinical_interpretation=f"Tissue activity modulated to {fold:.2f}x of normal resting level.",
        )
