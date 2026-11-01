"""
backend/app/molecular/spatial_engine.py
Master Orchestrator for the Spatial Pharmacology and Molecular Target Engagement Engine.

Simulates the complete mechanistic chain:
Organ Selection -> Regional Tissue -> Cell Population -> Subcellular Location ->
Molecular Target (Receptors, Enzymes, Transporters, Ion Channels) ->
Local Drug Exposure -> Binding Kinetics & Target Occupancy ->
Second Messenger Transduction -> Cellular Response ->
Organ-Level Physiology -> Whole-Body Systemic Consequences & Adverse Effects.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Union
from app.molecular.tissue.cell_types import (
    OrganType,
    SubcellularLocation,
    CellTypeSpec,
    CELL_TYPES_ATLAS,
    get_anatomical_hierarchy,
)
from app.molecular.tissue.tissue_compartments import ORGAN_PHYSIOLOGY_ATLAS
from app.molecular.tissue.local_exposure import (
    LocalExposureCalculator,
    LocalExposureProfile,
)
from app.molecular.targets.target_registry import (
    MolecularTarget,
    TargetClass,
    ReceptorFamily,
    ExpressionStatus,
    get_target_registry,
)
from app.molecular.binding.target_engagement import (
    TargetEngagementEvaluator,
    TargetEngagementResult,
)
from app.molecular.signaling.second_messengers import (
    SecondMessengerState,
    SecondMessengerTransducer,
)
from app.molecular.signaling.downstream_response import (
    CellularResponse,
    CellularResponseEvaluator,
)
from app.molecular.drug_response.tissue_response import (
    TissueResponse,
    TissueResponseIntegrator,
)
from app.molecular.drug_response.organ_response import (
    OrganResponseReport,
    OrganResponseEvaluator,
)


@dataclass
class SpatialSimulationResult:
    # 1. Anatomical Context
    organ: str
    region: str
    cell_type_id: str
    cell_type_name: str
    subcellular_location: str

    # 2. Target Profile & Expression
    target_id: str
    target_name: str
    target_class: str
    gene_symbol: str
    uniprot_id: str
    expression_status: str
    target_density_pmol_g: float
    copies_per_cell: Optional[float]
    evidence_source: str
    evidence_confidence: float

    # 3. Local Drug Exposure
    drug_name: str
    plasma_total_mg_L: float
    plasma_free_nM: float
    tissue_total_mg_L: float
    interstitial_free_nM: float
    intracellular_free_nM: float
    target_accessible_conc_nM: float

    # 4. Target Engagement & Occupancy
    fractional_occupancy: float
    modulation_type: str
    functional_modulation_pct: float
    effective_Kd_nM: float
    bound_target_density_pmol_g: float
    residence_time_min: float

    # 5. Second Messenger Transduction
    second_messengers: Dict[str, float]

    # 6. Cellular Response
    primary_cellular_effect: str
    quantitative_change_fold: float
    affected_cellular_pathways: List[str]
    cellular_description: str

    # 7. Organ-Level & Whole-Body Effects
    organ_endpoints: Dict[str, Any]
    whole_body_consequences: Dict[str, Any]
    potential_adverse_effects: List[Dict[str, str]]
    summary: str


class SpatialPharmacologyEngine:
    """
    Spatially resolved drug-target interaction engine for BioDigitalTwin.
    """
    def __init__(self):
        self.registry = get_target_registry()

    def simulate_spatial_interaction(
        self,
        organ: Union[str, OrganType],
        cell_type_id: str,
        target_id: str,
        drug_name: str,
        plasma_concentration_mg_L: float = 0.50,
        molecular_weight_g_mol: float = 450.0,
        fu_plasma: float = 0.15,
        partition_coefficient_Kp: float = 1.0,
        is_agonist: bool = True,
        is_blocker: bool = False,
        current_plasma_glucose_mg_dL: float = 120.0,
        current_sbp_mmHg: float = 130.0,
        current_hr_bpm: float = 75.0,
    ) -> SpatialSimulationResult:
        # 1. Resolve Organ & Cell Type
        if isinstance(organ, str):
            organ_enum = OrganType(organ.lower())
        else:
            organ_enum = organ

        cell_spec = CELL_TYPES_ATLAS.get(cell_type_id)
        if not cell_spec:
            raise ValueError(f"Unknown cell_type_id: '{cell_type_id}'. Available: {list(CELL_TYPES_ATLAS.keys())}")

        if cell_spec.organ != organ_enum:
            raise ValueError(f"Cell type '{cell_type_id}' belongs to {cell_spec.organ.value}, not {organ_enum.value}.")

        # 2. Resolve Target
        target = self.registry.get(target_id)
        if not target:
            raise ValueError(f"Unknown target_id: '{target_id}'. Registered: {list(self.registry._targets.keys())}")

        expression = target.get_expression(cell_type_id)
        subcellular_loc = expression.subcellular_location if expression else SubcellularLocation.PLASMA_MEMBRANE_NONPOLARIZED
        exp_status = expression.status if expression else ExpressionStatus.UNKNOWN_UNMEASURED
        density = target.get_effective_density_pmol_g(cell_type_id)
        copies = expression.copies_per_cell if expression else None
        evidence_src = expression.evidence.source_database if expression else "Literature Prediction"
        evidence_conf = expression.evidence.confidence_score if expression else 0.5

        # 3. Local Drug Exposure
        exposure = LocalExposureCalculator.calculate(
            organ=organ_enum,
            cell_type_id=cell_type_id,
            subcellular_location=subcellular_loc,
            plasma_total_mg_L=plasma_concentration_mg_L,
            molecular_weight_g_mol=molecular_weight_g_mol,
            fu_plasma=fu_plasma,
            partition_coefficient_Kp=partition_coefficient_Kp,
        )

        # 4. Target Engagement & Occupancy
        engagement = TargetEngagementEvaluator.evaluate(
            target=target,
            cell_type_id=cell_type_id,
            local_free_conc_nM=exposure.target_accessible_conc_nM,
            is_agonist=is_agonist,
        )

        # 5. Second Messenger Transduction
        rec_family_val = target.receptor_family.value if target.receptor_family else None
        second_msg = SecondMessengerTransducer.compute_steady_state(
            receptor_family=rec_family_val,
            functional_modulation_pct=engagement.functional_modulation_pct,
            is_agonist=is_agonist,
            is_blocker=is_blocker,
        )

        # 6. Cellular Response
        cell_resp = CellularResponseEvaluator.evaluate(
            cell_type_id=cell_type_id,
            target_id=target.target_id,
            target_class=target.target_class,
            engagement=engagement,
            second_messengers=second_msg,
        )

        # 7. Tissue Response
        tissue_resp = TissueResponseIntegrator.integrate(
            organ=organ_enum,
            region=cell_spec.region,
            cell_response=cell_resp,
            target_id=target.target_id,
        )

        # 8. Organ & Whole-Body Effects
        organ_report = OrganResponseEvaluator.evaluate(
            organ=organ_enum,
            target_id=target.target_id,
            tissue_response=tissue_resp,
            current_plasma_glucose_mg_dL=current_plasma_glucose_mg_dL,
            current_sbp_mmHg=current_sbp_mmHg,
            current_hr_bpm=current_hr_bpm,
        )

        return SpatialSimulationResult(
            organ=organ_enum.value,
            region=cell_spec.region,
            cell_type_id=cell_type_id,
            cell_type_name=cell_spec.name,
            subcellular_location=subcellular_loc.value,
            target_id=target.target_id,
            target_name=target.name,
            target_class=target.target_class.value,
            gene_symbol=target.gene_symbol,
            uniprot_id=target.uniprot_id,
            expression_status=exp_status.value,
            target_density_pmol_g=density,
            copies_per_cell=copies,
            evidence_source=evidence_src,
            evidence_confidence=evidence_conf,
            drug_name=drug_name,
            plasma_total_mg_L=exposure.plasma_total_mg_L,
            plasma_free_nM=exposure.plasma_free_nM,
            tissue_total_mg_L=exposure.tissue_total_mg_L,
            interstitial_free_nM=exposure.interstitial_free_nM,
            intracellular_free_nM=exposure.intracellular_free_nM,
            target_accessible_conc_nM=exposure.target_accessible_conc_nM,
            fractional_occupancy=round(engagement.fractional_occupancy, 4),
            modulation_type=engagement.modulation_type,
            functional_modulation_pct=round(engagement.functional_modulation_pct, 2),
            effective_Kd_nM=engagement.effective_Kd_nM,
            bound_target_density_pmol_g=round(engagement.bound_target_density_pmol_g, 3),
            residence_time_min=round(engagement.residence_time_min, 2),
            second_messengers={
                "cAMP_uM": second_msg.cAMP_uM,
                "cytosolic_Ca2_uM": second_msg.cytosolic_Ca2_uM,
                "IP3_uM": second_msg.IP3_uM,
                "PKA_activity_fraction": second_msg.PKA_activity_fraction,
                "Akt_phosphorylation_fraction": second_msg.Akt_phosphorylation_fraction,
                "MLCK_activity_fraction": second_msg.MLCK_activity_fraction,
            },
            primary_cellular_effect=cell_resp.primary_cellular_effect,
            quantitative_change_fold=cell_resp.quantitative_change_fold,
            affected_cellular_pathways=cell_resp.affected_cellular_pathways,
            cellular_description=cell_resp.description,
            organ_endpoints=organ_report.primary_organ_endpoints,
            whole_body_consequences=organ_report.whole_body_consequences,
            potential_adverse_effects=organ_report.potential_adverse_effects,
            summary=organ_report.summary,
        )


_global_spatial_engine: Optional[SpatialPharmacologyEngine] = None

def get_spatial_pharmacology_engine() -> SpatialPharmacologyEngine:
    global _global_spatial_engine
    if _global_spatial_engine is None:
        _global_spatial_engine = SpatialPharmacologyEngine()
    return _global_spatial_engine
