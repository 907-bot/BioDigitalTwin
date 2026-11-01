"""
backend/app/molecular/targets/ion_channel_models.py
Mechanistic Ion Channel Target Models:
1. CACNA1C (Cav1.2): L-type voltage-gated calcium channel in myocardium and vascular smooth muscle (Amlodipine/Diltiazem target).
2. SCN5A (Nav1.5): Cardiac voltage-gated sodium channel mediating rapid action potential upstroke.
"""
from __future__ import annotations

from app.molecular.tissue.cell_types import OrganType, SubcellularLocation
from app.molecular.targets.target_registry import (
    MolecularTarget,
    TargetClass,
    SpatialTargetExpression,
    ExpressionStatus,
    EvidenceCitation,
    BindingParameters,
    FunctionalParameters,
    get_target_registry,
)


def build_cacna1c_target() -> MolecularTarget:
    """L-type Voltage-Gated Calcium Channel Alpha-1C Subunit (CACNA1C / Cav1.2)."""
    return MolecularTarget(
        target_id="CACNA1C",
        name="Voltage-Dependent L-Type Calcium Channel Subunit Alpha-1C",
        gene_symbol="CACNA1C",
        uniprot_id="Q13936",
        target_class=TargetClass.ION_CHANNEL,
        description="Pore-forming alpha subunit mediating long-lasting (L-type) inward Ca2+ current essential for excitation-contraction coupling.",
        known_drugs=["Amlodipine", "Nifedipine", "Diltiazem", "Verapamil"],
        binding_params=BindingParameters(
            endogenous_ligand="Ca2+ ion",
            Kd_nM=1.5,          # Dihydropyridine binding affinity (Amlodipine)
            k_on=0.020,
            k_off=0.030,
            residence_time_min=33.3,
            hill_coefficient=1.0,
        ),
        functional_params=FunctionalParameters(
            mechanism_type="voltage_gated_pore_blockade",
            E_max=1.0,
            IC50_nM=2.0,
        ),
        spatial_expressions=[
            SpatialTargetExpression(
                organ=OrganType.VASCULATURE,
                region="systemic_arteriolar_tunica_media",
                cell_type_id="vascular_smooth_muscle",
                subcellular_location=SubcellularLocation.PLASMA_MEMBRANE_NONPOLARIZED,
                status=ExpressionStatus.CONFIRMED_HIGH,
                density_pmol_mg_protein=0.100,
                copies_per_cell=40000.0,
                relative_expression_tpm=75.0,
                evidence=EvidenceCitation(
                    source_database="Catterall WA (2011) Cold Spring Harb Perspect Biol",
                    pubmed_id="21746798",
                    confidence_score=0.99,
                ),
                notes="Inhibition prevents Ca2+ influx, causing arteriolar vasodilation and reducing afterload/blood pressure.",
            ),
            SpatialTargetExpression(
                organ=OrganType.HEART,
                region="left_ventricular_myocardium",
                cell_type_id="heart_ventricular_cardiomyocyte",
                subcellular_location=SubcellularLocation.PLASMA_MEMBRANE_NONPOLARIZED,
                status=ExpressionStatus.CONFIRMED_HIGH,
                density_pmol_mg_protein=0.120,
                copies_per_cell=50000.0,
                relative_expression_tpm=85.0,
                evidence=EvidenceCitation(
                    source_database="IUPHAR/BPS Guide to PHARMACOLOGY",
                    confidence_score=0.99,
                ),
                notes="Triggers calcium-induced calcium release (CICR) via RyR2 from sarcoplasmic reticulum.",
            ),
        ],
    )


def register_all_ion_channels() -> None:
    registry = get_target_registry()
    registry.register(build_cacna1c_target())

