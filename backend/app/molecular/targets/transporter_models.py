"""
backend/app/molecular/targets/transporter_models.py
Mechanistic Transporter Target Models:
1. SGLT2 (SLC5A2): Apical sodium-glucose cotransporter in renal proximal tubule.
2. GLUT4 (SLC2A4): Facilitated hexose transporter in muscle and adipose tissue.
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


def build_sglt2_target() -> MolecularTarget:
    """Sodium/Glucose Cotransporter 2 (SLC5A2 / SGLT2)."""
    return MolecularTarget(
        target_id="SGLT2",
        name="Solute Carrier Family 5 Member 2 (SGLT2)",
        gene_symbol="SLC5A2",
        uniprot_id="P31639",
        target_class=TargetClass.TRANSPORTER,
        description="High-capacity, low-affinity 1:1 Na+/glucose cotransporter reabsorbing ~90% of filtered glucose in proximal tubule.",
        known_drugs=["Empagliflozin", "Dapagliflozin", "Canagliflozin", "Ertugliflozin"],
        binding_params=BindingParameters(
            endogenous_ligand="D-Glucose",
            Kd_nM=5000000.0,    # Km for glucose ~5.0 mM (5,000,000 nM)
            k_on=0.001,
            k_off=5000.0,
            residence_time_min=0.0002,
            hill_coefficient=1.0,
        ),
        functional_params=FunctionalParameters(
            mechanism_type="competitive_transporter_inhibition",
            E_max=1.0,
            Km_uM=5000.0,       # 5.0 mM
            IC50_nM=3.1,        # Empagliflozin exhibits IC50 ~3.1 nM against human SGLT2
            Vmax_pmol_min_mg=375000.0,  # ~375 mg/min (TmG) whole-kidney baseline capacity
        ),
        spatial_expressions=[
            SpatialTargetExpression(
                organ=OrganType.KIDNEY,
                region="renal_cortex_proximal_convoluted_tubule",
                cell_type_id="kidney_proximal_tubule_epithelial",
                subcellular_location=SubcellularLocation.PLASMA_MEMBRANE_APICAL,
                status=ExpressionStatus.CONFIRMED_HIGH,
                density_pmol_mg_protein=0.450,
                copies_per_cell=180000.0,
                relative_expression_tpm=210.0,
                evidence=EvidenceCitation(
                    source_database="Vallon V (2011) Diabetologia; Wright EM (2011) Physiol Rev",
                    pubmed_id="21153632",
                    confidence_score=0.99,
                ),
                notes="Expressed strictly on the apical brush border membrane of S1 and S2 proximal tubule segments.",
            ),
        ],
    )


def build_glut4_target() -> MolecularTarget:
    """Glucose Transporter Type 4 (SLC2A4 / GLUT4)."""
    return MolecularTarget(
        target_id="GLUT4",
        name="Solute Carrier Family 2 Member 4 (GLUT4)",
        gene_symbol="SLC2A4",
        uniprot_id="P14672",
        target_class=TargetClass.TRANSPORTER,
        description="Insulin-regulated facilitative glucose transporter sequestered in intracellular vesicles until insulin stimulation.",
        known_drugs=["Insulin (indirect regulator)", "Metformin (AMPK-mediated translocation)"],
        binding_params=BindingParameters(
            endogenous_ligand="D-Glucose",
            Kd_nM=5000000.0,    # Km ~5.0 mM
            k_on=0.001,
            k_off=5000.0,
            residence_time_min=0.0002,
        ),
        functional_params=FunctionalParameters(
            mechanism_type="facilitative_diffusion",
            E_max=1.0,
            Km_uM=5000.0,
        ),
        spatial_expressions=[
            SpatialTargetExpression(
                organ=OrganType.SKELETAL_MUSCLE,
                region="vastus_lateralis_fascicle",
                cell_type_id="skeletal_myocyte",
                subcellular_location=SubcellularLocation.CYTOPLASM, # Vesicular until translocated
                status=ExpressionStatus.CONFIRMED_HIGH,
                density_pmol_mg_protein=0.600,
                copies_per_cell=250000.0,
                relative_expression_tpm=180.0,
                evidence=EvidenceCitation(
                    source_database="Human Protein Atlas; Bryant NJ et al. (2002) Nat Rev Mol Cell Bio",
                    pubmed_id="11988770",
                    confidence_score=0.99,
                ),
                notes="Translocates to sarcolemma in response to Akt phosphorylation of AS160 (TBC1D4).",
            ),
            SpatialTargetExpression(
                organ=OrganType.ADIPOSE,
                region="subcutaneous_and_visceral_depot",
                cell_type_id="white_adipocyte",
                subcellular_location=SubcellularLocation.CYTOPLASM,
                status=ExpressionStatus.CONFIRMED_HIGH,
                density_pmol_mg_protein=0.400,
                copies_per_cell=150000.0,
                relative_expression_tpm=140.0,
                evidence=EvidenceCitation(
                    source_database="Human Protein Atlas",
                    confidence_score=0.98,
                ),
                notes="Provides glucose substrate for de novo fatty acid esterification and glycerol-3-phosphate synthesis.",
            ),
        ],
    )


def register_all_transporters() -> None:
    registry = get_target_registry()
    registry.register(build_sglt2_target())
    registry.register(build_glut4_target())
