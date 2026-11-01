"""
backend/app/molecular/targets/receptor_models.py
Mechanistic Receptor Models across Major Superfamilies:
1. GPCR-Gs: GLP1R (Pancreatic Beta Cell), ADRB1 (Cardiomyocyte)
2. GPCR-Gq: AGTR1 (Vascular Smooth Muscle Cell)
3. Receptor Tyrosine Kinase (RTK): INSR (Hepatocyte & Skeletal Myocyte)
4. Nuclear Receptor: NR3C1 (Glucocorticoid Receptor)
5. Ligand-Gated Ion Channel: CHRNA1 (Skeletal Neuromuscular Junction)

Each receptor defines its exact spatial distribution across organs, tissues, and cell types,
along with thermodynamic binding constants and downstream functional signaling couplings.
"""
from __future__ import annotations

from typing import List
from app.molecular.tissue.cell_types import OrganType, SubcellularLocation
from app.molecular.targets.target_registry import (
    MolecularTarget,
    TargetClass,
    ReceptorFamily,
    SpatialTargetExpression,
    ExpressionStatus,
    EvidenceCitation,
    BindingParameters,
    FunctionalParameters,
    get_target_registry,
)


def build_glp1r_target() -> MolecularTarget:
    """
    Glucagon-Like Peptide-1 Receptor (GLP1R)
    Primary role: Incretin-stimulated glucose-dependent insulin secretion.
    """
    return MolecularTarget(
        target_id="GLP1R",
        name="Glucagon-Like Peptide 1 Receptor",
        gene_symbol="GLP1R",
        uniprot_id="P43220",
        target_class=TargetClass.RECEPTOR,
        receptor_family=ReceptorFamily.GPCR_GS,
        description="Class B1 G-protein coupled receptor mediating incretin response in pancreatic beta cells.",
        known_drugs=["Semaglutide", "Liraglutide", "Tirzepatide", "Exenatide", "Dulaglutide"],
        binding_params=BindingParameters(
            endogenous_ligand="GLP-1(7-36)amide",
            Kd_nM=0.40,
            k_on=0.050,         # 1 / (nM * min)
            k_off=0.020,        # 1 / min -> residence time 50 min
            residence_time_min=50.0,
            hill_coefficient=1.0,
        ),
        functional_params=FunctionalParameters(
            mechanism_type="operational_agonism",
            E_max=1.0,
            tau=12.0,           # High receptor reserve in healthy islets
            EC50_nM=0.35,
        ),
        spatial_expressions=[
            SpatialTargetExpression(
                organ=OrganType.PANCREAS,
                region="islet_of_langerhans",
                cell_type_id="pancreas_beta_cell",
                subcellular_location=SubcellularLocation.PLASMA_MEMBRANE_NONPOLARIZED,
                status=ExpressionStatus.CONFIRMED_HIGH,
                density_pmol_mg_protein=0.060,     # ~6.0 pmol/g tissue
                copies_per_cell=3500.0,
                relative_expression_tpm=48.5,
                evidence=EvidenceCitation(
                    source_database="Human Protein Atlas v23 / IUPHAR",
                    pubmed_id="29456073",
                    experimental_method="Targeted Mass Spectrometry & RNA-Seq",
                    confidence_score=0.98,
                ),
                notes="Stimulates cAMP production, triggering exocytosis of insulin granules.",
            ),
            SpatialTargetExpression(
                organ=OrganType.GASTROINTESTINAL,
                region="myenteric_and_submucosal_plexus",
                cell_type_id="enteric_neuron",
                subcellular_location=SubcellularLocation.PLASMA_MEMBRANE_NONPOLARIZED,
                status=ExpressionStatus.CONFIRMED_MEDIUM,
                density_pmol_mg_protein=0.020,
                copies_per_cell=1200.0,
                relative_expression_tpm=14.2,
                evidence=EvidenceCitation(
                    source_database="GTEx v8",
                    pubmed_id="30301824",
                    confidence_score=0.92,
                ),
                notes="Slows gastric emptying rate by vagal cholinergic modulation.",
            ),
            SpatialTargetExpression(
                organ=OrganType.KIDNEY,
                region="renal_cortex_proximal_convoluted_tubule",
                cell_type_id="kidney_proximal_tubule_epithelial",
                subcellular_location=SubcellularLocation.PLASMA_MEMBRANE_APICAL,
                status=ExpressionStatus.CONFIRMED_LOW,
                density_pmol_mg_protein=0.005,
                copies_per_cell=300.0,
                relative_expression_tpm=3.1,
                evidence=EvidenceCitation(
                    source_database="Human Protein Atlas",
                    confidence_score=0.85,
                ),
                notes="Promotes natriuresis via NHE3 transporter phosphorylation.",
            ),
            SpatialTargetExpression(
                organ=OrganType.SKELETAL_MUSCLE,
                region="vastus_lateralis_fascicle",
                cell_type_id="skeletal_myocyte",
                subcellular_location=SubcellularLocation.PLASMA_MEMBRANE_NONPOLARIZED,
                status=ExpressionStatus.ABSENT,
                copies_per_cell=0.0,
                relative_expression_tpm=0.05,
                evidence=EvidenceCitation(
                    source_database="Human Protein Atlas",
                    confidence_score=0.95,
                ),
                notes="GLP1R is NOT directly expressed in skeletal myocytes; insulin mediates peripheral glucose uptake.",
            ),
        ],
    )


def build_adrb1_target() -> MolecularTarget:
    """
    Beta-1 Adrenergic Receptor (ADRB1)
    Primary role: Sympathetic regulation of myocardial contractility and heart rate.
    """
    return MolecularTarget(
        target_id="ADRB1",
        name="Beta-1 Adrenergic Receptor",
        gene_symbol="ADRB1",
        uniprot_id="P08588",
        target_class=TargetClass.RECEPTOR,
        receptor_family=ReceptorFamily.GPCR_GS,
        description="Predominant adrenergic receptor in mammalian myocardium mediating inotropic and chronotropic responses.",
        known_drugs=["Atenolol", "Metoprolol", "Bisoprolol", "Carvedilol", "Epinephrine", "Dobutamine"],
        binding_params=BindingParameters(
            endogenous_ligand="Norepinephrine",
            Kd_nM=5.0,
            k_on=0.015,
            k_off=0.075,
            residence_time_min=13.3,
            hill_coefficient=1.0,
        ),
        functional_params=FunctionalParameters(
            mechanism_type="operational_agonism",
            E_max=1.0,
            tau=15.0,           # High spare receptor reserve in human left ventricle
            EC50_nM=2.5,
        ),
        spatial_expressions=[
            SpatialTargetExpression(
                organ=OrganType.HEART,
                region="left_ventricular_myocardium",
                cell_type_id="heart_ventricular_cardiomyocyte",
                subcellular_location=SubcellularLocation.PLASMA_MEMBRANE_NONPOLARIZED,
                status=ExpressionStatus.CONFIRMED_HIGH,
                density_pmol_mg_protein=0.120,     # ~12 pmol/g tissue
                copies_per_cell=75000.0,
                relative_expression_tpm=72.0,
                evidence=EvidenceCitation(
                    source_database="Brodde OE (1991) Pharmacol Rev; IUPHAR",
                    pubmed_id="1684841",
                    confidence_score=0.99,
                ),
                notes="Couples to Gs-adenylyl cyclase, phosphorylating L-type Ca2+ channels (Cav1.2) to increase stroke work.",
            ),
            SpatialTargetExpression(
                organ=OrganType.HEART,
                region="sinoatrial_node",
                cell_type_id="heart_sinoatrial_pacemaker",
                subcellular_location=SubcellularLocation.PLASMA_MEMBRANE_NONPOLARIZED,
                status=ExpressionStatus.CONFIRMED_HIGH,
                density_pmol_mg_protein=0.100,
                copies_per_cell=60000.0,
                relative_expression_tpm=65.0,
                evidence=EvidenceCitation(
                    source_database="IUPHAR/BPS",
                    confidence_score=0.97,
                ),
                notes="Accelerates phase 4 diastolic depolarization, elevating heart rate.",
            ),
            SpatialTargetExpression(
                organ=OrganType.KIDNEY,
                region="juxtaglomerular_apparatus",
                cell_type_id="juxtaglomerular_granular_cell",
                subcellular_location=SubcellularLocation.PLASMA_MEMBRANE_NONPOLARIZED,
                status=ExpressionStatus.CONFIRMED_MEDIUM,
                density_pmol_mg_protein=0.035,
                copies_per_cell=15000.0,
                relative_expression_tpm=22.0,
                evidence=EvidenceCitation(
                    source_database="Human Protein Atlas",
                    confidence_score=0.94,
                ),
                notes="Stimulates renin secretion upon renal sympathetic nerve activation.",
            ),
        ],
    )


def build_agtr1_target() -> MolecularTarget:
    """
    Angiotensin II Receptor Type 1 (AGTR1)
    Primary role: Vasoconstriction, systemic vascular resistance, and aldosterone secretion.
    """
    return MolecularTarget(
        target_id="AGTR1",
        name="Angiotensin II Receptor Type 1",
        gene_symbol="AGTR1",
        uniprot_id="P30556",
        target_class=TargetClass.RECEPTOR,
        receptor_family=ReceptorFamily.GPCR_GQ,
        description="Gq-coupled receptor mediating vasoconstriction, aldosterone release, and renal sodium retention.",
        known_drugs=["Losartan", "Valsartan", "Telmisartan", "Candesartan", "Angiotensin II"],
        binding_params=BindingParameters(
            endogenous_ligand="Angiotensin II",
            Kd_nM=2.0,
            k_on=0.025,
            k_off=0.050,
            residence_time_min=20.0,
            hill_coefficient=1.1,
        ),
        functional_params=FunctionalParameters(
            mechanism_type="operational_agonism",
            E_max=1.0,
            tau=6.0,
            EC50_nM=1.8,
        ),
        spatial_expressions=[
            SpatialTargetExpression(
                organ=OrganType.VASCULATURE,
                region="systemic_arteriolar_tunica_media",
                cell_type_id="vascular_smooth_muscle",
                subcellular_location=SubcellularLocation.PLASMA_MEMBRANE_NONPOLARIZED,
                status=ExpressionStatus.CONFIRMED_HIGH,
                density_pmol_mg_protein=0.180,     # ~18 pmol/g tissue
                copies_per_cell=45000.0,
                relative_expression_tpm=88.4,
                evidence=EvidenceCitation(
                    source_database="Timmermans PB et al. (1993) Pharmacol Rev; GTEx",
                    pubmed_id="8248340",
                    confidence_score=0.99,
                ),
                notes="Triggers IP3/DAG-dependent calcium mobilization, causing contraction and elevating SVR.",
            ),
            SpatialTargetExpression(
                organ=OrganType.KIDNEY,
                region="glomerulus",
                cell_type_id="kidney_podocyte",
                subcellular_location=SubcellularLocation.PLASMA_MEMBRANE_NONPOLARIZED,
                status=ExpressionStatus.CONFIRMED_MEDIUM,
                density_pmol_mg_protein=0.040,
                copies_per_cell=12000.0,
                relative_expression_tpm=18.0,
                evidence=EvidenceCitation(
                    source_database="Human Protein Atlas",
                    confidence_score=0.91,
                ),
                notes="Regulates intraglomerular hemodynamics and filtration barrier permeability.",
            ),
        ],
    )


def build_insr_target() -> MolecularTarget:
    """
    Insulin Receptor (INSR)
    Primary role: Systemic glucose uptake, glycogenesis, and lipogenesis.
    """
    return MolecularTarget(
        target_id="INSR",
        name="Insulin Receptor",
        gene_symbol="INSR",
        uniprot_id="P06213",
        target_class=TargetClass.RECEPTOR,
        receptor_family=ReceptorFamily.RECEPTOR_TYROSINE_KINASE,
        description="Heterotetrameric (alpha2-beta2) receptor tyrosine kinase coordinating anabolic glucose and lipid metabolism.",
        known_drugs=["Regular Insulin", "Insulin Glargine", "Insulin Lispro", "Insulin Aspart"],
        binding_params=BindingParameters(
            endogenous_ligand="Insulin",
            Kd_nM=0.70,
            k_on=0.040,
            k_off=0.028,
            residence_time_min=35.7,
            hill_coefficient=1.0,
        ),
        functional_params=FunctionalParameters(
            mechanism_type="tyrosine_kinase_activation",
            E_max=1.0,
            tau=8.0,
            EC50_nM=0.60,
        ),
        spatial_expressions=[
            SpatialTargetExpression(
                organ=OrganType.LIVER,
                region="hepatic_lobule_parenchyma",
                cell_type_id="liver_hepatocyte",
                subcellular_location=SubcellularLocation.PLASMA_MEMBRANE_BASOLATERAL,
                status=ExpressionStatus.CONFIRMED_HIGH,
                density_pmol_mg_protein=0.350,     # ~35 pmol/g tissue
                copies_per_cell=120000.0,
                relative_expression_tpm=125.0,
                evidence=EvidenceCitation(
                    source_database="Kahn CR (1981) J Clin Invest; Bevan P (2001) J Cell Sci",
                    pubmed_id="7019246",
                    confidence_score=0.99,
                ),
                notes="Suppresses glycogenolysis and gluconeogenesis via Akt/FoxO1 phosphorylation.",
            ),
            SpatialTargetExpression(
                organ=OrganType.SKELETAL_MUSCLE,
                region="vastus_lateralis_fascicle",
                cell_type_id="skeletal_myocyte",
                subcellular_location=SubcellularLocation.PLASMA_MEMBRANE_NONPOLARIZED,
                status=ExpressionStatus.CONFIRMED_HIGH,
                density_pmol_mg_protein=0.280,
                copies_per_cell=90000.0,
                relative_expression_tpm=95.0,
                evidence=EvidenceCitation(
                    source_database="Human Protein Atlas",
                    confidence_score=0.98,
                ),
                notes="Drives GLUT4 vesicle translocation to sarcolemma, accounting for ~70-80% of postprandial glucose disposal.",
            ),
            SpatialTargetExpression(
                organ=OrganType.ADIPOSE,
                region="subcutaneous_and_visceral_depot",
                cell_type_id="white_adipocyte",
                subcellular_location=SubcellularLocation.PLASMA_MEMBRANE_NONPOLARIZED,
                status=ExpressionStatus.CONFIRMED_HIGH,
                density_pmol_mg_protein=0.200,
                copies_per_cell=60000.0,
                relative_expression_tpm=80.0,
                evidence=EvidenceCitation(
                    source_database="Human Protein Atlas",
                    confidence_score=0.96,
                ),
                notes="Inhibits hormone-sensitive lipase (HSL), halting lipolysis and suppressing circulating free fatty acids.",
            ),
        ],
    )


def build_nr3c1_target() -> MolecularTarget:
    """
    Glucocorticoid Receptor (NR3C1)
    Primary role: Stress response, anti-inflammatory transcription, and hepatic gluconeogenesis.
    """
    return MolecularTarget(
        target_id="NR3C1",
        name="Nuclear Receptor Subfamily 3 Group C Member 1 (Glucocorticoid Receptor)",
        gene_symbol="NR3C1",
        uniprot_id="P04150",
        target_class=TargetClass.RECEPTOR,
        receptor_family=ReceptorFamily.NUCLEAR_RECEPTOR,
        description="Ligand-dependent transcription factor translocating from cytoplasm to nucleus upon cortisol binding.",
        known_drugs=["Dexamethasone", "Prednisone", "Hydrocortisone", "Methylprednisolone"],
        binding_params=BindingParameters(
            endogenous_ligand="Cortisol",
            Kd_nM=15.0,
            k_on=0.010,
            k_off=0.150,
            residence_time_min=6.7,
            hill_coefficient=1.0,
        ),
        functional_params=FunctionalParameters(
            mechanism_type="nuclear_translocation_transactivation",
            E_max=1.0,
            tau=5.0,
            EC50_nM=18.0,
        ),
        spatial_expressions=[
            SpatialTargetExpression(
                organ=OrganType.LIVER,
                region="hepatic_lobule_parenchyma",
                cell_type_id="liver_hepatocyte",
                subcellular_location=SubcellularLocation.CYTOPLASM,
                status=ExpressionStatus.CONFIRMED_HIGH,
                density_pmol_mg_protein=0.150,
                copies_per_cell=50000.0,
                relative_expression_tpm=65.0,
                evidence=EvidenceCitation(
                    source_database="Human Protein Atlas",
                    confidence_score=0.97,
                ),
                notes="Induces PEPCK and G6Pase transcription, driving hepatic glucose output under stress.",
            ),
        ],
    )


def build_chrna1_target() -> MolecularTarget:
    """
    Nicotinic Acetylcholine Receptor Subunit Alpha-1 (CHRNA1)
    Primary role: Ligand-gated ion flux triggering skeletal muscle contraction.
    """
    return MolecularTarget(
        target_id="CHRNA1",
        name="Neuronal/Muscle Nicotinic Acetylcholine Receptor Alpha-1",
        gene_symbol="CHRNA1",
        uniprot_id="P02708",
        target_class=TargetClass.RECEPTOR,
        receptor_family=ReceptorFamily.LIGAND_GATED_ION_CHANNEL,
        description="Pentameric ligand-gated cation channel mediating fast chemical synaptic transmission at neuromuscular junction.",
        known_drugs=["Acetylcholine", "Succinylcholine", "Rocuronium", "Vecuronium"],
        binding_params=BindingParameters(
            endogenous_ligand="Acetylcholine",
            Kd_nM=50.0,
            k_on=0.10,
            k_off=5.0,
            residence_time_min=0.20,
            hill_coefficient=2.0,       # Cooperative dual ACh binding
        ),
        functional_params=FunctionalParameters(
            mechanism_type="channel_opening_depolarization",
            E_max=1.0,
            tau=20.0,
            EC50_nM=45.0,
        ),
        spatial_expressions=[
            SpatialTargetExpression(
                organ=OrganType.SKELETAL_MUSCLE,
                region="vastus_lateralis_fascicle",
                cell_type_id="skeletal_myocyte",
                subcellular_location=SubcellularLocation.PLASMA_MEMBRANE_NONPOLARIZED,
                status=ExpressionStatus.CONFIRMED_HIGH,
                density_pmol_mg_protein=0.080,
                copies_per_cell=30000.0,
                relative_expression_tpm=40.0,
                evidence=EvidenceCitation(
                    source_database="IUPHAR/BPS",
                    confidence_score=0.99,
                ),
                notes="Opening triggers rapid Na+ and Ca2+ influx, evoking endplate potentials.",
            ),
        ],
    )


def register_all_receptors() -> None:
    """Register all defined receptor models in the global MolecularTargetRegistry."""
    registry = get_target_registry()
    registry.register(build_glp1r_target())
    registry.register(build_adrb1_target())
    registry.register(build_agtr1_target())
    registry.register(build_insr_target())
    registry.register(build_nr3c1_target())
    registry.register(build_chrna1_target())
