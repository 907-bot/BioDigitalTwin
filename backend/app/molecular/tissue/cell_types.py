"""
backend/app/molecular/tissue/cell_types.py
Hierarchical Anatomical and Cellular Classification for Spatial Pharmacology.

Defines:
- OrganType
- AnatomicalRegion
- CellTypeSpec
- SubcellularLocation
- AnatomicalHierarchy (Organ -> Region -> Cell Type)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any


class OrganType(str, Enum):
    KIDNEY = "kidney"
    HEART = "heart"
    PANCREAS = "pancreas"
    LIVER = "liver"
    LUNG = "lung"
    BRAIN = "brain"
    SKELETAL_MUSCLE = "skeletal_muscle"
    ADIPOSE = "adipose"
    VASCULATURE = "vasculature"
    GASTROINTESTINAL = "gastrointestinal"


class SubcellularLocation(str, Enum):
    PLASMA_MEMBRANE_APICAL = "plasma_membrane_apical"
    PLASMA_MEMBRANE_BASOLATERAL = "plasma_membrane_basolateral"
    PLASMA_MEMBRANE_NONPOLARIZED = "plasma_membrane_nonpolarized"
    CYTOPLASM = "cytoplasm"
    NUCLEUS = "nucleus"
    ENDOPLASMIC_RETICULUM = "endoplasmic_reticulum"
    SARCOPLASMIC_RETICULUM = "sarcoplasmic_reticulum"
    MITOCHONDRIA = "mitochondria"
    ENDOSOME_LYSOSOME = "endosome_lysosome"


@dataclass
class CellTypeSpec:
    cell_type_id: str
    name: str
    organ: OrganType
    region: str
    description: str
    cell_volume_fL: float = 2000.0          # Typical mammalian cell volume ~2000 fL (2 pL)
    surface_area_um2: float = 1200.0        # Typical cell surface area ~1200 um^2
    is_polarized: bool = False
    typical_density_cells_per_g: float = 1e8 # ~10^8 cells per gram of tissue
    markers: List[str] = field(default_factory=list)


# =============================================================================
# Curated Anatomical & Cellular Atlas for Spatial Pharmacology
# =============================================================================
CELL_TYPES_ATLAS: Dict[str, CellTypeSpec] = {
    # --- KIDNEY ---
    "kidney_proximal_tubule_epithelial": CellTypeSpec(
        cell_type_id="kidney_proximal_tubule_epithelial",
        name="Proximal Tubular Epithelial Cell",
        organ=OrganType.KIDNEY,
        region="renal_cortex_proximal_convoluted_tubule",
        description="High-capacity reabsorptive cell equipped with apical microvillar brush border.",
        cell_volume_fL=2400.0,
        surface_area_um2=3500.0,
        is_polarized=True,
        typical_density_cells_per_g=1.2e8,
        markers=["SLC5A2", "CUBN", "LRP2"],
    ),
    "kidney_podocyte": CellTypeSpec(
        cell_type_id="kidney_podocyte",
        name="Glomerular Podocyte",
        organ=OrganType.KIDNEY,
        region="glomerulus",
        description="Terminally differentiated visceral epithelial cell forming the filtration slit barrier.",
        cell_volume_fL=4000.0,
        surface_area_um2=2800.0,
        is_polarized=True,
        typical_density_cells_per_g=2e7,
        markers=["NPHS1", "NPHS2", "WT1"],
    ),

    # --- HEART ---
    "heart_ventricular_cardiomyocyte": CellTypeSpec(
        cell_type_id="heart_ventricular_cardiomyocyte",
        name="Ventricular Cardiomyocyte",
        organ=OrganType.HEART,
        region="left_ventricular_myocardium",
        description="Excitable contractile cell responsible for cardiac stroke work and ventricular ejection.",
        cell_volume_fL=25000.0, # Large cylindrical cells ~25-30 pL
        surface_area_um2=11000.0,
        is_polarized=False,
        typical_density_cells_per_g=2.5e7,
        markers=["TNNT2", "MYH7", "ADRB1", "CACNA1C"],
    ),
    "heart_sinoatrial_pacemaker": CellTypeSpec(
        cell_type_id="heart_sinoatrial_pacemaker",
        name="Sinoatrial Nodal Pacemaker Cell",
        organ=OrganType.HEART,
        region="sinoatrial_node",
        description="Specialized autorhythmic cell generating intrinsic cardiac pace via funny (If) and L-type Ca2+ currents.",
        cell_volume_fL=8000.0,
        surface_area_um2=4000.0,
        is_polarized=False,
        typical_density_cells_per_g=1e7,
        markers=["HCN4", "TBX3", "CACNA1D"],
    ),

    # --- PANCREAS ---
    "pancreas_beta_cell": CellTypeSpec(
        cell_type_id="pancreas_beta_cell",
        name="Pancreatic Beta Cell",
        organ=OrganType.PANCREAS,
        region="islet_of_langerhans",
        description="Endocrine cell producing, packaging, and secreting insulin in response to glucose and incretins.",
        cell_volume_fL=1200.0,
        surface_area_um2=800.0,
        is_polarized=False,
        typical_density_cells_per_g=8e7,
        markers=["INS", "GLP1R", "ABCC8", "KCNJ11"],
    ),

    # --- LIVER ---
    "liver_hepatocyte": CellTypeSpec(
        cell_type_id="liver_hepatocyte",
        name="Hepatocyte",
        organ=OrganType.LIVER,
        region="hepatic_lobule_parenchyma",
        description="Primary metabolic cell responsible for gluconeogenesis, glycogen synthesis, bile production, and xenobiotic metabolism.",
        cell_volume_fL=3500.0,
        surface_area_um2=2200.0,
        is_polarized=True,
        typical_density_cells_per_g=1.4e8,
        markers=["ALB", "CYP3A4", "HMGCR", "INSR"],
    ),

    # --- SKELETAL MUSCLE ---
    "skeletal_myocyte": CellTypeSpec(
        cell_type_id="skeletal_myocyte",
        name="Skeletal Muscle Myocyte",
        organ=OrganType.SKELETAL_MUSCLE,
        region="vastus_lateralis_fascicle",
        description="Multinucleated contractile fiber responsible for insulin-stimulated glucose disposal and physical locomotion.",
        cell_volume_fL=40000.0,
        surface_area_um2=15000.0,
        is_polarized=False,
        typical_density_cells_per_g=2e7,
        markers=["ACTA1", "SLC2A4", "INSR"],
    ),

    # --- VASCULATURE ---
    "vascular_smooth_muscle": CellTypeSpec(
        cell_type_id="vascular_smooth_muscle",
        name="Vascular Smooth Muscle Cell",
        organ=OrganType.VASCULATURE,
        region="systemic_arteriolar_tunica_media",
        description="Spindle-shaped cell regulating systemic vascular resistance and arterial blood pressure via contractile tone.",
        cell_volume_fL=1800.0,
        surface_area_um2=1400.0,
        is_polarized=False,
        typical_density_cells_per_g=1e8,
        markers=["ACTA2", "MYH11", "AGTR1", "CACNA1C"],
    ),
    "vascular_endothelial_cell": CellTypeSpec(
        cell_type_id="vascular_endothelial_cell",
        name="Vascular Endothelial Cell",
        organ=OrganType.VASCULATURE,
        region="capillary_bed_tunica_intima",
        description="Semi-permeable barrier cell mediating nitric oxide release and vascular permeability.",
        cell_volume_fL=1000.0,
        surface_area_um2=1200.0,
        is_polarized=True,
        typical_density_cells_per_g=2e8,
        markers=["PECAM1", "VWF", "NOS3", "ACE"],
    ),

    # --- ADIPOSE ---
    "white_adipocyte": CellTypeSpec(
        cell_type_id="white_adipocyte",
        name="White Adipocyte",
        organ=OrganType.ADIPOSE,
        region="subcutaneous_and_visceral_depot",
        description="Unilocular lipid-storing cell regulating free fatty acid release, leptin, and adiponectin secretion.",
        cell_volume_fL=30000.0, # Large unilocular lipid droplet
        surface_area_um2=12000.0,
        is_polarized=False,
        typical_density_cells_per_g=1.5e7,
        markers=["ADIPOQ", "LEP", "PPARG", "SLC2A4"],
    ),
}


def get_anatomical_hierarchy() -> Dict[str, Any]:
    """
    Returns nested hierarchy: Organ -> Regions -> Cell Types for the UI/simulation engine.
    """
    hierarchy: Dict[str, Dict[str, List[Dict[str, str]]]] = {}
    for spec in CELL_TYPES_ATLAS.values():
        organ_key = spec.organ.value
        region_key = spec.region
        if organ_key not in hierarchy:
            hierarchy[organ_key] = {}
        if region_key not in hierarchy[organ_key]:
            hierarchy[organ_key][region_key] = []
        hierarchy[organ_key][region_key].append({
            "cell_type_id": spec.cell_type_id,
            "name": spec.name,
            "description": spec.description,
            "markers": spec.markers,
        })
    return hierarchy
