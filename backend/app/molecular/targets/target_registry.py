"""
backend/app/molecular/targets/target_registry.py
Molecular Target Registry and Spatial Expression Specifications.

Supports all major target classes:
- Receptors (GPCRs, RTKs, Nuclear Receptors, Ligand-Gated Ion Channels)
- Enzymes (HMG-CoA Reductase, ACE, DPP-4)
- Transporters (SGLT2, GLUT4, etc.)
- Ion Channels (CACNA1C, SCN5A, etc.)
- Structural Proteins & Nucleic Acids

Distinguishes between CONFIRMED expression, ABSENT expression, and UNKNOWN/UNMEASURED data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from app.molecular.tissue.cell_types import OrganType, SubcellularLocation


class TargetClass(str, Enum):
    RECEPTOR = "receptor"
    ENZYME = "enzyme"
    TRANSPORTER = "transporter"
    ION_CHANNEL = "ion_channel"
    STRUCTURAL_PROTEIN = "structural_protein"
    NUCLEIC_ACID = "nucleic_acid"


class ReceptorFamily(str, Enum):
    GPCR_GS = "gpcr_gs"
    GPCR_GI = "gpcr_gi"
    GPCR_GQ = "gpcr_gq"
    GPCR_G12_13 = "gpcr_g12_13"
    RECEPTOR_TYROSINE_KINASE = "receptor_tyrosine_kinase"
    NUCLEAR_RECEPTOR = "nuclear_receptor"
    LIGAND_GATED_ION_CHANNEL = "ligand_gated_ion_channel"
    CYTOKINE_RECEPTOR = "cytokine_receptor"


class ExpressionStatus(str, Enum):
    CONFIRMED_HIGH = "confirmed_high"
    CONFIRMED_MEDIUM = "confirmed_medium"
    CONFIRMED_LOW = "confirmed_low"
    ABSENT = "absent"                   # Experimentally validated as not expressed
    UNKNOWN_UNMEASURED = "unknown"       # NOT measured; never assume zero!


@dataclass
class EvidenceCitation:
    source_database: str                # e.g., "Human Protein Atlas v23", "IUPHAR/BPS Guide to PHARMACOLOGY", "GTEx v8"
    pubmed_id: Optional[str] = None
    doi: Optional[str] = None
    experimental_method: str = "RNA-Seq and Immunohistochemistry"
    confidence_score: float = 1.0       # 0.0 to 1.0


@dataclass
class SpatialTargetExpression:
    """
    Spatially resolved target expression record within a specific anatomical cell population.
    """
    organ: OrganType
    region: str
    cell_type_id: str
    subcellular_location: SubcellularLocation
    status: ExpressionStatus
    # Quantitative densities (when available)
    density_pmol_mg_protein: Optional[float] = None
    copies_per_cell: Optional[float] = None
    relative_expression_tpm: Optional[float] = None  # Transcripts Per Million (TPM)
    evidence: EvidenceCitation = field(default_factory=lambda: EvidenceCitation(source_database="IUPHAR/BPS"))
    notes: str = ""

    @property
    def is_expressed(self) -> bool:
        return self.status in [
            ExpressionStatus.CONFIRMED_HIGH,
            ExpressionStatus.CONFIRMED_MEDIUM,
            ExpressionStatus.CONFIRMED_LOW,
        ]


@dataclass
class BindingParameters:
    """
    Thermodynamic and kinetic interaction constants between ligand/drug and molecular target.
    """
    endogenous_ligand: Optional[str] = None
    Kd_nM: float = 1.0                 # Equilibrium dissociation constant [nM]
    k_on: float = 0.05                 # Association rate [1 / (nM * min)]
    k_off: float = 0.05                # Dissociation rate [1 / min] (Kd = k_off / k_on)
    kon_molar_sec: Optional[float] = None  # in M^-1 * s^-1
    residence_time_min: float = 20.0   # 1 / k_off
    hill_coefficient: float = 1.0      # Cooperativity coefficient
    allosteric_modulators: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.k_on > 0:
            self.Kd_nM = self.k_off / self.k_on
        if self.k_off > 0:
            self.residence_time_min = 1.0 / self.k_off


@dataclass
class FunctionalParameters:
    """
    Signal transduction or catalytic efficiency parameters.
    """
    mechanism_type: str                # "agonism", "competitive_inhibition", "pore_blockade", "allosteric_enhancement"
    E_max: float = 1.0                 # Maximal intrinsic activity (1.0 = full agonist, <1.0 = partial)
    tau: float = 10.0                  # Black-Leff operational transducer ratio
    IC50_nM: Optional[float] = None    # Inhibitory concentration 50%
    EC50_nM: Optional[float] = None    # Effective concentration 50%
    Km_uM: Optional[float] = None      # Michaelis-Menten constant for enzymes/transporters
    Vmax_pmol_min_mg: Optional[float] = None # Maximal catalytic / transport capacity
    turnover_number_kcat_s: Optional[float] = None


@dataclass
class MolecularTarget:
    """
    Master specification for any molecular drug target.
    """
    target_id: str                     # e.g., "GLP1R", "SGLT2", "HMGCR", "CACNA1C"
    name: str                          # Full scientific name
    gene_symbol: str                   # Official HGNC Gene Symbol
    uniprot_id: str                    # UniProt accession
    target_class: TargetClass
    receptor_family: Optional[ReceptorFamily] = None
    spatial_expressions: List[SpatialTargetExpression] = field(default_factory=list)
    binding_params: BindingParameters = field(default_factory=BindingParameters)
    functional_params: FunctionalParameters = field(default_factory=lambda: FunctionalParameters(mechanism_type="operational_agonism"))
    description: str = ""
    known_drugs: List[str] = field(default_factory=list)

    def get_expression(self, cell_type_id: str) -> Optional[SpatialTargetExpression]:
        for exp in self.spatial_expressions:
            if exp.cell_type_id == cell_type_id:
                return exp
        return None

    def get_effective_density_pmol_g(self, cell_type_id: str) -> float:
        """
        Returns estimated target density in pmol/g tissue for the specified cell population.
        If unknown or absent, returns 0.0 or default estimate.
        """
        exp = self.get_expression(cell_type_id)
        if not exp or not exp.is_expressed:
            return 0.0
        if exp.density_pmol_mg_protein is not None:
            # Assume ~100 mg total protein per gram of wet tissue
            return exp.density_pmol_mg_protein * 100.0
        if exp.copies_per_cell is not None:
            # 1 mole = 6.022e23 molecules, ~1e8 cells per gram
            moles_per_cell = exp.copies_per_cell / 6.022e23
            return moles_per_cell * 1e8 * 1e12  # pmol/g
        # Fallback based on qualitative tier
        tier_map = {
            ExpressionStatus.CONFIRMED_HIGH: 25.0,
            ExpressionStatus.CONFIRMED_MEDIUM: 10.0,
            ExpressionStatus.CONFIRMED_LOW: 2.0,
        }
        return tier_map.get(exp.status, 5.0)


# =============================================================================
# Global Target Registry Singleton
# =============================================================================
class MolecularTargetRegistry:
    """
    Thread-safe registry indexing all validated molecular targets across the human organism.
    """
    _instance: Optional[MolecularTargetRegistry] = None

    def __init__(self):
        self._targets: Dict[str, MolecularTarget] = {}

    @classmethod
    def get_instance(cls) -> MolecularTargetRegistry:
        if cls._instance is None:
            cls._instance = MolecularTargetRegistry()
        return cls._instance

    def register(self, target: MolecularTarget) -> None:
        self._targets[target.target_id.upper()] = target

    def get(self, target_id: str) -> Optional[MolecularTarget]:
        return self._targets.get(target_id.upper())

    def list_targets(
        self,
        target_class: Optional[TargetClass] = None,
        organ: Optional[OrganType] = None,
        cell_type_id: Optional[str] = None,
    ) -> List[MolecularTarget]:
        results = []
        for target in self._targets.values():
            if target_class and target.target_class != target_class:
                continue
            if organ:
                has_organ = any(exp.organ == organ for exp in target.spatial_expressions)
                if not has_organ:
                    continue
            if cell_type_id:
                has_cell = any(exp.cell_type_id == cell_type_id and exp.is_expressed for exp in target.spatial_expressions)
                if not has_cell:
                    continue
            results.append(target)
        return results

    def get_summary_catalog(self) -> List[Dict[str, Any]]:
        return [
            {
                "target_id": t.target_id,
                "name": t.name,
                "gene_symbol": t.gene_symbol,
                "target_class": t.target_class.value,
                "receptor_family": t.receptor_family.value if t.receptor_family else None,
                "expressed_organs": list(set(exp.organ.value for exp in t.spatial_expressions if exp.is_expressed)),
                "known_drugs": t.known_drugs,
            }
            for t in self._targets.values()
        ]


def get_target_registry() -> MolecularTargetRegistry:
    return MolecularTargetRegistry.get_instance()
