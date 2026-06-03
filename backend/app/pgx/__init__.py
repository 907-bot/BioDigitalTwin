"""Phase 8 — Pharmacogenomics."""
from .genes import (
    PHARMACOGENES,
    MetabolizerStatus,
    PatientPGx,
    assign_genotypes,
    attach_pgx_to_cohort,
    get_patient_pgx,
)
from .registry import (
    DRUG_GENE_REGISTRY,
    DrugGeneRule,
    get_impact_factor,
    lookup_drug,
    lookup_gene,
)
from .router import router as pgx_router

__all__ = [
    "PHARMACOGENES",
    "MetabolizerStatus",
    "PatientPGx",
    "assign_genotypes",
    "attach_pgx_to_cohort",
    "get_patient_pgx",
    "DRUG_GENE_REGISTRY",
    "DrugGeneRule",
    "get_impact_factor",
    "lookup_drug",
    "lookup_gene",
    "pgx_router",
]
