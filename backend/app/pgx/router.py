"""FastAPI router for Phase 8 — Pharmacogenomics."""
from __future__ import annotations

import logging
import os
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from .genes import (
    PHARMACOGENES,
    PatientPGx,
    attach_pgx_to_cohort,
    get_patient_pgx,
)
from .registry import (
    DRUG_GENE_REGISTRY,
    get_impact_factor,
    lookup_drug,
    lookup_gene,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/phase8", tags=["Phase 8 — Pharmacogenomics"])


PATIENTS_CSV = "data/synthetic_patients.csv"


def _ensure_pgx_attached() -> None:
    """Make sure the cohort has PGx columns; attach them if missing."""
    if not os.path.exists(PATIENTS_CSV):
        raise HTTPException(404, "no cohort — call POST /generate-patients first")
    df = pd.read_csv(PATIENTS_CSV, nrows=0)
    missing = [g for g in PHARMACOGENES if g not in df.columns]
    if missing:
        attach_pgx_to_cohort(PATIENTS_CSV)


@router.get("/genes")
def list_genes():
    """List the pharmacogenes in the panel with their activity scores."""
    from .genes import MetabolizerStatus
    return {
        "genes": PHARMACOGENES,
        "statuses": [
            {"code": s.value, "label": s.name, "activity": s.activity}
            for s in MetabolizerStatus
        ],
    }


@router.get("/registry")
def list_drug_gene_pairs(gene: Optional[str] = None, drug: Optional[str] = None):
    """Curated drug-gene interaction registry (subset of PharmGKB 1A)."""
    rules = DRUG_GENE_REGISTRY
    if gene:
        rules = [r for r in rules if r.gene.upper() == gene.upper()]
    if drug:
        rules = [r for r in rules if drug.lower() in r.drug.lower()]
    return {
        "total": len(rules),
        "rules": [
            {
                "drug": r.drug,
                "gene": r.gene,
                "is_prodrug": r.is_prodrug,
                "pm_clinical": r.pm_clinical,
                "um_clinical": r.um_clinical,
                "cpic_level": r.cpic_level,
                "severity": r.severity,
                "impact_pm":  r.impact_factor["PM"],
                "impact_im":  r.impact_factor["IM"],
                "impact_em":  r.impact_factor["EM"],
                "impact_um":  r.impact_factor["UM"],
            }
            for r in rules
        ],
    }


@router.get("/patients/{patient_id}/pgx")
def patient_pgx(patient_id: str):
    """A patient's complete PGx profile (8 genes, metabolizer status, activity)."""
    _ensure_pgx_attached()
    pgx = get_patient_pgx(patient_id, PATIENTS_CSV)
    if pgx is None:
        raise HTTPException(404, f"patient '{patient_id}' not found")
    return {
        "patient_id": patient_id,
        "genes": [
            {
                "gene": g,
                "status": pgx.genotypes.get(g, "EM").value,
                "activity": pgx.activity_for(g),
            }
            for g in PHARMACOGENES
        ],
    }


class PGxCheckRequest(BaseModel):
    patient_id: str
    drugs: list[str]


@router.post("/patients/pgx-check")
def pgx_check(req: PGxCheckRequest):
    """
    For a patient on a list of drugs, return gene-drug warnings.
    Critical when:
      - a drug is in the registry
      - the patient is PM/UM for the relevant gene
    """
    _ensure_pgx_attached()
    pgx = get_patient_pgx(req.patient_id, PATIENTS_CSV)
    if pgx is None:
        raise HTTPException(404, f"patient '{req.patient_id}' not found")

    warnings = []
    for drug in req.drugs:
        for r in lookup_drug(drug):
            status = pgx.genotypes.get(r.gene)
            if status is None or status.value == "EM":
                continue
            warnings.append({
                "drug": r.drug,
                "gene": r.gene,
                "patient_status": status.value,
                "severity": r.severity,
                "cpic_level": r.cpic_level,
                "is_prodrug": r.is_prodrug,
                "impact_factor": r.impact_factor[status.value],
                "clinical_note": r.pm_clinical if status.value == "PM" else r.um_clinical,
            })

    # sort by severity
    sev_order = {"critical": 0, "major": 1, "moderate": 2, "minor": 3}
    warnings.sort(key=lambda w: sev_order.get(w["severity"], 9))

    return {
        "patient_id": req.patient_id,
        "drugs_checked": req.drugs,
        "warnings": warnings,
        "n_warnings": len(warnings),
        "highest_severity": warnings[0]["severity"] if warnings else "none",
    }
