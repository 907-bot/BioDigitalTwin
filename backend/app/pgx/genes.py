"""
Phase 8 — Pharmacogenomics.

Adds a CYP/Phase-II enzyme metabolizer status to each synthetic patient and
a drug-gene interaction registry. Used to modulate the SCM-based
counterfactual so that poor metabolizers see larger effects for
active-metabolite drugs and smaller effects for prodrugs.

References (curated, not exhaustive):
  - PharmGKB level-1A drug-gene pairs
  - CPIC guidelines for the highest-impact genes
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MetabolizerStatus(str, Enum):
    PM = "PM"   # Poor metabolizer     — activity ~0.1
    IM = "IM"   # Intermediate         — activity ~0.5
    EM = "EM"   # Extensive (normal)   — activity 1.0
    UM = "UM"   # Ultra-rapid          — activity ~2.0

    @property
    def activity(self) -> float:
        return {
            MetabolizerStatus.PM: 0.1,
            MetabolizerStatus.IM: 0.5,
            MetabolizerStatus.EM: 1.0,
            MetabolizerStatus.UM: 2.0,
        }[self]


# Key pharmacogenes — same panel CPIC covers
PHARMACOGENES: list[str] = [
    "CYP2D6",
    "CYP2C19",
    "CYP3A4",
    "CYP2C9",
    "CYP2B6",
    "SLCO1B1",
    "TPMT",
    "DPYD",
]


@dataclass(frozen=True)
class PatientPGx:
    patient_id: str
    genotypes: dict[str, MetabolizerStatus]

    def activity_for(self, gene: str) -> float:
        return self.genotypes.get(gene, MetabolizerStatus.EM).activity

    def summary(self) -> str:
        lines = [f"PGx profile for {self.patient_id}:"]
        for gene in PHARMACOGENES:
            status = self.genotypes.get(gene, MetabolizerStatus.EM)
            lines.append(f"  {gene}: {status.value} (activity {status.activity:.2f})")
        return "\n".join(lines)


def assign_genotypes(patient_id: str, risk_score: float, seed: int) -> dict[str, MetabolizerStatus]:
    """
    Assign pharmacogenomic status to a patient.

    Genotype frequencies follow published population distributions (CPIC):
      CYP2D6: PM 5-10%, IM 2-11%, EM 70-90%, UM 1-5%
      CYP2C19: PM 2-15%, IM 18-45%, EM 35-50%, UM 5-30%
      ...

    We skew toward PM/IM for high-risk patients (intuition: high-risk
    phenotypes correlate with multiple-system dysregulation, including
    drug metabolism), but the effect is mild.
    """
    rng = random.Random(f"{seed}:{patient_id}")
    skew = max(0.0, (risk_score - 0.5) * 0.3)  # +0..+0.15 toward PM/IM

    # (PM, IM, EM, UM) probabilities — cumulative
    probs: dict[str, tuple[float, float, float, float]] = {
        "CYP2D6":   (0.07,  0.10,  0.78,  0.05),
        "CYP2C19":  (0.10,  0.30,  0.40,  0.20),
        "CYP3A4":   (0.02,  0.10,  0.85,  0.03),
        "CYP2C9":   (0.03,  0.20,  0.75,  0.02),
        "CYP2B6":   (0.05,  0.25,  0.65,  0.05),
        "SLCO1B1":  (0.05,  0.20,  0.70,  0.05),
        "TPMT":     (0.03,  0.10,  0.85,  0.02),
        "DPYD":     (0.01,  0.05,  0.93,  0.01),
    }

    out: dict[str, MetabolizerStatus] = {}
    for gene, (p_pm, p_im, p_em, p_um) in probs.items():
        # Add skew: pull from PM/IM, push away from EM
        p_pm = min(0.4, p_pm + skew)
        p_im = min(0.5, p_im + skew / 2)
        p_em = max(0.2, p_em - skew)
        p_um = max(0.01, p_um)
        total = p_pm + p_im + p_em + p_um
        p_pm, p_im, p_em = p_pm / total, p_im / total, p_em / total
        r = rng.random()
        if r < p_pm:
            out[gene] = MetabolizerStatus.PM
        elif r < p_pm + p_im:
            out[gene] = MetabolizerStatus.IM
        elif r < p_pm + p_im + p_em:
            out[gene] = MetabolizerStatus.EM
        else:
            out[gene] = MetabolizerStatus.UM
    return out


def attach_pgx_to_cohort(csv_path: str = "data/synthetic_patients.csv",
                         seed: int = 42) -> pd.DataFrame:
    """
    Load the cohort and attach one PGx column per gene.
    Writes back to the same CSV (additive, idempotent).
    """
    df = pd.read_csv(csv_path)
    for gene in PHARMACOGENES:
        if gene in df.columns:
            continue
        col = []
        for _, row in df.iterrows():
            risk = float(row.get("risk_score", 0.5) or 0.5)
            genotypes = assign_genotypes(row["patient_id"], risk, seed)
            col.append(genotypes[gene].value)
        df[gene] = col
    df.to_csv(csv_path, index=False)
    logger.info("Attached PGx genotypes to %d patients", len(df))
    return df


def get_patient_pgx(patient_id: str, csv_path: str = "data/synthetic_patients.csv") -> Optional[PatientPGx]:
    df = pd.read_csv(csv_path)
    row = df[df["patient_id"] == patient_id]
    if row.empty:
        return None
    r = row.iloc[0]
    genotypes: dict[str, MetabolizerStatus] = {}
    for gene in PHARMACOGENES:
        val = r.get(gene)
        if pd.isna(val):
            genotypes[gene] = MetabolizerStatus.EM
            continue
        try:
            genotypes[gene] = MetabolizerStatus(val)
        except ValueError:
            genotypes[gene] = MetabolizerStatus.EM
    return PatientPGx(patient_id=patient_id, genotypes=genotypes)
