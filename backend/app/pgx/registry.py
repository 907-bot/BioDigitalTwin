"""
Drug-gene interaction registry (curated subset of PharmGKB level-1A pairs).

Each entry says:
  - what gene modulates this drug
  - whether the drug is a prodrug (needs activation) or an active drug
  - what the clinical effect is when the patient is a PM or UM
  - the CPIC evidence level

`impact_factor` is the multiplier we apply to the drug's effect in the
counterfactual:
  - active drug + PM metabolizer → stronger effect (impact > 1)
  - active drug + UM metabolizer → weaker effect (impact < 1)
  - prodrug + PM metabolizer → weaker effect (impact < 1)
  - prodrug + UM metabolizer → stronger effect (impact > 1)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DrugGeneRule:
    drug: str             # canonical drug name (lowercase, matches chembl & common)
    gene: str             # gene name (matches PHARMACOGENES)
    is_prodrug: bool      # True if the drug is inactive until metabolized
    pm_clinical: str      # what happens to a poor metabolizer
    um_clinical: str      # what happens to an ultra-rapid metabolizer
    cpic_level: str       # "A", "B", "C", "D"
    severity: str         # "critical", "major", "moderate", "minor"

    @property
    def impact_factor(self) -> dict[str, float]:
        """
        Multiplier on the drug's effect size in the counterfactual, indexed
        by metabolizer status. Applied as: effect = baseline * factor.
        """
        if self.is_prodrug:
            # PM doesn't activate it → much weaker effect
            return {"PM": 0.30, "IM": 0.60, "EM": 1.0, "UM": 1.80}
        else:
            # PM can't clear it → builds up → much stronger effect
            return {"PM": 3.00, "IM": 1.50, "EM": 1.0, "UM": 0.40}


# Curated registry — ~40 well-established pairs
DRUG_GENE_REGISTRY: list[DrugGeneRule] = [
    # --- Cardiovascular ---
    DrugGeneRule("warfarin",     "CYP2C9",  is_prodrug=False,
                 pm_clinical="Increased bleeding risk; reduce dose 30-50%",
                 um_clinical="Sub-therapeutic INR; may need higher dose",
                 cpic_level="A", severity="critical"),
    DrugGeneRule("warfarin",     "CYP4F2",  is_prodrug=False,
                 pm_clinical="Mildly increased warfarin sensitivity",
                 um_clinical="Mildly decreased warfarin sensitivity",
                 cpic_level="B", severity="moderate"),
    DrugGeneRule("clopidogrel",  "CYP2C19", is_prodrug=True,
                 pm_clinical="Reduced platelet inhibition; higher CV event risk",
                 um_clinical="Increased bleeding risk",
                 cpic_level="A", severity="critical"),
    DrugGeneRule("metoprolol",   "CYP2D6",  is_prodrug=False,
                 pm_clinical="Excessive β-blockade; bradycardia risk",
                 um_clinical="Sub-therapeutic effect; may need higher dose",
                 cpic_level="A", severity="major"),
    DrugGeneRule("carvedilol",   "CYP2D6",  is_prodrug=False,
                 pm_clinical="Increased β-blockade and dizziness",
                 um_clinical="Reduced effect",
                 cpic_level="B", severity="moderate"),
    DrugGeneRule("atorvastatin", "SLCO1B1", is_prodrug=False,
                 pm_clinical="Increased myopathy risk; reduce dose",
                 um_clinical="Standard dosing",
                 cpic_level="A", severity="major"),
    DrugGeneRule("simvastatin",  "SLCO1B1", is_prodrug=False,
                 pm_clinical="Increased myopathy/rhabdomyolysis risk",
                 um_clinical="Standard dosing",
                 cpic_level="A", severity="critical"),
    DrugGeneRule("rosuvastatin", "SLCO1B1", is_prodrug=False,
                 pm_clinical="Mildly increased exposure",
                 um_clinical="Standard dosing",
                 cpic_level="B", severity="moderate"),

    # --- Anticoagulants (DOACs) ---
    DrugGeneRule("apixaban",     "CYP3A4",  is_prodrug=False,
                 pm_clinical="Increased exposure; bleeding risk",
                 um_clinical="Decreased exposure",
                 cpic_level="B", severity="major"),
    DrugGeneRule("rivaroxaban",  "CYP3A4",  is_prodrug=False,
                 pm_clinical="Increased exposure; bleeding risk",
                 um_clinical="Decreased exposure",
                 cpic_level="B", severity="major"),

    # --- GI / PPI ---
    DrugGeneRule("omeprazole",   "CYP2C19", is_prodrug=True,
                 pm_clinical="Increased acid suppression (often desired)",
                 um_clinical="Therapeutic failure likely",
                 cpic_level="A", severity="major"),
    DrugGeneRule("pantoprazole", "CYP2C19", is_prodrug=True,
                 pm_clinical="Increased acid suppression",
                 um_clinical="Therapeutic failure likely",
                 cpic_level="B", severity="moderate"),

    # --- Psychotropics ---
    DrugGeneRule("sertraline",   "CYP2C19", is_prodrug=False,
                 pm_clinical="Increased SSRI exposure; sertraline toxicity risk",
                 um_clinical="Sub-therapeutic; SSRI failure",
                 cpic_level="A", severity="major"),
    DrugGeneRule("escitalopram", "CYP2C19", is_prodrug=False,
                 pm_clinical="Increased QT prolongation risk",
                 um_clinical="Therapeutic failure",
                 cpic_level="A", severity="major"),
    DrugGeneRule("fluoxetine",   "CYP2D6",  is_prodrug=False,
                 pm_clinical="Increased SSRI exposure",
                 um_clinical="Reduced effect",
                 cpic_level="B", severity="moderate"),
    DrugGeneRule("risperidone",  "CYP2D6",  is_prodrug=False,
                 pm_clinical="Increased extrapyramidal side effects",
                 um_clinical="Therapeutic failure",
                 cpic_level="B", severity="major"),
    DrugGeneRule("aripiprazole", "CYP2D6",  is_prodrug=False,
                 pm_clinical="Increased aripiprazole exposure",
                 um_clinical="Reduced effect",
                 cpic_level="B", severity="moderate"),
    DrugGeneRule("haloperidol",  "CYP2D6",  is_prodrug=False,
                 pm_clinical="Increased EPS risk",
                 um_clinical="Therapeutic failure",
                 cpic_level="B", severity="major"),

    # --- Pain / Opioids ---
    DrugGeneRule("codeine",      "CYP2D6",  is_prodrug=True,
                 pm_clinical="No analgesia (cannot activate)",
                 um_clinical="Severe toxicity / respiratory depression risk",
                 cpic_level="A", severity="critical"),
    DrugGeneRule("tramadol",     "CYP2D6",  is_prodrug=True,
                 pm_clinical="Reduced analgesia",
                 um_clinical="Serotonin syndrome / respiratory depression risk",
                 cpic_level="A", severity="critical"),
    DrugGeneRule("oxycodone",    "CYP2D6",  is_prodrug=True,
                 pm_clinical="Reduced analgesia",
                 um_clinical="Increased opioid effect and side effects",
                 cpic_level="B", severity="major"),
    DrugGeneRule("morphine",     "CYP2D6",  is_prodrug=False,
                 pm_clinical="Minimal impact",
                 um_clinical="Minimal impact",
                 cpic_level="C", severity="minor"),

    # --- Oncology / Immunosuppressants ---
    DrugGeneRule("azathioprine", "TPMT",    is_prodrug=True,
                 pm_clinical="Severe myelosuppression risk; avoid or 10% dose",
                 um_clinical="Standard or higher dose",
                 cpic_level="A", severity="critical"),
    DrugGeneRule("6-mercaptopurine", "TPMT", is_prodrug=True,
                 pm_clinical="Severe myelosuppression risk",
                 um_clinical="Standard or higher dose",
                 cpic_level="A", severity="critical"),
    DrugGeneRule("fluorouracil", "DPYD",    is_prodrug=True,
                 pm_clinical="Severe/lethal toxicity; avoid",
                 um_clinical="Standard dosing",
                 cpic_level="A", severity="critical"),
    DrugGeneRule("capecitabine", "DPYD",    is_prodrug=True,
                 pm_clinical="Severe toxicity; avoid",
                 um_clinical="Standard dosing",
                 cpic_level="A", severity="critical"),
    DrugGeneRule("tamoxifen",    "CYP2D6",  is_prodrug=True,
                 pm_clinical="Reduced endoxifen formation; consider alternative",
                 um_clinical="Standard or higher effect",
                 cpic_level="A", severity="major"),

    # --- Anti-infectives ---
    DrugGeneRule("efavirenz",    "CYP2B6",  is_prodrug=False,
                 pm_clinical="Increased CNS toxicity",
                 um_clinical="Therapeutic failure",
                 cpic_level="A", severity="major"),
    DrugGeneRule("voriconazole", "CYP2C19", is_prodrug=False,
                 pm_clinical="Increased hepatotoxicity",
                 um_clinical="Therapeutic failure",
                 cpic_level="A", severity="major"),

    # --- Diabetes (the ones we model) ---
    DrugGeneRule("metformin",    "CYP2D6",  is_prodrug=False,
                 pm_clinical="Minimal impact (renally cleared)",
                 um_clinical="Minimal impact",
                 cpic_level="C", severity="minor"),
    DrugGeneRule("glipizide",    "CYP2C9",  is_prodrug=False,
                 pm_clinical="Increased hypoglycemia risk",
                 um_clinical="Reduced effect",
                 cpic_level="B", severity="moderate"),

    # --- Anti-hypertensives ---
    DrugGeneRule("losartan",     "CYP2C9",  is_prodrug=True,
                 pm_clinical="Reduced active metabolite; weaker effect",
                 um_clinical="Increased antihypertensive effect",
                 cpic_level="B", severity="moderate"),
    DrugGeneRule("irbesartan",   "CYP2C9",  is_prodrug=False,
                 pm_clinical="Mildly increased exposure",
                 um_clinical="Mildly decreased exposure",
                 cpic_level="B", severity="minor"),
]


def lookup_drug(drug_name: str) -> list[DrugGeneRule]:
    name = drug_name.lower().strip()
    return [r for r in DRUG_GENE_REGISTRY if r.drug == name]


def lookup_gene(gene: str) -> list[DrugGeneRule]:
    return [r for r in DRUG_GENE_REGISTRY if r.gene == gene]


def get_impact_factor(drug: str, gene: str, status: str) -> float:
    """Get the multiplicative effect modifier for a drug-gene-status combo."""
    name = drug.lower().strip()
    for r in DRUG_GENE_REGISTRY:
        if r.drug == name and r.gene == gene:
            return r.impact_factor.get(status, 1.0)
    return 1.0
