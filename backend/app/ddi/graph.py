"""
CYP/transporter graph for transitive DDI detection.

If drug A inhibits CYP3A4 and drug B is a CYP3A4 substrate, they interact
even when not in the curated DDI table. This graph makes that visible.

Nodes are drugs; edges are role->substrate connections.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CYPNode:
    """A drug's role with respect to a specific CYP/transporter."""
    drug: str
    enzyme: str                  # e.g. "CYP3A4", "CYP2C9", "P-gp", "OATP1B1"
    role: str                    # "substrate" | "inhibitor" | "inducer"
    strength: str = "moderate"   # "strong" | "moderate" | "weak"


# Curated subset — most clinically relevant roles
CYP_GRAPH: list[CYPNode] = [
    # ---- CYP3A4 ----
    CYPNode("simvastatin",  "CYP3A4", "substrate", "sensitive"),
    CYPNode("atorvastatin", "CYP3A4", "substrate", "sensitive"),
    CYPNode("midazolam",    "CYP3A4", "substrate", "sensitive"),
    CYPNode("amlodipine",   "CYP3A4", "substrate", "moderate"),
    CYPNode("diltiazem",    "CYP3A4", "substrate", "moderate"),
    CYPNode("verapamil",    "CYP3A4", "substrate", "moderate"),
    CYPNode("apixaban",     "CYP3A4", "substrate", "sensitive"),
    CYPNode("rivaroxaban",  "CYP3A4", "substrate", "sensitive"),
    CYPNode("tacrolimus",   "CYP3A4", "substrate", "sensitive"),
    CYPNode("cyclosporine", "CYP3A4", "substrate", "sensitive"),
    CYPNode("quetiapine",   "CYP3A4", "substrate", "moderate"),
    CYPNode("aripiprazole", "CYP3A4", "substrate", "moderate"),
    CYPNode("ondansetron",  "CYP3A4", "substrate", "moderate"),
    CYPNode("clarithromycin","CYP3A4", "inhibitor", "strong"),
    CYPNode("erythromycin", "CYP3A4", "inhibitor", "strong"),
    CYPNode("ketoconazole", "CYP3A4", "inhibitor", "strong"),
    CYPNode("itraconazole", "CYP3A4", "inhibitor", "strong"),
    CYPNode("fluconazole",  "CYP3A4", "inhibitor", "moderate"),
    CYPNode("diltiazem",    "CYP3A4", "inhibitor", "moderate"),
    CYPNode("verapamil",    "CYP3A4", "inhibitor", "moderate"),
    CYPNode("grapefruit",   "CYP3A4", "inhibitor", "strong"),
    CYPNode("ritonavir",    "CYP3A4", "inhibitor", "strong"),
    CYPNode("rifampin",     "CYP3A4", "inducer",   "strong"),
    CYPNode("phenytoin",    "CYP3A4", "inducer",   "strong"),
    CYPNode("carbamazepine","CYP3A4", "inducer",   "strong"),
    CYPNode("st_johns_wort","CYP3A4", "inducer",   "strong"),

    # ---- CYP2C9 ----
    CYPNode("warfarin",     "CYP2C9", "substrate", "sensitive"),
    CYPNode("phenytoin",    "CYP2C9", "substrate", "sensitive"),
    CYPNode("glipizide",    "CYP2C9", "substrate", "sensitive"),
    CYPNode("losartan",     "CYP2C9", "substrate", "moderate"),
    CYPNode("fluconazole",  "CYP2C9", "inhibitor", "strong"),
    CYPNode("amiodarone",   "CYP2C9", "inhibitor", "moderate"),
    CYPNode("metronidazole","CYP2C9", "inhibitor", "moderate"),
    CYPNode("trimethoprim", "CYP2C9", "inhibitor", "moderate"),
    CYPNode("rifampin",     "CYP2C9", "inducer",   "strong"),

    # ---- CYP2C19 ----
    CYPNode("clopidogrel",  "CYP2C19", "substrate", "sensitive"),
    CYPNode("omeprazole",   "CYP2C19", "substrate", "sensitive"),
    CYPNode("pantoprazole", "CYP2C19", "substrate", "sensitive"),
    CYPNode("sertraline",   "CYP2C19", "substrate", "moderate"),
    CYPNode("escitalopram", "CYP2C19", "substrate", "moderate"),
    CYPNode("voriconazole", "CYP2C19", "substrate", "sensitive"),
    CYPNode("fluvoxamine",  "CYP2C19", "inhibitor", "strong"),
    CYPNode("omeprazole",   "CYP2C19", "inhibitor", "moderate"),
    CYPNode("fluconazole",  "CYP2C19", "inhibitor", "moderate"),
    CYPNode("rifampin",     "CYP2C19", "inducer",   "strong"),

    # ---- CYP2D6 ----
    CYPNode("metoprolol",   "CYP2D6", "substrate", "sensitive"),
    CYPNode("carvedilol",   "CYP2D6", "substrate", "sensitive"),
    CYPNode("codeine",      "CYP2D6", "substrate", "sensitive"),
    CYPNode("tramadol",     "CYP2D6", "substrate", "sensitive"),
    CYPNode("oxycodone",    "CYP2D6", "substrate", "moderate"),
    CYPNode("risperidone",  "CYP2D6", "substrate", "moderate"),
    CYPNode("haloperidol",  "CYP2D6", "substrate", "moderate"),
    CYPNode("tamoxifen",    "CYP2D6", "substrate", "sensitive"),
    CYPNode("fluoxetine",   "CYP2D6", "inhibitor", "strong"),
    CYPNode("paroxetine",   "CYP2D6", "inhibitor", "strong"),
    CYPNode("bupropion",    "CYP2D6", "inhibitor", "strong"),
    CYPNode("quinidine",    "CYP2D6", "inhibitor", "strong"),

    # ---- P-gp (transporter) ----
    CYPNode("digoxin",      "P-gp",   "substrate", "sensitive"),
    CYPNode("apixaban",     "P-gp",   "substrate", "moderate"),
    CYPNode("rivaroxaban",  "P-gp",   "substrate", "moderate"),
    CYPNode("verapamil",    "P-gp",   "inhibitor", "strong"),
    CYPNode("amiodarone",   "P-gp",   "inhibitor", "moderate"),
    CYPNode("clarithromycin","P-gp",  "inhibitor", "moderate"),
    CYPNode("ritonavir",    "P-gp",   "inhibitor", "strong"),
    CYPNode("rifampin",     "P-gp",   "inducer",   "strong"),

    # ---- OATP1B1 (hepatic uptake) ----
    CYPNode("simvastatin",  "OATP1B1", "substrate", "sensitive"),
    CYPNode("atorvastatin", "OATP1B1", "substrate", "moderate"),
    CYPNode("rosuvastatin", "OATP1B1", "substrate", "sensitive"),
    CYPNode("cyclosporine", "OATP1B1", "inhibitor", "strong"),
    CYPNode("gemfibrozil",  "OATP1B1", "inhibitor", "strong"),
    CYPNode("rifampin",     "OATP1B1", "inhibitor", "moderate"),
]


def get_role(drug: str, enzyme: str) -> str | None:
    """Return 'substrate', 'inhibitor', 'inducer' for a drug-enzyme pair, or None."""
    n = drug.lower().strip()
    for n_ in CYP_GRAPH:
        if n_.drug == n and n_.enzyme == enzyme:
            return n_.role
    return None


def find_substrates(enzyme: str) -> list[str]:
    return list({n.drug for n in CYP_GRAPH
                 if n.enzyme == enzyme and n.role == "substrate"})


def find_inhibitors(enzyme: str) -> list[str]:
    return list({n.drug for n in CYP_GRAPH
                 if n.enzyme == enzyme and n.role == "inhibitor"})


def find_inducers(enzyme: str) -> list[str]:
    return list({n.drug for n in CYP_GRAPH
                 if n.enzyme == enzyme and n.role == "inducer"})


def detect_transitive_interactions(drugs: list[str]) -> list[dict]:
    """
    For every drug that is a CYP/transporter inhibitor or inducer, check
    whether any other drug in the list is a substrate of the same enzyme.
    Returns inferred interactions with severity inferred from strength.
    """
    SEVERITY_BY_STRENGTH = {"strong": "major", "moderate": "moderate",
                            "weak": "minor", "sensitive": "moderate"}
    n = [d.lower().strip() for d in drugs]
    out: list[dict] = []
    for n_ in CYP_GRAPH:
        if n_.drug not in n:
            continue
        if n_.role not in ("inhibitor", "inducer"):
            continue
        for n2 in CYP_GRAPH:
            if n2.enzyme != n_.enzyme or n2.role != "substrate":
                continue
            if n2.drug not in n or n2.drug == n_.drug:
                continue
            sev = SEVERITY_BY_STRENGTH.get(n_.strength, "moderate")
            direction = ("inhibition" if n_.role == "inhibitor"
                         else "induction")
            effect = ("Increased exposure of substrate"
                      if direction == "inhibition"
                      else "Decreased exposure of substrate")
            out.append({
                "drug_a": n_.drug,
                "drug_b": n2.drug,
                "enzyme": n_.enzyme,
                "direction": direction,
                "severity": sev,
                "mechanism": f"{n_.enzyme} {direction} ({n_.strength})",
                "clinical_effect": effect,
                "source": "transitive_inference",
            })
    return out
