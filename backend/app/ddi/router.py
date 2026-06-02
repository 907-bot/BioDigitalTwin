"""FastAPI router for Phase 9 — Drug-Drug Interactions."""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .database import DDI_RULES, SEVERITY_RANK, find_direct, find_pair, normalize
from .graph import CYP_GRAPH, detect_transitive_interactions, get_role
from app.narrative import ddi as ddi_narrative

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/phase9", tags=["Phase 9 — Drug-Drug Interactions"])


@router.get("/rules")
def list_rules(severity: Optional[str] = None, drug: Optional[str] = None):
    """List curated DDI rules, optionally filtered."""
    rules = DDI_RULES
    if severity:
        rules = [r for r in rules if r.severity == severity.lower()]
    if drug:
        d = drug.lower()
        rules = [r for r in rules if r.drug_a == d or r.drug_b == d]
    return {
        "total": len(rules),
        "rules": [
            {
                "drug_a": r.drug_a,
                "drug_b": r.drug_b,
                "severity": r.severity,
                "mechanism": r.mechanism,
                "clinical_effect": r.clinical_effect,
                "source": r.cpic_or_fda,
                "on_set": r.on_set,
            }
            for r in sorted(rules, key=lambda r: SEVERITY_RANK.get(r.severity, 9))
        ],
    }


@router.get("/graph")
def cyp_graph():
    """The CYP/transporter role graph used for transitive inference."""
    return {
        "nodes": [
            {
                "drug": n.drug,
                "enzyme": n.enzyme,
                "role": n.role,
                "strength": n.strength,
            }
            for n in CYP_GRAPH
        ],
        "n_nodes": len(CYP_GRAPH),
    }


class DDICheckRequest(BaseModel):
    drugs: list[str]


@router.post("/check")
def check_polypharmacy(req: DDICheckRequest):
    """
    For a list of drugs, return the full interaction matrix.

    Combines:
      1. Curated direct interactions (the DDI_RULES table)
      2. Transitive interactions inferred from the CYP/transporter graph
    """
    if not req.drugs or len(req.drugs) < 2:
        raise HTTPException(400, "need at least 2 drugs")
    if len(req.drugs) > 30:
        raise HTTPException(400, "max 30 drugs")

    n = [normalize(d) for d in req.drugs]
    n_dedup: list[str] = []
    for d in n:
        if d and d not in n_dedup:
            n_dedup.append(d)
    if len(n_dedup) < 2:
        raise HTTPException(400, "need at least 2 unique drugs")

    interactions = []
    seen = set()
    for i, a in enumerate(n_dedup):
        for b in n_dedup[i + 1:]:
            direct = find_pair(a, b)
            for r in direct:
                key = (a, b, r.mechanism)
                if key in seen:
                    continue
                seen.add(key)
                interactions.append({
                    "drug_a": a,
                    "drug_b": b,
                    "severity": r.severity,
                    "mechanism": r.mechanism,
                    "clinical_effect": r.clinical_effect,
                    "source": r.cpic_or_fda,
                    "inferred": False,
                })

    # Transitive interactions from CYP graph
    transitive = detect_transitive_interactions(n_dedup)
    for t in transitive:
        key = (t["drug_a"], t["drug_b"], t["mechanism"])
        if key in seen:
            continue
        seen.add(key)
        interactions.append(t)

    # Sort: contraindicated → major → moderate → minor, then alphabetical
    interactions.sort(key=lambda i: (SEVERITY_RANK.get(i["severity"], 9),
                                     i["drug_a"], i["drug_b"]))

    # Per-drug severity summary
    drug_summary: dict[str, dict] = {d: {"max_severity": "none", "n_interactions": 0}
                                     for d in n_dedup}
    for i in interactions:
        for d in (i["drug_a"], i["drug_b"]):
            s = drug_summary[d]
            s["n_interactions"] += 1
            if SEVERITY_RANK.get(i["severity"], 9) < SEVERITY_RANK.get(s["max_severity"], 9):
                s["max_severity"] = i["severity"]

    overall = "none"
    for i in interactions:
        if SEVERITY_RANK.get(i["severity"], 9) < SEVERITY_RANK.get(overall, 9):
            overall = i["severity"]

    return {
        "drugs": n_dedup,
        "n_drugs": len(n_dedup),
        "n_interactions": len(interactions),
        "n_direct": sum(1 for i in interactions if not i.get("inferred")),
        "n_inferred": sum(1 for i in interactions if i.get("inferred")),
        "overall_severity": overall,
        "interactions": interactions,
        "per_drug_summary": drug_summary,
        "narrative": ddi_narrative.narrate_check(
            drugs=n_dedup, interactions=interactions, overall_severity=overall,
            n_direct=sum(1 for i in interactions if not i.get("inferred")),
            n_inferred=sum(1 for i in interactions if i.get("inferred")),
        ),
    }


class DDIPairRequest(BaseModel):
    drug_a: str
    drug_b: str


@router.post("/pair")
def check_pair(req: DDIPairRequest):
    """Direct check for a single pair — returns the worst interaction if any."""
    a, b = normalize(req.drug_a), normalize(req.drug_b)
    direct = find_pair(a, b)
    if direct:
        r = min(direct, key=lambda r: SEVERITY_RANK.get(r.severity, 9))
        pair_dict = {
            "drug_a": a, "drug_b": b, "severity": r.severity,
            "mechanism": r.mechanism, "clinical_effect": r.clinical_effect,
            "cpic_or_fda": r.cpic_or_fda, "on_set": r.on_set,
        }
        return {
            "drug_a": a,
            "drug_b": b,
            "interaction_found": True,
            "severity": r.severity,
            "mechanism": r.mechanism,
            "clinical_effect": r.clinical_effect,
            "source": r.cpic_or_fda,
            "narrative": ddi_narrative.narrate_pair(a, b, pair_dict, None),
        }
    # Try transitive
    transitive = detect_transitive_interactions([a, b])
    if transitive:
        t = min(transitive, key=lambda t: SEVERITY_RANK.get(t["severity"], 9))
        return {
            "drug_a": a,
            "drug_b": b,
            "interaction_found": True,
            "severity": t["severity"],
            "mechanism": t["mechanism"],
            "clinical_effect": t["clinical_effect"],
            "source": "transitive_inference",
            "narrative": ddi_narrative.narrate_pair(a, b, None, [a, t.get("enzyme", "?"), b]),
        }
    return {
        "drug_a": a,
        "drug_b": b,
        "interaction_found": False,
        "severity": "none",
        "mechanism": None,
        "clinical_effect": None,
        "source": None,
        "narrative": ddi_narrative.narrate_pair(a, b, None, None),
    }
    # Try transitive
    transitive = detect_transitive_interactions([a, b])
    if transitive:
        t = min(transitive, key=lambda t: SEVERITY_RANK.get(t["severity"], 9))
        return {
            "drug_a": a,
            "drug_b": b,
            "interaction_found": True,
            "severity": t["severity"],
            "mechanism": t["mechanism"],
            "clinical_effect": t["clinical_effect"],
            "source": "transitive_inference",
        }
    return {
        "drug_a": a,
        "drug_b": b,
        "interaction_found": False,
        "severity": "none",
        "mechanism": None,
        "clinical_effect": None,
        "source": None,
    }
