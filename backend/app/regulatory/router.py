"""FastAPI router for Phase 13 — Regulatory knowledge (FDA, FAERS, RxNorm)."""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from .faers import drug_summary, top_adverse_events
from .orange_book import (
    CURATED_ORANGE_BOOK,
    is_approved,
    lookup,
    normalize_rxnorm,
)
from .warnings import SAFETY_REGISTRY, get_safety, has_black_box

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/phase13", tags=["Phase 13 — Regulatory"])


@router.get("/drugs/{drug_name}/regulatory")
async def drug_regulatory(drug_name: str):
    """
    Full regulatory profile for a drug:
      - FDA approval (Orange Book)
      - Black-box warnings + contraindications + common AEs (curated)
      - Top adverse events from FAERS (live)
      - Pregnancy category
    """
    d = drug_name.lower().strip()
    safety = get_safety(d)
    ob = lookup(d)

    faers_summary = await drug_summary(d)
    top_ae = faers_summary.get("top_reactions", [])

    return {
        "drug": d,
        "orange_book": {
            "approved": is_approved(d),
            "entries": [
                {
                    "ingredient": e.ingredient,
                    "trade_name": e.trade_name,
                    "applicant": e.applicant,
                    "approval_date": e.approval_date,
                }
                for e in ob
            ],
        },
        "safety": {
            "black_box_warnings":    safety.black_box if safety else [],
            "contraindications":     safety.contraindications if safety else [],
            "common_adverse_events": safety.common_aes if safety else [],
            "pregnancy_category":    safety.pregnancy if safety else "unknown",
            "typical_dose":          safety.typical_dose if safety else "",
            "notes":                 safety.notes if safety else "",
        } if safety else None,
        "faers": {
            "total_reports":  faers_summary.get("total_reports"),
            "top_reactions":  top_ae,
        },
    }


@router.get("/drugs/{drug_name}/black-box")
async def black_box(drug_name: str):
    """Just the black-box warnings for a drug."""
    s = get_safety(drug_name)
    if not s or not s.black_box:
        return {"drug": drug_name.lower(), "has_black_box": False,
                "warnings": []}
    return {"drug": drug_name.lower(), "has_black_box": True,
            "warnings": s.black_box}


@router.get("/drugs/{drug_name}/faers")
async def faers_for_drug(drug_name: str, limit: int = Query(10, ge=1, le=50)):
    """Top adverse events for a drug from the live FAERS API."""
    events = await top_adverse_events(drug_name, limit=limit)
    return {
        "drug": drug_name.lower(),
        "n_reactions": len(events),
        "top_reactions": events,
    }


@router.get("/drugs/{drug_name}/approval")
async def approval_status(drug_name: str):
    """FDA approval status (curated Orange Book snapshot)."""
    ob = lookup(drug_name)
    return {
        "drug": drug_name.lower(),
        "fda_approved": is_approved(drug_name),
        "entries": [
            {"ingredient": e.ingredient, "trade_name": e.trade_name,
             "applicant": e.applicant, "approval_date": e.approval_date}
            for e in ob
        ],
    }


@router.get("/rxnorm/normalize")
async def rxnorm(name: str = Query(...)):
    """Normalize a drug name to RxNorm standard form."""
    res = await normalize_rxnorm(name)
    if res is None:
        raise HTTPException(404, f"RxNorm could not resolve '{name}'")
    return res


@router.get("/registry/snapshot")
def orange_book_snapshot():
    """The full curated Orange Book snapshot."""
    return {
        "n_drugs": len(CURATED_ORANGE_BOOK),
        "drugs": [
            {"ingredient": e.ingredient, "trade_name": e.trade_name,
             "applicant": e.applicant, "approval_date": e.approval_date}
            for e in CURATED_ORANGE_BOOK
        ],
    }
