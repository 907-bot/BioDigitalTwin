"""FastAPI router for Phase 12 — Clinical Trials (ClinicalTrials.gov v2)."""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from .client import get_trial, search_by_condition, search_by_drug

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/phase12", tags=["Phase 12 — Clinical Trials"])


@router.get("/trials/search")
async def trials_search(q: str = Query(..., description="condition or drug"),
                         max: int = Query(20, ge=1, le=100),
                         by: str = Query("condition", pattern="^(condition|drug)$")):
    """Search ClinicalTrials.gov. `by=condition` (default) or `by=drug`."""
    if by == "drug":
        results = await search_by_drug(q, max_results=max)
    else:
        results = await search_by_condition(q, max_results=max)
    return {
        "query": q,
        "by": by,
        "n_results": len(results),
        "trials": results,
        "narrative": __import__("app.narrative", fromlist=["trials"]).trials.narrate_search(
            query=q, by=by, n_results=len(results), trials=results,
        ),
    }


@router.get("/trials/{nct_id}")
async def trial_detail(nct_id: str):
    """Full record for a single NCT trial."""
    res = await get_trial(nct_id)
    if res is None:
        raise HTTPException(404, f"trial '{nct_id}' not found")
    return res
