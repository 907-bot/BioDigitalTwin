"""FastAPI router for Phase 15 — Extensible Disease Registry."""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .db import (
    create_disease,
    delete_disease,
    get_disease,
    get_registry,
    list_diseases,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/phase15", tags=["Phase 15 — Disease Registry"])


class DiseaseIn(BaseModel):
    key: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-z0-9_]+$")
    name: str
    description: str = ""
    target_proteins: list[str] = Field(default_factory=list)
    current_treatments: str = ""
    clinical_trials: int = 0
    unmet_need: str = "high"
    added_by: str = "user"


class DiseaseUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    target_proteins: Optional[list[str]] = None
    current_treatments: Optional[str] = None
    clinical_trials: Optional[int] = None
    unmet_need: Optional[str] = None


@router.get("/registry/diseases")
def list_registry():
    """List all diseases in the registry."""
    items = list_diseases()
    return {"n": len(items), "diseases": items,
            "narrative": __import__("app.narrative", fromlist=["registry"]).registry.narrate_summary(
                {"n_diseases": len(items), "n_target_proteins": sum(len(d.get("target_proteins") or []) for d in items),
                 "total_clinical_trials": sum(d.get("clinical_trials") or 0 for d in items),
                 "by_unmet_need": {}},
            )}


@router.get("/registry/diseases/{key}")
def get_one(key: str):
    d = get_disease(key)
    if not d:
        raise HTTPException(404, f"disease '{key}' not found")
    d["narrative"] = __import__("app.narrative", fromlist=["registry"]).registry.narrate_disease(d)
    return d


@router.post("/registry/diseases")
def create_one(d: DiseaseIn):
    try:
        return create_disease(d.model_dump())
    except ValueError as e:
        raise HTTPException(409, str(e))


@router.put("/registry/diseases/{key}")
def update_one(key: str, d: DiseaseUpdate):
    try:
        return update_disease(key, d.model_dump(exclude_none=True))
    except KeyError:
        raise HTTPException(404, f"disease '{key}' not found")


@router.delete("/registry/diseases/{key}")
def delete_one(key: str):
    ok = delete_disease(key)
    if not ok:
        raise HTTPException(404, f"disease '{key}' not found")
    return {"status": "deleted", "key": key}


@router.get("/registry/summary")
def summary():
    """Counts by unmet_need, target_proteins used, etc."""
    items = list_diseases()
    by_unmet: dict[str, int] = {}
    for e in items:
        by_unmet[e["unmet_need"]] = by_unmet.get(e["unmet_need"], 0) + 1
    proteins: set[str] = set()
    for e in items:
        proteins.update(e.get("target_proteins", []))
    return {
        "n_diseases": len(items),
        "n_target_proteins": len(proteins),
        "by_unmet_need": by_unmet,
        "total_clinical_trials": sum(e.get("clinical_trials", 0) for e in items),
    }


@router.get("/registry/diseases-lookup")
def diseases_lookup():
    """Fast {key: name} map for use by other modules."""
    return {e["key"]: e["name"] for e in list_diseases()}
