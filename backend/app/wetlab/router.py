"""FastAPI router for Phase 14 — Wet-Lab Validation."""
from __future__ import annotations

import logging

from fastapi import APIRouter
from pydantic import BaseModel

from .validation import batch_validate, validate_lead

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/phase14", tags=["Phase 14 — Wet-Lab Validation"])


class ValidateRequest(BaseModel):
    smiles: str


class BatchValidateRequest(BaseModel):
    smiles_list: list[str]


@router.post("/validate")
def validate(req: ValidateRequest):
    """Full wet-lab readiness report for a single SMILES."""
    return validate_lead(req.smiles)


@router.post("/validate-batch")
def validate_batch(req: BatchValidateRequest):
    """Validate many SMILES at once."""
    return {"n": len(req.smiles_list), "reports": batch_validate(req.smiles_list)}


@router.get("/rdkit-version")
def rdkit_status():
    from .validation import RDKIT_AVAILABLE
    if not RDKIT_AVAILABLE:
        return {"rdkit_available": False}
    from rdkit import Chem
    return {"rdkit_available": True,
            "version": Chem.rdBase.rdkitVersion}
