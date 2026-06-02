"""Phase 12 — Clinical Trials (ClinicalTrials.gov v2)."""
from .client import get_trial, search_by_condition, search_by_drug
from .router import router as trials_router

__all__ = ["get_trial", "search_by_condition", "search_by_drug", "trials_router"]
