"""Phase 13 — Regulatory knowledge (FDA, FAERS, RxNorm, warnings)."""
from .faers import drug_summary, top_adverse_events
from .orange_book import (
    CURATED_ORANGE_BOOK,
    OrangeBookEntry,
    is_approved,
    lookup,
    normalize_rxnorm,
)
from .warnings import SAFETY_REGISTRY, DrugSafety, get_safety, has_black_box
from .router import router as regulatory_router

__all__ = [
    "drug_summary", "top_adverse_events",
    "CURATED_ORANGE_BOOK", "OrangeBookEntry", "is_approved", "lookup",
    "normalize_rxnorm",
    "SAFETY_REGISTRY", "DrugSafety", "get_safety", "has_black_box",
    "regulatory_router",
]
