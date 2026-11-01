"""
Phase 18 — Samsung Health Data Integration & Training Module
Parses real Samsung Health CSV exports → feature engineering → ML training → predictions
"""
from .router import samsung_router

__all__ = ["samsung_router"]
