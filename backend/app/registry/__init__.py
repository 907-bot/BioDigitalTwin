"""Phase 15 — Extensible Disease Registry (Postgres-backed)."""
from .db import (
    DiseaseEntry,
    create_disease,
    delete_disease,
    get_disease,
    get_registry,
    list_diseases,
    update_disease,
)
from .router import router as registry_router

__all__ = [
    "DiseaseEntry",
    "create_disease", "delete_disease", "get_disease", "get_registry",
    "list_diseases", "update_disease",
    "registry_router",
]
