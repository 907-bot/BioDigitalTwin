"""
Phase 15 — Extensible Disease Registry.

Replaces the hard-coded UNTREATABLE_DISEASES in chembldiscovery with a
Postgres-backed CRUD store. Seeded with the 8 upstream entries on first
run. Supports add/edit/delete via the admin UI.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

import pandas as pd
from sqlalchemy import (
    Column, Integer, String, DateTime, Text, create_engine, Index,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import declarative_base, sessionmaker, Session

logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@postgres:5432/biodigital",
)

Base = declarative_base()


class DiseaseEntry(Base):
    __tablename__ = "registry_diseases"
    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(64), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, default="")
    target_proteins = Column(ARRAY(String), default=list)
    current_treatments = Column(Text, default="")
    clinical_trials = Column(Integer, default=0)
    unmet_need = Column(String(32), default="high")
    added_at = Column(DateTime, default=datetime.utcnow)
    added_by = Column(String(64), default="system")


# --- seed data (the 8 upstream untreatable diseases) ---
SEED: list[dict] = [
    {"key": "als", "name": "Amyotrophic Lateral Sclerosis",
     "description": "Motor neuron disease",
     "target_proteins": ["SOD1", "TDP-43", "FUS", "C9orf72"],
     "current_treatments": "Riluzole, edaravone",
     "clinical_trials": 156, "unmet_need": "critical"},
    {"key": "huntington", "name": "Huntington Disease",
     "description": "Neurodegenerative genetic disorder",
     "target_proteins": ["HTT", "Caspase-3", "mTOR"],
     "current_treatments": "Tetrabenazine, deutetrabenazine",
     "clinical_trials": 47, "unmet_need": "high"},
    {"key": "duchenne", "name": "Duchenne Muscular Dystrophy",
     "description": "Genetic muscle disorder",
     "target_proteins": ["DMD", "Utrophin", "Myostatin"],
     "current_treatments": "Prednisone, deflazacort, eteplirsen",
     "clinical_trials": 89, "unmet_need": "high"},
    {"key": "cf", "name": "Cystic Fibrosis",
     "description": "Genetic lung disease",
     "target_proteins": ["CFTR", "ENaC", "Sodium channel"],
     "current_treatments": "Ivacaftor, lumacaftor, elexacaftor",
     "clinical_trials": 234, "unmet_need": "medium"},
    {"key": "sickle_cell", "name": "Sickle Cell Disease",
     "description": "Blood disorder",
     "target_proteins": ["Hemoglobin", "BCL11A", "Fetal hemoglobin"],
     "current_treatments": "Hydroxyurea, voxelotor, L-Glutamine",
     "clinical_trials": 67, "unmet_need": "high"},
    {"key": "parkinson", "name": "Parkinson Disease",
     "description": "Neurodegenerative movement disorder",
     "target_proteins": ["Alpha-synuclein", "LRRK2", "DJ-1", "PINK1"],
     "current_treatments": "Levodopa, carbidopa, ropininirole",
     "clinical_trials": 412, "unmet_need": "high"},
    {"key": "alzheimers", "name": "Alzheimer Disease",
     "description": "Neurodegenerative dementia",
     "target_proteins": ["Amyloid-beta", "Tau", "BACE1", "APP"],
     "current_treatments": "Donepezil, memantine, lecanemab",
     "clinical_trials": 589, "unmet_need": "critical"},
    {"key": "multiple_sclerosis", "name": "Multiple Sclerosis",
     "description": "Autoimmune demyelinating disease",
     "target_proteins": ["Myelin", "CD20", "IL-17", "IFN-beta"],
     "current_treatments": "Interferon, glatiramer, ocrelizumab",
     "clinical_trials": 345, "unmet_need": "high"},
]


# In-memory fallback (used when Postgres is unreachable)
class _MemoryRegistry:
    def __init__(self) -> None:
        self._store: dict[str, dict] = {e["key"]: e for e in SEED}
        self._next_id = len(SEED) + 1

    def list(self) -> list[dict]:
        return list(self._store.values())

    def get(self, key: str) -> Optional[dict]:
        return self._store.get(key)

    def create(self, payload: dict) -> dict:
        if payload["key"] in self._store:
            raise ValueError(f"key '{payload['key']}' already exists")
        payload.setdefault("id", self._next_id)
        self._next_id += 1
        payload.setdefault("added_at", datetime.utcnow().isoformat())
        payload.setdefault("added_by", "user")
        self._store[payload["key"]] = payload
        return payload

    def update(self, key: str, payload: dict) -> dict:
        if key not in self._store:
            raise KeyError(key)
        merged = {**self._store[key], **payload, "key": key}
        self._store[key] = merged
        return merged

    def delete(self, key: str) -> bool:
        return self._store.pop(key, None) is not None


_mem = _MemoryRegistry()
_engine = None
_SessionLocal = None
_pg_available = False


def _init_engine() -> None:
    global _engine, _SessionLocal, _pg_available
    if _engine is not None:
        return
    try:
        _engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        Base.metadata.create_all(_engine)
        _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
        with _SessionLocal() as s:
            for entry in SEED:
                if not s.query(DiseaseEntry).filter_by(key=entry["key"]).first():
                    s.add(DiseaseEntry(**entry))
            s.commit()
        _pg_available = True
        logger.info("Connected to Postgres registry")
    except Exception as e:  # noqa: BLE001
        logger.warning("Postgres unavailable, using in-memory registry: %s", e)
        _pg_available = False


def _to_dict(e: DiseaseEntry | dict) -> dict:
    if isinstance(e, dict):
        d = dict(e)
    else:
        d = {
            "id": e.id, "key": e.key, "name": e.name,
            "description": e.description or "",
            "target_proteins": list(e.target_proteins or []),
            "current_treatments": e.current_treatments or "",
            "clinical_trials": e.clinical_trials or 0,
            "unmet_need": e.unmet_need or "high",
            "added_at": e.added_at.isoformat() if e.added_at else None,
            "added_by": e.added_by or "system",
        }
    return d


def list_diseases() -> list[dict]:
    _init_engine()
    if _pg_available and _SessionLocal is not None:
        with _SessionLocal() as s:
            return [_to_dict(e) for e in s.query(DiseaseEntry).order_by(DiseaseEntry.id).all()]
    return [_to_dict(e) for e in _mem.list()]


def get_disease(key: str) -> Optional[dict]:
    _init_engine()
    if _pg_available and _SessionLocal is not None:
        with _SessionLocal() as s:
            e = s.query(DiseaseEntry).filter_by(key=key).first()
            return _to_dict(e) if e else None
    e = _mem.get(key)
    return _to_dict(e) if e else None


def create_disease(payload: dict) -> dict:
    _init_engine()
    if _pg_available and _SessionLocal is not None:
        with _SessionLocal() as s:
            if s.query(DiseaseEntry).filter_by(key=payload["key"]).first():
                raise ValueError(f"key '{payload['key']}' already exists")
            entry = DiseaseEntry(**payload)
            s.add(entry)
            s.commit()
            s.refresh(entry)
            return _to_dict(entry)
    return _to_dict(_mem.create(payload))


def update_disease(key: str, payload: dict) -> dict:
    _init_engine()
    if _pg_available and _SessionLocal is not None:
        with _SessionLocal() as s:
            entry = s.query(DiseaseEntry).filter_by(key=key).first()
            if not entry:
                raise KeyError(key)
            for k, v in payload.items():
                if k == "id":
                    continue
                setattr(entry, k, v)
            s.commit()
            s.refresh(entry)
            return _to_dict(entry)
    return _to_dict(_mem.update(key, payload))


def delete_disease(key: str) -> bool:
    _init_engine()
    if _pg_available and _SessionLocal is not None:
        with _SessionLocal() as s:
            entry = s.query(DiseaseEntry).filter_by(key=key).first()
            if not entry:
                return False
            s.delete(entry)
            s.commit()
            return True
    return _mem.delete(key)


def get_registry() -> dict[str, dict]:
    """Return {key: entry} — used by the research module."""
    return {e["key"]: e for e in list_diseases()}
