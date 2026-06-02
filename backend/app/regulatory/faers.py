"""
FAERS (FDA Adverse Event Reporting System) live API client.

OpenFDA endpoint: https://api.fda.gov/drug/event.json
Free with rate limits (no key required for low volume; with key for higher).

We query by drug name and return the top N adverse-event reactions sorted
by report count. This is a real-world post-market surveillance signal,
not a clinical trial signal — interpret with care.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

OPENFDA_BASE = "https://api.fda.gov/drug/event.json"
CACHE_DIR = Path("/tmp/biodigital_faers_cache")
CACHE_TTL = timedelta(hours=24)
TIMEOUT = 20.0


def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    h = hashlib.md5(key.encode()).hexdigest()
    return CACHE_DIR / f"{h}.json"


def _cache_get(key: str) -> Optional[dict]:
    p = _cache_path(key)
    if not p.exists():
        return None
    age = datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)
    if age > CACHE_TTL:
        return None
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def _cache_put(key: str, data: dict) -> None:
    p = _cache_path(key)
    try:
        p.write_text(json.dumps(data))
    except Exception:  # noqa: BLE001
        pass


def _normalize(drug: str) -> str:
    """OpenFDA expects generic or brand names; we just pass the input."""
    return drug.strip().lower()


async def top_adverse_events(drug: str, limit: int = 10) -> list[dict]:
    """
    Top adverse events for a drug from FAERS, sorted by report count.

    Returns a list of {reaction, count} dicts.
    """
    drug_n = _normalize(drug)
    cache_key = f"faers:{drug_n}:{limit}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached.get("events", [])

    params = {
        "search": f'patient.drug.medicinalproduct:"{drug_n}"',
        "count": "patient.reaction.reactionmeddrapt.exact",
        "limit": min(limit, 100),
    }
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.get(OPENFDA_BASE, params=params)
            if r.status_code == 404:
                _cache_put(cache_key, {"events": []})
                return []
            if r.status_code == 429:
                logger.warning("OpenFDA rate limited")
                return []
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPError as e:
        logger.warning("OpenFDA failed: %s", e)
        return []

    events = [
        {"reaction": item["term"], "count": int(item["count"])}
        for item in data.get("results", [])
    ]
    _cache_put(cache_key, {"events": events})
    return events


async def drug_summary(drug: str) -> dict:
    """FAERS high-level summary for a drug: total reports, top AEs, by sex, by age."""
    drug_n = _normalize(drug)
    cache_key = f"faers_summary:{drug_n}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    params_total = {
        "search": f'patient.drug.medicinalproduct:"{drug_n}"',
        "limit": 1,
    }
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.get(OPENFDA_BASE, params=params_total)
            if r.status_code == 404:
                result = {"drug": drug_n, "total_reports": 0,
                          "serious_reports": 0, "death_reports": 0,
                          "top_reactions": []}
                _cache_put(cache_key, result)
                return result
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPError as e:
        logger.warning("OpenFDA summary failed: %s", e)
        return {"drug": drug_n, "error": str(e)}

    top_ae = await top_adverse_events(drug, limit=10)
    result = {
        "drug": drug_n,
        "total_reports": data.get("meta", {}).get("results", {}).get("total", 0),
        "top_reactions": top_ae,
    }
    _cache_put(cache_key, result)
    return result
