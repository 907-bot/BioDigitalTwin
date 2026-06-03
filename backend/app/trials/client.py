"""
Phase 12 — Clinical Trials integration.

Wraps the ClinicalTrials.gov v2 API (free, public, no auth). Stable
records are cached on disk for 24h.

Reference: https://clinicaltrials.gov/data-api/v2/about
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

CT_BASE = "https://clinicaltrials.gov/api/v2/studies"
CACHE_DIR = Path("/tmp/biodigital_ct_cache")
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
    except Exception as e:  # noqa: BLE001
        logger.debug("cache write failed: %s", e)


def _format_study(s: dict) -> dict:
    """Project a CT.gov v2 study record into a small, friendly dict."""
    proto = s.get("protocolSection", {})
    ident = proto.get("identificationModule", {})
    status = proto.get("statusModule", {})
    design = proto.get("designModule", {})
    elig = proto.get("eligibilityModule", {})
    sponsor = proto.get("sponsorCollaboratorsModule", {})
    outcomes = proto.get("outcomesModule", {})
    desc = proto.get("descriptionModule", {})
    return {
        "nct_id": ident.get("nctId"),
        "title": (ident.get("briefTitle") or "")[:300],
        "official_title": (ident.get("officialTitle") or "")[:300],
        "overall_status": status.get("overallStatus"),
        "phase": design.get("phases") or [],
        "study_type": design.get("studyType"),
        "enrollment": (design.get("enrollmentInfo") or {}).get("count"),
        "start_date": (status.get("startDateStruct") or {}).get("date"),
        "completion_date": (status.get("completionDateStruct") or {}).get("date"),
        "last_update": status.get("lastUpdatePostDateStruct", {}).get("date"),
        "primary_completion": (status.get("primaryCompletionDateStruct") or {}).get("date"),
        "conditions": proto.get("conditionsModule", {}).get("conditions", []),
        "interventions": [
            {"type": i.get("type"), "name": i.get("name")}
            for i in (proto.get("armsInterventionsModule", {})
                       .get("interventions", []) or [])
        ],
        "primary_outcomes": [
            {"measure": o.get("measure"), "time_frame": o.get("timeFrame")}
            for o in (outcomes.get("primaryOutcomes", []) or [])
        ],
        "secondary_outcomes": [
            {"measure": o.get("measure"), "time_frame": o.get("timeFrame")}
            for o in (outcomes.get("secondaryOutcomes", []) or [])
        ],
        "sponsor": (sponsor.get("leadSponsor") or {}).get("name"),
        "collaborators": [c.get("name") for c in (sponsor.get("collaborators") or [])],
        "eligibility_criteria": (elig.get("eligibilityCriteria") or "")[:1500],
        "healthy_volunteers": elig.get("healthyVolunteers"),
        "minimum_age": elig.get("minimumAge"),
        "maximum_age": elig.get("maximumAge"),
        "sex": elig.get("sex"),
        "brief_summary": (desc.get("briefSummary") or "")[:600],
    }


async def _fetch(params: dict[str, Any]) -> dict:
    key = json.dumps(params, sort_keys=True)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        r = await client.get(CT_BASE, params=params)
        r.raise_for_status()
        data = r.json()
    _cache_put(key, data)
    return data


async def search_by_condition(condition: str, max_results: int = 20) -> list[dict]:
    """
    Search for trials by condition/disease term.

    Example: 'type 2 diabetes', 'hypertension', 'alzheimer'.
    """
    params = {
        "query.cond": condition,
        "pageSize": min(max_results, 100),
        "format": "json",
        "sort": ["LastUpdatePostDate:desc"],
    }
    try:
        data = await _fetch(params)
    except httpx.HTTPError as e:
        logger.warning("CT.gov search failed: %s", e)
        return []
    return [_format_study(s) for s in data.get("studies", [])]


async def search_by_drug(intervention: str, max_results: int = 20) -> list[dict]:
    """Search for trials by intervention name (drug)."""
    params = {
        "query.intr": intervention,
        "pageSize": min(max_results, 100),
        "format": "json",
        "sort": ["LastUpdatePostDate:desc"],
    }
    try:
        data = await _fetch(params)
    except httpx.HTTPError as e:
        logger.warning("CT.gov drug search failed: %s", e)
        return []
    return [_format_study(s) for s in data.get("studies", [])]


async def get_trial(nct_id: str) -> Optional[dict]:
    """Fetch a single trial by NCT ID."""
    params = {"format": "json"}
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.get(f"{CT_BASE}/{nct_id}", params=params)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPError as e:
        logger.warning("CT.gov get_trial failed: %s", e)
        return None
    return _format_study(data)
