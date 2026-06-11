"""
Twin Service — whole-body digital twin, disease dynamics, UQ, XAI.

Combines: Phase 3 (personalization v2), Phase 3 (disease dynamics),
          Phase 11 (UQ), Phase 16 (XAI)
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Twin Service", version="0.1.0")
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "healthy", "service": "twin", "version": "0.1.0"}


# ── Disease Dynamics (Phase 3) ──
_lazy_simulator = None
def get_simulator():
    global _lazy_simulator
    if _lazy_simulator is None:
        from app.dynamics.disease_model import DiseaseSimulator
        _lazy_simulator = DiseaseSimulator()
    return _lazy_simulator


class SimulateRequest(BaseModel):
    initial_state: dict
    disease: str = Field(..., description="One of: t2d, hypertension, cvd, copd")
    horizon_days: int = Field(180, ge=1, le=1825)
    dt_hours: float = Field(6.0, gt=0, le=48)
    intervention: Optional[dict] = Field(default=None)
    intervention_name: Optional[str] = Field(default=None)
    rng_seed: int = 0
    sample_every_hours: Optional[float] = Field(default=None)


class CounterfactualRequest(BaseModel):
    initial_state: dict
    disease: str
    horizon_days: int = 365
    intervention: Optional[dict] = None
    intervention_name: Optional[str] = None
    dt_hours: float = 6.0
    rng_seed: int = 0


def _resolve_intervention(body) -> dict:
    if body.intervention_name:
        from app.dynamics.disease_model import INTERVENTIONS
        if body.intervention_name not in INTERVENTIONS:
            raise HTTPException(status_code=404, detail=f"unknown intervention: {body.intervention_name}")
        return INTERVENTIONS[body.intervention_name]
    return body.intervention or {}


def _format_simulation(res: dict, body, intervention: dict) -> dict:
    sample_every = body.sample_every_hours or body.dt_hours
    step = max(1, int(sample_every / body.dt_hours))
    times_d = [round(t / 24.0, 2) for t in res["times_h"][::step]]
    biomarkers = []
    from bio_digital_twin_core import NODE_INDEX
    by_id = NODE_INDEX
    for name, series in res["series"].items():
        b = by_id.get(name)
        biomarkers.append({
            "name": name, "label": b.name if b else name, "unit": b.unit if b else "",
            "healthy_lo": b.healthy_lo if b else None, "healthy_hi": b.healthy_hi if b else None,
            "baseline": res["final_state"].get(name, series[0] if series else None),
            "trajectory": [{"day": d, "value": round(float(v), 3)} for d, v in zip(times_d, series[::step])],
        })
    return {
        "disease": body.disease, "horizon_days": body.horizon_days,
        "steps": len(res["times_h"]),
        "disease_state": res["disease_state"],
        "final_risk": round(res["final_risk"], 4),
        "initial_risk": round(res["risks"][0], 4) if res["risks"] else None,
        "risk_evolution": [{"day": d, "risk": round(float(r), 4)} for d, r in zip(times_d, res["risks"][::step])],
        "biomarkers": biomarkers,
        "spike_view": {
            "dominant_biomarker": res["lif_dominant_biomarker"],
            "spike_count": res["spike_count"], "spike_rate_hz": res["spike_rate_hz"],
        },
        "intervention_applied": intervention or None,
    }


@app.get("/dynamics/diseases")
def list_diseases():
    from app.dynamics.disease_model import DISEASE_FORCINGS, bifurcation_summary
    return {"diseases": [{"id": k, "name": v.name, "bifurcation": bifurcation_summary(k)} for k, v in DISEASE_FORCINGS.items()]}


@app.get("/dynamics/interventions")
def list_interventions():
    from app.dynamics.disease_model import INTERVENTIONS
    return {"interventions": [{"name": k, "daily_delta": v} for k, v in INTERVENTIONS.items()]}


@app.get("/dynamics/attractors")
def list_attractors():
    from app.dynamics.disease_model import ATTRACTORS
    return {"attractors": [{"name": a.name, "description": a.description, "risk_range": [a.risk_lo, a.risk_hi]} for a in ATTRACTORS]}


@app.post("/dynamics/simulate")
def simulate(req: SimulateRequest):
    try:
        intervention = _resolve_intervention(req)
        sim = get_simulator()
        res = sim.simulate(initial_state=req.initial_state, disease=req.disease,
                           horizon_days=req.horizon_days, intervention=intervention, rng_seed=req.rng_seed)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _format_simulation(res, req, intervention)


@app.post("/dynamics/counterfactual")
def counterfactual(req: CounterfactualRequest):
    try:
        intervention = _resolve_intervention(req)
        sim = get_simulator()
        ctrl = sim.simulate(initial_state=req.initial_state, disease=req.disease,
                            horizon_days=req.horizon_days, intervention={}, rng_seed=req.rng_seed)
        tx = sim.simulate(initial_state=req.initial_state, disease=req.disease,
                          horizon_days=req.horizon_days, intervention=intervention, rng_seed=req.rng_seed)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "disease": req.disease, "horizon_days": req.horizon_days,
        "intervention_applied": intervention or None,
        "control": {"final_risk": round(ctrl["final_risk"], 4), "disease_state": ctrl["disease_state"], "final_state": ctrl["final_state"]},
        "treated": {"final_risk": round(tx["final_risk"], 4), "disease_state": tx["disease_state"], "final_state": tx["final_state"]},
        "counterfactual_effect": {
            "absolute_risk_reduction": round(ctrl["final_risk"] - tx["final_risk"], 4),
            "relative_risk_reduction": round((ctrl["final_risk"] - tx["final_risk"]) / max(ctrl["final_risk"], 1e-6), 4),
            "state_changed": ctrl["disease_state"] != tx["disease_state"],
        },
    }


# ── Personalization (v2, v4) ──
from app.personalization import personalization_router
from app.personalization import phase5_router

app.include_router(personalization_router)
app.include_router(phase5_router)


# ── UQ (Phase 11) ──
from app.uq import uq_router
app.include_router(uq_router)


# ── XAI (Phase 16) ──
from app.xai import router as xai_router
app.include_router(xai_router)
