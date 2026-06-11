"""
Narrative Service — human-readable text generation for all phases.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Narrative Service", version="0.1.0")
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "healthy", "service": "narrative", "version": "0.1.0"}


class NarrativeRequest(BaseModel):
    phase: str
    data: dict
    level: str = "lay"  # lay | scientist


@app.post("/narrative/generate")
def generate_narrative(req: NarrativeRequest):
    from app.narrative._utils import risk_from_severity, empty_narrative

    phase_handlers = {
        "pgx": lambda d: {"headline": "Pharmacogenomics profile generated", "lay": f"Analysis of {len(d.get('genes', []))} genes", "scientist": str(d), "risk_level": risk_from_severity(d.get("severity", 0))},
        "ddi": lambda d: {"headline": f"Drug interaction check: {len(d.get('interactions', []))} found", "lay": f"Found {len(d.get('interactions', []))} potential interactions", "scientist": str(d), "risk_level": risk_from_severity(d.get("severity", 0))},
        "pkpd": lambda d: {"headline": "PK/PD simulation complete", "lay": "Drug concentration and effect simulated", "scientist": str(d), "risk_level": "info"},
        "trial": lambda d: {"headline": f"Trial: {d.get('nct_id', 'unknown')}", "lay": d.get("brief_title", "Clinical trial data"), "scientist": str(d), "risk_level": "info"},
        "regulatory": lambda d: {"headline": f"Regulatory profile: {d.get('drug', 'unknown')}", "lay": "FDA approval and safety data", "scientist": str(d), "risk_level": risk_from_severity(d.get("severity", 0))},
        "wetlab": lambda d: {"headline": f"Wet-lab validation: {d.get('passed', False)}", "lay": "Computational molecular triage complete", "scientist": str(d), "risk_level": "info"},
        "uq": lambda d: {"headline": "Uncertainty quantification complete", "lay": f"Confidence: {d.get('confidence', 'N/A')}", "scientist": str(d), "risk_level": "info"},
    }

    handler = phase_handlers.get(req.phase, lambda d: empty_narrative())
    return handler(req.data)
