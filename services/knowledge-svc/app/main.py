"""
Knowledge Service — drug knowledge, regulatory, trials, wetlab.

Combines: PGx (8), DDI (9), PKPD (10), Trials (12), Regulatory (13),
          WetLab (14), Disease Registry (15)
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Knowledge Service", version="0.1.0")
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "healthy", "service": "knowledge", "version": "0.1.0"}


# Register knowledge-domain routers
from app.pgx import pgx_router
from app.ddi import ddi_router
from app.pkpd import pkpd_router
from app.trials import trials_router
from app.regulatory import regulatory_router
from app.wetlab import wetlab_router
from app.registry import registry_router
from app.xai import router as xai_router

app.include_router(pgx_router)
app.include_router(ddi_router)
app.include_router(pkpd_router)
app.include_router(trials_router)
app.include_router(regulatory_router)
app.include_router(wetlab_router)
app.include_router(registry_router)
app.include_router(xai_router)
