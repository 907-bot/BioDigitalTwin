"""
Ontology Service — single source of truth for biological ontology.

Routes:
  GET /ontology/biomarkers
  GET /ontology/organs
  GET /ontology/diseases
  GET /ontology/graph
  GET /ontology/positions
  GET /health
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from bio_digital_twin_core.ontology import BIOMARKERS, ORGANS, DISEASES, EDGES

app = FastAPI(title="Ontology Service", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "healthy", "service": "ontology", "version": "0.1.0"}


@app.get("/ontology/biomarkers")
def list_biomarkers():
    return {"biomarkers": [{"id": b.id, "name": b.name, "unit": b.unit, "healthy_lo": b.healthy_lo, "healthy_hi": b.healthy_hi} for b in BIOMARKERS]}


@app.get("/ontology/organs")
def list_organs():
    return {"organs": [{"id": o.id, "name": o.name} for o in ORGANS]}


@app.get("/ontology/diseases")
def list_diseases():
    return {"diseases": [{"id": d.id, "name": d.name} for d in DISEASES]}


@app.get("/ontology/graph")
def full_graph():
    return {
        "nodes": [{"id": n.id, "kind": n.kind, "name": n.name, "unit": n.unit, "healthy_lo": n.healthy_lo, "healthy_hi": n.healthy_hi} for n in (BIOMARKERS + ORGANS + DISEASES)],
        "edges": [{"src": e.src, "dst": e.dst, "rel": e.rel, "weight": e.weight} for e in EDGES],
    }


@app.get("/ontology/positions")
def organ_positions():
    return {
        "positions": {
            "heart": [-0.08, 1.42, 0.10],
            "vasculature": [0.00, 1.20, 0.05],
            "lungs": [0.00, 1.40, 0.06],
            "pancreas": [-0.04, 1.10, 0.08],
            "liver": [0.10, 1.18, 0.10],
            "kidney": [0.00, 1.05, -0.08],
            "hr": [-0.10, 1.46, 0.10],
            "hrv": [-0.06, 1.40, 0.10],
            "spo2": [0.00, 1.42, 0.08],
            "glucose": [-0.04, 1.08, 0.08],
            "systolic_bp": [0.00, 1.20, 0.05],
            "diastolic_bp": [0.00, 1.18, 0.05],
            "bmi": [0.00, 0.95, 0.20],
            "t2d": [-0.04, 1.06, 0.08],
            "hypertension": [0.00, 1.22, 0.04],
            "cvd": [-0.08, 1.40, 0.10],
            "copd": [0.00, 1.40, 0.10],
            "age": [0.00, 1.70, 0.06],
        }
    }
