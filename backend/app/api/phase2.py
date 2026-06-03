"""
Phase 2 router — Graph-based digital twin.

Endpoints:
  POST /api/phase2/graph/build
        Build the cohort graph from PostgreSQL (or freshly generated
        synthetic data) and persist to Neo4j.
  POST /api/phase2/graph/train
        Self-supervised training of the GraphSAGE patient encoder.
  GET  /api/phase2/patients/{id}/state
        Return the patient's graph state (embedding, abnormal biomarkers,
        affected organs, active diseases, nearest neighbours).
  GET  /api/phase2/patients/{id}/predict/{biomarker}
        Learned-prior estimate for a missing biomarker.
  GET  /api/phase2/graph/stats
        Topology statistics.
  GET  /api/phase2/ontology
        The static biological ontology.
"""
from __future__ import annotations

import json
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.crud import patient as patient_crud
from app.graph.builder import (
    GraphBuilder, get_or_build_cohort_graph, reset_cohort_graph,
    _normalise, compute_risk_score, _risk_label,
)
from app.graph.gnn import gnn_service
from app.graph.neo4j_client import neo4j_client
from app.graph.ontology import (
    ALL_NODES, BIOMARKERS, DISEASES, EDGES, ORGANS, NODE_INDEX,
)
from app.schemas import GraphBuildResponse, PatientGraphState
from app.synthetic.generator import generate_patients

router = APIRouter(prefix="/api/phase2", tags=["phase2"])


# --- helpers ----------------------------------------------------------
def _load_or_generate_cohort(db: Session, n: int, seed: int) -> "pd.DataFrame":
    import pandas as pd
    count = patient_crud.count_patients(db)
    if count >= n:
        objs = patient_crud.list_patients(db, limit=n)
        df = pd.DataFrame([{
            "patient_id": o.patient_id, "age": o.age, "gender": o.gender,
            "bmi": o.bmi, "hr": o.hr, "hrv": o.hrv, "spo2": o.spo2,
            "glucose": o.glucose, "systolic_bp": o.systolic_bp,
            "diastolic_bp": o.diastolic_bp,
        } for o in objs])
    else:
        df = generate_patients(n=n, seed=seed)
        for _, row in df.iterrows():
            patient_crud.upsert_patient(db, row.to_dict())
        patient_crud.commit(db)
    return df


def _patient_subgraph(patient: dict, graph) -> dict:
    pid = patient["patient_id"]
    abnorm = []
    for b in BIOMARKERS:
        v = patient.get(b.name)
        if v is None:
            continue
        in_range = b.healthy_lo <= float(v) <= b.healthy_hi
        abnorm.append({
            "biomarker": b.name, "value": float(v), "unit": b.unit,
            "healthy_lo": b.healthy_lo, "healthy_hi": b.healthy_hi,
            "is_abnormal": not in_range,
            "deviation_norm": round(
                abs(_normalise(float(v), b.healthy_lo, b.healthy_hi)), 3),
        })
    abnorm = [a for a in abnorm if a["is_abnormal"]]
    abnorm.sort(key=lambda a: a["deviation_norm"], reverse=True)

    affected_organs = []
    active_diseases = []
    for e in EDGES:
        if e.src == "bmi" or e.src in [b.name for b in BIOMARKERS]:
            src_val = patient.get(e.src)
            if src_val is None:
                continue
            tgt = NODE_INDEX.get(e.dst)
            if tgt is None:
                continue
            if e.rel == "REGULATED_BY":
                affected_organs.append({
                    "organ": tgt.id, "name": tgt.name,
                    "via": e.src, "weight": e.weight,
                })
            elif e.rel == "ELEVATED_IN":
                b = next(b for b in BIOMARKERS if b.name == e.src)
                if float(src_val) > b.healthy_hi:
                    active_diseases.append({
                        "disease": tgt.id, "name": tgt.name,
                        "via": e.src, "weight": e.weight, "direction": "elevated",
                    })
            elif e.rel == "DEPRESSED_IN":
                b = next(b for b in BIOMARKERS if b.name == e.src)
                if float(src_val) < b.healthy_lo:
                    active_diseases.append({
                        "disease": tgt.id, "name": tgt.name,
                        "via": e.src, "weight": e.weight, "direction": "depressed",
                    })

    # de-dupe organs / diseases keeping highest-weight entry
    def _dedupe(items, key):
        best = {}
        for it in items:
            k = it[key]
            if k not in best or it["weight"] > best[k]["weight"]:
                best[k] = it
        return sorted(best.values(), key=lambda x: -x["weight"])

    return {
        "abnormal_biomarkers": abnorm,
        "affected_organs": _dedupe(affected_organs, "organ"),
        "active_diseases": _dedupe(active_diseases, "disease"),
    }


def _neighbours(embedding_map: dict[str, list[float]], pid: str, k: int = 5) -> list[dict]:
    if pid not in embedding_map:
        return []
    v = np.array(embedding_map[pid])
    out = []
    for other, emb in embedding_map.items():
        if other == pid:
            continue
        d = float(np.linalg.norm(v - np.array(emb)))
        out.append({"patient_id": other, "distance": round(d, 4)})
    out.sort(key=lambda x: x["distance"])
    return out[:k]


# --- endpoints --------------------------------------------------------
@router.get("/health")
def health():
    return {
        "status": "ok", "phase": "2", "service": "graph-digital-twin",
        "neo4j_available": neo4j_client.available,
        "gnn_loaded": gnn_service.is_loaded(),
    }


@router.post("/graph/build", response_model=GraphBuildResponse)
def build_graph(
    n_patients: int = Query(500, ge=10, le=5000),
    seed: int = Query(42),
    persist_neo4j: bool = Query(True),
    db: Session = Depends(get_db),
):
    df = _load_or_generate_cohort(db, n=n_patients, seed=seed)
    reset_cohort_graph()
    graph = get_or_build_cohort_graph(df)
    gnn_service.attach_graph(graph)

    counts = {
        "patients": len(graph.patient_id_to_node),
        "biomarkers": len(BIOMARKERS),
        "organs": len(ORGANS),
        "diseases": len(DISEASES),
        "edges": int(graph.edge_index.shape[1]),
    }
    neo4j_status = False
    if persist_neo4j:
        builder = GraphBuilder()
        info = builder.persist_to_neo4j(df)
        neo4j_status = info.get("neo4j_loaded", False)
    return GraphBuildResponse(neo4j_loaded=neo4j_status, **counts)


@router.post("/graph/train")
def train_gnn(epochs: int = Query(200, ge=10, le=2000),
              lr: float = Query(1e-3, gt=0)):
    if not gnn_service.is_loaded():
        raise HTTPException(status_code=400, detail="graph not built yet")
    res = gnn_service.train(epochs=epochs, lr=lr)
    return {"status": "trained", **res}


@router.get("/patients/{patient_id}/state", response_model=PatientGraphState)
def patient_state(patient_id: str, db: Session = Depends(get_db)):
    obj = patient_crud.get_patient(db, patient_id)
    if obj is None:
        raise HTTPException(status_code=404, detail="patient not found")
    if not gnn_service.is_loaded():
        raise HTTPException(status_code=400, detail="graph not built")
    emb_map = gnn_service.embed_patients()
    if patient_id not in emb_map:
        raise HTTPException(status_code=404, detail="patient not in cohort graph")

    patient = {
        "patient_id": obj.patient_id, "age": obj.age, "gender": obj.gender,
        "bmi": obj.bmi, "hr": obj.hr, "hrv": obj.hrv, "spo2": obj.spo2,
        "glucose": obj.glucose, "systolic_bp": obj.systolic_bp,
        "diastolic_bp": obj.diastolic_bp,
    }
    sub = _patient_subgraph(patient, gnn_service.bundle.graph)
    score = float(obj.risk_score) if obj.risk_score is not None else \
            compute_risk_score(__import__("pandas").Series(patient))
    label = obj.risk_label or _risk_label(score)

    return PatientGraphState(
        patient_id=patient_id,
        risk_score=score,
        risk_label=label,
        embedding=emb_map[patient_id],
        abnormal_biomarkers=sub["abnormal_biomarkers"],
        affected_organs=sub["affected_organs"],
        active_diseases=sub["active_diseases"],
        neighbor_similar=_neighbours(emb_map, patient_id),
    )


@router.get("/patients/{patient_id}/predict/{biomarker}")
def predict_biomarker(patient_id: str, biomarker: str, db: Session = Depends(get_db)):
    if not gnn_service.is_loaded():
        raise HTTPException(status_code=400, detail="graph not built")
    obj = patient_crud.get_patient(db, patient_id)
    if obj is None:
        raise HTTPException(status_code=404, detail="patient not found")
    res = gnn_service.predict_missing_biomarker(patient_id, biomarker)
    if "error" in res:
        raise HTTPException(status_code=400, detail=res["error"])
    return res


@router.get("/graph/stats")
def graph_stats():
    if not gnn_service.is_loaded():
        return {"loaded": False}
    g = gnn_service.bundle.graph
    n_patients = sum(1 for k in g.node_kind if k == "patient")
    return {
        "loaded": True,
        "n_nodes": len(g.node_ids),
        "n_patients": n_patients,
        "n_edges": int(g.edge_index.shape[1]),
        "node_kinds": {
            "patient": n_patients,
            "biomarker": sum(1 for k in g.node_kind if k == "biomarker"),
            "organ":     sum(1 for k in g.node_kind if k == "organ"),
            "disease":   sum(1 for k in g.node_kind if k == "disease"),
        },
        "embedding_dim": gnn_service.bundle.out_dim,
        "neo4j_available": neo4j_client.available,
    }


@router.get("/ontology")
def ontology():
    return {
        "biomarkers": [b.__dict__ for b in BIOMARKERS],
        "organs":     [o.__dict__ for o in ORGANS],
        "diseases":   [d.__dict__ for d in DISEASES],
        "edges":      [e.__dict__ for e in EDGES],
    }
