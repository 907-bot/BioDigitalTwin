"""
Patient Service — patient CRUD, graph/GNN, causal inference.

Combines: Phase 1 (patient generation), Phase 2 (graph/GNN), Phase 4 (causal SCM)
"""
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Patient Service", version="0.1.0")
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"])

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)


@app.get("/health")
def health():
    return {"status": "healthy", "service": "patient", "version": "0.1.0"}


# ── Phase 1: Patient Generation ──
@app.post("/patients/generate")
def generate_patients(n: int = Query(500, ge=10, le=5000)):
    np.random.seed(42)
    data = {
        "patient_id": [f"P{str(i).zfill(6)}" for i in range(1, n + 1)],
        "age": np.random.normal(45, 15, n).astype(int).clip(18, 85),
        "gender": np.random.choice(["Male", "Female"], n, p=[0.52, 0.48]),
        "bmi": np.random.normal(26.5, 5.5, n).clip(15, 45),
        "hr": np.random.normal(72, 8, n).astype(int).clip(50, 110),
        "hrv": np.random.normal(45, 18, n).astype(int).clip(10, 120),
        "spo2": np.random.normal(96.5, 1.8, n).clip(88, 100),
        "glucose": np.random.normal(105, 25, n).astype(int).clip(70, 250),
        "systolic_bp": np.random.normal(125, 15, n).astype(int).clip(90, 180),
        "diastolic_bp": np.random.normal(78, 10, n).astype(int).clip(60, 110),
        "created_at": datetime.now().isoformat(),
    }
    df = pd.DataFrame(data)
    df.to_csv("data/synthetic_patients.csv", index=False)
    return {"status": "success", "generated": n, "sample": df.head(3).to_dict(orient="records")}


@app.get("/patients")
def list_patients():
    if not os.path.exists("data/synthetic_patients.csv"):
        raise HTTPException(status_code=404, detail="No patients — call POST /patients/generate first")
    df = pd.read_csv("data/synthetic_patients.csv")
    return {"patients": df.head(100).to_dict(orient="records"), "total": len(df)}


@app.get("/patients/{patient_id}")
def get_patient(patient_id: str):
    if not os.path.exists("data/synthetic_patients.csv"):
        raise HTTPException(status_code=404, detail="No patients")
    df = pd.read_csv("data/synthetic_patients.csv")
    row = df[df["patient_id"] == patient_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found")
    return row.iloc[0].to_dict()


# ── Phase 2: Graph/GNN ──
@app.post("/graph/build")
def build_graph_endpoint(threshold: float = Query(0.80, ge=0.5, le=0.99),
                          max_neighbors: int = Query(15, ge=1, le=50),
                          use_neo4j: bool = Query(False)):
    from app.graph.builder import load_patients, compute_risk_flags, compute_similarity_matrix, build_edge_list
    try:
        df = load_patients()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    df = compute_risk_flags(df)
    sim_matrix = compute_similarity_matrix(df)
    patient_ids = df["patient_id"].tolist()
    edges = build_edge_list(sim_matrix, patient_ids, threshold, max_neighbors)
    n_nodes = len(patient_ids)
    n_edges = len(edges)
    stats = {
        "node_count": n_nodes, "edge_count": n_edges,
        "avg_degree": round((2 * n_edges) / max(n_nodes, 1), 2),
        "density": round((n_edges * 2) / max(n_nodes * (n_nodes - 1), 1), 6),
    }
    summary = {"threshold": threshold, "max_neighbors": max_neighbors, **stats}
    with open("data/graph_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(edges, columns=["src", "dst", "similarity"]).to_csv("data/patient_edges.csv", index=False)
    result = {"status": "success", "nodes_created": n_nodes, "edges_created": n_edges, "stats": stats}
    if use_neo4j:
        from app.graph.builder import get_driver, write_patients_to_neo4j, write_edges_to_neo4j
        driver = get_driver()
        try:
            write_patients_to_neo4j(driver, df)
            write_edges_to_neo4j(driver, edges, df)
            result["neo4j_status"] = "written"
        finally:
            driver.close()
    return result


@app.post("/graph/train")
def train_gnn_endpoint(background_tasks: BackgroundTasks,
                        epochs: int = Query(100, ge=10, le=500),
                        threshold: float = Query(0.80, ge=0.5, le=0.99),
                        async_train: bool = Query(False)):
    try:
        from app.graph.trainer import train_gnn
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"torch-geometric not installed: {e}")
    if async_train:
        background_tasks.add_task(lambda: train_gnn(epochs=epochs, threshold=threshold))
        return {"status": "training_started", "epochs": epochs}
    result = train_gnn(epochs=epochs, threshold=threshold)
    return {"status": "success", **result}


@app.get("/graph/stats")
def graph_stats():
    if not os.path.exists("data/graph_summary.json"):
        raise HTTPException(status_code=404, detail="Graph not built yet")
    with open("data/graph_summary.json") as f:
        return json.load(f)


@app.get("/graph/embedding/{patient_id}")
def get_patient_embedding(patient_id: str):
    try:
        from app.graph.trainer import get_embedding
        embedding = get_embedding(patient_id)
    except (FileNotFoundError, KeyError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"patient_id": patient_id, "embedding_dim": len(embedding), "embedding": embedding}


@app.get("/graph/similar/{patient_id}")
def similar_patients(patient_id: str, k: int = Query(10, ge=1, le=50)):
    from app.graph.trainer import load_embeddings
    try:
        embeddings, id_index = load_embeddings()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No embeddings — train GNN first")
    from sklearn.metrics.pairwise import cosine_similarity
    idx = id_index.get(patient_id)
    if idx is None:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found")
    sims = cosine_similarity([embeddings[idx]], embeddings)[0]
    top = np.argsort(sims)[::-1][1:k + 1]
    index_to_id = {v: k for k, v in id_index.items()}
    return {"patient_id": patient_id, "similar": [{"id": index_to_id[i], "similarity": round(float(sims[i]), 4)} for i in top]}


@app.get("/graph/cluster-summary")
def cluster_summary(n_clusters: int = Query(5, ge=2, le=20)):
    try:
        from app.graph.trainer import load_embeddings
        from sklearn.cluster import KMeans
        embeddings, id_index = load_embeddings()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)
    index_to_id = {v: k for k, v in id_index.items()}
    patients_df = pd.read_csv("data/synthetic_patients.csv").set_index("patient_id")
    clusters = []
    for c in range(n_clusters):
        idxs = np.where(labels == c)[0]
        pids = [index_to_id[i] for i in idxs if i in index_to_id]
        sub = patients_df.loc[[p for p in pids if p in patients_df.index]]
        clusters.append({
            "cluster_id": c, "patient_count": len(sub),
            "avg_age": round(sub["age"].mean(), 1), "avg_bmi": round(sub["bmi"].mean(), 2),
            "avg_glucose": round(sub["glucose"].mean(), 1), "sample_ids": pids[:5],
        })
    return {"n_clusters": n_clusters, "clusters": clusters}


# ── Phase 4: Causal Inference ──
@app.get("/causal/graph")
def causal_graph():
    from app.causal.scm import build_causal_dag
    g = build_causal_dag()
    return {
        "n_nodes": g.number_of_nodes(), "n_edges": g.number_of_edges(),
        "nodes": [{"id": n, **g.nodes[n]} for n in g.nodes],
        "edges": [{"src": u, "dst": v, **g.edges[u, v]} for u, v in g.edges],
    }


@app.post("/causal/scm")
def build_scm(force: bool = Query(False)):
    from app.causal.scm import fit_cohort_scm, reset_scm
    if force:
        reset_scm()
    df = _cohort_df()
    scm = fit_cohort_scm(df, force=force)
    return {"status": "ok", "n_nodes_fitted": len(scm.coefficients), "fit_metrics": scm.fit_metrics}


@app.post("/causal/ate")
def compute_ate(treatment: str, outcome: str, common_causes: Optional[str] = Query(None)):
    from app.causal.scm import ate_estimate
    df = _cohort_df()
    ccs = common_causes.split(",") if common_causes else None
    res = ate_estimate(df, treatment=treatment, outcome=outcome, common_causes=ccs)
    if "error" in res:
        raise HTTPException(status_code=400, detail=res["error"])
    return res


@app.post("/causal/counterfactual")
def patient_counterfactual(patient_id: str, treatment: str, outcome: str, value: float = 1.0):
    from app.causal.scm import fit_cohort_scm, patient_counterfactual
    df = _cohort_df()
    row = df[df["patient_id"] == patient_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found")
    r = row.iloc[0]
    from app.graph.ontology import BIOMARKERS
    observed = {b.id: float(r[b.id]) for b in BIOMARKERS}
    observed["bmi"] = float(r["bmi"])
    observed["age"] = float(r["age"])
    scm = fit_cohort_scm(df)
    res = patient_counterfactual(scm, observed=observed, treatment=treatment, value=value, outcome=outcome)
    if "error" in res:
        raise HTTPException(status_code=400, detail=res["error"])
    res["patient_id"] = patient_id
    return res


@app.get("/causal/treatments")
def list_treatments():
    from app.causal.scm import TREATMENT_TARGETS
    return {"treatments": [{"name": k, "target_diseases": v} for k, v in TREATMENT_TARGETS.items()]}


@app.get("/causal/outcomes")
def list_outcomes():
    from app.causal.scm import OUTCOMES_FOR_DISEASE
    return {"outcomes": [{"disease": k, "biomarkers": v} for k, v in OUTCOMES_FOR_DISEASE.items()]}


def _cohort_df():
    csv = "data/synthetic_patients.csv"
    if not os.path.exists(csv):
        raise HTTPException(status_code=404, detail="No cohort — call POST /patients/generate first")
    return pd.read_csv(csv)
