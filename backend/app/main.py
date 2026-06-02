"""
Bio-Digital Twin  —  Phases 1-8 API
"""
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, Body
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Bio-Digital Twin API", version="0.8.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Lazy-loaded Phase 3 simulator (torch import is expensive)
_simulator = None
def get_simulator():
    global _simulator
    if _simulator is None:
        from app.dynamics.disease_model import DiseaseSimulator
        _simulator = DiseaseSimulator()
    return _simulator

@app.get("/")
def root():
    return {
        "message": "Bio-Digital Twin API",
        "version": "0.3.0",
        "phases": {
            "phase_1": "POST /generate-patients",
            "phase_2": {
                "build_graph": "POST /phase2/build-graph",
                "train_gnn": "POST /phase2/train-gnn",
                "similar": "GET /phase2/similar-patients/{patient_id}",
                "subgraph": "GET /phase2/patient-subgraph/{patient_id}",
                "embedding": "GET /phase2/embedding/{patient_id}",
                "graph_stats": "GET /phase2/graph-stats",
                "cluster_summary": "GET /phase2/cluster-summary",
            },
            "phase_3": {
                "diseases": "GET /phase3/diseases",
                "interventions": "GET /phase3/interventions",
                "attractors": "GET /phase3/attractors",
                "disease_attractor": "GET /phase3/diseases/{name}/attractor",
                "simulate": "POST /phase3/simulate",
                "patient_simulate": "GET /phase3/patients/{patient_id}/simulate",
                "counterfactual": "POST /phase3/counterfactual",
            },
            "phase_4": {
                "causal_graph": "GET /phase4/causal-graph",
                "build_scm": "POST /phase4/build-scm",
                "ate": "POST /phase4/ate",
                "cate": "POST /phase4/cate",
                "refute": "POST /phase4/refute",
                "patient_counterfactual": "POST /phase4/patient-counterfactual",
                "treatments": "GET /phase4/treatments",
                "outcomes": "GET /phase4/outcomes",
            },
            "phase_5": {
                "chat": "POST /phase5/chat",
                "tools": "GET /phase5/tools",
                "reset": "POST /phase5/reset",
                "history": "GET /phase5/history",
            },
            "phase_8": {
                "genes": "GET /phase8/genes",
                "registry": "GET /phase8/registry",
                "patient_pgx": "GET /phase8/patients/{patient_id}/pgx",
                "pgx_check": "POST /phase8/patients/pgx-check",
            },
            "phase_9": {
                "rules": "GET /phase9/rules",
                "graph": "GET /phase9/graph",
                "check": "POST /phase9/check",
                "pair": "POST /phase9/pair",
            },
            "phase_10": {
                "drugs": "GET /phase10/drugs",
                "drug": "GET /phase10/drugs/{name}",
                "pk": "POST /phase10/pk/simulate",
                "pd": "POST /phase10/pd/simulate",
                "population": "POST /phase10/population",
            },
            "phase_11": {
                "patient_counterfactual": "POST /phase11/patient-counterfactual",
                "ate": "POST /phase11/ate",
            },
            "phase_12": {
                "trials_search": "GET /phase12/trials/search",
                "trial_detail": "GET /phase12/trials/{nct_id}",
            },
            "phase_13": {
                "drug_regulatory": "GET /phase13/drugs/{drug_name}/regulatory",
                "black_box": "GET /phase13/drugs/{drug_name}/black-box",
                "faers": "GET /phase13/drugs/{drug_name}/faers",
                "approval": "GET /phase13/drugs/{drug_name}/approval",
                "rxnorm": "GET /phase13/rxnorm/normalize",
                "orange_book_snapshot": "GET /phase13/registry/snapshot",
            },
            "phase_14": {
                "validate": "POST /phase14/validate",
                "validate_batch": "POST /phase14/validate-batch",
                "rdkit_status": "GET /phase14/rdkit-version",
            },
            "phase_15": {
                "registry_list": "GET /phase15/registry/diseases",
                "registry_get": "GET /phase15/registry/diseases/{key}",
                "registry_create": "POST /phase15/registry/diseases",
                "registry_update": "PUT /phase15/registry/diseases/{key}",
                "registry_delete": "DELETE /phase15/registry/diseases/{key}",
                "registry_summary": "GET /phase15/registry/summary",
            },
        },
    }

@app.get("/health")
def health():
    return {"status": "healthy", "phase": "1+2+3+4+5+8+9+10+11+12+13+14+15"}

@app.post("/generate-patients")
def generate_patients(n: int = Query(500, ge=10, le=5000)):
    np.random.seed(42)
    data = {"patient_id": [f"P{str(i).zfill(6)}" for i in range(1, n+1)], "age": np.random.normal(45,15,n).astype(int).clip(18,85), "gender": np.random.choice(["Male","Female"],n,p=[0.52,0.48]), "bmi": np.random.normal(26.5,5.5,n).clip(15,45), "hr": np.random.normal(72,8,n).astype(int).clip(50,110), "hrv": np.random.normal(45,18,n).astype(int).clip(10,120), "spo2": np.random.normal(96.5,1.8,n).clip(88,100), "glucose": np.random.normal(105,25,n).astype(int).clip(70,250), "systolic_bp": np.random.normal(125,15,n).astype(int).clip(90,180), "diastolic_bp": np.random.normal(78,10,n).astype(int).clip(60,110), "created_at": datetime.now().isoformat()}
    df = pd.DataFrame(data)
    df.to_csv("data/synthetic_patients.csv", index=False)
    return {"status": "success", "generated": n, "sample": df.head(3).to_dict(orient="records"), "file_saved": "data/synthetic_patients.csv"}

@app.post("/phase2/build-graph")
def build_graph(threshold: float = Query(0.80, ge=0.5, le=0.99), max_neighbors: int = Query(15, ge=1, le=50), use_neo4j: bool = Query(False)):
    from .graph.builder import load_patients, compute_risk_flags, compute_similarity_matrix, build_edge_list, neo4j_available
    import json
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
    stats = {"node_count": n_nodes, "edge_count": n_edges, "avg_degree": round((2*n_edges)/max(n_nodes,1),2), "density": round((n_edges*2)/max(n_nodes*(n_nodes-1),1),6), "connected_components": -1}
    summary = {"threshold": threshold, "max_neighbors": max_neighbors, **stats, "edge_sample": [{"src": e[0], "dst": e[1], "similarity": round(e[2],4)} for e in edges[:5]]}
    with open("data/graph_summary.json","w") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(edges, columns=["src","dst","similarity"]).to_csv("data/patient_edges.csv", index=False)
    result = {"status": "success", "nodes_created": n_nodes, "edges_created": n_edges, "similarity_threshold": threshold, "stats": stats, "edges_file": "data/patient_edges.csv"}
    if use_neo4j:
        if not neo4j_available():
            result["neo4j_warning"] = "Neo4j unreachable — graph written to CSV only."
        else:
            from .graph.builder import get_driver, write_patients_to_neo4j, write_edges_to_neo4j
            driver = get_driver()
            try:
                write_patients_to_neo4j(driver, df)
                write_edges_to_neo4j(driver, edges, df)
                result["neo4j_status"] = "written"
            finally:
                driver.close()
    return result

@app.post("/phase2/train-gnn")
def train_gnn_endpoint(background_tasks: BackgroundTasks, epochs: int = Query(100, ge=10, le=500), threshold: float = Query(0.80, ge=0.5, le=0.99), encoder_type: str = Query("gcn", pattern="^(gcn|gat)$"), lr: float = Query(1e-3, gt=0), async_train: bool = Query(False)):
    try:
        from .graph.trainer import train_gnn
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"torch-geometric not installed: {e}")
    if async_train:
        background_tasks.add_task(lambda: train_gnn(epochs=epochs, threshold=threshold, encoder_type=encoder_type, lr=lr))
        return {"status": "training_started", "epochs": epochs}
    try:
        result = train_gnn(epochs=epochs, threshold=threshold, encoder_type=encoder_type, lr=lr)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ImportError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "success", **result}

@app.get("/phase2/similar-patients/{patient_id}")
def similar_patients(patient_id: str, k: int = Query(10, ge=1, le=50), source: str = Query("graph", pattern="^(graph|embedding)$")):
    if source == "embedding":
        try:
            from .graph.trainer import get_top_k_similar_by_embedding
            results = get_top_k_similar_by_embedding(patient_id, k=k)
        except (FileNotFoundError, KeyError) as e:
            raise HTTPException(status_code=404, detail=str(e))
        return {"patient_id": patient_id, "source": "gnn_embedding", "k": k, "similar_patients": [{"patient_id": pid, "similarity": sim} for pid, sim in results]}
    edge_path = "data/patient_edges.csv"
    if not os.path.exists(edge_path):
        raise HTTPException(status_code=404, detail="Edge list not found. Call POST /phase2/build-graph first.")
    edges_df = pd.read_csv(edge_path)
    mask = (edges_df["src"] == patient_id) | (edges_df["dst"] == patient_id)
    patient_edges = edges_df[mask].copy()
    if patient_edges.empty:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found or has no edges.")
    patient_edges["neighbor"] = np.where(patient_edges["src"] == patient_id, patient_edges["dst"], patient_edges["src"])
    top_k = patient_edges[["neighbor","similarity"]].sort_values("similarity", ascending=False).head(k)
    patients_df = pd.read_csv("data/synthetic_patients.csv")
    merged = top_k.merge(patients_df[["patient_id","age","bmi","glucose","hr","systolic_bp","spo2"]], left_on="neighbor", right_on="patient_id", how="left")
    return {"patient_id": patient_id, "source": "graph_edges", "k": len(merged), "similar_patients": merged.drop(columns=["patient_id"]).to_dict(orient="records")}

@app.get("/phase2/patient-subgraph/{patient_id}")
def patient_subgraph(patient_id: str, depth: int = Query(1, ge=1, le=2), max_nodes: int = Query(30, ge=5, le=100)):
    edge_path = "data/patient_edges.csv"
    if not os.path.exists(edge_path):
        raise HTTPException(status_code=404, detail="Edge list not found. Call POST /phase2/build-graph first.")
    edges_df = pd.read_csv(edge_path)
    patients_df = pd.read_csv("data/synthetic_patients.csv")
    patient_lookup = patients_df.set_index("patient_id").to_dict(orient="index")
    if patient_id not in patient_lookup:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found.")
    def get_neighbors(pid):
        mask = (edges_df["src"] == pid) | (edges_df["dst"] == pid)
        return [{"id": row["dst"] if row["src"]==pid else row["src"], "similarity": row["similarity"]} for _,row in edges_df[mask].iterrows()]
    visited = {patient_id}
    frontier = [patient_id]
    subgraph_edges = []
    for _ in range(depth):
        next_frontier = []
        for pid in frontier:
            for nbr in get_neighbors(pid):
                nid = nbr["id"]
                subgraph_edges.append({"source": pid, "target": nid, "similarity": nbr["similarity"]})
                if nid not in visited:
                    visited.add(nid)
                    next_frontier.append(nid)
                if len(visited) >= max_nodes:
                    break
            if len(visited) >= max_nodes:
                break
        frontier = next_frontier[:max_nodes]
    seen_pairs = set()
    unique_edges = []
    for e in subgraph_edges:
        pair = tuple(sorted([e["source"], e["target"]]))
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            unique_edges.append(e)
    nodes = [{"id": pid, "is_center": pid==patient_id, "age": patient_lookup.get(pid,{}).get("age"), "bmi": round(float(patient_lookup.get(pid,{}).get("bmi",0)),1), "glucose": patient_lookup.get(pid,{}).get("glucose"), "hr": patient_lookup.get(pid,{}).get("hr"), "spo2": round(float(patient_lookup.get(pid,{}).get("spo2",0)),1)} for pid in visited]
    return {"center_patient": patient_id, "depth": depth, "nodes": nodes, "edges": unique_edges, "node_count": len(nodes), "edge_count": len(unique_edges)}

@app.get("/phase2/embedding/{patient_id}")
def get_patient_embedding(patient_id: str):
    try:
        from .graph.trainer import get_embedding
        embedding = get_embedding(patient_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"GNN deps not installed: {e}")
    return {"patient_id": patient_id, "embedding_dim": len(embedding), "embedding": embedding}

@app.get("/phase2/graph-stats")
def graph_stats():
    import json
    if not os.path.exists("data/graph_summary.json"):
        raise HTTPException(status_code=404, detail="Graph not built yet. Call POST /phase2/build-graph first.")
    with open("data/graph_summary.json") as f:
        return json.load(f)

@app.get("/phase2/cluster-summary")
def cluster_summary(n_clusters: int = Query(5, ge=2, le=20)):
    try:
        from .graph.trainer import load_embeddings
        from sklearn.cluster import KMeans
        embeddings, id_index = load_embeddings()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ImportError as e:
        raise HTTPException(status_code=500, detail=str(e))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)
    index_to_id = {v: k for k, v in id_index.items()}
    patients_df = pd.read_csv("data/synthetic_patients.csv").set_index("patient_id")
    clusters = []
    for c in range(n_clusters):
        idxs = np.where(labels == c)[0]
        pids = [index_to_id[i] for i in idxs if i in index_to_id]
        sub = patients_df.loc[[p for p in pids if p in patients_df.index]]
        if sub.empty:
            continue
        clusters.append({"cluster_id": c, "patient_count": len(sub), "avg_age": round(sub["age"].mean(),1), "avg_bmi": round(sub["bmi"].mean(),2), "avg_glucose": round(sub["glucose"].mean(),1), "avg_hr": round(sub["hr"].mean(),1), "avg_spo2": round(sub["spo2"].mean(),2), "pct_female": round((sub["gender"]=="Female").mean()*100,1), "sample_ids": pids[:5]})
    return {"n_clusters": n_clusters, "total_patients": len(embeddings), "clusters": clusters}


# =============================================================================
# Phase 3 — Disease Dynamics
# =============================================================================
from pydantic import BaseModel, Field

class SimulateRequest(BaseModel):
    initial_state: dict
    disease: str = Field(..., description="One of: t2d, hypertension, cvd, copd")
    horizon_days: int = Field(180, ge=1, le=1825)
    dt_hours: float = Field(6.0, gt=0, le=48)
    intervention: Optional[dict] = Field(default=None, description="{biomarker: delta_per_day}")
    intervention_name: Optional[str] = Field(default=None, description="Named intervention from /phase3/interventions")
    rng_seed: int = 0
    sample_every_hours: Optional[float] = Field(default=None, description="Downsample trajectory; default = dt_hours")

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
            raise HTTPException(status_code=404,
                                detail=f"unknown intervention: {body.intervention_name}")
        return INTERVENTIONS[body.intervention_name]
    return body.intervention or {}


def _format_simulation(res: dict, body, intervention: dict) -> dict:
    sample_every = body.sample_every_hours or body.dt_hours
    step = max(1, int(sample_every / body.dt_hours))
    times_d = [round(t / 24.0, 2) for t in res["times_h"][::step]]

    biomarkers = []
    from app.graph.ontology import BIOMARKERS
    by_id = {b.id: b for b in BIOMARKERS}
    for name, series in res["series"].items():
        b = by_id.get(name)
        biomarkers.append({
            "name": name,
            "label": b.name if b else name,
            "unit": b.unit if b else "",
            "healthy_lo": b.healthy_lo if b else None,
            "healthy_hi": b.healthy_hi if b else None,
            "baseline": res["final_state"].get(name, series[0] if series else None),
            "trajectory": [
                {"day": d, "value": round(float(v), 3)}
                for d, v in zip(times_d, series[::step])
            ],
        })

    return {
        "disease": body.disease,
        "horizon_days": body.horizon_days,
        "steps": len(res["times_h"]),
        "sample_step_days": round(sample_every / 24.0, 3),
        "disease_state": res["disease_state"],
        "final_risk": round(res["final_risk"], 4),
        "initial_risk": round(res["risks"][0], 4) if res["risks"] else None,
        "risk_evolution": [
            {"day": d, "risk": round(float(r), 4)}
            for d, r in zip(times_d, res["risks"][::step])
        ],
        "biomarkers": biomarkers,
        "spike_view": {
            "dominant_biomarker": res["lif_dominant_biomarker"],
            "spike_count": res["spike_count"],
            "spike_rate_hz": res["spike_rate_hz"],
        },
        "intervention_applied": intervention or None,
    }


@app.get("/phase3/diseases")
def list_diseases():
    from app.dynamics.disease_model import DISEASE_FORCINGS, bifurcation_summary
    return {
        "diseases": [
            {
                "id": k,
                "name": v.name,
                "bifurcation": bifurcation_summary(k),
            }
            for k, v in DISEASE_FORCINGS.items()
        ],
    }


@app.get("/phase3/interventions")
def list_interventions():
    from app.dynamics.disease_model import INTERVENTIONS
    return {"interventions": [{"name": k, "daily_delta": v} for k, v in INTERVENTIONS.items()]}


@app.get("/phase3/attractors")
def list_attractors():
    from app.dynamics.disease_model import ATTRACTORS
    return {
        "attractors": [
            {"name": a.name, "description": a.description,
             "risk_range": [a.risk_lo, a.risk_hi]}
            for a in ATTRACTORS
        ],
    }


@app.get("/phase3/diseases/{disease}/attractor")
def disease_attractor(disease: str):
    from app.dynamics.disease_model import DISEASE_FORCINGS, bifurcation_summary
    if disease not in DISEASE_FORCINGS:
        raise HTTPException(status_code=404, detail=f"unknown disease: {disease}")
    return {
        "disease": disease,
        "name": DISEASE_FORCINGS[disease].name,
        "bifurcation": bifurcation_summary(disease),
    }


@app.post("/phase3/simulate")
def simulate(req: SimulateRequest):
    try:
        intervention = _resolve_intervention(req)
        sim = get_simulator()
        res = sim.simulate(
            initial_state=req.initial_state,
            disease=req.disease,
            horizon_days=req.horizon_days,
            intervention=intervention,
            rng_seed=req.rng_seed,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _format_simulation(res, req, intervention)


@app.get("/phase3/patients/{patient_id}/simulate")
def simulate_patient(
    patient_id: str,
    disease: str = Query("t2d"),
    horizon_days: int = Query(365, ge=1, le=1825),
    intervention_name: Optional[str] = Query(None),
    intervention: Optional[str] = Query(None, description="JSON dict, e.g. {\"glucose\":-8}"),
    dt_hours: float = Query(6.0, gt=0, le=48),
    rng_seed: int = Query(0),
):
    csv_path = "data/synthetic_patients.csv"
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="no synthetic patients — call POST /generate-patients first")
    df = pd.read_csv(csv_path)
    row = df[df["patient_id"] == patient_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"patient '{patient_id}' not found")
    row = row.iloc[0]

    interv_dict = None
    if intervention:
        import json
        try:
            interv_dict = json.loads(intervention)
        except Exception:
            raise HTTPException(status_code=400, detail="intervention must be a JSON object")

    initial = {
        "hr": float(row["hr"]), "hrv": float(row["hrv"]),
        "spo2": float(row["spo2"]), "glucose": float(row["glucose"]),
        "systolic_bp": float(row["systolic_bp"]),
        "diastolic_bp": float(row["diastolic_bp"]),
        "bmi": float(row["bmi"]),
    }

    req = SimulateRequest(
        initial_state=initial, disease=disease,
        horizon_days=horizon_days, dt_hours=dt_hours,
        intervention=interv_dict, intervention_name=intervention_name,
        rng_seed=rng_seed,
    )
    intervention = _resolve_intervention(req)
    try:
        sim = get_simulator()
        res = sim.simulate(
            initial_state=initial, disease=disease,
            horizon_days=horizon_days, intervention=intervention,
            rng_seed=rng_seed,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "patient_id": patient_id,
        "patient_baseline": initial,
        **_format_simulation(res, req, intervention),
    }


@app.post("/phase3/counterfactual")
def counterfactual(req: CounterfactualRequest):
    """Run two parallel simulations: control vs intervention.
    Returns the delta in final risk and the full trajectories for comparison."""
    try:
        intervention = _resolve_intervention(req)
        sim = get_simulator()
        ctrl = sim.simulate(
            initial_state=req.initial_state, disease=req.disease,
            horizon_days=req.horizon_days, intervention={},
            rng_seed=req.rng_seed,
        )
        tx = sim.simulate(
            initial_state=req.initial_state, disease=req.disease,
            horizon_days=req.horizon_days, intervention=intervention,
            rng_seed=req.rng_seed,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "disease": req.disease,
        "horizon_days": req.horizon_days,
        "intervention_applied": intervention or None,
        "control": {
            "final_risk": round(ctrl["final_risk"], 4),
            "disease_state": ctrl["disease_state"],
            "final_state": ctrl["final_state"],
        },
        "treated": {
            "final_risk": round(tx["final_risk"], 4),
            "disease_state": tx["disease_state"],
            "final_state": tx["final_state"],
        },
        "counterfactual_effect": {
            "absolute_risk_reduction": round(ctrl["final_risk"] - tx["final_risk"], 4),
            "relative_risk_reduction": round(
                (ctrl["final_risk"] - tx["final_risk"]) / max(ctrl["final_risk"], 1e-6), 4),
            "state_changed": ctrl["disease_state"] != tx["disease_state"],
            "from_state": ctrl["disease_state"],
            "to_state":   tx["disease_state"],
        },
    }


# =============================================================================
# Phase 4 — Counterfactual Simulation (Causal AI)
# =============================================================================
def _cohort_df():
    csv = "data/synthetic_patients.csv"
    if not os.path.exists(csv):
        raise HTTPException(status_code=404, detail="no cohort — call POST /generate-patients first")
    return pd.read_csv(csv)


class ATERequest(BaseModel):
    treatment: str
    outcome: str
    common_causes: Optional[list[str]] = None

class CATERequest(BaseModel):
    treatment: str
    outcome: str
    effect_modifiers: list[str] = ["bmi"]
    common_causes: Optional[list[str]] = None

class RefuteRequest(BaseModel):
    treatment: str
    outcome: str
    common_causes: Optional[list[str]] = None
    method: str = Field(default="random_common_cause",
                       pattern="^(random_common_cause|placebo)$")

class PatientCounterfactualRequest(BaseModel):
    patient_id: str
    treatment: str
    outcome: str
    value: float = 1.0


@app.get("/phase4/causal-graph")
def causal_graph():
    from app.causal.scm import build_causal_dag
    g = build_causal_dag()
    return {
        "n_nodes": g.number_of_nodes(),
        "n_edges": g.number_of_edges(),
        "nodes": [{"id": n, **g.nodes[n]} for n in g.nodes],
        "edges": [{"src": u, "dst": v, **g.edges[u, v]} for u, v in g.edges],
    }


@app.post("/phase4/build-scm")
def build_scm(force: bool = Query(False)):
    from app.causal.scm import fit_cohort_scm, reset_scm
    if force:
        reset_scm()
    df = _cohort_df()
    scm = fit_cohort_scm(df, force=force)
    return {
        "status": "ok",
        "n_nodes_fitted": len(scm.coefficients),
        "fit_metrics": scm.fit_metrics,
    }


@app.post("/phase4/ate")
def phase4_ate(req: ATERequest):
    from app.causal.scm import ate_estimate
    df = _cohort_df()
    res = ate_estimate(df, treatment=req.treatment, outcome=req.outcome,
                       common_causes=req.common_causes)
    if "error" in res:
        raise HTTPException(status_code=400, detail=res["error"])
    return res


@app.post("/phase4/cate")
def phase4_cate(req: CATERequest):
    from app.causal.scm import cate_estimate
    df = _cohort_df()
    res = cate_estimate(df, treatment=req.treatment, outcome=req.outcome,
                        effect_modifiers=req.effect_modifiers,
                        common_causes=req.common_causes)
    if "error" in res:
        raise HTTPException(status_code=400, detail=res["error"])
    return res


@app.post("/phase4/refute")
def phase4_refute(req: RefuteRequest):
    from app.causal.scm import refute_ate
    df = _cohort_df()
    res = refute_ate(df, treatment=req.treatment, outcome=req.outcome,
                     common_causes=req.common_causes, method=req.method)
    if "error" in res:
        raise HTTPException(status_code=400, detail=res["error"])
    return res


@app.post("/phase4/patient-counterfactual")
def phase4_patient_counterfactual(req: PatientCounterfactualRequest):
    from app.causal.scm import fit_cohort_scm, patient_counterfactual
    df = _cohort_df()
    row = df[df["patient_id"] == req.patient_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"patient '{req.patient_id}' not found")
    r = row.iloc[0]
    from app.graph.ontology import BIOMARKERS
    observed = {b.id: float(r[b.id]) for b in BIOMARKERS}
    observed["bmi"] = float(r["bmi"])
    observed["age"] = float(r["age"])
    scm = fit_cohort_scm(df)
    res = patient_counterfactual(scm, observed=observed,
                                 treatment=req.treatment,
                                 value=float(req.value),
                                 outcome=req.outcome)
    if "error" in res:
        raise HTTPException(status_code=400, detail=res["error"])
    res["patient_id"] = req.patient_id
    return res


@app.get("/phase4/treatments")
def phase4_treatments():
    from app.causal.scm import TREATMENT_TARGETS
    return {"treatments": [{"name": k, "target_diseases": v}
                           for k, v in TREATMENT_TARGETS.items()]}


@app.get("/phase4/outcomes")
def phase4_outcomes():
    from app.causal.scm import OUTCOMES_FOR_DISEASE
    return {"outcomes": [{"disease": k, "biomarkers": v}
                          for k, v in OUTCOMES_FOR_DISEASE.items()]}


# =============================================================================
# Phase 5 — LLM Agent
# =============================================================================
class ChatRequest(BaseModel):
    session_id: str = "default"
    message: str
    use_mock: Optional[bool] = None   # auto-detect if None


@app.get("/phase5/tools")
def phase5_tools():
    from app.agent.llm import list_tools
    return {"tools": list_tools()}


@app.post("/phase5/chat")
def phase5_chat(req: ChatRequest):
    from app.agent.llm import chat as agent_chat
    return agent_chat(req.session_id, req.message, use_mock=req.use_mock)


@app.post("/phase5/reset")
def phase5_reset(session_id: str = Query("default")):
    from app.agent.llm import reset_memory
    reset_memory(session_id)
    return {"status": "reset", "session_id": session_id}


@app.get("/phase5/history")
def phase5_history(session_id: str = Query("default")):
    from app.agent.llm import get_memory
    mem = get_memory(session_id)
    return {"session_id": session_id,
            "turns": len(mem.messages) // 2,
            "messages": mem.as_list()}


# =============================================================================
# Phase 8 — Pharmacogenomics
# =============================================================================
from app.pgx import pgx_router  # noqa: E402
app.include_router(pgx_router)


# =============================================================================
# Phase 9 — Drug-Drug Interactions
# =============================================================================
from app.ddi import ddi_router  # noqa: E402
app.include_router(ddi_router)


# =============================================================================
# Phase 10 — PK/PD
# =============================================================================
from app.pkpd import pkpd_router  # noqa: E402
app.include_router(pkpd_router)


# =============================================================================
# Phase 11 — Uncertainty Quantification
# =============================================================================
from app.uq import uq_router  # noqa: E402
app.include_router(uq_router)


# =============================================================================
# Phase 12 — Clinical Trials (ClinicalTrials.gov v2)
# =============================================================================
from app.trials import trials_router  # noqa: E402
app.include_router(trials_router)


# =============================================================================
# Phase 13 — Regulatory (FDA, FAERS, RxNorm)
# =============================================================================
from app.regulatory import regulatory_router  # noqa: E402
app.include_router(regulatory_router)


# =============================================================================
# Phase 14 — Wet-Lab Validation (PAINS/Brenk/SAS/IC50/tox)
# =============================================================================
from app.wetlab import wetlab_router  # noqa: E402
app.include_router(wetlab_router)


# =============================================================================
# Phase 15 — Disease Registry (Postgres-backed)
# =============================================================================
from app.registry import registry_router  # noqa: E402
app.include_router(registry_router)

