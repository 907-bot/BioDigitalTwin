"""
Phase 5 Router: Autonomous Biological Intelligence Platform (ABIP) API.
Exposes all 9 pillars as REST endpoints.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/personalization/v4", tags=["Phase 5 — ABIP"])


# ── Lazy imports ─────────────────────────────────────────────


_kg = None
_md = None
_hg = None
_ct = None
_agents = None
_twins: Dict[str, Any] = {}
_ts = None
_fm = None
_fl = None
_vf = None


def _lazy(attr, module, cls_name, *init_args, **init_kw):
    target = globals().get(attr)
    if target is None:
        import importlib
        mod = importlib.import_module(module)
        cls = getattr(mod, cls_name)
        target = cls(*init_args, **init_kw)
        globals()[attr] = target
    return target


# ── Schemas ──────────────────────────────────────────────────


class SearchRequest(BaseModel):
    query: str
    max_results: int = 20


class SubgraphRequest(BaseModel):
    entity: str
    depth: int = 2


class PathRequest(BaseModel):
    source: str
    target: str
    max_length: int = 4


class CausalDiscoveryRequest(BaseModel):
    data: List[List[float]]
    variable_names: List[str]
    methods: List[str] = ["correlation"]


class HypothesisRequest(BaseModel):
    n_hypotheses: int = 5
    patient_group: Optional[str] = None


class TrialSimulationRequest(BaseModel):
    trial_name: str = "Custom Trial"
    n_patients: int = 100
    duration_days: int = 180


class DebateRequest(BaseModel):
    question: str
    context: Optional[Dict[str, Any]] = None


class ObservationRequest(BaseModel):
    patient_id: str
    timestamp: float = 0.0
    variables: Dict[str, float]


class CreateAdaptiveTwinRequest(BaseModel):
    patient_id: str
    physio: Optional[List[float]] = None
    params: Optional[List[float]] = None


class TissueSimRequest(BaseModel):
    tissue_types: List[str] = ["liver"]
    duration_days: int = 30
    params: Optional[Dict[str, float]] = None


class FoundationModelRequest(BaseModel):
    patient_states: List[List[float]]
    n_steps: int = 10


class CounterfactualRequest(BaseModel):
    current_state: List[float]
    intervention_params: Dict[str, float]


class ValidationRequest(BaseModel):
    levels: Optional[List[str]] = None


# ── 1. Knowledge Graph ────────────────────────────────────────


@router.post("/knowledge-graph/search")
async def kg_search(req: SearchRequest):
    kg = _lazy('_kg', 'app.personalization.phase5.knowledge_graph_engine', 'KnowledgeGraphEngine')
    results = kg.search_nodes(req.query, top_k=req.max_results)
    return {"query": req.query, "results": [{"name": r.name, "type": str(r.node_type), "confidence": getattr(r, 'confidence', 1.0)} for r in results]}


@router.get("/knowledge-graph/entity/{entity}")
async def kg_entity(entity: str):
    kg = _lazy('_kg', 'app.personalization.phase5.knowledge_graph_engine', 'KnowledgeGraphEngine')
    node = kg.get_node(entity)
    if node is None:
        raise HTTPException(404, f"Entity '{entity}' not found")
    edges = kg.get_edges(entity)
    return {
        "name": node.name,
        "node_type": str(node.node_type),
        "edges": [
            {"target": e.target, "edge_type": str(e.edge_type), "confidence": e.confidence}
            for e in edges
        ],
    }


@router.post("/knowledge-graph/subgraph")
async def kg_subgraph(req: SubgraphRequest):
    kg = _lazy('_kg', 'app.personalization.phase5.knowledge_graph_engine', 'KnowledgeGraphEngine')
    sub = kg.get_subgraph([req.entity], depth=req.depth)
    g = sub.graph if hasattr(sub, 'graph') else sub
    nodes = list(g.nodes()) if hasattr(g, 'nodes') else []
    edges = [{"source": u, "target": v} for u, v in (g.edges() if hasattr(g, 'edges') else [])]
    return {"center": req.entity, "nodes": nodes, "edges": edges}


@router.post("/knowledge-graph/paths")
async def kg_paths(req: PathRequest):
    kg = _lazy('_kg', 'app.personalization.phase5.knowledge_graph_engine', 'KnowledgeGraphEngine')
    paths = kg.find_paths(req.source, req.target, max_depth=req.max_length)
    return {
        "source": req.source,
        "target": req.target,
        "paths": [
            {"path": [{"source": e.source, "target": e.target, "type": str(e.edge_type)} for e in p], "length": len(p)}
            for p in paths
        ],
    }


@router.post("/knowledge-graph/mine-literature")
async def kg_mine_literature(query: str = Query("diabetes insulin resistance")):
    kg = _lazy('_kg', 'app.personalization.phase5.knowledge_graph_engine', 'KnowledgeGraphEngine')
    n_added = kg.ingest_from_literature(query)
    return {"mined": int(n_added) if not isinstance(n_added, (list, tuple)) else len(n_added), "query": query}


@router.post("/knowledge-graph/mine-trials")
async def kg_mine_trials(condition: str = Query("type 2 diabetes")):
    kg = _lazy('_kg', 'app.personalization.phase5.knowledge_graph_engine', 'KnowledgeGraphEngine')
    n_added = kg.ingest_clinical_trials(condition)
    return {"mined": int(n_added) if not isinstance(n_added, (list, tuple)) else len(n_added), "condition": condition}


# ── 2. Mechanism Discovery ────────────────────────────────────


@router.post("/mechanisms/discover")
async def discover_mechanisms(req: CausalDiscoveryRequest):
    md = _lazy('_md', 'app.personalization.phase5.mechanism_discovery', 'MechanismDiscoveryEngine')
    results = md.discover_from_cross_sectional(np.array(req.data), req.variable_names)
    return {
        "mechanisms": [
            {
                "source": getattr(r, 'source', ''),
                "target": getattr(r, 'target', ''),
                "method": getattr(r, 'method', 'correlation'),
                "strength": getattr(r, 'strength', 0.0),
            }
            for r in (results or [])
        ],
    }


@router.get("/mechanisms/graph")
async def get_mechanism_graph():
    md = _lazy('_md', 'app.personalization.phase5.mechanism_discovery', 'MechanismDiscoveryEngine')
    mg = md.mechanism_graph
    return {
        "nodes": mg.variables if hasattr(mg, 'variables') else [],
        "mechanisms": [
            {
                "source": m.get("source"),
                "target": m.get("target"),
                "strength": m.get("strength"),
                "method": m.get("method", "unknown"),
            }
            for m in (mg.mechanisms if hasattr(mg, 'mechanisms') else [])
        ],
    }


# ── 3. Hypothesis Generator ───────────────────────────────────


@router.post("/hypotheses/generate")
async def generate_hypotheses(req: HypothesisRequest):
    hg = _lazy('_hg', 'app.personalization.phase5.hypothesis_generator', 'HypothesisGenerator')
    import pandas as pd
    data = np.random.randn(20, 5)
    df = pd.DataFrame(data, columns=["glucose", "insulin", "bmi", "crp", "sbp"])
    hypotheses = hg.from_novel_association(df, target_variable="glucose")
    top = hypotheses[:req.n_hypotheses]
    return {
        "hypotheses": [
            {
                "title": h.title,
                "hypothesis_type": h.hypothesis_type if hasattr(h, 'hypothesis_type') else "association",
                "mechanism": h.mechanism if hasattr(h, 'mechanism') else "",
                "confidence": h.confidence if hasattr(h, 'confidence') else 0.0,
                "impact_score": h.impact_score if hasattr(h, 'impact_score') else 0.0,
            }
            for h in top
        ],
    }


@router.get("/hypotheses")
async def list_hypotheses():
    hg = _lazy('_hg', 'app.personalization.phase5.hypothesis_generator', 'HypothesisGenerator')
    all_h = hg.get_all_hypotheses()
    return {
        "hypotheses": [
            {
                "title": h.title if hasattr(h, 'title') else str(h),
                "hypothesis_type": h.hypothesis_type if hasattr(h, 'hypothesis_type') else "unknown",
                "confidence": h.confidence if hasattr(h, 'confidence') else 0.0,
            }
            for h in all_h
        ]
    }


# ── 4. Clinical Trial Simulator ───────────────────────────────


@router.post("/trials/simulate")
async def simulate_trial(req: TrialSimulationRequest):
    ct = _lazy('_ct', 'app.personalization.phase5.clinical_trial_simulator', 'ClinicalTrialSimulator')
    result = ct.simulate_trial(n_patients=req.n_patients, duration_days=req.duration_days)
    return {
        "trial_name": req.trial_name,
        "n_patients": req.n_patients,
        "duration_days": req.duration_days,
        "result": str(result),
    }


@router.post("/trials/comparative")
async def comparative_trial(
    trial_a: str = Query("metformin"),
    trial_b: str = Query("lifestyle"),
    n_patients: int = Query(500, ge=10, le=10000),
):
    ct = _lazy('_ct', 'app.personalization.phase5.clinical_trial_simulator', 'ClinicalTrialSimulator')
    from app.personalization.phase5.clinical_trial_simulator import simulate_comparative_trial, TrialEndpoint
    endpoint = TrialEndpoint(name="HbA1c reduction", baseline=7.5, target=7.0, direction="decrease")
    result = simulate_comparative_trial(ct, trial_a, trial_b, n_patients, [endpoint])
    return {
        "trial_a": getattr(result, 'trial_a_name', trial_a),
        "trial_b": getattr(result, 'trial_b_name', trial_b),
        "superiority": getattr(result, 'superiority', None),
        "effect_size": getattr(result, 'effect_size', None),
    }


# ── 5. Multi-Agent System ─────────────────────────────────────


@router.post("/agents/debate")
async def debate(req: DebateRequest):
    agents = _lazy('_agents', 'app.personalization.phase5.multi_agent_system', 'MultiAgentSystem')
    parts = req.question.split()
    cause = parts[0] if len(parts) > 0 else "exposure"
    effect = parts[-1] if len(parts) > 1 else "outcome"
    result = agents.debate_mechanism(cause=cause, effect=effect, proposed_mechanism=req.question)
    return {
        "question": req.question,
        "consensus": getattr(result, 'consensus', str(result)),
        "confidence": getattr(result, 'confidence', 0.0),
        "agent_votes": getattr(result, 'agent_votes', {}),
        "reasoning": getattr(result, 'reasoning', str(result)),
    }


@router.get("/agents/agents")
async def list_agents():
    agents = _lazy('_agents', 'app.personalization.phase5.multi_agent_system', 'MultiAgentSystem')
    return {
        "agents": [
            {"name": a.name, "role": str(role.value) if hasattr(role, 'value') else str(role)}
            for role, a in agents.agents.items()
        ],
    }


# ── 6. Adaptive Twin ──────────────────────────────────────────


@router.post("/adaptive-twin/create")
async def create_adaptive_twin_endpoint(req: CreateAdaptiveTwinRequest):
    if req.patient_id in _twins:
        raise HTTPException(400, f"Adaptive twin exists for {req.patient_id}")
    physio = np.array(req.physio) if req.physio else np.random.randn(30) * 0.1 + 100
    from app.personalization.dynamics import DEFAULT_PARAMS
    params = np.array(req.params) if req.params else DEFAULT_PARAMS.copy()
    from app.personalization.phase5.adaptive_twin import create_adaptive_twin
    twin = create_adaptive_twin(physio, params)
    _twins[req.patient_id] = twin
    return {"patient_id": req.patient_id, "status": "created", "state_dim": int(len(physio)), "param_dim": int(len(params))}


@router.post("/adaptive-twin/observe")
async def observe_twin(req: ObservationRequest):
    twin = _twins.get(req.patient_id)
    if twin is None:
        raise HTTPException(404, f"No adaptive twin for {req.patient_id}")
    from app.personalization.phase5.adaptive_twin import Observation
    obs = Observation(timestamp=req.timestamp, variables=req.variables)
    info = twin.observe(obs)
    return {"patient_id": req.patient_id, "result": info if isinstance(info, dict) else {"updated": True}}


@router.get("/adaptive-twin/{patient_id}")
async def get_adaptive_state(patient_id: str):
    twin = _twins.get(patient_id)
    if twin is None:
        raise HTTPException(404, f"No adaptive twin for {patient_id}")
    state, params = twin.get_state()
    return {"patient_id": patient_id, "state": state.tolist(), "params": params.tolist()}


@router.get("/adaptive-twin/{patient_id}/accuracy")
async def get_accuracy(patient_id: str, variable: str = Query("glucose")):
    twin = _twins.get(patient_id)
    if twin is None:
        raise HTTPException(404, f"No adaptive twin for {patient_id}")
    accuracy = {"n_predictions": 0, "mean_error": 0.0}
    if hasattr(twin, 'validate_prediction'):
        try:
            stats = twin.validate_prediction(variable, actual=0.0)
            accuracy = stats if isinstance(stats, dict) else {"result": str(stats)}
        except Exception:
            accuracy = {"note": "insufficient prediction history"}
    return {"patient_id": patient_id, "variable": variable, "accuracy": accuracy}


@router.get("/adaptive-twin/{patient_id}/evolution")
async def get_evolution(patient_id: str):
    twin = _twins.get(patient_id)
    if twin is None:
        raise HTTPException(404, f"No adaptive twin for {patient_id}")
    events = twin.get_evolution_summary() if hasattr(twin, 'get_evolution_summary') else []
    return {"patient_id": patient_id, "events": events}


# ── 7. Tissue Simulation ──────────────────────────────────────


@router.post("/tissues/simulate")
async def simulate_tissue(req: TissueSimRequest):
    ts = _lazy('_ts', 'app.personalization.phase5.tissue_simulation', 'TissueSimulator')
    organ_inputs = req.params or {"glucose": 100.0, "insulin": 10.0, "SBP": 120.0}
    traj = {}
    for _ in range(min(req.duration_days * 24 * 60 // 30, 100)):
        state = ts.step(dt=30.0, organ_inputs=organ_inputs)
        for k, v in state.items():
            if k not in traj:
                traj[k] = []
            if hasattr(v, '__dict__'):
                traj[k].append(str(v))
            else:
                traj[k].append(str(v))
    return {"tissue_types": list(ts.tissues.keys()) if hasattr(ts, 'tissues') else req.tissue_types, "steps": len(list(traj.values())[0]) if traj else 0, "trajectory": {k: v[:5] for k, v in traj.items()}}


@router.get("/tissues/types")
async def list_tissue_types():
    from app.personalization.phase5.tissue_simulation import TISSUE_TYPES
    return {"tissue_types": list(TISSUE_TYPES)}


# ── 8. Foundation Model ───────────────────────────────────────


@router.post("/foundation-model/encode")
async def fm_encode(req: FoundationModelRequest):
    import torch
    fm = _lazy('_fm', 'app.personalization.phase5.foundation_model', 'PhysiologyFoundationModel')
    data = torch.tensor(req.patient_states, dtype=torch.float32)
    if data.dim() == 2:
        data = data.unsqueeze(0)
    rep = fm.encode_patient(data)
    return {"representation": rep.detach().numpy().tolist(), "dim": rep.shape[-1]}


@router.post("/foundation-model/predict")
async def fm_predict(req: FoundationModelRequest):
    import torch
    fm = _lazy('_fm', 'app.personalization.phase5.foundation_model', 'PhysiologyFoundationModel')
    data = torch.tensor(req.patient_states, dtype=torch.float32)
    if data.dim() == 2:
        data = data.unsqueeze(0)
    pred = fm.predict_state(data)
    return {"predicted": pred.detach().numpy().tolist()}


@router.post("/foundation-model/forecast")
async def fm_forecast(req: FoundationModelRequest):
    import torch
    fm = _lazy('_fm', 'app.personalization.phase5.foundation_model', 'PhysiologyFoundationModel')
    data = torch.tensor(req.patient_states, dtype=torch.float32)
    if data.dim() == 2:
        data = data.unsqueeze(0)
    traj = fm.forecast_patient(data, req.n_steps)
    return {"trajectory": traj.detach().numpy().tolist(), "n_steps": req.n_steps}


# ── 9. Federated Learning ─────────────────────────────────────


@router.post("/federated/round")
async def federated_round(req: ValidationRequest):
    fl = _lazy('_fl', 'app.personalization.phase5.federated_learning', 'FederatedLearningEngine', 25, 5)
    result = fl.federated_averaging(client_fraction=0.5)
    return {
        "round": result.get("round", 0) if isinstance(result, dict) else 0,
        "privacy_budget": result.get("privacy_budget", 0.0) if isinstance(result, dict) else 0.0,
    }


@router.get("/federated/clients")
async def list_federated_clients():
    fl = _lazy('_fl', 'app.personalization.phase5.federated_learning', 'FederatedLearningEngine', 25, 5)
    for cid in ["c1", "c2", "c3"]:
        fl.register_client(cid, epsilon=1.0)
    clients = fl.get_active_clients()
    return {
        "clients": [
            {"id": c.client_id, "observations": getattr(c, 'n_local_observations', 0)}
            for c in clients
        ],
    }


@router.get("/federated/population")
async def get_population_knowledge():
    fl = _lazy('_fl', 'app.personalization.phase5.federated_learning', 'FederatedLearningEngine', 25, 5)
    mean, cov = fl.get_population_prior()
    return {"mean": mean.tolist(), "covariance": cov.tolist()}


# ── 10. Validation ────────────────────────────────────────────


@router.post("/validation/run")
async def run_validation(req: ValidationRequest):
    vf = _lazy('_vf', 'app.personalization.phase5.validation_framework', 'ValidationFrameworkV2')
    vf.run_all()
    report = vf.get_report()
    return {
        "status": "completed",
        "n_criteria": len(getattr(report, 'criteria', [])) if isinstance(report, object) else 0,
        "report": str(report),
    }


@router.get("/validation/results")
async def get_validation_results():
    vf = _lazy('_vf', 'app.personalization.phase5.validation_framework', 'ValidationFrameworkV2')
    report = vf.get_report() if hasattr(vf, 'get_report') else {}
    return {"report": report}


# ── Health ────────────────────────────────────────────────────


@router.get("/health")
async def health():
    return {
        "status": "healthy",
        "layer": "personalization",
        "version": "5.0.0",
        "description": "Phase 5 Autonomous Biological Intelligence Platform",
        "capabilities": [
            "Self-Updating Knowledge Graph",
            "Mechanism Discovery",
            "Scientific Hypothesis Generator",
            "Virtual Clinical Trials",
            "Multi-Agent Reasoning",
            "Adaptive Twin Evolution",
            "Tissue Simulation",
            "Foundation Model",
            "Federated Learning",
            "Validation V2",
        ],
    }
