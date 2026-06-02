"""
Turns a row from the synthetic patient table into:
  - an in-memory torch_geometric Data object for the GNN
  - a Neo4j subgraph (patient -> biomarkers -> organs/diseases)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from app.graph.ontology import (
    ALL_NODES, BIOMARKERS, DISEASES, EDGES, ORGANS, NODE_INDEX, Node,
)
from app.graph.neo4j_client import neo4j_client


# --- helpers -----------------------------------------------------------
def _normalise(value: float, lo: float, hi: float) -> float:
    """Map [lo, hi] -> [-1, 1], clamping outside."""
    if hi == lo:
        return 0.0
    return float(np.clip((value - (hi + lo) / 2) / ((hi - lo) / 2), -1.0, 1.0))


def _risk_label(score: float) -> str:
    if score < 0.25:
        return "low"
    if score < 0.55:
        return "moderate"
    if score < 0.80:
        return "high"
    return "critical"


def compute_risk_score(row: pd.Series) -> float:
    """Aggregate deviation from healthy range into [0, 1]."""
    deltas = []
    for b in BIOMARKERS:
        v = row.get(b.name)
        if v is None or pd.isna(v):
            continue
        deltas.append(abs(_normalise(float(v), b.healthy_lo, b.healthy_hi)))
    if not deltas:
        return 0.0
    return float(np.clip(np.mean(deltas), 0.0, 1.0))


# --- in-memory cohort graph -------------------------------------------
@dataclass
class CohortGraph:
    x: torch.Tensor              # (N_nodes_total, F) node feature matrix
    edge_index: torch.Tensor     # (2, E) COO edges
    node_ids: list[str]
    patient_id_to_node: dict[str, int]
    node_kind: list[str]         # 'patient'|'biomarker'|'organ'|'disease' per row
    risk: dict[str, float]       # patient_id -> score


class GraphBuilder:
    """Builds a (patient | biomarker | organ | disease) heterogeneous graph
    shared across the whole cohort."""

    BIOMARKER_FEATURE_NAMES = [
        "value_norm", "is_abnormal", "delta_from_healthy",
    ]

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.node_ids: list[str] = []
        self.node_kind: list[str] = []
        self.patient_id_to_node: dict[str, int] = {}
        self.biomarker_value: dict[str, float] = {}
        self.risk: dict[str, float] = {}

    # ---- building ---------------------------------------------------
    def add_patient(self, row: pd.Series) -> int:
        pid = row["patient_id"]
        idx = len(self.node_ids)
        self.node_ids.append(pid)
        self.node_kind.append("patient")
        self.patient_id_to_node[pid] = idx
        self.risk[pid] = compute_risk_score(row)
        return idx

    def add_ontology_skeleton(self) -> None:
        """Add the static biomarker/organ/disease nodes + edges."""
        for b in BIOMARKERS:
            self.node_ids.append(b.id); self.node_kind.append("biomarker")
        for o in ORGANS:
            self.node_ids.append(o.id); self.node_kind.append("organ")
        for d in DISEASES:
            self.node_ids.append(d.id); self.node_kind.append("disease")

    def add_patient_values(self, row: pd.Series) -> None:
        pid = row["patient_id"]
        for b in BIOMARKERS:
            v = row.get(b.name)
            if v is None or pd.isna(v):
                continue
            self.biomarker_value[b.id] = float(v)
        self.biomarker_value  # touch

    def add_edges(self) -> tuple[np.ndarray, np.ndarray]:
        id_index = {nid: i for i, nid in enumerate(self.node_ids)}
        srcs, dsts = [], []

        # patient -> biomarker (HAS_VALUE)
        for pid, pidx in self.patient_id_to_node.items():
            for b in BIOMARKERS:
                bidx = id_index[b.id]
                srcs.append(pidx); dsts.append(bidx)

        # ontology edges (biomarker<->organ, biomarker<->disease, disease<->disease)
        for e in EDGES:
            s = id_index.get(e.src); d = id_index.get(e.dst)
            if s is None or d is None:
                continue
            srcs.append(s); dsts.append(d)
            srcs.append(d); dsts.append(s)  # undirected

        return np.array(srcs, dtype=np.int64), np.array(dsts, dtype=np.int64)

    def build(self, df: pd.DataFrame) -> CohortGraph:
        self.reset()
        self.add_ontology_skeleton()
        for _, row in df.iterrows():
            self.add_patient(row)
            self.add_patient_values(row)
        src, dst = self.add_edges()

        # ---- features -------------------------------------------------
        n = len(self.node_ids)
        # patients get [age_z, gender_male, bmi_norm, n_abnormal, risk]
        # biomarkers get [value_norm, is_abnormal, delta]
        # organs/diseases get zero vectors (they will receive messages)
        F = 8
        x = np.zeros((n, F), dtype=np.float32)

        for _, row in df.iterrows():
            pidx = self.patient_id_to_node[row["patient_id"]]
            x[pidx, 0] = (float(row["age"]) - 45) / 15
            x[pidx, 1] = 1.0 if row["gender"] == "Male" else 0.0
            x[pidx, 2] = _normalise(float(row["bmi"]),
                                    BIOMARKERS[-1].healthy_lo,
                                    BIOMARKERS[-1].healthy_hi)
            abnorm = 0
            for b in BIOMARKERS:
                v = row.get(b.name)
                if v is None or pd.isna(v):
                    continue
                if not (b.healthy_lo <= float(v) <= b.healthy_hi):
                    abnorm += 1
            x[pidx, 3] = abnorm / len(BIOMARKERS)
            x[pidx, 4] = self.risk[row["patient_id"]]

        for b in BIOMARKERS:
            bidx = self.node_ids.index(b.id)
            v = self.biomarker_value.get(b.id)
            if v is None:
                continue
            x[bidx, 5] = _normalise(v, b.healthy_lo, b.healthy_hi)
            x[bidx, 6] = 0.0 if b.healthy_lo <= v <= b.healthy_hi else 1.0
            x[bidx, 7] = (v - (b.healthy_hi + b.healthy_lo) / 2) / \
                         max(1e-3, (b.healthy_hi - b.healthy_lo) / 2)

        edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
        x_t = torch.from_numpy(x)

        return CohortGraph(
            x=x_t,
            edge_index=edge_index,
            node_ids=self.node_ids,
            patient_id_to_node=self.patient_id_to_node,
            node_kind=self.node_kind,
            risk=self.risk,
        )

    # ---- neo4j persistence -----------------------------------------
    def persist_to_neo4j(self, df: pd.DataFrame) -> dict:
        """Push the cohort into Neo4j. Returns counts."""
        if not neo4j_client.available:
            return {"neo4j_loaded": False}

        neo4j_client.wipe()

        for b in BIOMARKERS:
            neo4j_client.upsert_node("Biomarker", "id", {
                "id": b.id, "name": b.name, "unit": b.unit,
                "healthy_lo": b.healthy_lo, "healthy_hi": b.healthy_hi,
            })
        for o in ORGANS:
            neo4j_client.upsert_node("Organ", "id",
                                     {"id": o.id, "name": o.name})
        for d in DISEASES:
            neo4j_client.upsert_node("Disease", "id",
                                     {"id": d.id, "name": d.name})

        for _, row in df.iterrows():
            neo4j_client.upsert_node("Patient", "patient_id", {
                "patient_id": row["patient_id"],
                "age": int(row["age"]),
                "gender": row["gender"],
                "bmi": float(row["bmi"]),
                "risk_score": float(self.risk[row["patient_id"]]),
                "risk_label": _risk_label(self.risk[row["patient_id"]]),
            })
            for b in BIOMARKERS:
                v = row.get(b.name)
                if v is None or pd.isna(v):
                    continue
                neo4j_client.upsert_edge(
                    "Patient", row["patient_id"],
                    "Biomarker", b.id,
                    "HAS_VALUE",
                    {"value": float(v), "unit": b.unit},
                )

        for e in EDGES:
            src_label = NODE_INDEX[e.src].kind.capitalize()
            dst_label = NODE_INDEX[e.dst].kind.capitalize()
            neo4j_client.upsert_edge(
                src_label, e.src, dst_label, e.dst, e.rel,
                {"weight": e.weight},
            )

        return {
            "neo4j_loaded": True,
            "patients": neo4j_client.count("Patient"),
            "biomarkers": neo4j_client.count("Biomarker"),
            "organs": neo4j_client.count("Organ"),
            "diseases": neo4j_client.count("Disease"),
        }


# module-level singleton (state lives in app.state in production, but
# a singleton is fine for the MVP since uvicorn --reload is one process)
_GRAPH: CohortGraph | None = None
_BUILDER = GraphBuilder()


def get_or_build_cohort_graph(df: pd.DataFrame) -> CohortGraph:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _BUILDER.build(df)
    return _GRAPH


def reset_cohort_graph() -> None:
    global _GRAPH
    _GRAPH = None
