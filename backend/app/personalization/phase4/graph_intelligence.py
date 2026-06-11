"""
Phase 4: Graph Intelligence Layer.

Unified biomedical knowledge graph connecting:
  - Drugs (RDKit structures + targets)
  - Diseases (phenotypes + pathways)
  - Proteins/genes (UniProt)
  - Pathways (Reactome)
  - Biomarkers
  - Patient similarity

Uses NetworkX for in-memory graph, PyTorch Geometric for GNN embeddings,
with Neo4j schema ready for production deployment.
"""

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass, field

try:
    import torch_geometric as pyg
    from torch_geometric.data import Data, Batch, HeteroData
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


# ── Node and Edge Types ───────────────────────────────────────

NODE_TYPES = ["drug", "disease", "protein", "pathway", "biomarker", "patient"]
EDGE_TYPES = [
    ("drug", "targets", "protein"),
    ("drug", "treats", "disease"),
    ("protein", "participates_in", "pathway"),
    ("pathway", "associated_with", "disease"),
    ("biomarker", "indicates", "disease"),
    ("patient", "has_disease", "disease"),
    ("patient", "similar_to", "patient"),
    ("protein", "interacts_with", "protein"),
]


@dataclass
class GraphEntity:
    """A node in the biomedical knowledge graph."""
    id: str
    type: str
    name: str
    features: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


class BioKnowledgeGraph:
    """
    Unified biomedical knowledge graph.

    In-memory using NetworkX; Neo4j schema provided for scale-out.
    Supports GNN-based representation learning via PyG.
    """

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self._entities: Dict[str, GraphEntity] = {}

    # ── Entity Management ──

    def add_entity(self, entity: GraphEntity) -> None:
        self._entities[entity.id] = entity
        self.graph.add_node(entity.id, type=entity.type, name=entity.name, **entity.features)

    def add_relation(
        self, source_id: str, target_id: str, relation: str,
        weight: float = 1.0, properties: Dict = None,
    ) -> None:
        self.graph.add_edge(source_id, target_id, relation=relation,
                           weight=weight, **(properties or {}))

    def get_entity(self, entity_id: str) -> Optional[GraphEntity]:
        return self._entities.get(entity_id)

    def get_neighbors(self, entity_id: str, relation: str = None) -> List[str]:
        if relation:
            return [
                n for n in self.graph.neighbors(entity_id)
                if self.graph.has_edge(entity_id, n) and
                any(d.get("relation") == relation for d in self.graph[entity_id][n].values())
            ]
        return list(self.graph.neighbors(entity_id))

    # ── Built-in Knowledge Population ──

    def populate_diabetes_graph(self) -> None:
        """Populate with core diabetes/CVD/kidney knowledge."""
        # Drugs
        for drug_id, name, targets in [
            ("DB00331", "Metformin", ["P06213", "P01308"]),
            ("DB01076", "Atorvastatin", ["P04035"]),
            ("DB08877", "Empagliflozin", ["P31639"]),
        ]:
            self.add_entity(GraphEntity(drug_id, "drug", name))
            for t in targets:
                self.add_relation(drug_id, t, "targets")

        # Proteins
        for prot_id, name in [
            ("P06213", "INSR"), ("P01308", "Insulin"),
            ("P04035", "HMGCR"), ("P31639", "SGLT2"),
            ("P06239", "PPARG"), ("P01375", "TNF"),
            ("P05231", "IL6"), ("P01344", "IGF1"),
        ]:
            self.add_entity(GraphEntity(prot_id, "protein", name))

        # Pathways
        for pw_id, name in [
            ("PW0001", "Insulin Signaling"),
            ("PW0002", "Gluconeogenesis"),
            ("PW0003", "Inflammatory Response"),
            ("PW0004", "Lipid Metabolism"),
            ("PW0005", "Renal Function"),
        ]:
            self.add_entity(GraphEntity(pw_id, "pathway", name))

        # Protein → Pathway
        for prot_id, pw_id in [
            ("P06213", "PW0001"), ("P01308", "PW0001"),
            ("P06239", "PW0004"), ("P01375", "PW0003"),
            ("P05231", "PW0003"),
        ]:
            self.add_relation(prot_id, pw_id, "participates_in")

        # Diseases
        for dis_id, name in [
            ("DOID_9352", "Type 2 Diabetes"),
            ("DOID_10763", "Hypertension"),
            ("DOID_1319", "Chronic Kidney Disease"),
            ("DOID_10652", "Atrial Fibrillation"),
        ]:
            self.add_entity(GraphEntity(dis_id, "disease", name))

        # Drugs → Diseases
        self.add_relation("DB00331", "DOID_9352", "treats", weight=0.8)
        self.add_relation("DB01076", "DOID_9352", "treats", weight=0.3)
        self.add_relation("DB08877", "DOID_9352", "treats", weight=0.9)

        # Pathways → Diseases
        for pw_id, dis_id in [
            ("PW0001", "DOID_9352"), ("PW0003", "DOID_9352"),
            ("PW0004", "DOID_9352"), ("PW0005", "DOID_1319"),
        ]:
            self.add_relation(pw_id, dis_id, "associated_with")

    # ── NetworkX to PyG Conversion ──

    def to_pyg_data(self) -> 'Data':
        """Convert to PyTorch Geometric Data object for GNN training."""
        if not HAS_PYG:
            raise ImportError("PyTorch Geometric required")

        nodes = list(self.graph.nodes())
        node_to_idx = {n: i for i, n in enumerate(nodes)}

        # Node features: one-hot type encoding + degree
        type_set = list(set(
            self.graph.nodes[n].get("type", "unknown") for n in nodes
        ))
        type_to_idx = {t: i for i, t in enumerate(type_set)}

        x = torch.zeros((len(nodes), len(type_set) + 1))
        for i, n in enumerate(nodes):
            ntype = self.graph.nodes[n].get("type", "unknown")
            x[i, type_to_idx.get(ntype, 0)] = 1.0
            x[i, -1] = self.graph.degree(n)

        # Edge index
        edge_index = [[], []]
        edge_attr = []
        for u, v, data in self.graph.edges(data=True):
            if u in node_to_idx and v in node_to_idx:
                edge_index[0].append(node_to_idx[u])
                edge_index[1].append(node_to_idx[v])
                edge_attr.append(data.get("weight", 1.0))

        return Data(
            x=x,
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float).view(-1, 1),
        )

    # ── Neo4j Schema (for production) ──

    NEO4J_SCHEMA = """
    // Phase 4 Neo4j Schema
    CREATE CONSTRAINT drug_id IF NOT EXISTS FOR (d:Drug) REQUIRE d.id IS UNIQUE;
    CREATE CONSTRAINT disease_id IF NOT EXISTS FOR (d:Disease) REQUIRE d.id IS UNIQUE;
    CREATE CONSTRAINT protein_id IF NOT EXISTS FOR (p:Protein) REQUIRE p.id IS UNIQUE;
    CREATE CONSTRAINT pathway_id IF NOT EXISTS FOR (p:Pathway) REQUIRE p.id IS UNIQUE;
    CREATE CONSTRAINT biomarker_id IF NOT EXISTS FOR (b:Biomarker) REQUIRE b.id IS UNIQUE;
    CREATE CONSTRAINT patient_id IF NOT EXISTS FOR (p:Patient) REQUIRE p.id IS UNIQUE;

    CREATE INDEX drug_name IF NOT EXISTS FOR (d:Drug) ON (d.name);
    CREATE INDEX disease_name IF NOT EXISTS FOR (d:Disease) ON (d.name);

    // Relationships
    // (Drug)-[:TARGETS {affinity: float}]->(Protein)
    // (Drug)-[:TREATS {efficacy: float}]->(Disease)
    // (Protein)-[:PARTICIPATES_IN]->(Pathway)
    // (Pathway)-[:ASSOCIATED_WITH]->(Disease)
    // (Biomarker)-[:INDICATES {sensitivity: float}]->(Disease)
    // (Patient)-[:HAS_DISEASE {severity: float}]->(Disease)
    // (Patient)-[:SIMILAR_TO {score: float}]->(Patient)
    // (Protein)-[:INTERACTS_WITH {type: string}]->(Protein)
    """


# ── GNN Encoder ───────────────────────────────────────────────

class GraphEncoder(nn.Module):
    """
    GNN encoder for biomedical knowledge graph embedding.

    Produces node embeddings that capture drug-disease-pathway
    relationships. Used for patient similarity and drug repurposing.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 32):
        super().__init__()
        if not HAS_PYG:
            raise ImportError("PyTorch Geometric required")

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        self.conv3 = SAGEConv(hidden_dim, output_dim)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return F.normalize(x, dim=1)


# ── Patient Similarity ────────────────────────────────────────

class PatientSimilarityGraph:
    """
    Patient-patient similarity graph based on twin states.
    Used for cohort discovery and treatment outcome prediction.
    """

    def __init__(self, similarity_threshold: float = 0.7):
        self.graph = nx.Graph()
        self.threshold = similarity_threshold

    def add_patient(self, patient_id: str, twin_state: np.ndarray,
                    parameters: np.ndarray, metadata: Dict = None) -> None:
        features = np.concatenate([twin_state, parameters])
        self.graph.add_node(patient_id, features=features, **(metadata or {}))

    def compute_similarities(self) -> None:
        """Compute pairwise cosine similarity between patients."""
        nodes = list(self.graph.nodes(data=True))
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                fi = nodes[i][1].get("features")
                fj = nodes[j][1].get("features")
                if fi is not None and fj is not None:
                    sim = np.dot(fi, fj) / (np.linalg.norm(fi) * np.linalg.norm(fj) + 1e-10)
                    if sim > self.threshold:
                        self.graph.add_edge(nodes[i][0], nodes[j][0], weight=float(sim))

    def get_similar_patients(self, patient_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if patient_id not in self.graph:
            return []
        similarities = []
        for neighbor in self.graph.neighbors(patient_id):
            w = self.graph[patient_id][neighbor].get("weight", 0)
            similarities.append((neighbor, w))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


# ── Drug Repurposing Suggestion ───────────────────────────────

def suggest_drugs_for_patient(
    kg: BioKnowledgeGraph,
    patient_diseases: List[str],
    encoder: Optional[GraphEncoder] = None,
    top_k: int = 5,
) -> List[Dict]:
    """
    Suggest drugs for a patient's diseases using graph proximity.
    """
    suggestions = []
    for disease_id in patient_diseases:
        # Find drugs that treat this disease
        for drug_id in kg.graph.nodes():
            if kg.graph.nodes[drug_id].get("type") != "drug":
                continue
            if kg.graph.has_edge(drug_id, disease_id):
                edge_data = kg.graph[drug_id][disease_id]
                weight = max(d.get("weight", 1.0) for d in edge_data.values())
                suggestions.append({
                    "drug": kg._entities[drug_id].name if drug_id in kg._entities else drug_id,
                    "disease": kg._entities[disease_id].name if disease_id in kg._entities else disease_id,
                    "confidence": float(weight),
                })
    suggestions.sort(key=lambda x: x["confidence"], reverse=True)
    return suggestions[:top_k]
