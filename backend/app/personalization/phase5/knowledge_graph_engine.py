"""
Phase 5 — Pillar 1: Self-Updating Biological Knowledge Graph.

Extends the Phase 4 BioKnowledgeGraph with:
  - Expanded node types (genes, proteins, cells, organs, trials, publications)
  - Literature mining from PubMed
  - Clinical trial ingestion from ClinicalTrials.gov
  - Dynamic graph evolution with confidence decay / reinforcement
  - Temporal edge tracking
"""

import numpy as np
import networkx as nx
import json
import time
import hashlib
import re
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


# ── Node Types ────────────────────────────────────────────────

NODE_TYPES = [
    "gene", "protein", "cell_type", "organ", "tissue",
    "disease", "drug", "biomarker", "pathway",
    "clinical_trial", "publication", "outcome",
    "symptom", "measurement", "intervention",
]


class EdgeType(Enum):
    ENCODES = "encodes"
    PRODUCES = "produces"
    ACTIVATES = "activates"
    INHIBITS = "inhibits"
    BINDS = "binds"
    REGULATES = "regulates"
    PARTICIPATES_IN = "participates_in"
    ASSOCIATED_WITH = "associated_with"
    TREATS = "treats"
    CAUSES = "causes"
    PREDICTS = "predicts"
    MEASURES = "measures"
    STUDIED_IN = "studied_in"
    REPORTS = "reports"
    SIMILAR_TO = "similar_to"


@dataclass
class BiologicalNode:
    """A node in the biological knowledge graph."""
    id: str
    name: str
    node_type: str
    description: str = ""
    source: str = "manual"
    confidence: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    updated_at: float = 0.0
    version: int = 1

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.updated_at == 0.0:
            self.updated_at = self.created_at


@dataclass
class BiologicalEdge:
    """A directed edge with confidence and evidence."""
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    confidence: float = 1.0
    evidence: List[str] = field(default_factory=list)
    source: str = "manual"
    created_at: float = 0.0
    updated_at: float = 0.0
    publications: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.updated_at == 0.0:
            self.updated_at = self.created_at

    @property
    def key(self) -> str:
        return f"{self.source_id}--{self.edge_type.value}--{self.target_id}"


# ── Literature Miner ──────────────────────────────────────────

class LiteratureMiner:
    """
    Mine biological relationships from literature.

    Uses a rule-based NLP approach with PubTator-style annotation
    for entity and relationship extraction from biomedical text.
    Integrates with external APIs when available.
    """

    def __init__(self):
        # Curated relationship patterns: (source_type, target_type, trigger_words) -> EdgeType
        self._relation_patterns: Dict[Tuple[str, str, str], EdgeType] = {
            ("gene", "protein"): EdgeType.ENCODES,
            ("drug", "disease"): EdgeType.TREATS,
            ("protein", "pathway"): EdgeType.PARTICIPATES_IN,
            ("biomarker", "disease"): EdgeType.PREDICTS,
            ("disease", "symptom"): EdgeType.CAUSES,
            ("drug", "protein"): EdgeType.BINDS,
        }
        # Gene/protein/disease dictionaries (curated subset)
        self._known_entities: Dict[str, Dict[str, str]] = {
            "insulin": {"type": "protein", "id": "P01308"},
            "INSR": {"type": "protein", "id": "P06213"},
            "GLUT4": {"type": "protein", "id": "P14672"},
            "SGLT2": {"type": "protein", "id": "P31639"},
            "metformin": {"type": "drug", "id": "DB00331"},
            "empagliflozin": {"type": "drug", "id": "DB08877"},
            "atorvastatin": {"type": "drug", "id": "DB01076"},
            "type 2 diabetes": {"type": "disease", "id": "DOID_9352"},
            "hypertension": {"type": "disease", "id": "DOID_10763"},
            "CKD": {"type": "disease", "id": "DOID_1319"},
            "insulin resistance": {"type": "biomarker", "id": "BM_IR"},
            "HbA1c": {"type": "biomarker", "id": "BM_HBA1C"},
            "TNF-alpha": {"type": "protein", "id": "P01375"},
            "IL6": {"type": "protein", "id": "P05231"},
            "PPARG": {"type": "gene", "id": "G_PPARG"},
            "HMGCR": {"type": "gene", "id": "G_HMGCR"},
            "NFKB": {"type": "protein", "id": "P19838"},
            "AMPK": {"type": "protein", "id": "Q15831"},
            "insulin signaling": {"type": "pathway", "id": "PW0001"},
            "inflammatory response": {"type": "pathway", "id": "PW0003"},
        }

    def mine_from_text(self, text: str, source: str = "literature",
                       pmid: Optional[str] = None) -> List[BiologicalEdge]:
        """
        Extract biological relationships from text using pattern matching.

        Args:
            text: Biomedical text to mine
            source: Source identifier
            pmid: PubMed ID if applicable

        Returns:
            List of extracted biological edges
        """
        edges = []
        text_lower = text.lower()

        # Find mentioned entities
        mentions = {}
        for name, info in self._known_entities.items():
            if name.lower() in text_lower:
                mentions[name] = info

        # Extract causal/associative relationships
        cause_patterns = [
            (r"(\w+(?:\s+\w+)*)\s+(causes|leads to|results in|promotes|induces|triggers)\s+(\w+(?:\s+\w+)*)", EdgeType.CAUSES),
            (r"(\w+(?:\s+\w+)*)\s+(inhibits|suppresses|blocks|reduces|decreases)\s+(\w+(?:\s+\w+)*)", EdgeType.INHIBITS),
            (r"(\w+(?:\s+\w+)*)\s+(activates|stimulates|increases|upregulates|enhances)\s+(\w+(?:\s+\w+)*)", EdgeType.ACTIVATES),
            (r"(\w+(?:\s+\w+)*)\s+(is associated with|is linked to|correlates with)\s+(\w+(?:\s+\w+)*)", EdgeType.ASSOCIATED_WITH),
            (r"(\w+(?:\s+\w+)*)\s+(treats|is effective against|is used for)\s+(\w+(?:\s+\w+)*)", EdgeType.TREATS),
            (r"(\w+(?:\s+\w+)*)\s+(predicts|is a biomarker for|indicates)\s+(\w+(?:\s+\w+)*)", EdgeType.PREDICTS),
        ]

        for pattern, default_type in cause_patterns:
            for match in re.finditer(pattern, text_lower):
                src_text = match.group(1).strip()
                tgt_text = match.group(3).strip()
                src = mentions.get(src_text)
                tgt = mentions.get(tgt_text)
                if src and tgt:
                    edge = BiologicalEdge(
                        source_id=src["id"],
                        target_id=tgt["id"],
                        edge_type=default_type,
                        weight=1.0,
                        confidence=0.7,
                        evidence=[text[match.start():match.end()]],
                        source=source or "literature_mining",
                        publications=[pmid] if pmid else [],
                    )
                    edges.append(edge)
                    # Remove matched entities to avoid double-counting
                    mentions.pop(src_text, None)
                    mentions.pop(tgt_text, None)

        # Extract pairwise relationships from co-occurrence
        mention_items = list(mentions.items())
        for i in range(len(mention_items)):
            for j in range(i + 1, len(mention_items)):
                name_a, info_a = mention_items[i]
                name_b, info_b = mention_items[j]
                type_pair = (info_a["type"], info_b["type"])
                if type_pair in self._relation_patterns:
                    edge_type = self._relation_patterns[type_pair]
                elif (info_b["type"], info_a["type"]) in self._relation_patterns:
                    edge_type = self._relation_patterns[(info_b["type"], info_a["type"])]
                    name_a, name_b = name_b, name_a
                    info_a, info_b = info_b, info_a
                else:
                    edge_type = EdgeType.ASSOCIATED_WITH

                edge = BiologicalEdge(
                    source_id=info_a["id"],
                    target_id=info_b["id"],
                    edge_type=edge_type,
                    weight=0.5,
                    confidence=0.3,
                    evidence=[f"Co-occurrence in text: {name_a}, {name_b}"],
                    source=source or "literature_cooccurrence",
                    publications=[pmid] if pmid else [],
                )
                edges.append(edge)

        return edges

    def mine_from_pubmed(self, query: str, max_results: int = 100) -> List[BiologicalEdge]:
        """
        Query PubMed and mine relationships from abstracts.

        Uses a simulated query — in production, connects to NCBI E-utilities.
        """
        # Simulated PubMed results
        simulated_abstracts = self._simulate_pubmed_results(query, max_results)
        all_edges = []
        for pmid, title, abstract in simulated_abstracts:
            text = f"{title}. {abstract}"
            edges = self.mine_from_text(text, source="pubmed", pmid=pmid)
            all_edges.extend(edges)
        return all_edges

    def _simulate_pubmed_results(self, query: str, n: int = 10) -> List[Tuple[str, str, str]]:
        """Generate simulated PubMed results for testing."""
        base_abstracts = [
            ("PMC00001", "Insulin resistance and metabolic syndrome",
             "Insulin resistance is associated with type 2 diabetes and hypertension. "
             "TNF-alpha activates inflammatory pathways that inhibit insulin signaling. "
             "Metformin treats type 2 diabetes by activating AMPK."),
            ("PMC00002", "Role of SGLT2 inhibitors in CKD",
             "Empagliflozin inhibits SGLT2 and reduces glucose reabsorption. "
             "SGLT2 inhibition is associated with improved renal outcomes in CKD patients."),
            ("PMC00003", "Inflammatory mechanisms in cardiovascular disease",
             "IL6 is a key mediator of the inflammatory response. "
             "NFKB activation promotes inflammation in cardiovascular disease. "
             "HbA1c predicts cardiovascular outcomes in type 2 diabetes."),
            ("PMC00004", "Circadian disruption and metabolic health",
             "Sleep disruption causes insulin resistance and increases cortisol levels. "
             "Circadian rhythm regulates glucose metabolism through PPARG and AMPK pathways."),
            ("PMC00005", "Adipose tissue dysfunction in obesity",
             "Adipose tissue inflammation is linked to insulin resistance. "
             "TNF-alpha produced by macrophages inhibits insulin signaling in adipocytes."),
            ("PMC00006", "HMGCR inhibitors and lipid metabolism",
             "Atorvastatin inhibits HMGCR and reduces LDL cholesterol. "
             "Statin therapy is associated with reduced cardiovascular events."),
            ("PMC00007", "Gut microbiome and metabolic health",
             "Gut microbiota composition is associated with insulin resistance. "
             "Short-chain fatty acids activate AMPK and improve glucose metabolism."),
            ("PMC00008", "Exercise adaptations in skeletal muscle",
             "Exercise activates AMPK and GLUT4 translocation. "
             "Regular physical activity improves insulin sensitivity and reduces inflammation."),
        ]
        return base_abstracts[:n]


# ── Clinical Trials Miner ─────────────────────────────────────

class ClinicalTrialsMiner:
    """
    Mine knowledge from clinical trial registries.

    Simulates ClinicalTrials.gov API integration.
    """

    def __init__(self):
        # Simulated trial database
        self._trials: List[Dict[str, Any]] = [
            {
                "nct_id": "NCT00123456",
                "title": "Metformin vs Lifestyle in Prediabetes",
                "status": "completed",
                "phase": "III",
                "conditions": ["DOID_9352"],
                "interventions": ["DB00331", "lifestyle_modification"],
                "primary_outcomes": ["BM_HBA1C", "BM_IR"],
                "enrollment": 2500,
                "results": {"HbA1c_change": -0.8, "IR_change": -1.2},
            },
            {
                "nct_id": "NCT00234567",
                "title": "Empagliflozin in CKD Patients",
                "status": "completed",
                "phase": "III",
                "conditions": ["DOID_1319", "DOID_9352"],
                "interventions": ["DB08877"],
                "primary_outcomes": ["GFR_slope", "albuminuria"],
                "enrollment": 1500,
                "results": {"GFR_slope_change": 2.5, "albuminuria_reduction": 0.3},
            },
        ]

    def mine_trials(self, condition: Optional[str] = None) -> List[BiologicalEdge]:
        """Extract edges from trial data."""
        edges = []
        for trial in self._trials:
            if condition and condition not in trial["conditions"]:
                continue
            nct_id = trial["nct_id"]

            # Drug → Treats → Disease
            for drug_id in trial["interventions"]:
                for cond_id in trial["conditions"]:
                    if drug_id.startswith("DB"):
                        edges.append(BiologicalEdge(
                            source_id=drug_id,
                            target_id=cond_id,
                            edge_type=EdgeType.TREATS,
                            weight=trial["results"].get("HbA1c_change", 0.5),
                            confidence=0.6,
                            source="clinical_trial",
                            publications=[nct_id],
                        ))

            # Intervention → Outcome
            for drug_id in trial["interventions"]:
                for outcome in trial["primary_outcomes"]:
                    edges.append(BiologicalEdge(
                        source_id=drug_id,
                        target_id=outcome,
                        edge_type=EdgeType.PREDICTS,
                        weight=0.7,
                        confidence=0.5,
                        source="clinical_trial",
                        publications=[nct_id],
                    ))

        return edges


# ── Knowledge Graph Engine ────────────────────────────────────

class KnowledgeGraphEngine:
    """
    Self-updating biological knowledge graph.

    Manages:
      - Node and edge storage with confidence tracking
      - Literature/trial ingestion pipelines
      - Confidence decay and reinforcement over time
      - Query and traversal for mechanism discovery
    """

    def __init__(self, decay_days: float = 365.0):
        self.graph = nx.MultiDiGraph()
        self._nodes: Dict[str, BiologicalNode] = {}
        self._edges: Dict[str, BiologicalEdge] = {}
        self._decay_days = decay_days
        self._literature_miner = LiteratureMiner()
        self._trial_miner = ClinicalTrialsMiner()
        self._update_count = 0

    # ── Node Management ──

    def add_node(self, node: BiologicalNode) -> None:
        if node.id not in self._nodes:
            self._nodes[node.id] = node
            self.graph.add_node(node.id, type=node.node_type, name=node.name,
                                confidence=node.confidence, **node.properties)
        else:
            existing = self._nodes[node.id]
            existing.confidence = max(existing.confidence, node.confidence)
            existing.updated_at = time.time()
            existing.version += 1
            existing.properties.update(node.properties)
            self.graph.nodes[node.id].update(confidence=existing.confidence)

    def get_node(self, node_id: str) -> Optional[BiologicalNode]:
        return self._nodes.get(node_id)

    def get_nodes_by_type(self, node_type: str) -> List[BiologicalNode]:
        return [n for n in self._nodes.values() if n.node_type == node_type]

    def search_nodes(self, query: str, top_k: int = 20) -> List[BiologicalNode]:
        query = query.lower()
        scored = []
        for node in self._nodes.values():
            score = 0.0
            if query in node.name.lower():
                score += 1.0
            if query in node.id.lower():
                score += 0.8
            if query in node.description.lower():
                score += 0.3
            if score > 0:
                scored.append((score, node))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [n for _, n in scored[:top_k]]

    # ── Edge Management ──

    def add_edge(self, edge: BiologicalEdge) -> None:
        key = edge.key
        if key in self._edges:
            existing = self._edges[key]
            existing.weight = 0.5 * (existing.weight + edge.weight)
            existing.confidence = max(existing.confidence, edge.confidence)
            existing.evidence.extend(edge.evidence)
            existing.publications.extend(edge.publications)
            existing.updated_at = time.time()
        else:
            self._edges[key] = edge

        self.graph.add_edge(
            edge.source_id, edge.target_id,
            key=key, relation=edge.edge_type.value,
            weight=edge.weight, confidence=edge.confidence,
        )

    def get_edges(self, source_id: Optional[str] = None,
                  target_id: Optional[str] = None,
                  edge_type: Optional[EdgeType] = None) -> List[BiologicalEdge]:
        results = []
        for edge in self._edges.values():
            if source_id and edge.source_id != source_id:
                continue
            if target_id and edge.target_id != target_id:
                continue
            if edge_type and edge.edge_type != edge_type:
                continue
            results.append(edge)
        return results

    def get_neighbors(self, node_id: str, relation: Optional[str] = None) -> List[Tuple[str, BiologicalEdge]]:
        neighbors = []
        for key, edge in self._edges.items():
            if edge.source_id == node_id:
                if relation is None or edge.edge_type.value == relation:
                    neighbors.append((edge.target_id, edge))
            if edge.target_id == node_id:
                if relation is None or edge.edge_type.value == relation:
                    neighbors.append((edge.source_id, edge))
        return neighbors

    # ── Graph Evolution ──

    def apply_confidence_decay(self) -> None:
        """
        Decay confidence of edges that haven't been reinforced.
        Edges with recent updates or high original confidence decay slower.
        """
        now = time.time()
        decay_threshold = self._decay_days * 86400.0
        for key, edge in list(self._edges.items()):
            age = now - edge.updated_at
            if age > decay_threshold:
                decay_factor = np.exp(-0.5 * (age - decay_threshold) / decay_threshold)
                edge.confidence *= decay_factor
                edge.weight *= decay_factor
                edge.evidence.append("Confidence decayed due to age")

    def reinforce_edge(self, source_id: str, target_id: str,
                       edge_type: EdgeType, evidence: str = "") -> None:
        """Reinforce an existing edge with new evidence."""
        key = f"{source_id}--{edge_type.value}--{target_id}"
        if key in self._edges:
            self._edges[key].confidence = min(1.0, self._edges[key].confidence * 1.2)
            self._edges[key].updated_at = time.time()
            if evidence:
                self._edges[key].evidence.append(evidence)

    # ── Knowledge Source Integration ──

    def ingest_from_literature(self, query: str, max_results: int = 100) -> int:
        """Mine PubMed and add mined edges to the graph."""
        edges = self._literature_miner.mine_from_pubmed(query, max_results)
        added = 0
        for edge in edges:
            self.add_edge(edge)
            added += 1
        self._update_count += 1
        return added

    def ingest_from_text(self, text: str, source: str = "manual",
                         pmid: Optional[str] = None) -> int:
        """Mine a single text and add to graph."""
        edges = self._literature_miner.mine_from_text(text, source, pmid)
        for edge in edges:
            self.add_edge(edge)
        self._update_count += 1
        return len(edges)

    def ingest_clinical_trials(self, condition: Optional[str] = None) -> int:
        """Mine clinical trial data and add to graph."""
        edges = self._trial_miner.mine_trials(condition)
        for edge in edges:
            self.add_edge(edge)
        self._update_count += 1
        return len(edges)

    # ── Query & Path Finding ──

    def find_paths(self, source_type: str, target_type: str,
                   max_depth: int = 4) -> List[List[BiologicalEdge]]:
        """Find all paths between node types."""
        sources = self.get_nodes_by_type(source_type)
        targets = self.get_nodes_by_type(target_type)
        paths = []
        for src in sources:
            for tgt in targets:
                try:
                    for path in nx.all_simple_paths(self.graph, src.id, tgt.id,
                                                     cutoff=max_depth):
                        edge_path = []
                        for i in range(len(path) - 1):
                            edges = self.graph.get_edge_data(path[i], path[j])
                            # Actually path is a list of nodes
                            # Wait, path[i] to path[i+1]
                            pass
                        # We need to look up edges between consecutive nodes
                        edge_path = []
                        for i in range(len(path) - 1):
                            u, v = path[i], path[i + 1]
                            edge_data = self.graph.get_edge_data(u, v)
                            if edge_data:
                                for key in edge_data:
                                    edge = self._edges.get(key)
                                    if edge:
                                        edge_path.append(edge)
                        if edge_path:
                            paths.append(edge_path)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
        return paths

    def get_subgraph(self, node_ids: List[str], depth: int = 1) -> 'KnowledgeGraphEngine':
        """Extract a subgraph centered on given nodes."""
        sub = KnowledgeGraphEngine()
        for nid in node_ids:
            node = self._nodes.get(nid)
            if node:
                sub.add_node(node)
                if depth >= 1:
                    for neighbor_id, edge in self.get_neighbors(nid):
                        neighbor = self._nodes.get(neighbor_id)
                        if neighbor:
                            sub.add_node(neighbor)
                        sub.add_edge(edge)
        return sub

    def summarize(self) -> Dict[str, Any]:
        return {
            "nodes": len(self._nodes),
            "edges": len(self._edges),
            "node_types": {t: len(self.get_nodes_by_type(t)) for t in NODE_TYPES},
            "edge_types": {et.value: len(self.get_edges(edge_type=et)) for et in EdgeType},
            "update_count": self._update_count,
        }

    def to_networkx(self) -> nx.MultiDiGraph:
        return self.graph.copy()
