"""
Phase 4: Molecular Twin.

Uses PyTorch Geometric (GNN) over biological pathway graphs
for molecular interaction prediction, and RDKit for drug
structure processing. Links molecular events to cellular outcomes.

Architecture:
  PathwayGraph (networkx) → PyG Data → GNN → PathwayActivation → CellSignal
  Drug (RDKit Mol) → MorganFP → DrugTargetPredictor → TargetInhibition → PathwayEffect
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

try:
    import torch_geometric as pyg
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, RDKFingerprint
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


# ── Molecular State ──────────────────────────────────────────

@dataclass
class MolecularState:
    """
    Molecular-level state of the twin.

    Gene expression (normalized log2-fold from baseline):
      - 50 core metabolic/inflammatory genes
    Protein activity (0-1 normalized):
      - Key signaling proteins (IRS-1, Akt, AMPK, JNK, NF-κB, etc.)
    Metabolite levels (normalized):
      - Glucose, FFA, lactate, ketones, amino acids
    Pathway activation (0-1):
      - Insulin signaling, gluconeogenesis, lipolysis, inflammation, oxidative stress
    """
    gene_expression: np.ndarray         # shape (n_genes,)
    protein_activity: np.ndarray        # shape (n_proteins,)
    metabolite_levels: np.ndarray       # shape (n_metabolites,)
    pathway_activation: np.ndarray      # shape (n_pathways,)
    drug_target_binding: np.ndarray     # shape (n_drugs,) — current drug occupancy

    @property
    def dim(self) -> int:
        return (len(self.gene_expression) + len(self.protein_activity) +
                len(self.metabolite_levels) + len(self.pathway_activation) +
                len(self.drug_target_binding))

    def to_array(self) -> np.ndarray:
        return np.concatenate([
            self.gene_expression, self.protein_activity,
            self.metabolite_levels, self.pathway_activation,
            self.drug_target_binding,
        ])

    @classmethod
    def from_array(cls, arr: np.ndarray, dims: Tuple[int, ...]) -> 'MolecularState':
        offset = 0
        ge = arr[offset:offset+dims[0]]; offset += dims[0]
        pa = arr[offset:offset+dims[1]]; offset += dims[1]
        ml = arr[offset:offset+dims[2]]; offset += dims[2]
        pwa = arr[offset:offset+dims[3]]; offset += dims[3]
        dt = arr[offset:offset+dims[4]]
        return cls(ge, pa, ml, pwa, dt)

    @classmethod
    def healthy_resting(cls) -> 'MolecularState':
        return cls(
            gene_expression=np.zeros(50),
            protein_activity=np.array([
                0.7, 0.6, 0.3, 0.2, 0.15,  # IRS-1, Akt, AMPK, JNK, NF-κB
                0.5, 0.3, 0.4, 0.6, 0.2,  # GLUT4, SREBP1, PPARg, PGC1a, FOXO1
            ]),
            metabolite_levels=np.array([0.5, 0.3, 0.2, 0.1, 0.1]),  # G, FFA, lactate, ketones, AA
            pathway_activation=np.array([
                0.3, 0.2, 0.15, 0.2, 0.2, 0.1, 0.05,
                # insulin, gluco neo, lipolysis, inflam, ox stress, mito, autophagy
            ]),
            drug_target_binding=np.zeros(3),
        )


# Dimension constants
N_GENES = 50
N_PROTEINS = 10
N_METABOLITES = 5
N_PATHWAYS = 7
N_DRUGS = 3
MOLECULAR_DIM = N_GENES + N_PROTEINS + N_METABOLITES + N_PATHWAYS + N_DRUGS


# ── Pathway Knowledge Graph ───────────────────────────────────

PROTEIN_NAMES = [
    "IRS1", "AKT", "AMPK", "JNK", "NFKB",
    "GLUT4", "SREBP1", "PPARG", "PGC1A", "FOXO1",
]

PATHWAY_NAMES = [
    "insulin_signaling", "gluconeogenesis", "lipolysis",
    "inflammation", "oxidative_stress", "mitochondrial", "autophagy",
]

# Built-in pathway graph: proteins → pathways → cellular outcomes
PATHWAY_EDGES = [
    ("IRS1", "insulin_signaling", "activates"),
    ("AKT", "insulin_signaling", "activates"),
    ("AKT", "gluconeogenesis", "inhibits"),
    ("AMPK", "mitochondrial", "activates"),
    ("AMPK", "lipolysis", "activates"),
    ("JNK", "inflammation", "activates"),
    ("NFKB", "inflammation", "activates"),
    ("NFKB", "oxidative_stress", "activates"),
    ("FOXO1", "gluconeogenesis", "activates"),
    ("FOXO1", "autophagy", "activates"),
    ("PPARG", "lipolysis", "inhibits"),
    ("SREBP1", "lipolysis", "activates"),
    ("PGC1A", "mitochondrial", "activates"),
    ("insulin_signaling", "AKT", "activates"),  # feedback
    ("inflammation", "JNK", "activates"),         # feedforward
    ("oxidative_stress", "NFKB", "activates"),
]


def build_pathway_graph() -> nx.DiGraph:
    """Build the molecular pathway knowledge graph."""
    G = nx.DiGraph()
    all_nodes = set()
    for src, dst, rel in PATHWAY_EDGES:
        G.add_edge(src, dst, relation=rel)
        all_nodes.add(src)
        all_nodes.add(dst)
    # Add isolated nodes
    for name in PROTEIN_NAMES + PATHWAY_NAMES:
        if name not in G:
            G.add_node(name)
    return G


class PathwayGNN(nn.Module):
    """
    Graph Neural Network over the pathway knowledge graph.
    Predicts pathway activation from protein activity and gene expression.

    Architecture: GCN → GAT → global_readout → MLP
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_pathways: int = N_PATHWAYS):
        super().__init__()
        if not HAS_PYG:
            raise ImportError("PyTorch Geometric required for PathwayGNN")

        self.node_embed = nn.Linear(input_dim, hidden_dim)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        self.pathway_head = nn.Linear(hidden_dim, num_pathways)
        self.dropout = nn.Dropout(0.2)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        x = self.node_embed(x)
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index))
        # Pathway nodes are last num_pathways nodes
        pathway_idx = torch.tensor(
            list(range(data.num_nodes - N_PATHWAYS, data.num_nodes)),
            dtype=torch.long, device=x.device,
        )
        pathway_features = x[pathway_idx]
        return torch.sigmoid(self.pathway_head(pathway_features))


# ── Drug-Target Model ─────────────────────────────────────────

class DrugTargetPredictor:
    """
    Predict drug-target interactions using Morgan fingerprints
    and a learned affinity model. Wraps RDKit for featurization.

    In production, this would be a trained GNN or transformer
    on ChEMBL/DrugBank data. Here we use a similarity-based
    approach with known target profiles.
    """

    # Known drug-target profiles (protein_index → Ki_inhibition)
    KNOWN_DRUG_TARGETS: Dict[str, Dict[int, float]] = {
        "metformin": {
            0: 0.0,   # IRS1 — no direct effect
            3: -0.3,  # JNK — mild inhibition via AMPK
            4: -0.2,  # NFKB — mild inhibition
            5: 0.3,   # GLUT4 — increased translocation
            6: -0.2,  # SREBP1 — reduced lipogenesis
        },
        "atorvastatin": {
            6: -0.8,  # SREBP1 — strong inhibition (HMGCR)
            9: 0.1,   # FOXO1 — mild effect
        },
        "empagliflozin": {
            5: 0.4,   # GLUT4 — increased (indirect via SGLT2)
            1: 0.1,   # AKT — mild increase
        },
    }

    def __init__(self):
        if not HAS_RDKIT:
            raise ImportError("RDKit required for DrugTargetPredictor")

    def smiles_to_fingerprint(self, smiles: str) -> np.ndarray:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(2048)
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))

    def predict_target_profile(self, drug_name: str) -> np.ndarray:
        """Return predicted target inhibition profile (n_proteins,)."""
        profile = np.zeros(N_PROTEINS)
        targets = self.KNOWN_DRUG_TARGETS.get(drug_name, {})
        for protein_idx, effect in targets.items():
            if 0 <= protein_idx < N_PROTEINS:
                profile[protein_idx] = effect
        return profile

    def apply_drug_effect(
        self, mol_state: MolecularState, drug_name: str, dose_fraction: float = 1.0,
    ) -> MolecularState:
        """Apply drug effect to molecular state."""
        profile = self.predict_target_profile(drug_name) * dose_fraction
        new_proteins = np.clip(mol_state.protein_activity + profile, 0.0, 1.0)
        # Recompute pathway activation from modified proteins
        new_pathways = self._estimate_pathways(new_proteins)

        drug_idx = {"metformin": 0, "atorvastatin": 1, "empagliflozin": 2}.get(drug_name, -1)
        new_drug_binding = mol_state.drug_target_binding.copy()
        if drug_idx >= 0:
            new_drug_binding[drug_idx] = dose_fraction

        return MolecularState(
            gene_expression=mol_state.gene_expression,
            protein_activity=new_proteins,
            metabolite_levels=mol_state.metabolite_levels,
            pathway_activation=new_pathways,
            drug_target_binding=new_drug_binding,
        )

    def _estimate_pathways(self, proteins: np.ndarray) -> np.ndarray:
        """Simple linear estimate of pathway activation from proteins."""
        pathways = np.zeros(N_PATHWAYS)
        # Insulin signaling: IRS1 + AKT
        pathways[0] = np.clip(0.6 * proteins[0] + 0.6 * proteins[1], 0, 1)
        # Gluconeogenesis: FOXO1 - AKT inhibition
        pathways[1] = np.clip(0.7 * proteins[9] - 0.4 * proteins[1], 0, 1)
        # Lipolysis: AMPK + SREBP1 - PPARG
        pathways[2] = np.clip(0.4 * proteins[2] + 0.5 * proteins[6] - 0.4 * proteins[7], 0, 1)
        # Inflammation: JNK + NFKB
        pathways[3] = np.clip(0.5 * proteins[3] + 0.5 * proteins[4], 0, 1)
        # Oxidative stress: NFKB
        pathways[4] = np.clip(0.6 * proteins[4], 0, 1)
        # Mitochondrial: AMPK + PGC1A
        pathways[5] = np.clip(0.5 * proteins[2] + 0.5 * proteins[8], 0, 1)
        # Autophagy: FOXO1
        pathways[6] = np.clip(0.5 * proteins[9], 0, 1)
        return pathways


# ── Molecular Dynamics ────────────────────────────────────────

def compute_molecular_dynamics(
    mol_state: MolecularState,
    cellular_stress: float = 0.0,
    drug_inputs: Dict[str, float] = None,
    dt: float = 1.0,
) -> MolecularState:
    """
    Molecular-level dynamics: gene expression regulation,
    protein activity changes, metabolite fluxes.

    Args:
        mol_state: Current molecular state
        cellular_stress: Stress signal from cellular layer (0-1)
        drug_inputs: Drug doses {name: dose_fraction}
        dt: Time step in minutes

    Returns:
        Updated molecular state
    """
    if drug_inputs is None:
        drug_inputs = {}

    predictor = DrugTargetPredictor()
    new_state = mol_state

    # Apply drugs
    for drug_name, dose in drug_inputs.items():
        if dose > 0:
            new_state = predictor.apply_drug_effect(new_state, drug_name, dose)

    # Protein dynamics: relaxation toward target with stress modulation
    target_proteins = np.array([0.7, 0.6, 0.3, 0.2, 0.15, 0.5, 0.3, 0.4, 0.6, 0.2])
    stress_effect = cellular_stress * np.array([
        0.0, 0.0, 0.2, 0.5, 0.4,   # JNK↑, NFKB↑ under stress
        -0.2, 0.0, 0.0, -0.1, 0.2,  # GLUT4↓, FOXO1↑ under stress
    ])
    target_proteins = np.clip(target_proteins + stress_effect, 0.0, 1.0)

    tau = 120.0  # protein turnover time constant (min)
    new_proteins = new_state.protein_activity + (target_proteins - new_state.protein_activity) / tau * dt

    # Gene expression: slow dynamics (hours)
    tau_gene = 360.0
    new_genes = new_state.gene_expression + (-0.1 * new_state.gene_expression + cellular_stress * 0.2) / tau_gene * dt

    # Metabolite dynamics: fast
    new_metabolites = new_state.metabolite_levels.copy()

    # Recompute pathway activation from new protein state
    new_pathways = predictor._estimate_pathways(new_proteins)

    return MolecularState(
        gene_expression=np.clip(new_genes, -2, 2),
        protein_activity=np.clip(new_proteins, 0, 1),
        metabolite_levels=np.clip(new_metabolites, 0, 1),
        pathway_activation=np.clip(new_pathways, 0, 1),
        drug_target_binding=new_state.drug_target_binding,
    )


# ── Molecular → Cellular Interface ────────────────────────────

def molecular_to_cellular_signals(
    mol_state: MolecularState,
) -> Dict[str, float]:
    """
    Map molecular state to cellular-level signals.

    Returns:
        insulin_signal: 0-1 (strength of insulin pathway)
        inflammatory_signal: 0-1 (NFKB + JNK activity)
        metabolic_stress: 0-1 (oxidative stress + mitochondrial dysfunction)
        growth_signals: 0-1 (anabolic pathway activation)
    """
    pw = mol_state.pathway_activation
    return {
        "insulin_signal": float(pw[0]),
        "inflammatory_signal": float(np.clip(pw[3] + pw[4] * 0.5, 0, 1)),
        "metabolic_stress": float(np.clip(pw[4] + (1 - pw[5]) * 0.5, 0, 1)),
        "growth_signals": float(np.clip(pw[0] + pw[5], 0, 1)),
    }
