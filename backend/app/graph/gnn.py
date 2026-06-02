"""
GraphSAGE-based patient encoder.

Architecture:
    x ∈ R^{N x F}
        │
   SAGEConv(F -> H)         (sample + aggregate over 1-hop)
        │
   SAGEConv(H -> H)
        │
   SAGEConv(H -> D)         (output D-dim embedding per node)
        │
   patient head              (linear -> sigmoid -> risk ∈ [0,1])
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from app.core.config import settings
from app.graph.builder import CohortGraph


class PatientGNN(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden, aggr="mean")
        self.conv2 = SAGEConv(hidden, hidden, aggr="mean")
        self.conv3 = SAGEConv(hidden, out_dim, aggr="mean")
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x, edge_index))
        h = self.dropout(h)
        h = F.relu(self.conv2(h, edge_index))
        h = self.dropout(h)
        h = self.conv3(h, edge_index)
        return h


@dataclass
class GNNBundle:
    model: PatientGNN
    graph: CohortGraph
    in_dim: int
    out_dim: int


class GNNService:
    """Owns the trained model and the cohort graph it operates on."""

    def __init__(self) -> None:
        self.bundle: GNNBundle | None = None
        self.ckpt_path = os.path.join(settings.MODEL_DIR, "patient_gnn.pt")

    def is_loaded(self) -> bool:
        return self.bundle is not None

    def attach_graph(self, graph: CohortGraph) -> None:
        in_dim = graph.x.shape[1]
        out_dim = settings.GNN_EMBEDDING_DIM
        model = PatientGNN(in_dim, settings.GNN_HIDDEN, out_dim)
        self.bundle = GNNBundle(model=model, graph=graph,
                                in_dim=in_dim, out_dim=out_dim)

    def train(self, epochs: int | None = None, lr: float | None = None) -> dict:
        """Self-supervised reconstruction of patient features from the graph."""
        if self.bundle is None:
            raise RuntimeError("Graph not attached. Call attach_graph() first.")

        epochs = epochs or settings.GNN_EPOCHS
        lr = lr or settings.GNN_LR

        graph = self.bundle.graph
        model = self.bundle.model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        x = graph.x.to(device)
        edge_index = graph.edge_index.to(device)

        # patient node indices
        patient_idx = [i for i, k in enumerate(graph.node_kind) if k == "patient"]
        patient_idx_t = torch.tensor(patient_idx, dtype=torch.long, device=device)

        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        last_loss = float("nan")
        for ep in range(1, epochs + 1):
            model.train()
            opt.zero_grad()
            h = model(x, edge_index)
            # predict own features (denoising auto-encoder style)
            pred = h[patient_idx_t]
            target = x[patient_idx_t, : self.bundle.out_dim].detach()  # use first out_dim cols
            if target.shape[1] != pred.shape[1]:
                # if out_dim > patient feature dim, project
                target = target[:, : pred.shape[1]] if target.shape[1] >= pred.shape[1] \
                    else F.pad(target, (0, pred.shape[1] - target.shape[1]))
            loss = loss_fn(pred, target)
            loss.backward()
            opt.step()
            last_loss = float(loss.item())

        torch.save({
            "state_dict": model.state_dict(),
            "in_dim": self.bundle.in_dim,
            "out_dim": self.bundle.out_dim,
        }, self.ckpt_path)

        return {"epochs": epochs, "final_loss": last_loss, "device": str(device),
                "ckpt": self.ckpt_path}

    def load_if_exists(self) -> bool:
        if not os.path.exists(self.ckpt_path):
            return False
        ckpt = torch.load(self.ckpt_path, map_location="cpu")
        model = PatientGNN(ckpt["in_dim"], settings.GNN_HIDDEN, ckpt["out_dim"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        if self.bundle is None:
            return False
        self.bundle.model = model
        return True

    @torch.no_grad()
    def embed_patients(self) -> dict[str, list[float]]:
        if self.bundle is None:
            return {}
        model = self.bundle.model.eval()
        x = self.bundle.graph.x
        edge_index = self.bundle.graph.edge_index
        h = model(x, edge_index).cpu().numpy()
        out: dict[str, list[float]] = {}
        for pid, idx in self.bundle.graph.patient_id_to_node.items():
            out[pid] = h[idx].tolist()
        return out

    @torch.no_grad()
    def predict_missing_biomarker(self, patient_id: str, biomarker: str) -> dict:
        """Zero out the patient's biomarker feature and re-encode.
        The delta in the embedding gives a learned prior for the missing value."""
        if self.bundle is None or patient_id not in self.bundle.graph.patient_id_to_node:
            return {"error": "patient not found"}
        graph = self.bundle.graph
        model = self.bundle.model.eval()
        x_orig = graph.x.clone()
        bidx = graph.node_ids.index(biomarker)
        x_masked = x_orig.clone()
        x_masked[bidx, 5:] = 0.0  # zero biomarker features
        h_full = model(x_orig, graph.edge_index)
        h_mask = model(x_masked, graph.edge_index)
        pidx = graph.patient_id_to_node[patient_id]
        delta = float((h_full[pidx] - h_mask[pidx]).norm().item())
        # map delta -> predicted normalized value (heuristic)
        node = next((b for b in __import__("app.graph.ontology", fromlist=["BIOMARKERS"]).BIOMARKERS
                     if b.id == biomarker), None)
        if node is None:
            return {"error": "biomarker unknown"}
        observed = float(x_orig[bidx, 5])
        predicted_norm = float(np_clip(observed + delta * 0.1, -1.0, 1.0))
        predicted = predicted_norm * (node.healthy_hi - node.healthy_lo) / 2 + \
                    (node.healthy_hi + node.healthy_lo) / 2
        return {
            "patient_id": patient_id,
            "biomarker": biomarker,
            "predicted": round(predicted, 2),
            "observed": round(observed * (node.healthy_hi - node.healthy_lo) / 2 +
                              (node.healthy_hi + node.healthy_lo) / 2, 2),
            "delta_norm": round(delta, 4),
            "confidence": round(max(0.0, 1.0 - delta), 3),
        }


def np_clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


gnn_service = GNNService()
