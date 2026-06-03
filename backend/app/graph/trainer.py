import os
import json
import numpy as np
import pandas as pd
from app.graph.builder import _BUILDER, load_patients
from app.graph.gnn import gnn_service

EMBEDDING_FILE = "data/patient_embeddings.json"

def train_gnn(epochs: int = 100, threshold: float = 0.80, encoder_type: str = "gcn", lr: float = 1e-3) -> dict:
    # Load patients
    df = load_patients()
    
    # Build the cohort graph
    graph = _BUILDER.build(df)
    
    # Attach graph to GNN service and train
    gnn_service.attach_graph(graph)
    stats = gnn_service.train(epochs=epochs, lr=lr)
    
    # Generate and save embeddings
    embeddings_dict = gnn_service.embed_patients()
    os.makedirs("data", exist_ok=True)
    with open(EMBEDDING_FILE, "w") as f:
        json.dump(embeddings_dict, f)
        
    return stats

def load_embeddings() -> tuple[np.ndarray, dict[str, int]]:
    if not os.path.exists(EMBEDDING_FILE):
        # If not trained yet, let's train it with default parameters
        train_gnn(epochs=50)
    
    with open(EMBEDDING_FILE, "r") as f:
        embeddings_dict = json.load(f)
        
    patient_ids = list(embeddings_dict.keys())
    embeddings = np.array([embeddings_dict[pid] for pid in patient_ids], dtype=np.float32)
    id_index = {pid: idx for idx, pid in enumerate(patient_ids)}
    return embeddings, id_index

def get_embedding(patient_id: str) -> list[float]:
    if not os.path.exists(EMBEDDING_FILE):
        train_gnn(epochs=50)
        
    with open(EMBEDDING_FILE, "r") as f:
        embeddings_dict = json.load(f)
    if patient_id not in embeddings_dict:
        raise KeyError(f"Patient '{patient_id}' not found in embeddings.")
    return embeddings_dict[patient_id]

def get_top_k_similar_by_embedding(patient_id: str, k: int = 10) -> list[tuple[str, float]]:
    embeddings, id_index = load_embeddings()
    if patient_id not in id_index:
        raise KeyError(f"Patient '{patient_id}' not found in GNN embeddings.")
        
    target_idx = id_index[patient_id]
    target_emb = embeddings[target_idx]
    
    # Compute cosine similarities
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    norm_embeddings = embeddings / norms
    
    target_norm = norm_embeddings[target_idx]
    similarities = np.dot(norm_embeddings, target_norm)
    
    # Sort indices by similarity descending
    sorted_indices = np.argsort(similarities)[::-1]
    
    results = []
    for idx in sorted_indices:
        if idx == target_idx:
            continue
        pid = list(id_index.keys())[idx]
        sim = float(similarities[idx])
        results.append((pid, sim))
        if len(results) >= k:
            break
    return results
