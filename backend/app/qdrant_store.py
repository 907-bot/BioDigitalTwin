"""
Qdrant Vector Database Integration — Bio-Digital Twin
Stores and queries 64-dimensional / 128-dimensional patient embeddings and clinical evidence vectors.
Supports live Qdrant server connection (qdrant:6333) with seamless in-memory fallback.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

QDRANT_HOST = os.environ.get("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_URL = os.environ.get("QDRANT_URL", f"http://{QDRANT_HOST}:{QDRANT_PORT}")
COLLECTION_NAME = "patient_embeddings"
EMBEDDING_DIM = 64


class InMemQdrantStore:
    """In-memory fallback vector store implementing Qdrant-compatible API."""

    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim
        self.vectors: Dict[str, np.ndarray] = {}
        self.payloads: Dict[str, Dict[str, Any]] = {}

    def upsert(self, patient_id: str, vector: List[float], payload: Dict[str, Any]):
        vec = np.array(vector, dtype=np.float32)
        if len(vec) != self.dim:
            # Pad or truncate to expected dimension
            if len(vec) < self.dim:
                vec = np.pad(vec, (0, self.dim - len(vec)))
            else:
                vec = vec[: self.dim]
        # Normalize for cosine similarity
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        self.vectors[patient_id] = vec
        self.payloads[patient_id] = payload

    def search(self, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        if not self.vectors:
            return []
        q = np.array(query_vector, dtype=np.float32)
        if len(q) != self.dim:
            if len(q) < self.dim:
                q = np.pad(q, (0, self.dim - len(q)))
            else:
                q = q[: self.dim]
        norm = np.linalg.norm(q)
        if norm > 1e-6:
            q = q / norm

        results = []
        for pid, vec in self.vectors.items():
            sim = float(np.dot(q, vec))
            results.append({
                "id": pid,
                "score": round(sim, 4),
                "payload": self.payloads.get(pid, {}),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def count(self) -> int:
        return len(self.vectors)


class QdrantPatientStore:
    """
    Qdrant vector store manager.
    Connects to HTTP Qdrant REST/gRPC API if reachable; falls back cleanly to InMemQdrantStore.
    """

    def __init__(self):
        self.client = None
        self.use_fallback = True
        self.fallback_store = InMemQdrantStore(dim=EMBEDDING_DIM)
        self._init_client()

    def _init_client(self):
        try:
            import qdrant_client
            from qdrant_client.http import models as rest_models

            # Try connecting to Qdrant server
            client = qdrant_client.QdrantClient(url=QDRANT_URL, timeout=3.0)
            # Check server connection
            client.get_collections()
            self.client = client
            self.use_fallback = False
            self._ensure_collection(rest_models)
            logger.info(f"Connected to Qdrant server at {QDRANT_URL}")
        except Exception as e:
            logger.info(f"Qdrant server unreachable ({e}); operating in in-memory vector mode")
            self.use_fallback = True

    def _ensure_collection(self, rest_models):
        try:
            cols = [c.name for c in self.client.get_collections().collections]
            if COLLECTION_NAME not in cols:
                self.client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=rest_models.VectorParams(
                        size=EMBEDDING_DIM,
                        distance=rest_models.Distance.COSINE,
                    ),
                )
                logger.info(f"Created Qdrant collection: '{COLLECTION_NAME}'")
        except Exception as e:
            logger.warning(f"Error ensuring collection: {e}")

    def upsert_patient_embedding(
        self,
        patient_id: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Upsert a 64-dim patient embedding vector into Qdrant.
        """
        payload = metadata or {}
        payload["patient_id"] = patient_id

        if self.use_fallback:
            self.fallback_store.upsert(patient_id, embedding, payload)
            return {
                "status": "success",
                "storage": "in_memory_qdrant",
                "patient_id": patient_id,
                "dim": len(embedding),
                "total_vectors": self.fallback_store.count(),
            }

        try:
            from qdrant_client.http import models as rest_models

            # Ensure embedding length matches EMBEDDING_DIM
            vec = list(embedding)
            if len(vec) < EMBEDDING_DIM:
                vec.extend([0.0] * (EMBEDDING_DIM - len(vec)))
            else:
                vec = vec[:EMBEDDING_DIM]

            # Use integer hash or str id
            import zlib
            point_id = zlib.crc32(patient_id.encode("utf-8"))

            self.client.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    rest_models.PointStruct(
                        id=point_id,
                        vector=vec,
                        payload=payload,
                    )
                ],
            )
            return {
                "status": "success",
                "storage": "qdrant_server",
                "patient_id": patient_id,
                "vector_id": point_id,
                "dim": len(vec),
            }
        except Exception as e:
            logger.error(f"Qdrant server upsert failed: {e}")
            self.fallback_store.upsert(patient_id, embedding, payload)
            return {
                "status": "success_fallback",
                "storage": "in_memory_qdrant",
                "patient_id": patient_id,
            }

    def search_similar_patients(
        self, query_embedding: List[float], limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for top-k similar patients by embedding vector using Cosine Similarity.
        """
        if self.use_fallback:
            return self.fallback_store.search(query_embedding, limit=limit)

        try:
            vec = list(query_embedding)
            if len(vec) < EMBEDDING_DIM:
                vec.extend([0.0] * (EMBEDDING_DIM - len(vec)))
            else:
                vec = vec[:EMBEDDING_DIM]

            res = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=vec,
                limit=limit,
            )
            return [
                {
                    "id": p.payload.get("patient_id", str(p.id)),
                    "score": round(float(p.score), 4),
                    "payload": p.payload,
                }
                for p in res
            ]
        except Exception as e:
            logger.error(f"Qdrant search error: {e}")
            return self.fallback_store.search(query_embedding, limit=limit)

    def get_stats(self) -> Dict[str, Any]:
        """Return store metadata."""
        if self.use_fallback:
            return {
                "engine": "In-Memory Qdrant Vector Store",
                "collection": COLLECTION_NAME,
                "total_vectors": self.fallback_store.count(),
                "embedding_dim": EMBEDDING_DIM,
                "qdrant_url": QDRANT_URL,
                "server_connected": False,
            }

        try:
            info = self.client.get_collection(COLLECTION_NAME)
            return {
                "engine": "Qdrant Vector Database Server",
                "collection": COLLECTION_NAME,
                "total_vectors": info.points_count,
                "embedding_dim": EMBEDDING_DIM,
                "qdrant_url": QDRANT_URL,
                "server_connected": True,
                "status": str(info.status),
            }
        except Exception as e:
            return {
                "engine": "Qdrant Vector Database",
                "server_connected": False,
                "error": str(e),
            }


# Singleton instance
_qdrant_store: Optional[QdrantPatientStore] = None

def get_qdrant_store() -> QdrantPatientStore:
    global _qdrant_store
    if _qdrant_store is None:
        _qdrant_store = QdrantPatientStore()
    return _qdrant_store
