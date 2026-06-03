"""Phase 16 — Explainable AI (XAI).

Layered explanations on top of the SCM, GNN, PK/PD, and PGx stack.

Endpoints
---------
POST /phase16/explain/counterfactual  - reason about a patient counterfactual
POST /phase16/explain/ddi              - explain a DDI
POST /phase16/explain/pk               - explain PK/PD in patient terms
POST /phase16/explain/pgx              - explain a pharmacogenomic warning
POST /phase16/explain/patient          - holistic patient explanation (all layers)
GET  /phase16/methods                  - list explanation methods
"""
from .router import router

__all__ = ["router"]
