# Bio‑Digital Twin Platform

**A drug‑discovery and digital‑twin platform built with FastAPI, Next.js, and a suite of 16+ complementary phases.**

## Quick start (Docker)

```bash
# Build all service images
docker compose build

# Start the full stack (api, frontend, postgres, neo4j, redis, qdrant)
docker compose up -d

# View logs
docker compose logs -f api frontend

# Stop everything
docker compose down

# Rebuild after a code change
docker compose build api frontend && docker compose up -d api frontend
```

Once the stack is running:
- API docs → http://localhost:8000/docs
- Frontend → http://localhost:3000
- Neo4j browser → http://localhost:7474 (user: `neo4j`, pass: `password`)
- Postgres → `localhost:5432` (db: `biodigital`, user/pass: `postgres`)
- Redis → `localhost:6380`
- Qdrant → http://localhost:6333

Health check: `curl http://localhost:8000/health` → `{"status":"healthy","phase":"1+2+3+4+5+8+9+10+11+12+13+14+15+16"}`

---

## What is this?

The Bio‑Digital Twin project is a **modular, phase‑based platform** that walks a user through the drug‑discovery lifecycle:

| Phase | Feature | Endpoint prefix |
|-------|---------|-----------------|
| 1 | Synthetic Patient Generator | `/phase1` (core) |
| 2 | Graph Digital Twin (GNN) | `/phase2` (graph) |
| 3 | Disease Dynamics (ODE/LIF) | `/phase3` (dynamics) |
| 4 | Causal AI (SCM + ATE/CATE) | `/phase4` (causal) |
| 5 | LLM Agent (Ollama) | `/phase5` (agent) |
| 6 | Dashboard Aggregator | `/phase6` (api) |
| 8 | Pharmacogenomics | `/phase8` (pgx) |
| 9 | Polypharmacy / DDI | `/phase9` (ddi) |
|10 | PK/PD Simulator | `/phase10` (pkpd) |
|11 | Uncertainty Quantification | `/phase11` (uq) |
|12 | Clinical Trials search | `/phase12` (trials) |
|13 | Regulatory profile (FDA/FAERS/RxNorm) | `/phase13` (regulatory) |
|14 | Wet‑lab molecule triage | `/phase14` (wetlab) |
|15 | Disease Registry (Postgres CRUD) | `/phase15` (registry) |
|16 | Explainable AI (XAI) | `/phase16` (xai) |

Each phase is an independent FastAPI router that returns structured JSON; every response also includes a two‑tier **narrative** block (`headline`, `lay`, `scientist`, `risk_level`) that explains the result in plain English and for scientists.

The Next.js frontend mirrors the phases as pages (`/cohort`, `/simulate`, `/causal`, `/chat`, `/pharmacogenomics`, …, `/explain`) with a persistent left‑sidebar that lets you jump between them.

---

## Core concepts

- **Narrative layer** – Cross‑cutting `backend/app/narrative/*` modules turn raw numbers into plain‑English and scientist‑level explanations with a risk badge. The frontend `Narrative` component lets you toggle between the two views.
- **3‑D causal anatomy** – On the `/causal` page you can switch between a 2‑D radial DAG and a 3‑D view where each causal node (organ, biomarker, disease, age) is positioned on a stylized human body. Click a node to highlight its causal neighbours.
- **Extensible data** – Postgres holds the disease registry; Neo4j stores the similarity graph; Redis caches LLM agent sessions and API responses; Qdrant stores patient embeddings for fast similarity lookup.
- **Typical workflow** – Generate a cohort → explore similarity → simulate dynamics → ask causal “what‑if” questions → consult the LLM agent → run PK/PD, UQ, PGx, DDI checks → look up trials, regulatory info, wet‑lab validation → finally, get an XAI explanation that ties everything together.

---

## Development

### Prerequisites
- Docker ≥ 24.0
- (Optional) Python ≥ 3.11, Node ≥ 20 for local work without Docker

### Backend (FastAPI)
```bash
cd backend
# create a venv if you like
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# run with reload
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend (Next.js)
```bash
cd frontend
npm install
npm run dev   # starts at http://localhost:3000
```

### Testing
- Backend: `pytest` (run from repository root)
- Frontend: `npm run test` (if added)

---

## License

MIT – see the `LICENSE` file for details.
