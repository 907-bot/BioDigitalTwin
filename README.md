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
|17 | Multi-Organ World Model (Lungs, Liver, GI, ANS) | `/physiology` (multi-organ) |

Each phase is an independent FastAPI router that returns structured JSON; every response also includes a two‑tier **narrative** block (`headline`, `lay`, `scientist`, `risk_level`) that explains the result in plain English and for scientists.

The Next.js frontend mirrors the phases as pages (`/cohort`, `/simulate`, `/causal`, `/chat`, `/pharmacogenomics`, …, `/explain`) with a persistent left‑sidebar that lets you jump between them.

---

## 🔬 Scientific Foundation & Publication Framework

> **Primary Falsifiable Research Question**:  
> *"Does estimating a small, demonstrably identifiable set of patient-specific parameters improve short-horizon glucose-risk prediction over simpler baselines in a defined population?"*

### Auditable Pre-Clinical Boundaries
- **36-State Physiological Architecture**: Authoritative 36-state and 29-parameter space across metabolic, cardiovascular, renal, circadian, adipose, immune, and musculoskeletal systems (`backend/app/personalization/state.py`).
- **Parameter Identifiability**: Structural and practical identifiability diagnostics (`backend/app/personalization/identifiability.py`) prove that simultaneously estimating all 29 parameters from sparse observations causes severe ill-conditioning. Active UKF estimation isolates 7 demonstrably identifiable parameters ($S_I, \text{EGP}_0, \beta_{cell}, R_{thresh}, G_{baro}, \text{GFR}_{base}, A_{circ}$), anchoring the remaining 22 to population priors.
- **Multi-Organ World Model**: Coupled convective circulation (`BloodTransportLayer`) coordinating dedicated organ modules for pulmonary gas exchange (`RespiratoryOrganModel`), glycogen/HGP turnover (`HepaticOrganModel`), SGLT1/incretin GI absorption (`GastrointestinalOrganModel`), and baroreflex/HPA/RAAS axis (`AutonomicEndocrineModel`), validated to ASME V&V 40 standards (`/physiology`).
- **Baseline Benchmarking**: Short-horizon (30-min and 60-min) hypoglycemia risk ($G < 70\text{ mg/dL}$) predictions are benchmarked against naive persistence ($G_{t+\Delta t} = G_t$) and autoregressive (AR3) models (`backend/app/personalization/benchmark_baselines.py`), measuring event sensitivity, false alerts per patient-day, and lead time.
- **4-Source Uncertainty Quantification**: Every trajectory rigorously quantifies aleatoric, epistemic, numerical solver, and structural model discrepancy (`backend/app/personalization/uncertainty.py`).
- **Dataset Boundary**: MIMIC-IV provides inpatient/ICU and emergency-department validation. Outpatient CGM forecasting is evaluated on independent continuous ambulatory monitoring traces.
- **Hypothesis Separation**: Counterfactual treatment recommendations and long-term complication reduction claims are separated into subsequent prospective interventional studies.

### Auditable Benchmarking Commands
```bash
# 1. Run Structural & Practical Identifiability Audit
backend/.venv/bin/python backend/app/personalization/identifiability.py

# 2. Run Short-Horizon Baseline Benchmarks (Persistence vs AR3 vs Twin)
backend/.venv/bin/python backend/app/personalization/benchmark_baselines.py
```

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
