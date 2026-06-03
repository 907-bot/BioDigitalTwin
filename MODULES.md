# Bio-Digital Twin — Module Reference

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

Once up:
- API     → http://localhost:8000   (Swagger docs at `/docs`)
- Frontend → http://localhost:3000
- Neo4j   → http://localhost:7474   (user: `neo4j`, pass: `password`)
- Postgres → `localhost:5432`        (db: `biodigital`, user/pass: `postgres`)
- Redis   → `localhost:6380`
- Qdrant  → http://localhost:6333

Health check:
```bash
curl http://localhost:8000/health
# {"status":"healthy","phase":"1+2+3+4+5+8+9+10+11+12+13+14+15+16"}
```

---

## Architecture overview

A multi-phase drug-discovery and digital-twin platform. The FastAPI backend
exposes 16 numbered phase routers, each owning one capability. The Next.js
frontend renders a corresponding page per phase, with a persistent left
sidebar that lists all of them.

```
bio-digital-twin/
├── backend/
│   ├── app/
│   │   ├── main.py            ← FastAPI entrypoint, mounts all routers
│   │   ├── core/              ← Phase 1: cohort generator
│   │   ├── graph/             ← Phase 2: graph digital twin (GNN + similarity)
│   │   ├── dynamics/          ← Phase 3: disease dynamics + LIF
│   │   ├── causal/            ← Phase 4: causal AI (SCM + ATE/CATE)
│   │   ├── agent/             ← Phase 5: LLM agent (Ollama)
│   │   ├── api/               ← Phase 6: dashboard aggregator
│   │   ├── pgx/               ← Phase 8: pharmacogenomics
│   │   ├── ddi/               ← Phase 9: drug-drug interactions
│   │   ├── pkpd/              ← Phase 10: PK/PD simulation
│   │   ├── uq/                ← Phase 11: uncertainty quantification
│   │   ├── trials/            ← Phase 12: clinical trials search
│   │   ├── regulatory/        ← Phase 13: FDA / FAERS / RxNorm
│   │   ├── wetlab/            ← Phase 14: wet-lab molecule triage
│   │   ├── registry/          ← Phase 15: disease registry (CRUD)
│   │   ├── xai/               ← Phase 16: explainable AI layer
│   │   └── narrative/         ← Cross-cutting two-tier (lay+scientist) text
│   ├── data/                  ← Seed CSVs, ontology, drug tables, registry seed
│   ├── tests/                 ← pytest suites
│   ├── notebooks/             ← Reference Jupyter notebooks
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── app/               ← Next.js App Router
│   │   │   ├── page.tsx               (home: 15-card grid)
│   │   │   ├── cohort/                (phase 1+2)
│   │   │   ├── simulate/              (phase 3)
│   │   │   ├── causal/                (phase 4, with 2D/3D toggle)
│   │   │   ├── chat/                  (phase 5)
│   │   │   ├── pharmacogenomics/      (phase 8)
│   │   │   ├── polypharmacy/          (phase 9)
│   │   │   ├── pkpd/                  (phase 10)
│   │   │   ├── uncertainty/           (phase 11)
│   │   │   ├── trials/                (phase 12)
│   │   │   ├── regulatory/            (phase 13)
│   │   │   ├── wetlab/                (phase 14)
│   │   │   ├── registry/              (phase 15)
│   │   │   └── explain/               (phase 16)
│   │   ├── components/        ← AppShell, Sidebar, Panels, Narrative, Causal3D…
│   │   └── lib/api.ts         ← Typed wrappers for all backend endpoints
│   └── Dockerfile
├── docker-compose.yml
└── data/                      ← Cross-stack seed data
```

API design: every endpoint returns a structured JSON object. Every object
in the latest revision also carries a top-level `narrative: {headline,
lay, scientist, risk_level}` block produced by the cross-cutting
`backend/app/narrative/` layer.

---

## Core platform (phases 1–6)

### Phase 1 — Synthetic Patient Generator  (`/phase1`, `core/`)

Generates a realistic cohort of synthetic patients. Each patient has
demographics, vital signs, lab values, comorbidities, current medications,
and a longitudinal visit history.

- `PatientGenerator` samples from joint distributions learned from public
  datasets (NHANES-style priors).
- Outputs deterministic, seedable CSVs in `data/synthetic_patients.csv`.
- `Patient` dataclass carries `id`, `age`, `sex`, `weight_kg`,
  `serum_creatinine_mg_dl`, comorbidities, allergies, and a per-gene
  metabolizer status (PM/IM/EM/UM).

UI: `/cohort` shows summary stats, distribution histograms, and lets you
generate new cohorts with custom size + seed.

### Phase 2 — Graph Digital Twin  (`/phase2`, `graph/`)

Encodes the cohort as a heterogeneous patient-similarity graph and learns
node embeddings with a GraphSAGE-style GNN.

- Patients, conditions, drugs, and proteins are node types.
- Edges: co-morbidity, co-prescription, lab-correlation, and ontology.
- `PatientSimilarityIndex` provides k-nearest-neighbour lookup over the
  learned embeddings (cosine over 128-dim vectors).
- Backed by Neo4j for the structural graph; embeddings stored in Qdrant.

UI: `/cohort` second tab — pick a patient, see their top-K similar
patients and a 2-D UMAP projection.

### Phase 3 — Disease Dynamics  (`/phase3`, `dynamics/`)

ODE + LIF (leaky integrate-and-fire) simulator for short-term biomarker
trajectories under a treatment.

- `simulate_dynamics(patient, treatment, dt, t_end)` returns per-biomarker
  trajectories (glucose, HR, BP, SpO₂) using compartmental ODEs.
- LIF neuron model used for the neural biomarkers (HRV, sleep-stage proxy).
- Surfaces organ-level state changes that feed into the causal graph.

UI: `/simulate` — time-series chart of every biomarker for the next 24 h.

### Phase 4 — Causal AI  (`/phase4`, `causal/`)

Structural causal model built on top of the biological ontology.

- DAG of 18 nodes: 6 organs (heart, lungs, liver, kidneys, pancreas,
  vasculature), 7 biomarkers (HR, HRV, SpO₂, glucose, SBP, DBP, BMI),
  4 diseases (T2D, hypertension, CVD, COPD), and `age`.
- `compute_ate(outcome, treatment)` and `compute_cate(outcome, treatment,
  conditions)` for population and conditional average treatment effects.
- Counterfactual engine answers "what would have happened to this patient
  if X were Y?" using do-calculus on the fitted SCM.

UI: `/causal` — 2-D radial DAG **or** 3-D anatomy-mapped view (see
"3D causal anatomy" below). Click any node to see its causal neighbours
highlighted.

### Phase 5 — LLM Agent  (`/phase5`, `agent/`)

An Ollama-backed agent (default model: `llama3.1`) that can call the
backend's own tools.

- ReAct-style loop: think → tool call → observe → answer.
- Tools: patient lookup, PGx check, DDI check, PK simulate, trial search,
  regulatory profile, wet-lab validate, registry search.
- Conversation history persisted in Redis; sessions keyed by
  `session_id`.
- Optional streaming token output.

UI: `/chat` — chat-style interface; agent responses include which tools
were called and the latency.

### Phase 6 — Dashboard Aggregator  (`/phase6`, `api/`)

Lightweight aggregator that bundles the most recent results from every
phase into a single dashboard payload. Used by the Next.js home page grid.

---

## Drug-discovery advancements (phases 8–15)

### Phase 8 — Pharmacogenomics  (`/phase8`, `pgx/`)

8-gene CYP / Phase-II panel. Maps patient metabolizer status to drug
exposure adjustments.

- Genes: `CYP2D6, CYP2C19, CYP2C9, CYP3A4, CYP3A5, CYP1A2, CYP2B6, DPYD, TPMT, UGT1A1`.
- Metabolizer states: PM (poor), IM (intermediate), EM (extensive), UM (ultra).
- Activity factor per (gene, status) modulates the impact on the drug
  effect — for prodrugs (codeine → morphine) low activity = no effect,
  for clearance drugs (warfarin) low activity = accumulation.
- Endpoints:
  - `POST /phase8/patients/pgx-check` — given a patient + drug list,
    returns warnings with severity, impact factor, and clinical note.
  - `POST /phase8/patients/pgx-profile` — returns full genotype/activity
    table for a patient.
- Curated `DrugGeneRule` table from CPIC + DPWG guidelines.

UI: `/pharmacogenomics` — table of warnings + per-warning narrative.

### Phase 9 — Polypharmacy / Drug-Drug Interactions  (`/phase9`, `ddi/`)

60+ curated DDI pairs from FDA table, plus transitive inference through
the CYP / transporter graph.

- Direct pairs: e.g. warfarin + ciprofloxacin = major (CYP inhibition).
- Transitive: A inhibits CYP3A4 → ↑ simvastatin exposure = inferred
  moderate interaction.
- Severity scale: `contraindicated > major > moderate > minor > none`.
- Each interaction includes mechanism, clinical effect, and source.
- Endpoints:
  - `POST /phase9/check` — list of drugs in → N×N interaction matrix +
    list of warnings.
  - `POST /phase9/pair` — direct pair lookup.

UI: `/polypharmacy` — severity-coloured matrix + per-interaction cards.

### Phase 10 — PK / PD Simulator  (`/phase10`, `pkpd/`)

Industry-grade pharmacokinetic + pharmacodynamic simulator.

**PK (pharmacokinetics):**
- 2-compartment model with first-order absorption, ODE solved by
  `scipy.integrate.solve_ivp` (LSODA).
- Population parameter table (`DrugRecord.pk`) for ~30 common drugs
  (ka, CL, Vc, Vp, Q, F).
- Allometric scaling by patient weight (exponent 0.75 on CL, 1.0 on V).
- Cockcroft-Gault eGFR for renal adjustment of CL.
- Inter-individual BSV via log-normal eta on CL, Vc, ka.
- Steady-state metrics: Cmax_ss, Cmin_ss, accumulation_ratio,
  time_to_steady_state_h, AUC₀-∞, Vss/F, CL/F.
- Validation checks (mass-balance, parameter bounds, steady-state reach).

**PD (pharmacodynamics):**
- Sigmoid-Emax (Hill) model with optional effect-compartment delay (ke0).
- Linear, log-linear, and Emax variants supported.
- Effect-compartment ODE for hysteresis (peak effect ≠ peak concentration).
- Endpoints:
  - `POST /phase10/pk/simulate`
  - `POST /phase10/pd/simulate`
  - `POST /phase10/pd/predict` (single-point effect at a given concentration)

UI: `/pkpd` — input grid, PK summary cards, SVG concentration-time
chart, validation checks, PD curve, narrative blocks for both.

### Phase 11 — Uncertainty Quantification  (`/phase11`, `uq/`)

Bootstrap confidence intervals on individual counterfactual predictions.

- N× resamples the cohort, refits the SCM each time, and reports the
  effect distribution with:
  - mean, std, 50% and 90% confidence intervals
  - **confidence_label** ∈ {high, medium, low} derived from CI width
  - **direction_stability** = fraction of bootstrap samples with the
    same effect sign
  - **ci_width_relative** = (hi − lo) / |mean|
- Endpoints:
  - `POST /phase11/patient-counterfactual` — bootstrap on one patient
  - `POST /phase11/ate` — bootstrap on the ATE
  - `GET  /phase11/coverage` — empirical coverage of the CI

UI: `/uncertainty` — CI bar chart, confidence/direction/width cards,
narrative.

### Phase 12 — Clinical Trials  (`/phase12`, `trials/`)

Live search against the ClinicalTrials.gov v2 API with a 24-hour
disk-cache fallback.

- Search by `condition` or `drug/intervention`.
- Returns: NCT ID, brief title, official title, phase (array),
  overall_status, enrollment, start_date, completion_date, conditions,
  study_type.
- Cached as JSON files under `data/cache/trials/` keyed by SHA1(query).
- Endpoints:
  - `GET  /phase12/trials/search?q=…&by=condition|drug&max=15`
  - `GET  /phase12/trials/{nct_id}` — single trial detail

UI: `/trials` — list of cards with status/phase chips + CT.gov deep link.

### Phase 13 — Regulatory  (`/phase13`, `regulatory/`)

Bundles FDA orange book, safety labels, FAERS top reactions, and
RxNorm links for a drug.

- **FDA Orange Book** — curated snapshot of approved formulations
  (trade name, ingredient, applicant, approval date).
- **Safety** — black-box warnings, contraindications, common adverse
  events (frequency-tagged), pregnancy category, typical dose.
- **FAERS** (live OpenFDA) — top 10 reactions with relative counts and
  total report volume.
- Endpoints:
  - `GET  /phase13/drugs/{drug}/regulatory` — full profile
  - `GET  /phase13/drugs/{drug}/orange-book`
  - `GET  /phase13/drugs/{drug}/faers`
  - `GET  /phase13/drugs/{drug}/safety`
  - `GET  /phase13/rxnorm/{drug}` — RxNorm concept ID + synonyms

UI: `/regulatory` — three-column safety/OB/FAERS layout with narrative.

### Phase 14 — Wet-Lab Molecule Triage  (`/phase14`, `wetlab/`)

RDKit-powered physicochemical + filter + target + dose-response
prediction for a SMILES string.

- Physicochemical: MW, LogP, HBD, HBA, TPSA, rotatable bonds, ring
  counts, aromaticity.
- Drug-likeness: Lipinski Ro5 violations, Veber rules.
- Filters: PAINS, Brenk, Synthetic Accessibility Score (SAS).
- Predicted dose-response curve: IC50 via a target-specific model
  (lookup table) + Hill coefficient.
- Probable targets: Tanimoto similarity against a reference drug
  library (chEMBL-derived fingerprints).
- Toxicity alerts: rule-based structural flags.
- Composite `overall_score` (0-100) and `verdict` ∈ {ready_for_screen,
  moderate_caution, significant_concerns}.
- Endpoints:
  - `POST /phase14/validate` — full SMILES → properties + score
  - `POST /phase14/ic50` — single-target IC50 prediction
  - `POST /phase14/tanimoto` — similarity to a reference set

UI: `/wetlab` — preset drugs (aspirin, atorvastatin, metformin,
imatinib, ibuprofen, diazepam), physicochemical table, drug-likeness
cards, filters, log-scale dose-response SVG, target bar chart, toxicity
alerts.

### Phase 15 — Disease Registry  (`/phase15`, `registry/`)

Postgres-backed CRUD catalog of disease entries. Seeded from upstream
phase data, open for custom additions.

- Schema: `key, name, description, target_proteins[], current_treatments,
  unmet_need ∈ {critical, high, medium, low}, clinical_trials (int),
  added_by ∈ {system, user}, added_at (timestamp)`.
- Endpoints:
  - `GET    /phase15/registry/diseases` — list
  - `GET    /phase15/registry/diseases/{key}` — single
  - `POST   /phase15/registry/diseases` — create
  - `PUT    /phase15/registry/diseases/{key}` — update
  - `DELETE /phase15/registry/diseases/{key}` — delete
  - `GET    /phase15/registry/summary` — totals + by-unmet-need breakdown

UI: `/registry` — list of cards, edit/create modal, system vs custom
flag, target-protein chips.

---

## Phase 16 — Explainable AI  (`/phase16`, `xai/`)

A composable explanation layer that takes a structured request, gathers
evidence from the relevant upstream phase, runs feature attribution, and
returns a structured reasoning chain.

- **SHAP-lite attribution** via leave-one-out perturbation.
- **Reasoning chain**: question → evidence → conclusion, plus
  alternative hypotheses.
- **Confidence-from-CI** label.
- Endpoints:
  - `POST /phase16/explain/counterfactual`
  - `POST /phase16/explain/ddi`
  - `POST /phase16/explain/pk`
  - `POST /phase16/explain/pgx`
  - `POST /phase16/explain/patient`  (composite across PGx + DDI + UQ)
  - `GET  /phase16/methods`

UI: `/explain` — 5 modes, feature-attribution bar chart, reasoning chain
UI, alternative hypotheses, narrative.

---

## Cross-cutting: Narrative layer  (`narrative/`)

Every response from the latest revision of every router carries a
`narrative` block with:

```json
{
  "headline": "one-line summary",
  "lay":       "plain-English explanation for non-experts (1-3 sentences)",
  "scientist": "technical detail for clinicians/researchers (2-5 sentences)",
  "risk_level": "low | moderate | high | critical"
}
```

Generated by `backend/app/narrative/{pgx,ddi,pkpd,uq,trials,
regulatory,wetlab,registry}.py` and rendered on the frontend by
`frontend/src/components/Narrative.tsx` — a card with a colour-coded
risk badge and a "Plain English ⇄ For scientists" toggle.

The narrative is derived from the actual response payload (severity,
deltas, top features, etc.) — not hardcoded strings.

---

## Cross-cutting: 3D causal anatomy  (`Causal3D.tsx` + `HumanBody.tsx`)

The `/causal` page has a 2-D / 3-D view toggle. The 3-D view (default)
shows the causal graph placed on a stylized human body:

- Body built from `@react-three/drei` primitives: sphere head, capsule
  torso / arms / legs, cylinder neck, sphere hands/feet.
- Each causal-graph node is mapped to an `ORGAN_POSITIONS` entry in
  `BodyGeometry.ts`:
  - organs (`heart`, `lungs`, `liver`, `kidney`, `pancreas`, `vasculature`)
  - biomarkers (`hr`, `hrv`, `spo2`, `glucose`, `sbp`, `dbp`, `bmi`)
  - diseases (`t2d`, `hypertension`, `cvd`, `copd`)
  - demographic (`age` → head)
- Edges arc slightly forward to avoid intersecting the body.
- OrbitControls — drag to rotate, scroll to zoom, click a node to
  highlight its causal neighbours and dim the rest.
- Sidebar shows the selected node's metadata.

---

## Frontend structure

- **`AppShell.tsx`** — client component wrapping the layout with
  sidebar toggle, header, and footer. Footer text reports the running
  version + phase count.
- **`Sidebar.tsx`** — left-side slide-in with two groups (Core platform,
  Advancements) covering all 16 phases. Hamburger button toggles.
- **`Panels.tsx`** — `Card`, `RiskChip`, `Stat`, `ErrorBox`, `Live`
  (pulsing-dot "Xs ago" timer mounted on every page after a result).
- **`Narrative.tsx`** — see above.
- **`Causal3D.tsx`, `HumanBody.tsx`, `BodyGeometry.ts`** — 3D view.
- **`lib/api.ts`** — typed wrappers for every endpoint + `CausalGraph`,
  `PatientPGx`, `PKMetrics`, `UQCausalResult`, etc.

Pages mirror the phase numbers:

| Path                | Phase |
|---------------------|------:|
| `/cohort`           | 1 + 2 |
| `/simulate`         | 3     |
| `/causal`           | 4     |
| `/chat`             | 5     |
| `/pharmacogenomics` | 8     |
| `/polypharmacy`     | 9     |
| `/pkpd`             | 10    |
| `/uncertainty`      | 11    |
| `/trials`           | 12    |
| `/regulatory`       | 13    |
| `/wetlab`           | 14    |
| `/registry`         | 15    |
| `/explain`          | 16    |

---

## Tech stack

- **Backend** — FastAPI, SQLAlchemy + asyncpg, Pydantic v2, scikit-learn,
  PyTorch (GNN), DoWhy / EconML, RDKit, scipy, NumPy, pandas, httpx.
- **Frontend** — Next.js 14 (App Router), React 18, Tailwind, Recharts,
  three.js + @react-three/fiber + @react-three/drei, clsx.
- **Stores** — Postgres (registry), Neo4j (graph), Redis (cache +
  agent sessions), Qdrant (embeddings).
- **LLM** — Ollama with `llama3.1` (override `OLLAMA_URL` to point at
  another host).
- **External APIs** — ClinicalTrials.gov v2, OpenFDA, NIH RxNorm,
  ChEMBL fingerprints (bundled).

---

## Versioning

- `backend/app/main.py` exposes `__version__`. Bump in the same commit
  that adds a phase.
- API `/health` returns the active phase list
  (`"1+2+3+4+5+8+9+10+11+12+13+14+15+16"` at v0.9.0).
- Frontend `Sidebar.tsx` and `AppShell.tsx` footer display the version.
