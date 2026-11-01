# Human Digital Twin: Samsung Health Training Pipeline + Biochemical Researcher Platform

## Goal

Train the BioDigitalTwin's existing UKF-based personalization engine on the real Samsung Health wearable export (44 CSV files, ~26K records spanning Jan 2025–Aug 2026), and add a **Biochemical Researcher Module** — a professional-grade API layer that lets researchers run in-silico human trials on the trained twin to predict metabolism, simulate gene alterations, and test medication effects.

---

## Data Inventory (Samsung Health Export)

| Signal | File | Records | Key Fields |
|---|---|---|---|
| **Heart Rate** | `tracker.heart_rate` | ~6,665 | heart_rate, start_time |
| **Steps** | `tracker.pedometer_step_count` | ~7,866 | step_count, day_time |
| **Stress** | `shealth.stress` | ~1,326 | score, min, max, start_time |
| **Sleep Stages** | `health.sleep_stage` | ~138 | stage (40001=light,40002=REM,40003=deep) |
| **Activity** | `activity.day_summary` | ~607 | calorie, distance, active_time, floor_count |
| **Weight/Body Comp** | `health.weight` | ~12 | weight, body_fat_mass, skeletal_muscle_mass, basal_metabolic_rate, height |
| **Calories Burned** | `calories_burned.details` | ~608 | active_calorie, rest_calorie, tef_calorie |
| **Exercise** | `shealth.exercise` | ~631 | calorie, heart_rate (avg/max), distance, exercise_type |
| **HRV** | `health.hrv` | 1 | binning_data (JSON) |
| **Floors** | `floors_climbed`, `floors_day_summary` | ~1,263, ~252 | floor_count |
| **SpO2** | `tracker.oxygen_saturation` | ~10 | spo2 |

**User Profile (from weight data):** height=173cm, weight=68–82kg, body_fat~18–46%, skeletal_muscle~29–42kg, BMR=1541–1736 kcal

---

## Architecture Overview

```
samsung_healthdata/ (CSV export)
        ↓
[1] SamsungHealthDataLoader  (new notebook + service)
        ↓ merged daily feature vectors
[2] PersonalizationEngine.seed_from_real_data()  (enhance existing)
        ↓ trained twin state
[3] BiochemicalResearcherAPI  (new FastAPI router)
     ├── /researcher/trials/run           — in-silico human trials
     ├── /researcher/metabolism/predict   — metabolic flux simulation  
     ├── /researcher/gene/simulate        — gene alteration effects
     └── /researcher/medication/simulate  — drug/medication effects
```

---

## Proposed Changes

### Component 1: Samsung Health Data Pipeline

#### [NEW] `notebooks/samsung_health_training.ipynb` (or .py script)
A Jupyter notebook / standalone Python script that:
1. **Loads & parses** all Samsung CSVs (skipping BOM headers)
2. **Merges** signals into daily feature vectors aligned by `day_time`
3. **Computes derived features**: HRV from HR variability, resting HR, sleep quality index, stress-to-recovery ratio
4. **Extracts UKF seed vector** mapping to the 30-dim physiological state

#### [MODIFY] `backend/app/wearable/samsung_health.py`
- Add `load_from_csv_export(data_dir: str)` — parses the actual exported CSV format (BOM-aware, skip metadata header rows)
- Add `build_daily_feature_vectors()` — merges all signals into per-day training samples
- Add `extract_ukf_observations()` — maps Samsung signals → UKF 15-dim observation vector

**Observation vector mapping** (existing OBS_DIM=15):
```
[0]  glucose (estimated from HR+activity proxy)
[1]  SBP     (derived from stress + HR)  
[2]  DBP     (derived from stress + HR)
[3]  HR      ← heart_rate directly
[4]  HRV     ← inter-beat from HR variability
[5]  GFR     (estimated from weight/age)
[6]  Na      (default population prior)
[7]  K       (default population prior)
[8]  Osm     (default population prior)
[9]  FFA     (estimated from body_fat_pct)
[10] LDL     (estimated from body composition)
[11] HDL     (estimated from exercise + body_fat)
[12] TG      (estimated from calorie surplus)
[13] cortisol (proxy from stress score)
[14] sleep_pressure ← from sleep duration/stage
```

#### [NEW] `backend/app/wearable/csv_loader.py`
Pure parser for the Samsung Health CSV export format. Handles BOM, skips non-data header rows, normalizes timestamps.

---

### Component 2: Training Script

#### [NEW] `scripts/train_samsung_twin.py`
A runnable training script that:
1. Loads all Samsung CSVs via the new `csv_loader`
2. Builds chronological observation sequences
3. Runs the `PersonalizationEngine` in streaming mode (feeding daily observations)
4. Seeds with body composition from the weight CSV (height=173cm, weight, body_fat, skeletal_muscle, BMR)
5. Exports trained twin state as `data/trained_twin_state.json`
6. Prints convergence diagnostics

---

### Component 3: Biochemical Researcher Module

#### [NEW] `backend/app/researcher/__init__.py`
#### [NEW] `backend/app/researcher/models.py`
Request/response Pydantic models for all researcher endpoints.

#### [NEW] `backend/app/researcher/router.py`
FastAPI router with 4 research endpoints:

**`POST /researcher/trials/run`** — In-silico Human Trial
- Takes: trial protocol (cohort_size, intervention, duration_days, target_outcome)
- Uses existing `PersonalizationEngine` + `SCM` (causal) to simulate counterfactuals
- Returns: time-series of metabolic/cardio/immune states for each simulated subject

**`POST /researcher/metabolism/predict`** — Metabolic Flux Prediction
- Takes: current twin state + dietary/activity intervention
- Simulates TCA cycle proxy (glucose→energy flux), lipid oxidation, protein synthesis
- Returns: predicted metabolic rates, energy balance, substrate utilization

**`POST /researcher/gene/simulate`** — Gene Alteration Simulation
- Takes: gene name (e.g. AMPK, GLUT4, PPARG), alteration type (overexpression/knockout), magnitude
- Modifies relevant UKF parameters (SI, beta, fat_mass dynamics)
- Returns: before/after physiological state comparison, predicted metabolic phenotype

**`POST /researcher/medication/simulate`** — Medication Effect Simulation
- Takes: drug name, dose, route, duration
- Applies pharmacodynamic perturbations to the twin state
- Returns: predicted response trajectory with confidence intervals

#### [NEW] `backend/app/researcher/metabolism.py`
Metabolic flux analysis engine — estimates glucose oxidation rate, lipid mobilization, and energy balance from the twin's state vector.

#### [NEW] `backend/app/researcher/gene_perturbation.py`
Maps gene names to UKF parameter perturbations. Supports AMPK, GLUT4, PPARG, PCSK9, APOE, CYP enzymes.

#### [NEW] `backend/app/researcher/trial_engine.py`
Virtual cohort generator — creates synthetic "digital twins" by sampling parameter distributions, runs parallel UKF simulations, and applies statistical tests to trial outcomes.

---

### Component 4: Frontend Researcher Dashboard

#### [NEW] `frontend/src/components/researcher/`
- `ResearcherDashboard.tsx` — main researcher workspace
- `TrialDesigner.tsx` — UI for configuring virtual trials
- `MetabolicFluxViewer.tsx` — visualization of metabolic pathways
- `GeneAlteration.tsx` — gene perturbation interface
- `MedicationSimulator.tsx` — drug dosing simulator

---

### Component 5: Main API Registration

#### [MODIFY] `backend/app/main.py`
Register the new researcher router at `/researcher` prefix.

---

## Key Features for Biochemical Researcher

| Feature | Purpose | Implementation |
|---|---|---|
| **Samsung Data Training** | Ground truth personalization | CSV loader → UKF streaming update |
| **Metabolic Flux Prediction** | Predict glucose/lipid dynamics | Extended twin state + flux equations |
| **Gene Alteration** | AMPK/GLUT4/PPARG knockouts | Parameter perturbation in UKF |
| **Medication Simulation** | PK/PD modeling | Existing pkpd module + twin coupling |
| **Virtual Cohort Trials** | Population-level effects | Synthetic twin generator |
| **Counterfactual "What If"** | Compare interventions | Existing SCM causal module |
| **Convergence Report** | Validate training quality | UKF diagnostics + data quality score |

---

## Verification Plan

### Automated Tests
```bash
cd backend && python scripts/train_samsung_twin.py --dry-run
cd backend && pytest tests/test_samsung_loader.py -v
cd backend && pytest tests/test_researcher_api.py -v
```

### Manual Verification
1. Run the training script — verify `data/trained_twin_state.json` is created
2. Check convergence diagnostics (param_stability < 0.3 = good)
3. Call `POST /researcher/metabolism/predict` and verify physiologically plausible response
4. Call `POST /researcher/gene/simulate` with AMPK overexpression — verify glucose drops and fat oxidation increases

---

## Open Questions

> [!IMPORTANT]
> **HRV Data**: Only 1 HRV record exists (Aug 2025). Should HRV be estimated from the HR time series (RMSSD approximation) or use the default population prior when unavailable?

> [!IMPORTANT]
> **Glucose Proxy**: No CGM data in the Samsung export. The model will estimate glucose from HR, activity, and stress signals using a linear proxy. Do you want to use a fixed fasting glucose baseline (from the weight data) or a dynamic estimated value?

> [!NOTE]
> **Multiple Devices**: The data contains readings from 2-3 different Samsung devices (different `deviceuuid` values). The training will merge all devices by timestamp. This is expected behavior since it's longitudinal single-subject data.

> [!NOTE]
> **Researcher Auth**: The `/researcher/*` endpoints will be added to the existing `APIKeyAuthMiddleware` protected prefixes, requiring the `X-API-Key` header.
