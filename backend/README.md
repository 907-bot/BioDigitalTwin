# BioDigitalTwin — Backend Physiological World Model Architecture

This directory contains the computational physiology, state-space estimation, uncertainty quantification, and multi-organ closed-loop simulation engines for **BioDigitalTwin**.

---

## 1. Authoritative State & Parameter Contracts

To ensure dimensional and mathematical integrity across the Unscented Kalman Filter (UKF), Maximum A Posteriori (MAP) parameter estimation, and multi-scale APIs, the physiological state space is formalized as:

- **State Space Dimension ($n = 36$)**:
  - **Metabolic (0..4)**: Plasma glucose ($G$), plasma insulin ($I$), insulin sensitivity proxy ($X$), interstitial glucose ($G_t$), hepatic glucose production ($HGP$).
  - **Cardiovascular (5..9)**: Systolic BP ($SBP$), diastolic BP ($DBP$), heart rate ($HR$), heart rate variability ($HRV$), stroke volume ($SV$).
  - **Renal & Electrolytes (10..13)**: Glomerular filtration rate ($GFR$), serum sodium ($Na^+$), serum potassium ($K^+$), plasma osmolality ($Osm$).
  - **Circadian Pacemaker (14..18)**: CLOCK/BMAL1, PER/CRY, plasma cortisol, plasma melatonin, homeostatic sleep pressure ($S$).
  - **Adipose & Lipid Transport (19..23)**: Adipose fat mass ($FM$), free fatty acids ($FFA$), LDL cholesterol, HDL cholesterol, serum triglycerides ($TG$).
  - **Immune & Inflammation (24..29)**: Interleukin-6 ($IL-6$), TNF-$\alpha$, NF-$\kappa$B activity, M1/M2 macrophage polarization ratio, C-reactive protein ($CRP$), systemic inflammation composite ($Inflam$).
  - **Musculoskeletal (30..35)**: Functional muscle mass ($MM$), muscular strength ($F_{max}$), myofibrillar hypertrophy factor, bone mineral density ($BMD$), osteoblast bone formation activity ($OB$), osteoclast bone resorption activity ($OC$).

- **Observation Space Dimension ($m = 15$)**:
  Sparse clinical and wearable telemetry including CGM glucose, cuff SBP/DBP, photoplethysmography HR/HRV, serum Na/K/Osm, lipids, and inflammatory markers.

- **Authoritative Parameter Space ($p = 29$)**:
  The system explicitly separates active patient-specific identifiability from population anchoring:
  - **7 Actively Personalized Parameters (MAP Estimated)**: Insulin sensitivity ($S_I$), basal hepatic glucose production ($EGP_0$), beta-cell glucose responsiveness ($\beta_{cell}$), renal glucose clearance threshold ($R_{thresh}$), arterial baroreflex gain ($G_{baro}$), baseline GFR ($GFR_{base}$), circadian oscillator amplitude ($A_{circ}$).
  - **22 Population-Anchored Parameters**: Rigorously bounded physiological priors based on age, biological sex, BMI, and clinical phenotype.

---

## 2. Multi-Organ Physiological World Model (`app/physiology/`)

Rather than treating organ systems as decoupled mathematical phenomenological approximations, the multi-organ world model establishes a causally coupled, convective transport framework adhering to the `PhysiologicalOrganModel` contract.

```
                    ┌─────────────────────────┐
                    │    Lungs / Respiration  │ (O2 uptake, CO2 clearance,
                    │    (Pulmonary Mechanics)│  alveolar gas eq, pH)
                    └───────────┬─────────────┘
                                │ Q_pulmonary
                                ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                   Closed-Loop Blood Transport Layer                    │
 │            (Convective flow, Severinghaus O2-Hb, HH acid-base)         │
 └───┬─────────────┬─────────────┬─────────────┬────────────┬─────────────┘
     │ Q_brain     │ Q_coronary  │ Q_splanchnic│ Q_hepatic  │ Q_muscle/bone
     ▼             ▼             ▼             ▼            ▼
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐  ┌──────────────┐
│  Brain  │   │  Heart  │   │ Gut / GI│   │  Liver  │  │Musculoskeletal│
│(ANS/HPA)│   │(CO, MAP)│   │ (SGLT1) │   │ (HGP)   │  │(Adaptation)  │
└─────────┘   └─────────┘   └────┬────┘   └────▲────┘  └──────────────┘
                                 │             │
                                 └─────────────┘
                                  Portal Vein
```

### Modular Organ Subsystems:
1. **Convective Blood Transport (`transport.py`)**:
   - Closed-loop 6-compartment systemic hemodynamics ($CO = HR \times SV$, $MAP = CO \times TPR + CVP$).
   - Dynamic exercise-induced regional flow redistribution (arteriolar metabolic hyperemia: skeletal muscle perfusion surges up to 75% of CO; renal/splanchnic perfusion selectively vasoconstricts).
   - Biochemical equilibrium: Severinghaus oxygen-hemoglobin saturation and Henderson-Hasselbalch arterial $pH = 6.10 + \log_{10}([HCO_3^-] / (0.0306 \times PaCO_2))$.

2. **Dedicated Respiratory System (`respiratory.py`)**:
   - Pulmonary mechanics with dead space fraction ($V_D / V_T$).
   - Alveolar gas equation ($P_A O_2 = F_i O_2 (P_B - 47) - P_a CO_2 / RQ$).
   - Central & peripheral chemoreceptor feedback drives alveolar ventilation to maintain isocapnic arterial homeostasis ($PaCO_2 \approx 40$ mmHg).

3. **Dedicated Hepatic System (`hepatic.py`)**:
   - Glycogen turnover with reversible phosphorylase and synthase kinetics.
   - Hepatic Glucose Production (HGP) suppressed by portal insulin and stimulated by glucagon.
   - Well-stirred hepatic clearance model for drug metabolism and enzyme turnover.

4. **Dedicated Gastrointestinal System (`gastrointestinal.py`)**:
   - Elashoff non-linear power exponential gastric emptying model.
   - Intestinal SGLT1 active carrier-mediated glucose absorption.
   - Endocrine L-cell GLP-1 and K-cell GIP incretin release coupled with DPP-4 enzymatic clearance.

5. **Dedicated Autonomic & Endocrine Axis (`autonomic_endocrine.py`)**:
   - Arterial baroreceptor reflex modulating heart rate, stroke volume, and arteriolar tone.
   - Sympathetic Nervous System (SNS) and Parasympathetic Nervous System (PNS) antagonism.
   - HPA axis (CRH $\rightarrow$ ACTH $\rightarrow$ Cortisol circadian secretion) and RAAS cascade (Renin $\rightarrow$ Angiotensin II $\rightarrow$ Aldosterone).

---

## 3. Scientific Verification, Validation, and Uncertainty Quantification (VVUQ)

Implemented following the **ASME V&V 40** computational modeling standard:

- **Numerical Solver Convergence**:
  4th-Order Runge-Kutta (RK4) integration with Courant-Friedrichs-Lewy (CFL) rate limiting guarantees numerical stability without unphysiological oscillations across large simulation steps ($\Delta t = 5$ min).

- **Mass & Flow Conservation**:
  Regional organ blood flows strictly sum to Cardiac Output ($\sum Q_i = CO$). Ingested carbohydrates, gastric contents, intestinal lumen glucose, and absorbed portal mass conserve carbohydrate mass balance within $< 5\%$.

- **Genuine Inverse Parameter Identifiability**:
  Identifiability auditing utilizes bounded non-linear optimization (L-BFGS-B / SLSQP) against synthetic and observational cohorts, verifying rank diagnostics and parameter recoverability.

- **4-Source Uncertainty Quantification**:
  Every digital twin projection quantifies aleatoric sensor noise, epistemic parameter variance, numerical solver truncation error, and structural model discrepancy.

---

## 4. Physiological Core v2 & Hybrid Multi-Organ Architecture (`app/physiology/core/`)

Building upon the gap analysis, **Physiological Core v2** institutes the organ-first architecture with strict physical conservation contracts:

1. **Canonical State Registry (`core/state_registry.py`)**:
   - Replaces numerical integer indices with stable, hierarchical named strings (e.g., `"blood.glucose"`, `"cardio.heart_rate_bpm"`, `"liver.glycogen_mass_g"`).
   - Validates explicit SI/clinical units, physical lower/upper bounds, and owning compartments.
   - Provides high-performance vector conversions (`state_to_vector`, `vector_to_state`).

2. **Shared Physiological Environment (`core/shared_environment.py`)**:
   - Represents the physical transport networks and circulating pools linking all organs.
   - **Circulation**: Hemodynamics ($CO = HR \times SV$, $MAP = CO \times TPR$), dynamic regional flows strictly summing to cardiac output ($\sum Q_i = CO$).
   - **Blood Chemistry & Gases**: Severinghaus $O_2$-$Hb$ saturation, Henderson-Hasselbalch arterial $pH = 6.10 + \log_{10}([HCO_3^-] / (0.0306 \times PaCO_2))$.
   - **Fluid Balance**: Extracellular fluid (ECF), intracellular fluid (ICF), total body water (TBW), plasma volume.
   - **Hormone & Drug Transport**: Insulin, glucagon, GLP-1, cortisol, renin-angiotensin-aldosterone axis.

3. **Standard Organ Contract (`core/organ_interface.py`)**:
   - Abstract `OrganModel` interface enforcing `compute_derivatives`, `exchange_fluxes`, `validate_state`, and `get_observables`.
   - Organ models **never** directly overwrite shared blood state; they exchange quantities strictly via `Flux(source, destination, quantity, value, unit, mechanism)`.

4. **Coupled Simulation Engine (`core/simulation_engine.py`)**:
   - Coupled numerical time-stepping ($\frac{d\mathbf{x}}{dt} = f(\mathbf{x}, \boldsymbol{\theta}, \mathbf{u})$) with RK4 and Forward Euler solvers.
   - Systemic glucose mass balance:
     $$\frac{dG}{dt} = \frac{R_{gut} + R_{hepatic} - U_{periphery} - E_{renal}}{V_{ecf}}$$
   - Active discrete event scheduling (meals, physical exercise, drug dosing).

5. **Constrained Hybrid ML Layer (`hybrid/residual_learner.py`)**:
   - Formulates hybrid dynamics: $\frac{d\mathbf{x}}{dt} = f_{mechanistic}(\mathbf{x}, \boldsymbol{\theta}, \mathbf{u}) + g_{ML}(\mathbf{x}, \mathbf{u})$.
   - Bounds ML residual corrections to $\le \pm 20\%$ to prevent unphysiological drift and preserve conservation laws.
   - `PhysiologicalAnomalyDetector` flagging rate-of-change telemetry anomalies (e.g. CGM jump $> 4.5$ mg/dL/min).

---

## 5. API Endpoints

### V1 Physiology Endpoints (`/physiology/*`):
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/physiology/organs/status` | `GET` | Inspect registered organs, state descriptors, SI units, and bounds |
| `/physiology/simulation/step` | `POST` | Advance coupled multi-organ twin forward by $\Delta t$ with RK4/Euler |
| `/physiology/challenge/meal` | `POST` | Simulate clinical Oral Glucose Tolerance Test (OGTT) |
| `/physiology/challenge/exercise` | `POST` | Simulate acute physical exertion challenge and cardiorespiratory response |
| `/physiology/vvuq/report` | `GET` | Retrieve ASME V&V 40 mass conservation and stability audit |

### V2 Coupled Multi-Organ Endpoints (`/physiology/v2/*`):
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/physiology/v2/state-registry` | `GET` | Authoritative dictionary of all registered states, parameters, units, and literature provenance |
| `/physiology/v2/simulation/run` | `POST` | Advance coupled multi-organ twin with scheduled events (meals, exercise, drug doses) and RK4 solver |
| `/physiology/v2/challenge/meal` | `POST` | Standard Oral Glucose Tolerance Challenge (OGTT) with coordinated GI, Pancreas, Hepatic, and Renal responses |
| `/physiology/v2/challenge/exercise` | `POST` | Exercise Stress Protocol with central command baroreflex reset, tachycardia, and respiratory compensation |
| `/physiology/v2/conservation/audit` | `GET` | Mathematical audit of whole-body flow and mass conservation ($\sum Q_i == CO$, fluid balances) |
| `/physiology/v2/anomaly/detect` | `POST` | Clinical telemetry validator flagging impossible rates of change and sensor dropouts |
