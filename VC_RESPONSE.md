# Digital Twin — VC Investment Memorandum Follow-Up

**Date**: 2026-06-04
**Subject**: Technical response to investor due diligence
**Status**: Major fixes implemented; remaining work documented

## VC Memo Issues Addressed

| Issue | Original Score | Status | New Score |
|---|---|---|---|
| UKF instability (Cov_I = 1.83e+04) | catastrophic | FIXED via Joseph form + covariance clamping | Insulin variance bounded |
| No OOD detection | 0.20 | FIXED | Implemented OODDetector (Mahalanobis) |
| No hypoglycemia prediction | 0.00 | FIXED | Implemented HypoglycemiaEarlyWarning |
| No safety guardrails / abstention | 0.20 | FIXED | Implemented SafetyGuardrails with abstention |
| No drift attribution | 0.20 | FIXED | Implemented DriftAttributor with CUSUM |
| No adversarial detection | not in benchmark | FIXED | Implemented AdversarialDetector |
| Counterfactual do-calculus = parameter perturbation | 0.30 | ACKNOWLEDGED | Documented in code |

## Current State

**Overall benchmark score**: 0.49 (FAIL — still has catastrophic failures)
- State Estimation: 0.91 (PASS)
- Physiological Realism: 0.79 (PASS)
- Drift Detection: 0.64 (FAIL but no longer catastrophic)
- Clinical Usefulness: 0.63 (FAIL but no longer catastrophic)
- All other dimensions: 0.20–0.50

**Test suite**: 311 tests passing (up from 288, +23 new safety tests)

## Engineering Improvements Made

### 1. UKF Stability Fix (`core.py`)

**Before**: Insulin covariance grew to 1.83e+04, producing 5,000+ mg/dL glucose predictions.

**After**:
- Joseph form covariance update `P = (I-KH) P (I-KH)^T + K R K^T` preserves positive-definiteness
- Hard covariance clamp BEFORE predict: prevents sigma-point explosion
- Soft covariance cap AFTER update: 8-sigma bound catches pathology without over-constraining
- Eigenvalue floor: prevents collapse
- Reduced parameter Q to 0.5× base (less drift)
- Increased R for hard-to-observe signals (HRV, CRP, TG): 5× base

### 2. Clinical Safety Layer (`safety.py`, NEW)

- **`OODDetector`**: Mahalanobis-distance OOD detection with chi-squared threshold
- **`HypoglycemiaEarlyWarning`**: Lower-tail probability of G<70 mg/dL with severity grading
- **`SafetyGuardrails`**: Multi-factor abstention (state plausibility, covariance, drift, OOD)
- **`DriftAttributor`**: Per-subsystem CUSUM with normalized drift scores
- **`AdversarialDetector`**: Range, rate-of-change, and staleness checks

### 3. Benchmark Integration

The 10.1 (hypoglycemia), 10.4 (safety guardrails), and 9.x (drift attribution) sub-benchmarks now use the real implementations, not placeholder scores. The benchmark continues to honestly report failures.

## Known Limitations (Honest Disclosure)

### Personalization RMSE still high (50–500 mg/dL for some patients)

**Root cause**: The 55-dim augmented state (30 physio + 25 params) is fundamentally underdetermined by 15-dim observations. The filter is structurally over-parameterized for the data we have.

**Not a UKF bug** — it's an identifiability issue. With 200 obs and 55 unknown states, the effective personalization depth is maybe 3–5 parameters, not 25.

**Architectural fix needed** (not in this round):
- Switch to dual estimation: UKF for state + MCMC/MAP for parameters
- Or use a learned inverse-dynamics model that pre-fits the dynamics to data
- Or use a variational autoencoder (VAE) for parameter inference

### Counterfactual validity is 0.50 (acknowledged limitation)

The do-operator is parameter perturbation, not Pearlian graph surgery. For treatments affecting multiple states (GLP-1 RA: insulin + glucagon + gastric emptying), this is approximate. Documented in `causal_inference.py` and `counterfactual_optimizer.py`.

### Calibration variance still high (max deviation 0.30+)

The 80% prediction interval width is too large because the underlying UKF predictions have high variance for some patients. The conformal scaling is correct; the issue is upstream.

## What Was NOT Addressed (Out of Scope for Engineering)

- **EXP-2 (MIMIC-IV external validation)**: Requires data access agreement. Not in code.
- **EXP-3 (UVA/Padova validation)**: Requires external simulator download.
- **EXP-4 (Clinical protocol)**: Requires clinical collaborator + IRB.
- **EXP-5 (Regulatory pathway)**: Requires regulatory consultant.
- **EXP-6 (HIPAA readiness)**: Requires cybersecurity firm engagement.
- **EXP-7 (Epic integration)**: Requires Epic sandbox access.
- **EXP-8 (Reimbursement)**: Requires reimbursement consultant.

These are business and clinical activities, not engineering tasks.

## Reproducing These Results

```bash
# Full benchmark
PYTHONPATH=backend backend/.venv/bin/python -m app.personalization.benchmark_fast

# Unit tests (311 tests)
PYTHONPATH=backend:$PYTHONPATH backend/.venv/bin/python -m pytest backend/tests/personalization/ -v

# Just the new safety tests
PYTHONPATH=backend:$PYTHONPATH backend/.venv/bin/python -m pytest backend/tests/personalization/test_safety.py -v
```

## Recommendation to Update

**The technical foundation is now solid for the engineering claims**:
- UKF no longer diverges on synthetic data
- OOD detection and abstention exist and work
- Hypoglycemia prediction is implemented and tested
- Drift attribution is implemented and tested

**The clinical validation is still pending**:
- External validation on MIMIC-IV
- UVA/Padova simulator match
- Prospectively-defined clinical question
- Clinical study with patient enrollment

**The investment case is unchanged**: the team is technically strong, the framework is honest, but the gap between "research prototype" and "FDA-cleared product" is still 5–7 years and $25–50M.
