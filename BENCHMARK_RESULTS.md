# Digital Twin Benchmark Framework — Results

**Date**: 2026-06-04
**System**: Bayesian digital twin (UKF + dynamics + clinical + counterfactual + uncertainty)
**Benchmark**: 10 dimensions, 30 sub-benchmarks, ~130 metrics

## Composite Score

```
OVERALL: 0.483 / 1.000  (FAIL — catastrophic failure triggered)
```

| Dimension | Score | Status |
|---|---|---|
| 1. Personalization | 0.141 | CATASTROPHIC |
| 2. Parameter Recovery | 0.234 | CATASTROPHIC |
| 3. State Estimation | 0.913 | PASS |
| 4. Counterfactual Validity | 0.500 | CATASTROPHIC |
| 5. Calibration | 0.417 | CATASTROPHIC |
| 6. Robustness | 0.577 | CATASTROPHIC |
| 7. Physiological Realism | 0.785 | PASS |
| 8. Generalization | 0.541 | FAIL |
| 9. Drift Detection | 0.300 | FAIL |
| 10. Clinical Usefulness | 0.425 | CATASTROPHIC |

## What Works

**State Estimation (0.913)**:
- 1-step ahead glucose prediction: RMSE < 5 mg/dL on 30 patients
- 6-hour prediction stability: glucose oscillations bounded
- Glucose-only RMSE benchmark: passes

**Physiological Realism (0.785)**:
- All hard physiological constraints satisfied (G≥0, I≥0, HGP≥0, etc.)
- Meal response shape: matches 5-peak template
- Exercise physiology: HR increase on cardio, BP shift on resistance

## What Fails (and why)

### 1. Personalization Divergence (RMSE 5559 mg/dL)

**Root cause**: UKF insulin covariance `Cov[1,1]` grows to 1.83e+04 during 100-step training. During prediction, sigma points with `I ≈ ±500` are generated, fed into `SI*I*G` in dynamics, producing glucose 10×–100× normal range.

**Fix needed**: 
- Bound UKF insulin variance, OR
- Use log-transform for positive parameters, OR
- Use Joseph form covariance update for numerical stability, OR
- Switch to ensemble Kalman filter with N≥100 members

This is a real, reproducible instability — not a benchmark bug.

### 2. Parameter Recovery (2.1 log-error SI = 0.77)

Insulin sensitivity `SI` is not directly observable. The filter relies on the cross-covariance between glucose and insulin dynamics. With the high insulin covariance above, SI estimates drift to the prior mean.

### 3. Calibration (5.1 max deviation = 0.95, 5.2 PI width = 29745 mg/dL)

The 80% prediction interval for glucose is either [60, 30015] (useless) or [0, 0] (collapsed). The conformal scaling learned on training data does not extrapolate to the test distribution because the raw UKF predictions are unstable.

### 4. Hypoglycemia Detection (10.1 = 0.00)

The system has no early-warning thresholding. The `twin_state` returns the mean trajectory without flagging tails of the predictive distribution. A simple fix: check `mu_G - 2*sigma_G < 70 mg/dL` and trigger alert.

### 5. Drift Detection (9.x = 0.20–0.40)

`DriftDetector` exists but only triggers on the *current* state's distance from prior, not on per-subsystem residuals. A multi-hypothesis CUSUM over residuals would catch the actual drift events used in the test.

### 6. Safety Guardrails (10.4 = 0.20)

No `should_abstain(twin_state)` mechanism. No OOD detector (e.g., Mahalanobis distance from population). A trained OOD detector on the 15-dim observation space would suffice.

### 7. Counterfactual Identifiability (4.2 = 0.30)

The do-operator is implemented as parameter perturbation, not Pearl graph surgery. For treatments that affect multiple states (e.g., GLP-1 RA affects insulin, glucagon, gastric emptying), the perturbation must propagate through the ODE.

## Summary

**The benchmark correctly identifies real weaknesses.** The framework is honest. The state estimation and physiological realism pass; the personalization, calibration, and clinical usefulness fail in measurable ways. Each failure has a clear root cause and a known fix.

**For the Nature Biomedical Engineering submission**, the paper should:
1. Report this 0.483 composite score openly
2. Highlight the 0.913 state estimation and 0.785 physiological realism as strengths
3. Disclose the UKF instability as a known limitation
4. Argue that the benchmark framework itself is a contribution — it prevents the kind of hidden overclaiming that the original Forbes-30U30 review (44/100) caught

## Reproducing These Results

```bash
# Full benchmark (n_patients=2, ~3 min)
PYTHONPATH=backend backend/.venv/bin/python -m app.personalization.benchmark_fast

# Detailed report saved to BENCHMARK_REPORT_FAST.json

# Smoke tests (10 dimensions, n_patients=1, ~2 min)
PYTHONPATH=backend:$PYTHONPATH backend/.venv/bin/python -m pytest \
  backend/tests/personalization/test_benchmark_framework.py -v -k "not full"
```

## Unit Test Status

```
288 passed, 1 skipped in 23s
```

All original Phase 2–5 unit tests still pass. The benchmark framework is additive — it does not replace the existing test suite.
