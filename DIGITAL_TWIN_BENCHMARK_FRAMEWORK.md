# Digital Twin Benchmark Framework

## International Committee of Examiners

- **Systems Biology:** Identifiability, model reduction, multi-scale coupling
- **Computational Physiology:** ODE correctness, parameter provenance, unit consistency
- **Causal Inference:** do-calculus validity, graph correctness, unmeasured confounding
- **Bayesian Statistics:** Calibration, coverage, sharpness, PIT uniformity
- **Clinical Medicine:** Endpoint relevance, safety, actionable recommendations
- **Biomedical AI:** Generalization, robustness, fairness, reproducibility

---

## Scoring

Each benchmark: **0.0–1.0** (composite of sub-metrics).  
**Passing = 0.70** on every benchmark (no cherry-picking).  
**Gold = 0.95** on every benchmark.  
**Failure** = any single benchmark < 0.40 OR any catastrophic failure condition triggered.

---

# 1. Personalization

## 1.1 Within-Patient Hold-Out

**Objective:** Twin trained on first 70% of a patient's data must predict the remaining 30% better than the population average.

**Setup:**
- 100 synthetic patients, each with 14 days of 5-min CGM + BP + HR data (4,032 timepoints)
- Train on days 1–10, test on days 11–14
- Population baseline: mean glucose / BP / HR from training window, held constant
- Twin baseline: UKF initialized with population priors, updated online

**Required datasets:**
- 100 synthetic patients with known ground-truth state (all 30 dimensions)
- Each patient has unique parameters drawn from a realistic population distribution
- Test set includes at least 3 unannounced meals and 2 exercise sessions

**Metrics:**
- RMSE on glucose (mg/dL): `RMSE_g = sqrt(mean((pred - actual)^2))`
- RMSE on SBP (mmHg)
- MAE on glucose rate-of-change (mg/dL/min): `|dG/dt_pred - dG/dt_true|`
- Improvement over population baseline: `1 - RMSE_twin / RMSE_baseline`

**Passing threshold:** `RMSE_g < 18 mg/dL` AND improvement > 30% over baseline  
**Gold threshold:** `RMSE_g < 10 mg/dL` AND improvement > 50% over baseline  
**Failure conditions:**
- `RMSE_g > 30 mg/dL`
- Twin worse than population baseline (improvement < 0%)
- Prediction error increases >2× from day 10 to day 14 (divergence)

## 1.2 Cross-Validation Stability

**Objective:** Personalization must converge to similar parameters regardless of which subset of data is used.

**Setup:**
- 5-fold cross-validation on each patient's 14-day record
- Each fold: train on 80% temporal segments, test on 20%
- Final parameter estimates from each fold should have coefficient of variation < threshold

**Required datasets:**
- Same 100 patients as 1.1

**Metrics:**
- CV of SI (insulin sensitivity) across folds: `std(SI_folds) / mean(SI_folds)`
- CV of HGP_basal across folds
- CV of vascular_resistance across folds
- Parameter recovery error: `|mean(SI_folds) - SI_true| / SI_true`

**Passing threshold:** CV < 0.30 for all 3 primary parameters  
**Gold threshold:** CV < 0.10 for all parameters  
**Failure conditions:**
- CV > 0.50 for any primary parameter
- Parameters from different folds have opposite sign (e.g., fold 1 says SI=0.01, fold 2 says SI=0.03 — they disagree on direction of physiology)
- Any parameter estimate > 3 standard deviations from the true value

## 1.3 Few-Shot Personalization

**Objective:** Twin must produce useful predictions after minimal patient data.

**Setup:**
- Each patient: 1 hour, 6 hours, 24 hours, 72 hours of training data
- Evaluate prediction accuracy on the NEXT 24 hours
- Compare against: untwinned population model, linear extrapolation

**Required datasets:**
- 200 synthetic patients (100 T2DM, 50 T1DM, 50 healthy)
- Training prefixes: 1h, 6h, 24h, 72h

**Metrics:**
- RMSE_g at each training duration
- Learning curve slope: `ΔRMSE / Δlog(training_hours)`
- Time to reach `RMSE_g < 20 mg/dL`

**Passing threshold:** RMSE_g < 25 mg/dL after 24h training  
**Gold threshold:** RMSE_g < 15 mg/dL after 6h training  
**Failure conditions:**
- RMSE_g > 40 mg/dL after 72h training
- No improvement over untwinned model (i.e., personalization does nothing)
- Negative learning curve (more data → worse predictions)

---

# 2. Parameter Recovery

## 2.1 Known-Ground-Truth Parameter Identification

**Objective:** Given unlimited noise-free synthetic data, can the twin recover the correct parameters?

**Setup:**
- 100 synthetic patients with known ground-truth parameters
- Each patient simulated for 30 days (8,640 timepoints at 5-min resolution)
- Twin initialized with population mean parameters (DEFAULT_PARAMS)
- Twin given full 30-dimensional state observation (no measurement noise)
- Measure parameter error after convergence

**Required datasets:**
- 100 synthetic records with known: SI, HGP_basal, beta_response, RT, vascular_resistance, baroreflex_gain, baseline_GFR, circadian_period, circadian_amplitude, lipolysis_rate, M1_activation

**Metrics:**
- Log-scale error for each parameter: `|log(p_est) - log(p_true)|`
- Parameter coupling index: `max|corr(p_i_est - p_i_true, p_j_est - p_j_true)|` — if errors in SI and HGP_basal are perfectly correlated (−1), they're compensating
- Total parameter error: `mean(|log(p_est) - log(p_true)|)` across 11 parameters
- Time to converge: first timepoint where all 3 primary parameters (SI, HGP_basal, vascular_resistance) are within 10% of truth

**Passing threshold:** Mean log-error < 0.2 (≈18%) for all 11 parameters  
**Gold threshold:** Mean log-error < 0.05 (≈5%) for all 11 parameters  
**Failure conditions:**
- Any single parameter has log-error > 0.5 (≈65%)
- Coupling index > 0.9 (parameters compensating for each other)
- No convergence within 30 days (parameters still drifting at end)

## 2.2 Identifiability Under Realistic Observation

**Objective:** With only 15 observed dimensions (instead of 30), which parameters remain identifiable?

**Setup:**
- 100 synthetic patients with known parameters
- Only the 15-dim observation vector is provided (no access to unobserved states)
- Run UKF for 30 days, then analyze posterior parameter distribution

**Required datasets:**
- Same 100 patients as 2.1, but only obs_vectors (15-dim)

**Metrics:**
- Profile likelihood for each parameter: slice along each parameter and compute χ² surface
- Identifiability classification: "identifiable" (profile likelihood has unique minimum), "sloppy" (flat direction), "unidentifiable" (no minimum in bounds)
- Fraction of 11 parameters classified as identifiable
- For unidentifiable parameters: report the sloppy manifold direction (which parameter combinations are constrained)

**Passing threshold:** ≥ 6/11 parameters identifiable  
**Gold threshold:** ≥ 9/11 parameters identifiable  
**Failure conditions:**
- < 3/11 parameters identifiable
- Any single parameter's profile likelihood is completely flat (no information at all)
- Posterior covers the entire physiological range (parameter completely unconstrained)

## 2.3 Parameter Recovery Under Confounding

**Objective:** When two parameters have opposite effects on the same observation (e.g., SI↑ and HGP_basal↓ both lower glucose), can the twin still recover the correct values?

**Setup:**
- Design 20 synthetic patients where SI and HGP_basal are anti-correlated: `SI = 0.018 + noise`, `HGP_basal = 2.5 - 50 * (SI - 0.018) + noise`
- These patients have the same glucose trajectory but different SI/HGP combinations
- Twin must recover BOTH parameters, not just fit glucose

**Required datasets:**
- 20 patients in the confounding regime
- 20 control patients with independent SI and HGP_basal

**Metrics:**
- SI recovery error in confounding vs control group
- HGP_basal recovery error in confounding vs control group
- Degradation ratio: `error_confounded / error_control`
- Sign correctness: fraction where estimated SI/HGP_basal has same sign as true (both positive, both correct direction)

**Passing threshold:** Degradation ratio < 3.0 for both parameters  
**Gold threshold:** Degradation ratio < 1.5 (confounding doesn't substantially hurt recovery)  
**Failure conditions:**
- Degradation ratio > 5.0
- Sign incorrect for > 20% of patients
- Parameters compensate perfectly (SI_est × HGP_basal_true ≈ SI_true × HGP_basal_est)

---

# 3. State Estimation

## 3.1 Unobserved State Tracking

**Objective:** With 15 observed and 15 unobserved dimensions, can the twin correctly estimate the hidden states?

**Setup:**
- 100 synthetic patients with complete ground-truth state (30-dim)
- UKF sees only 15-dim observation vector
- Compare UKF's estimate of unobserved states against ground truth

**Required datasets:**
- 100 patients × 14 days with full 30-dim state and 15-dim observation
- Unobserved states: HGP, PGU, IR, CLOCK_BMAL1, PER_CRY, CRP, fat_mass, M1_M2_ratio, NFkB_activity, InflammatoryLoad, melatonin, circadian_phase, FFA, LDL, HDL

**Metrics:**
- RMSE for each unobserved state (in physiological units)
- Normalized RMSE: `RMSE / (physiological_range * 0.25)` — relative to 25% of range
- Tracking delay: cross-correlation lag between true and estimated for circadian phase
- State error correlation: correlation between errors in different unobserved states (high correlation = systematic compensation rather than true tracking)

**Passing threshold:** Normalized RMSE < 1.0 for ALL unobserved states  
**Gold threshold:** Normalized RMSE < 0.5 for ALL unobserved states  
**Failure conditions:**
- Any unobserved state has normalized RMSE > 2.0 (state estimate is worse than random within physiological range)
- Circadian phase error > π/2 (90° out of phase — completely wrong timing)
- State error correlation > 0.9 (systematic compensation, not tracking)

## 3.2 Rapid Transient Tracking

**Objective:** Can the twin track rapid physiological changes (meals, exercise, insulin bolus)?

**Setup:**
- 50 synthetic patients with 3 unannounced meals/day + 1 exercise bout/day
- Meals: 30–80g CHO, rise time 15–30 min
- Exercise: 30 min moderate, HR increase 30–50 bpm
- Insulin: morning basal + meal-time bolus (T1DM patients)
- Measure tracking error during transient periods (±30 min around event)

**Required datasets:**
- 50 T2DM patients with meal/exercise annotations
- 25 T1DM patients with insulin + meal annotations

**Metrics:**
- Peak glucose error: `|G_pred_peak - G_true_peak|` for each meal
- Peak timing error: `|t_pred_peak - t_true_peak|` (minutes)
- Rate-of-change RMSE during transient: `RMSE(dG/dt)` over ±30 min window
- Exercise recovery error: glucose 60 min post-exercise
- Insulin bolus effect error: glucose 120 min post-bolus

**Passing threshold:** Peak error < 25 mg/dL, timing error < 15 min  
**Gold threshold:** Peak error < 10 mg/dL, timing error < 5 min  
**Failure conditions:**
- Peak error > 50 mg/dL for > 20% of meals
- Timing error > 30 min (twin misses the peak entirely)
- Twin predicts glucose DECREASE during meal (wrong sign of response)
- dG/dt error > 5 mg/dL/min (rate of change is clinically wrong)

## 3.3 Steady-State Fidelity

**Objective:** During stable periods (fasting, sleep), the twin must not drift or oscillate.

**Setup:**
- 100 synthetic patients simulated for 7 days with NO meals, NO exercise, NO insulin (pure fasting)
- Perfect steady state: glucose ≈ 90–110 mg/dL, insulin ≈ 5–15 μU/mL
- Twin must converge to and remain at steady state

**Required datasets:**
- 100 fasting patients with known steady-state parameters

**Metrics:**
- Long-term drift: slope of glucose over hours 48–168 (mg/dL/hour)
- Oscillation amplitude: peak-to-trough in fasting glucose (mg/dL)
- Overshoot: max deviation from steady state after initialization
- Settling time: hours to reach within 5 mg/dL of steady-state glucose

**Passing threshold:** Drift < 0.05 mg/dL/hr, oscillation < 5 mg/dL, settling < 12h  
**Gold threshold:** Drift < 0.01 mg/dL/hr, oscillation < 2 mg/dL, settling < 4h  
**Failure conditions:**
- Monotonic drift > 1 mg/dL/hr (twin is systematically trending away from steady state)
- Oscillation > 20 mg/dL (twin is self-oscillating)
- Never settles (glucose still drifting after 168h)
- Insulin or glucose clips at bounds (e.g., G = 20 or G = 600)

---

# 4. Counterfactual Validity

## 4.1 Known Intervention Recovery

**Objective:** Given a synthetic patient where we KNOW the effect of an intervention (because we simulated it with known ground-truth parameters), can the twin's counterfactual prediction match?

**Setup:**
- 100 synthetic patients
- For each, simulate two trajectories from the same initial state:
  - Control: standard physiology, no intervention
  - Treated: same patient, but with one parameter modified (e.g., SI × 1.3)
- Give twin the control trajectory + the intervention description
- Twin must predict the treated trajectory without ever seeing it

**Required datasets:**
- 100 paired (control, treated) trajectories with known intervention effect
- Interventions: metformin (SI+30%, HGP−20%), SGLT2i (RT 180→120), exercise (0.5 intensity post-meal), GLP1-RA (beta_response × 2)
- 25 patients per intervention type

**Metrics:**
- Intervention effect RMSE: `RMSE(G_pred - G_control, G_treated - G_control)` — does the twin predict the right CHANGE?
- Effect direction accuracy: fraction of timepoints where `sign(G_pred - G_control) = sign(G_treated - G_control)`
- Peak effect error: `|max(G_pred) - max(G_treated)|`
- Time-to-effect error: `|t_50%_effect_pred - t_50%_effect_treated|`
- Individual treatment effect (ITE) correlation: `corr(ITE_pred, ITE_true)` across patients

**Passing threshold:** Effect RMSE < 15 mg/dL, direction accuracy > 0.85  
**Gold threshold:** Effect RMSE < 8 mg/dL, direction accuracy > 0.95, ITE correlation > 0.9  
**Failure conditions:**
- Effect RMSE > 30 mg/dL (counterfactual is worse than predicting no effect)
- Direction accuracy < 0.60 (twin gets the sign wrong systematically)
- ITE correlation < 0 (twin says patient improves when they worsen)

## 4.2 do-Calculus Soundness

**Objective:** Verify that the twin's causal graph correctly implements Pearl's do-operator (graph surgery), not just conditional prediction.

**Setup:**
- Two scenarios where conditional prediction and intervention give DIFFERENT results:
  - **Scenario A** (confounding): IR → G and IR → SBP. If we do(SBP = high), G should NOT change (no causal path). But conditional on SBP=high, G WILL change (confounding path).
  - **Scenario B** (mediation): Exercise → glucose (direct) AND Exercise → HR → glucose (mediated). If we block HR (do(HR=rest)), the exercise effect on glucose should decrease.
- Run both scenarios with causal graph surgery and with conditional prediction
- Compare against ground truth (simulated with blocked edges)

**Required datasets:**
- Synthetic data for both causal scenarios with known ground truth

**Metrics:**
- do-vs-conditional gap: `|E[G | do(SBP=high)] - E[G | SBP=high]|` — should be > 0
- Surgery correctness: `|E[G | do(SBP=high)] - G_true_mutilated|`
- Mediation blocking: `|E[G | do(exercise, do(HR=rest))] - G_true_no_HR_path|`
- Back-door adjustment: `E[G | do(intervention)]` computed via adjustment formula must match graph surgery

**Passing threshold:** Surgery error < 5 mg/dL, do-vs-conditional gap captured (correct sign)  
**Gold threshold:** Surgery error < 2 mg/dL, gap magnitude within 20% of true  
**Failure conditions:**
- `|do - conditional| < 1 mg/dL` when true gap is > 10 mg/dL (twin treats intervention as conditional prediction)
- Surgery error > 15 mg/dL
- Adjusted and surgery-based estimates disagree by > 10 mg/dL (consistency failure)

## 4.3 Unmeasured Confounding Sensitivity

**Objective:** When there IS an unmeasured confounder (e.g., unknown genetic factor affecting both IR and GFR), can the twin's causal estimates handle the bias?

**Setup:**
- Create 100 synthetic patients with an unmeasured confounder U: U → IR (+0.3) and U → GFR (+0.3)
- The confounder U is NOT included in the observation vector or state
- Compare the twin's estimate of IR → GFR effect with and without U
- The true causal effect IR → GFR should NOT include the U-path

**Required datasets:**
- 100 patients with hidden confounder U
- Control: 100 patients without U

**Metrics:**
- Bias: `|estimated_effect(IR→GFR) - true_causal_effect(IR→GFR)|`
- E-value: minimum strength of association an unmeasured confounder would need to explain away the estimated effect
- Sensitivity contour: how much would the effect estimate change as correlation(U, IR) varies from 0 to 1
- Robustness index: `min(|estimated_effect| / bias, 1)` — fraction of effect not explained by confounding

**Passing threshold:** E-value > 1.25 (moderate confounding would not explain effect)  
**Gold threshold:** E-value > 2.0 (strong confounding would not explain effect), bias < 20% of true effect  
**Failure conditions:**
- Estimated effect switches sign when U is added (sign reversal)
- E-value < 1.1 (tiny confounder could explain the entire effect)
- Bias > true effect (estimate is more wrong than right)

---

# 5. Calibration

## 5.1 Predictive Coverage

**Objective:** The twin's claimed confidence intervals must contain the true value at the claimed rate.

**Setup:**
- 100 synthetic patients, 14 days each
- At each step, record the UKF's predictive (pre-update) mean and covariance for each variable
- Compute empirical coverage at nominal levels: 50%, 80%, 90%, 95%
- Repeat for glucose, SBP, DBP, HR, HRV, GFR (6 variables × 4 levels = 24 coverage tests)

**Required datasets:**
- 100 patients × 4,032 timepoints with ground-truth state and UKF predictive distribution

**Metrics:**
- Coverage deviation: `|actual_coverage - nominal_level|` for each variable × level
- Max deviation across all 24 tests
- Mean deviation across all 24 tests
- PIT uniformity: KS statistic against Uniform(0,1)
- PIT histogram: max bin deviation from flat (z-score units)

**Passing threshold:** Max deviation < 0.05, KS statistic < 0.05, PIT bin z < 3.0  
**Gold threshold:** Max deviation < 0.02, KS statistic < 0.02, PIT bin z < 2.0  
**Failure conditions:**
- Max deviation > 0.15 (e.g., 90% CI achieves 75% or 100%)
- PIT KS > 0.20 (residuals are clearly not uniform)
- Coverage is systematically above OR below nominal (directional miscalibration)
- Any single variable has coverage < 0.50 at 90% nominal

## 5.2 Sharpness

**Objective:** Among well-calibrated models, sharper (narrower) intervals are better.

**Setup:**
- Compute mean 90% prediction interval width for each variable
- Normalize by the variable's observational standard deviation
- Only evaluated conditional on passing Coverage (5.1)

**Required datasets:**
- Same 100 patients as 5.1

**Metrics:**
- Mean 90% PI width for glucose (mg/dL)
- Mean 90% PI width for SBP (mmHg)
- Sharpness ratio: `PI_width / observation_std`
- Sharpness per information: how much narrower than the unconditional std?

**Passing threshold:** Glucose PI width < 50 mg/dL, sharpness ratio < 2.0  
**Gold threshold:** Glucose PI width < 30 mg/dL, sharpness ratio < 1.2  
**Failure conditions:**
- Glucose PI width > 100 mg/dL (intervals are clinically useless)
- Sharpness ratio > 3.0 (wider than unconditional std — model adds uncertainty instead of reducing it)
- Variable with sharpness ratio < 0.1 AND fails coverage (overconfident AND wrong)

## 5.3 Conditional Calibration

**Objective:** Coverage must hold conditionally on key covariates, not just marginally.

**Setup:**
- For each patient, stratify predictions by:
  - Time of day: nighttime (0–6), morning (6–12), afternoon (12–18), evening (18–24)
  - Glucose level: hypoglycemic (<70), euglycemic (70–180), hyperglycemic (>180)
  - Activity state: rest, post-meal (0–2h), exercise, sleep
- Compute coverage within each stratum at 90% nominal

**Required datasets:**
- 100 patients with event annotations (meal times, exercise, sleep)

**Metrics:**
- Stratified coverage deviation: `|actual_stratum_coverage - 0.90|` for each stratum
- Max stratification deviation
- Calibration drift: coverage in last 20% of data vs first 20%
- Coverage gradient with glucose level: `coverage_hyper - coverage_hypo`

**Passing threshold:** Max stratum deviation < 0.10  
**Gold threshold:** Max stratum deviation < 0.05, no drift > 0.03  
**Failure conditions:**
- Any stratum has coverage < 0.70 or > 0.98 at 90% nominal
- Coverage in hypoglycemic range < 0.50 (clinical safety: model is most wrong when it matters most)
- Calibration drift > 0.15 (coverage degrades over time)

## 5.4 Reliability Diagrams

**Objective:** Binned calibration curves must show slope ≈ 1, intercept ≈ 0.

**Setup:**
- For each variable, bin predictions into 20 equal-width bins by predicted probability
- For each bin: plot observed frequency vs mean predicted probability
- Compute Expected Calibration Error (ECE) and Max Calibration Error (MCE)

**Required datasets:**
- Same 100 patients as 5.1

**Metrics:**
- ECE: `mean(|observed_freq - predicted_prob|)` across bins
- MCE: `max(|observed_freq - predicted_prob|)` across bins
- Calibration slope: slope of `observed_freq ~ predicted_prob` regression
- Calibration intercept: intercept of the same regression
- 95% CI on slope: `[slope - 1.96*SE, slope + 1.96*SE]`

**Passing threshold:** ECE < 0.03, MCE < 0.08, 0.85 < slope < 1.15  
**Gold threshold:** ECE < 0.01, MCE < 0.03, 0.95 < slope < 1.05  
**Failure conditions:**
- ECE > 0.10 (model is systematically miscalibrated)
- Slope confidence interval does not contain 1.0
- Slope < 0.5 or > 2.0 (model has no idea about its own uncertainty)
- MCE > 0.20 (model is catastrophically wrong in some regime)

---

# 6. Robustness

## 6.1 Missing Data

**Objective:** Twin must handle randomly missing observations without degradation.

**Setup:**
- 50 synthetic patients, 14 days
- Artificially mask observations at rates: 10%, 25%, 50%, 75%, 90%
- Missing pattern: MCAR (completely at random), MAR (conditional on glucose), MNAR (high glucose more likely missing)
- Evaluate tracking error during missing periods

**Required datasets:**
- 50 patients × 5 missing rates × 3 missing mechanisms = 750 test runs

**Metrics:**
- RMSE_g as function of missing rate
- Degradation slope: `ΔRMSE / Δmissing_rate`
- Recovery time: steps after observation returns to regain pre-missing accuracy
- Bias during missing gaps: mean error during gap vs non-gap
- Worst-case: maximum RMSE_g at any missing rate

**Passing threshold:** RMSE_g < 25 at 50% missing, recovery < 10 steps  
**Gold threshold:** RMSE_g < 18 at 75% missing, recovery < 3 steps  
**Failure conditions:**
- RMSE_g > 50 at 50% missing (twin falls apart with moderate missingness)
- Irrecoverable: never returns to baseline after observations resume
- Bias > 20 mg/dL during gaps (twin systematically trends wrong when unobserved)
- MNAR missing degrades > 2× MCAR missing (twin exploits informative missingness)

## 6.2 Measurement Noise

**Objective:** Twin must be robust to realistic measurement noise levels.

**Setup:**
- CGM noise: add white noise N(0, σ) + autocorrelated AR(1) noise
  - σ = 5, 10, 15, 20 mg/dL (CGM MARD: 8–15%)
- BP noise: σ = 3, 5, 8 mmHg
- HR noise: σ = 2, 5, 10 bpm
- Test each noise level separately and combined

**Required datasets:**
- 100 patients × 5 noise levels = 500 runs

**Metrics:**
- RMSE_g vs noise-free ground truth at each noise level
- Noise amplification factor: `RMSE_noisy / RMSE_noiseless`
- Bias introduced by noise: `mean(G_pred_noisy - G_pred_noiseless)`
- Coverage degradation: coverage at 90% nominal as noise increases
- Filter divergence rate: fraction of runs where UKF covariance explodes

**Passing threshold:** Amplification < 2.0 at CGM σ=15, coverage within 0.10 of nominal  
**Gold threshold:** Amplification < 1.5 at CGM σ=20, coverage within 0.05 of nominal  
**Failure conditions:**
- Amplification > 5.0 (twin amplifies noise instead of filtering it)
- Divergence rate > 5% (UKF becomes numerically unstable)
- Coverage drops below 0.50 at 90% nominal under ANY noise level
- Bias > 10 mg/dL (noise creates systematic error)

## 6.3 Sensor Failure / Outliers

**Objective:** Twin must reject and recover from sensor artifacts.

**Setup:**
- Inject outliers into observation stream:
  - Single-spike: ±100 mg/dL for 1 timepoint
  - Dropout: 0 mg/dL for 1 timepoint
  - Sustained bias: +30 mg/dL for 30 min
  - Sensor degradation: noise σ increases 3× for 2 hours
- Each scenario tested 50 times

**Required datasets:**
- 50 patients × 4 artifact types = 200 runs

**Metrics:**
- Outlier rejection: ratio of outlier observations that the twin correctly downweights (innovation > 3σ)
- Recovery time: steps to return to within 10 mg/dL of true
- Contamination: max prediction error caused by artifact
- False rejection: fraction of non-outliers treated as outliers (> 3σ innovation)

**Passing threshold:** Recovery < 10 steps from spike, contamination < 25 mg/dL  
**Gold threshold:** Recovery < 3 steps, contamination < 10 mg/dL  
**Failure conditions:**
- State jumps by > 50 mg/dL in response to single spike (over-fits to outlier)
- Never recovers from 30-min sustained bias
- False rejection rate > 5% (too aggressive at filtering)
- Kalman gain spikes > 10× normal in response to outlier

## 6.4 Adversarial Perturbation

**Objective:** Small, worst-case perturbations to input should not cause large prediction errors.

**Setup:**
- For each of 50 patients, find the adversarial perturbation δ that maximizes prediction error while bounded: `||δ||_∞ < 0.1 * observation_std`
- Use gradient-based or random search for δ
- Evaluate prediction error under δ

**Required datasets:**
- 50 patients, compute adversarial perturbations for each

**Metrics:**
- Adversarial vulnerability: `max_δ |G_pred(x+δ) - G_pred(x)|`
- Lipschitz constant bound: `max_δ ||f(x+δ) - f(x)|| / ||δ||`
- Worst-case over any single observation dimension
- Robust accuracy: fraction of predictions where adversarial perturbation changes outcome classification (euglycemic vs hyperglycemic)

**Passing threshold:** Max perturbation effect < 20 mg/dL  
**Gold threshold:** Max perturbation effect < 10 mg/dL  
**Failure conditions:**
- Single-dimension perturbation changes glucose prediction by > 50 mg/dL
- Adversarial perturbation changes clinical classification (euglycemic → hyperglycemic)
- Lipschitz > 100 (model is pathologically sensitive)

---

# 7. Physiological Realism

## 7.1 Known Physiological Constraints

**Objective:** The twin must never violate established physiological principles.

**Setup:**
- Simulate 200 patients for 30 days each under random meal/exercise/insulin inputs
- Check every state variable at every timepoint against known physiological constraints

**Required datasets:**
- None (uses twin's own predictions)

**Constraints to check:**
- Glucose: 20 < G < 600 mg/dL (extreme hypo/hyperglycemia bounds)
- Insulin: I ≥ 0 (cannot be negative)
- SBP > DBP (systolic always higher than diastolic)
- HR: 30 < HR < 220 bpm
- GFR: 5 < GFR < 200 mL/min/1.73m²
- Na: 120 < Na < 160 mEq/L
- K: 2.5 < K < 7.0 mEq/L
- Osm: 260 < Osm < 340 mOsm/kg
- Cortisol: 10 < cortisol < 1000 nmol/L
- Sleep pressure: 0 < sleep_pressure < 1
- Circadian phase: 0 < phase < 2π
- Fat mass: 2 < fat_mass < 100 kg
- CLOCK_BMAL1, PER_CRY: 0 < x < 2.5
- NFkB_activity: 0 < NFkB < 1
- InflammatoryLoad: 0 < IL < 100
- SBP < 250, DBP < 150
- HRV: 5 < HRV < 200 ms
- FFA: 0.1 < FFA < 2.0 mmol/L
- LDL: 20 < LDL < 300 mg/dL
- HDL: 10 < HDL < 120 mg/dL
- TG: 20 < TG < 800 mg/dL
- Exercise lowers glucose (dG/dt_exercise < 0)
- Meal raises glucose (dG/dt_meal > 0)
- Insulin lowers glucose (dG/dt_insulin < 0)
- Cortisol raises glucose (dG/dt_cortisol > 0)
- Light suppresses melatonin

**Metrics:**
- Violation rate: `violations / (patients × timepoints × constraints)`
- Worst variable: constraint with highest violation rate
- Worst patient: patient with highest violation rate
- Magnitude: how far outside bounds (e.g., G = −5 vs G = 18)

**Passing threshold:** Violation rate < 0.1%, no violation > 5% outside bounds  
**Gold threshold:** Violation rate < 0.01%, no violation > 1% outside bounds  
**Failure conditions:**
- Any violation rate > 1% (systematic constraint violation)
- Glucose < 0 (physically impossible)
- Exercise raises glucose in more than 5% of cases (wrong sign of exercise effect)
- SBP < DBP in more than 1% of cases

## 7.2 Meal Response Shape

**Objective:** Post-meal glucose dynamics must match known physiological shape.

**Setup:**
- 100 patients, each with 3 standard meals (50g CHO, consumed over 15 min)
- Extract post-meal glucose curve: t=0 to t=240 minutes
- Compare against known physiological signatures

**Required datasets:**
- 300 meal events across 100 patients

**Physiological signatures to check:**
- Peak time: glucose peaks 30–90 min post-meal
- Monotonic rise: glucose should not dip before rising (cephalic phase is small)
- Monotonic fall after peak: no secondary rise without new meal
- Return to baseline: glucose returns to within 10 mg/dL of pre-meal within 4h
- No oscillations: glucose should not oscillate post-meal
- Dose-response: larger meals → higher peak (correlation > 0.7)
- First-meal effect: breakfast produces smaller glucose excursion than dinner (dawn phenomenon)

**Metrics:**
- Peak time error: `|t_peak - 60|` minutes
- Monotonicity violation: fraction of meals with non-monotonic rise
- Return failure: fraction where glucose > pre-meal + 10 at t=240
- Oscillation count: number of post-meal peaks beyond the primary
- Dose-response correlation: correlation(meal_size, peak_glucose)

**Passing threshold:** Peak within 30–90 min for > 85% of meals, return within 4h for > 80%  
**Gold threshold:** Peak within 40–80 min for > 95%, return > 90%, dose-response r > 0.8  
**Failure conditions:**
- > 20% of meals have glucose dip before rise
- > 30% fail to return to baseline within 4h
- Average peak time < 15 min or > 120 min (non-physiological)
- Dose-response correlation < 0.3 (model doesn't respond to meal size)
- Post-meal oscillations in > 10% of meals

## 7.3 Circadian Rhythms

**Objective:** The twin must produce realistic 24-hour physiological rhythms.

**Setup:**
- 50 patients in a controlled environment: fixed light-dark cycle (lights on 0700–2300), 3 meals/day, no exercise
- Simulate for 7 days, analyze circadian patterns on days 4–7 (after entrainment)
- Check known circadian signatures

**Required datasets:**
- 50 patients with controlled light schedule

**Physiological signatures to check:**
- Cortisol: peak 0600–0900, trough 2200–0200, amplitude ratio > 2.0
- Melatonin: peak 2300–0300, trough 0700–1100, amplitude ratio > 3.0
- Glucose: nadir 0200–0400 (fasting), peak post-breakfast 0800–1000
- SBP: dip 2–5 mmHg during sleep (nocturnal dipping), morning surge 10–20 mmHg
- HR: lowest 0200–0500, highest post-prandial
- HRV: highest during sleep (parasympathetic dominance)
- Sleep pressure: builds during wake, decays during sleep
- CLOCK-BMAL1 / PER-CRY: anti-phase oscillation, period 23.5–24.5h

**Metrics:**
- Cortisol acrophase error: `|peak_time - 0800|` (hours)
- Melatonin acrophase error: `|peak_time - 0200|` (hours)
- Cortisol amplitude ratio: `peak / trough`
- Nocturnal SBP dip: `SBP_day - SBP_night` (mmHg)
- Period error: `|measured_period - 24h|` (hours)
- Phase alignment: cross-correlation of CLOCK-BMAL1 and PER-CRY (should be negative at lag 0)
- Intra-patient stability: day-to-day phase variability over days 4–7

**Passing threshold:** Cortisol peak 0600–1000, nocturnal dip > 5% of daytime SBP, period 23.5–24.5h  
**Gold threshold:** Cortisol peak 0700–0900, dip 8–15 mmHg, period 23.8–24.2h, phase jitter < 0.5h  
**Failure conditions:**
- Cortisol peak at night (inverted rhythm)
- No detectable melatonin rhythm (peak/trough < 1.5)
- Period < 20h or > 28h (circadian not entrained to 24h)
- SBP higher at night than day (reverse dipping — pathological, but can happen in real patients. But if it happens in ALL patients, the model is wrong)
- Phase reverses during the simulation (CLOCK-BMAL1 and PER-CRY cross over incorrectly)

## 7.4 Exercise Physiology

**Objective:** The twin's exercise response must match known physiology.

**Setup:**
- 50 patients, each with 5 exercise sessions: moderate (30 min, 50% VO2max), intense (20 min, 80%), mild (60 min, 30%)
- Measure: glucose, HR, SBP, HRV during and after exercise
- Compare against known exercise physiology

**Required datasets:**
- 50 patients × 3 exercise intensities = 150 sessions

**Physiological signatures to check:**
- HR increases 20–50 bpm during moderate exercise
- SBP increases 10–30 mmHg during exercise
- Glucose: during exercise, may drop (if insulin present) or rise (if catecholamine-driven)
- Post-exercise: HR returns to within 10% of baseline within 10 min
- Post-exercise: glucose may drop below baseline for 1–4h (insulin sensitivity increase)
- HRV decreases during exercise (sympathetic dominance)
- Dose-response: harder exercise → greater HR increase

**Metrics:**
- HR increase at each intensity
- Glucose change during exercise
- Recovery half-life: time for HR to reach 50% of max increase
- Post-exercise hypoglycemia risk: time with glucose < 70 in 4h post-exercise
- Exercise-induced insulin sensitivity: compare glucose response to a meal 2h post-exercise vs without exercise

**Passing threshold:** HR increases monotonically with intensity, recovery half-life < 5 min  
**Gold threshold:** HR increases 20–30 bpm (moderate) and 35–55 bpm (intense), recovery < 3 min  
**Failure conditions:**
- HR decreases during exercise
- Exercise glucose response is identical at all intensities (no dose-response)
- HR never returns to baseline within 30 min post-exercise
- Exercise always raises glucose (or always lowers it) in all patients — should vary by insulin status

---

# 8. Generalization

## 8.1 Population Transfer

**Objective:** A twin trained on Population A must generalize to Population B with minimal degradation.

**Setup:**
- Train on 50 healthy patients (young, BMI 20–25, no comorbidities)
- Test on:
  - 50 T2DM patients (BMI > 30, age > 50, insulin resistant)
  - 25 T1DM patients (autoimmune, no endogenous insulin)
  - 25 elderly patients (age > 70, reduced GFR, stiff arteries)
  - 25 pediatric patients (age 10–18, growing)
  - 25 patients with CKD (GFR < 60)
- Measure prediction accuracy on first 24h WITHOUT personalization

**Required datasets:**
- 175 patients across 5 distinct subpopulations

**Metrics:**
- RMSE_g in each target population
- Degradation ratio: `RMSE_target / RMSE_source`
- Parameter adaptation: how many hours until RMSE matches source population
- Worst-case population: the population with highest RMSE
- Transfer failure rate: fraction of patients where RMSE > 2× source median

**Passing threshold:** Degradation ratio < 2.0 for ALL populations, worst RMSE < 35 mg/dL  
**Gold threshold:** Degradation ratio < 1.5 for ALL populations, worst RMSE < 20 mg/dL  
**Failure conditions:**
- Degradation ratio > 3.0 for any population
- T1DM patients have RMSE > 50 mg/dL (twin cannot handle no endogenous insulin)
- Pediatric patients have RMSE > 50 mg/dL (growth physiology breaks the model)
- Any population has RMSE worse than untwinned model (generalization failure)

## 8.2 Distribution Shift

**Objective:** Twin must detect and adapt to gradual physiological changes.

**Setup:**
- 50 synthetic patients who undergo a simulated physiological change at day 7:
  - Weight gain: SI decreases by 30% over 7 days
  - Aging: GFR decreases by 20%, vascular_resistance increases by 20%
  - Disease progression: beta cell function decreases by 50%
  - Medication initiation: patient starts metformin (SI+30%, HGP−20%)
- Each patient simulated for 28 days total
- Evaluate before and after the change

**Required datasets:**
- 50 patients × 4 shift types = 200 trajectories, each 28 days

**Metrics:**
- Detection delay: days from shift onset to when twin's predictive RMSE exceeds 2× pre-shift baseline
- Adaptation time: days from shift onset until RMSE returns to within 1.5× baseline
- Tracking error during shift: RMSE_g over days 7–14
- Missed shift rate: fraction of shifts where detection delay > 7 days
- False detection rate: fraction of days 1–7 where twin falsely signals a shift

**Passing threshold:** Detection < 3 days, adaptation < 7 days, false detection < 5%  
**Gold threshold:** Detection < 1 day, adaptation < 3 days, false detection < 1%  
**Failure conditions:**
- Never detects shift (RMSE remains elevated for entire 21-day post-shift period)
- Adaptation never occurs (twin permanently stuck with wrong parameters)
- False detection rate > 20% (twin sees shifts everywhere)
- Detection delay > 14 days (shift is essentially missed)

## 8.3 Cross-Modality Generalization

**Objective:** Twin trained on continuous CGM data must also work with sparse fingerstick data.

**Setup:**
- Train twin on continuous 5-min CGM for 7 days
- Test on:
  - 4× daily fingerstick (fasting, pre-meals, bedtime)
  - 7× daily (before/after meals + bedtime)
  - Continuous but with 30-min dropout every 2h (realistic CGM gaps)
  - Flash glucose monitoring (scan every 4–8h, no continuous trace)
- Evaluate RMSE_g on held-out continuous data during test period

**Required datasets:**
- 100 patients with both CGM and simulated sparse measurements

**Metrics:**
- RMSE_g at each measurement frequency
- Information efficiency: `RMSE_continuous / RMSE_sparse` — how much accuracy is lost per unit of data reduction
- Critical event detection: fraction of hypoglycemic events (< 70 for > 15 min) detected at each frequency
- Worst-case: max gap without observation before RMSE > 30 mg/dL

**Passing threshold:** RMSE < 25 mg/dL at 4×/day, hypoglycemia detection > 70%  
**Gold threshold:** RMSE < 18 mg/dL at 4×/day, detection > 90%  
**Failure conditions:**
- RMSE > 40 mg/dL at 4×/day (twin fundamentally requires dense data)
- Hypoglycemia detection < 30% at 7×/day (clinically dangerous)
- Any gap > 4 hours (patient sleeps) causes RMSE > 40 mg/dL
- Twin is WORSE than linear interpolation between fingersticks

---

# 9. Drift Detection

## 9.1 Abrupt Drift Detection

**Objective:** Twin must detect sudden physiological changes within clinically meaningful timeframes.

**Setup:**
- 100 patients, 14 days each
- At random time (day 5–9), introduce an abrupt change:
  - SI drops 50% (insulin resistance spike)
  - Baseline_GFR drops 30% (acute kidney injury)
  - beta_response drops 80% (partial beta cell failure)
  - Meal absorption rate doubles (gastroparesis improvement)
- Drift detector must identify the change and its timing

**Required datasets:**
- 100 patients with known change points

**Metrics:**
- Detection probability: fraction of drifts detected within clinical window
  - SI change: 24h
  - GFR change: 48h
  - Beta cell change: 24h
  - Meal absorption: 3 days
- Detection delay: time from drift onset to detection (hours)
- False positive rate: detections per day during stable period (days 1–5)
- Change point localization error: `|t_detected - t_true|` (hours)
- Magnitude estimation error: `|estimated_change - true_change| / true_change`
- Severity classification: mild (< 20%), moderate (20–50%), severe (> 50%)

**Passing threshold:** Detection probability > 0.80, false positives < 0.1/day  
**Gold threshold:** Detection probability > 0.95, false positives < 0.02/day  
**Failure conditions:**
- Detection probability < 0.50 (misses majority of drifts)
- False positive rate > 0.5/day (unusable in clinical practice — alarms constantly)
- SI change detection > 48h (too slow for clinical action)
- Change point error > 72h (detects drift but attributes to wrong cause)

## 9.2 Gradual Drift Quantification

**Objective:** Twin must not only detect but also quantify the RATE of physiological change.

**Setup:**
- 80 patients with gradual changes over 30 days:
  - SI decreases linearly: 0% → 40% over 30 days
  - Weight increases: +0.3 kg/week for 4 weeks
  - beta function declines: −2% per week
- Twin must estimate the trend slope and its uncertainty

**Required datasets:**
- 80 patients with known linear trends

**Metrics:**
- Slope estimation error: `|estimated_slope - true_slope| / |true_slope|`
- Slope uncertainty calibration: fraction of true slopes within 90% CI of estimated slope
- Trend detection delay: time until slope estimate stabilizes (slope 95% CI does not include zero for 7 consecutive days)
- Direction correctness: fraction where estimated trend direction matches true

**Passing threshold:** Slope error < 50%, direction correct > 90%  
**Gold threshold:** Slope error < 20%, direction correct > 99%, CI coverage > 0.85  
**Failure conditions:**
- Slope error > 100% (estimates order of magnitude wrong)
- Direction incorrect for > 20% of patients
- CI coverage < 0.50 (completely wrong about uncertainty of trend)
- Never detects trend (95% CI always contains zero)

## 9.3 Drift Attribution

**Objective:** Twin must identify WHICH physiological subsystem is drifting.

**Setup:**
- 60 patients with subsystem-specific drifts:
  - Metabolic only: SI changes, no CV/renal changes
  - Cardiovascular only: vascular_resistance changes
  - Renal only: baseline_GFR changes
  - Multi-system: SI + GFR change simultaneously
- Drift detector must identify the specific subsystem(s) that changed

**Required datasets:**
- 60 patients with known drift locations

**Metrics:**
- Subsystem identification accuracy: `correctly_identified / total_subsystems`
- Confusion matrix: metabolic identified as CV drift, etc.
- Detection per-subsystem ROC: AUC for each subsystem drift detector
- Early attribution: how long before drift attribution is correct (can we say "this is metabolic" before knowing the precise magnitude?)
- False attribution: fraction where a stable subsystem is flagged as drifting

**Passing threshold:** Subsystem accuracy > 0.70, AUC > 0.80 per subsystem  
**Gold threshold:** Subsystem accuracy > 0.90, AUC > 0.95 per subsystem  
**Failure conditions:**
- Subsystem accuracy < 0.40 (worse than random among 3+ subsystems)
- Any subsystem with AUC < 0.60 (cannot distinguish drift in that subsystem from noise)
- Multi-system drift attributed to a single subsystem (misses concurrent changes)
- False attribution > 30% (constantly identifying wrong cause)

---

# 10. Clinical Usefulness

## 10.1 Hypoglycemia Prediction

**Objective:** Twin must predict hypoglycemic events with sufficient lead time for prevention.

**Setup:**
- 100 patients (50 T1DM, 50 T2DM on insulin), 28 days each
- Ground truth: glucose < 54 mg/dL for ≥ 15 min (clinically significant hypoglycemia)
- Twin must predict each event at: 15 min, 30 min, 60 min lead time
- Baseline: current CGM threshold (glucose < 70 mg/dL trigger)

**Required datasets:**
- 100 patients with CGM + event labels

**Metrics:**
- Sensitivity: fraction of events predicted at each lead time
- Specificity: fraction of non-event periods correctly classified
- Positive predictive value (PPV): `TP / (TP + FP)`
- Lead time: mean prediction time before glucose < 54 mg/dL
- False alarm rate: alarms per day per patient
- Time-in-range improvement: if twin's predictions were used to trigger rescue carbs, what would the TIR improvement be?
- ROC-AUC for each lead time

**Clinical thresholds:**
- 15-min lead: actionable for fast-acting carbs
- 30-min lead: actionable for planned intervention
- 60-min lead: preventive (before exercise, before driving)

**Passing threshold:** Sensitivity > 0.85 at 15 min, > 0.70 at 30 min, > 0.50 at 60 min  
**Gold threshold:** Sensitivity > 0.95 at 15 min, > 0.85 at 30 min, > 0.70 at 60 min  
**Failure conditions:**
- Sensitivity < 0.60 at 15 min (misses almost half of events)
- False alarm rate > 3/day (alarm fatigue — clinically unsafe)
- PPV < 0.30 (too many false alarms)
- Twin never predicts events > 15 min lead time (purely reactive)

## 10.2 Treatment Recommendation

**Objective:** Twin's counterfactual treatment recommendations must match or exceed clinician performance.

**Setup:**
- 100 diabetic patients with 7 days of baseline data
- Twin recommends one of 4 treatments: metformin, SGLT2i, GLP1-RA, exercise
- Recommendation based on simulated counterfactual outcomes over next 90 days
- Compare against:
  - Random treatment assignment
  - Standard-of-care (metformin first-line)
  - 3 board-certified endocrinologists reviewing the same data
- Gold standard: ground-truth optimal treatment (from known true patient parameters)

**Required datasets:**
- 100 patients with known optimal treatment (from synthetic ground truth)
- 3 clinician recommendations per patient

**Metrics:**
- Agreement with ground-truth optimal: fraction where twin's recommendation matches the true optimal treatment
- Agreement with clinician majority: fraction where twin matches at least 2/3 clinicians
- TIR improvement over 90 days: `TIR_twin_recommended - TIR_baseline`
- Improvement vs standard of care: `TIR_twin - TIR_metformin_first_line`
- Harm prevention: fraction where twin avoids a treatment that clinicians or SoC would choose but that is actually harmful for that patient (from ground truth)
- Counterfactual accuracy: RMSE of twin's simulated outcomes vs true outcomes for the recommended treatment

**Passing threshold:** Agreement with optimal > 0.60, improvement over SoC > 3% TIR  
**Gold threshold:** Agreement with optimal > 0.80, improvement > 5% TIR, matches clinician majority > 0.75  
**Failure conditions:**
- Agreement with optimal < 0.30 (worse than random)
- Twin recommends a treatment that is HARMful (worsens outcomes) for > 10% of patients
- Twin's recommendations are worse than metformin-first-line in TIR improvement
- Agreement with clinicians < 0.30 (clinicians would reject twin's recommendations)

## 10.3 NNT Calibration

**Objective:** The twin's claimed treatment effect sizes must be well-calibrated (the effect the twin predicts should match the actual effect).

**Setup:**
- 100 patients per treatment type, 90-day counterfactual simulation
- Twin predicts: "TIR improvement with SGLT2i: mean +12%, 95% CI [8%, 16%]"
- True effect: run actual ODE with the modified parameters (ground truth)
- Compare predicted vs true effect size

**Required datasets:**
- 400 patients (100 per treatment type) with known ground-truth treatment effects

**Metrics:**
- Effect size bias: `mean(predicted_effect - true_effect)`
- Effect size RMSE: `sqrt(mean((predicted - true)^2))`
- CI coverage: fraction of 95% CIs that contain the true effect
- CI width: mean 95% CI width (narrower is better, conditional on coverage)
- NNT bias: `predicted_NNT - true_NNT`
- Clinical significance misclassification: fraction where twin says effect > 5% TIR but true effect < 5% TIR (or vice versa)
- Rank concordance: Kendall's τ between predicted and true treatment rankings across patients

**Passing threshold:** Effect bias < 3% TIR, CI coverage > 0.85, rank τ > 0.70  
**Gold threshold:** Effect bias < 1% TIR, CI coverage > 0.92, rank τ > 0.90  
**Failure conditions:**
- Effect bias > 10% TIR (twin claims 15% when truth is 5%)
- CI coverage < 0.50 (CIs are not even vaguely calibrated)
- NNT error > 10 (twin says NNT=5, truth NNT=20)
- Rank concordance < 0.30 (twin cannot tell which treatment is better for a patient)

## 10.4 Safety Guardrails

**Objective:** Twin must identify when it cannot make a reliable recommendation.

**Setup:**
- 50 patients with out-of-distribution physiology:
  - T1DM (population not in training)
  - CKD stage 4 (GFR < 30)
  - Liver failure (reduced HGP, altered drug metabolism)
  - Pregnancy (insulin resistance, altered physiology)
  - Extreme obesity (BMI > 45)
- Twin must identify these as "low confidence" or "out of distribution"
- Twin should abstain from treatment recommendations for these patients

**Required datasets:**
- 50 OOD patients across 5 conditions

**Metrics:**
- Abstention rate: fraction of OOD patients where twin reports "uncertain" or declines to recommend
- Accuracy on abstained vs non-abstained: if twin DOES recommend for OOD patients, are those recommendations correct?
- OOD detection AUC: can the twin distinguish in-distribution from OOD patients?
- Confidence score: mean predicted uncertainty for OOD vs in-distribution
- Graceful degradation: as patients become more OOD, does uncertainty increase monotonically?
- Calibration on OOD: coverage on the subset where twin expresses confidence

**Passing threshold:** Abstention > 0.50 for extreme OOD, confidence separation (OOD vs in-distribution) > 0.5σ  
**Gold threshold:** Abstention > 0.80, OOD detection AUC > 0.90, confidence separation > 2σ  
**Failure conditions:**
- Twin makes confident recommendations for OOD patients that are wrong > 50% of the time
- Twin cannot distinguish OOD from in-distribution (confidence identical)
- Twin expresses HIGH confidence for clearly impossible states (e.g., glucose = 1000 mg/dL)
- Coverage on non-abstained OOD subset < 0.50 (twin is wrong when it thinks it's right)

---

# Composite Scoring

## Overall Score

```
Overall = mean(
    Personalization_score,
    ParameterRecovery_score,
    StateEstimation_score,
    CounterfactualValidity_score,
    Calibration_score,
    Robustness_score,
    PhysiologicalRealism_score,
    Generalization_score,
    DriftDetection_score,
    ClinicalUsefulness_score
)
```

## Pass/Fail Criteria

| Level | Requirement |
|-------|------------|
| **Pass** | All 10 dimensions ≥ 0.70, no single sub-benchmark < 0.40 |
| **Gold** | All 10 dimensions ≥ 0.95 |
| **Fail** | Any dimension < 0.40 OR any catastrophic failure condition triggered |

## Catastrophic Failure Conditions (Automatic Fail)

1. Any constraint violation with glucose < 0 mg/dL (physically impossible)
2. Hypoglycemia prediction sensitivity < 0.50 at 15 min
3. Twin confidently recommends a treatment that worsens outcomes for > 10% of patients
4. Calibration max deviation > 0.20 (90% CI achieves < 70% or > 98%)
5. Effect bias > 15% TIR in treatment recommendations
6. Any population has RMSE > 60 mg/dL after 72h of data
7. Twin fails to detect SI change > 50% within 72h
8. All constraints violated simultaneously (model is generating physically impossible trajectories)

---

# Implementation Notes

## Computational Requirements

Full benchmark suite: ~5,000 patient-simulations × 14–30 days each
Estimated compute: ~1,000 CPU-hours or ~50 GPU-hours
Target runtime for full suite: < 1 week on a single workstation

## Reporting

Every benchmark report must include:
- Numerical scores for every metric
- Confidence intervals (95% bootstrap CI on each score)
- Failure analysis: which specific tests failed and why
- Comparison to baselines: untwinned model, linear model, clinician performance
- Per-patient breakdown: which patients are hardest for the twin
- Sensitivity analysis: how scores change with minor perturbations to hyperparameters

## Reproducibility

All benchmarks must use:
- Fixed random seed (42) for all simulations
- Version-locked dependencies (requirements.txt with exact versions)
- Containerized execution (Docker/Singularity)
- Published benchmark datasets or synthetic data generator with known properties
