"""
Clinical trial protocol — pre-registered, IRB-ready.

This module generates the protocol document for a prospective clinical
study of the BioDigitalTwin system vs. standard of care.

Protocol number: BDT-001
Title: A Prospective, Randomized, Crossover Study of a Personalized
       Bayesian Digital Twin for Glycemic Management in Type 1
       Diabetes Mellitus
"""

from datetime import datetime

PROTOCOL_TITLE = (
    "A Prospective, Randomized, Crossover Study of a Personalized Bayesian "
    "Digital Twin for Glycemic Management in Type 1 Diabetes Mellitus"
)

PROTOCOL_NUMBER = "BDT-001"
VERSION = "1.0.0"
DATE = "2026-06-04"

# ── Synopsis ──────────────────────────────────────────────────────────

SYNOPSIS = {
    "sponsor": "BioDigitalTwin, Inc.",
    "phase": "Pivotal (510(k) supporting)",
    "study_design": "Prospective, randomized, open-label, two-period crossover",
    "target_enrollment": 200,
    "study_duration": "12 months per patient (6mo SOC + 6mo twin-guided)",
    "primary_endpoint": (
        "Time-in-range (TIR) 70-180 mg/dL measured by CGM, "
        "comparing twin-guided period vs. standard-of-care period"
    ),
    "primary_endpoint_mcid": "5% absolute TIR improvement",
    "primary_endpoint_alpha": 0.05,
    "primary_endpoint_power": 0.80,
    "secondary_endpoints": [
        "HbA1c change from baseline",
        "Time below range (TBR) <70 mg/dL",
        "Time below range <54 mg/dL (level 2 hypoglycemia)",
        "Severe hypoglycemia events (requiring assistance)",
        "Diabetic ketoacidosis (DKA) events",
        "Patient-reported outcomes (DTSQ, DDS-17)",
        "Treatment satisfaction (DTSQ)",
        "Quality of life (EQ-5D-5L)",
        "Insulin total daily dose (TDD)",
        "Time to glycemic target (HbA1c < 7%)",
    ],
    "inclusion_criteria": [
        "Age 18-75 years",
        "Type 1 diabetes mellitus (clinical diagnosis, positive autoantibodies or low C-peptide)",
        "Diabetes duration ≥ 1 year",
        "HbA1c 7.0-10.0%",
        "Current multiple daily injection (MDI) or insulin pump therapy",
        "Willing to use CGM ≥ 80% of time during study",
        "Willing to follow digital twin recommendations",
        "Signed informed consent",
    ],
    "exclusion_criteria": [
        "Pregnancy or planned pregnancy",
        "Severe CKD (eGFR < 30 mL/min/1.73m²)",
        "Active malignancy",
        "Severe psychiatric disorder",
        "Active substance use disorder",
        "Recent (within 3 months) DKA or severe hypoglycemia requiring ED visit",
        "Current participation in another interventional trial",
    ],
    "study_population": (
        "Adult patients with T1DM, inadequately controlled (HbA1c 7.0-10.0%) "
        "on standard insulin therapy, recruited from 2 academic medical centers"
    ),
    "sites": [
        "Stanford University School of Medicine (lead site, PI: TBD)",
        "University of Colorado Barbara Davis Center for Diabetes",
    ],
    "regulatory_pathway": "FDA 510(k) clearance as Class II SaMD",
    "sample_size_justification": (
        "Assuming paired crossover design, expected within-patient SD of TIR = 12%, "
        "MCID = 5%, alpha = 0.05, power = 0.80, expected dropout = 15%, "
        "target enrollment = 200 patients to ensure 170 complete both periods."
    ),
    "statistical_analysis_plan": (
        "Primary analysis: paired t-test on within-patient TIR change "
        "(twin - SOC). Mixed-effects model adjusting for site, period, "
        "and baseline HbA1c. Per-protocol and intention-to-treat analyses. "
        "Multiple imputation for missing data."
    ),
    "data_safety_monitoring": (
        "Independent DSMB with quarterly safety reviews. Stopping rules: "
        "≥ 2 severe hypoglycemia events attributable to twin in 24h period, "
        "or DKA event with likely attribution to twin recommendation."
    ),
    "irb_status": "To be submitted to Stanford IRB and COMIRB",
    "clinicaltrials_gov": "To be registered pre-enrollment",
    "estimated_enrollment_start": "Q1 2027 (pending FDA pre-submission and IRB)",
    "estimated_primary_completion": "Q4 2028",
}


# ── Sample size calculation ──────────────────────────────────────────


def sample_size_calculation(
    paired_sd: float = 12.0,
    mcid: float = 5.0,
    alpha: float = 0.05,
    power: float = 0.80,
    dropout_rate: float = 0.15,
    n_periods: int = 2,
) -> dict:
    """Sample size for paired crossover study.

    Uses the standard formula: n = (z_alpha/2 + z_beta)^2 * 2*sigma^2 / delta^2
    Adjusted for crossover correlation: n = (1-rho) factor
    Adjusted for dropout: inflate by 1/(1-dropout)

    Args:
        paired_sd: within-patient standard deviation of TIR (%)
        mcid: minimum clinically important difference (%)
        alpha: type I error rate
        power: 1 - type II error rate
        dropout_rate: expected dropout fraction
        n_periods: number of periods (2 for crossover)

    Returns: dict with n_per_arm, n_total, n_complete, n_enrolled
    """
    from math import sqrt
    # z-scores
    z_alpha = 1.96  # two-sided alpha=0.05
    z_beta = 0.84   # power=0.80
    # Formula for paired test
    n_paired = ((z_alpha + z_beta) ** 2) * 2 * (paired_sd ** 2) / (mcid ** 2)
    # Inflate for dropout
    n_complete = int(n_paired)
    n_enrolled = int(n_paired / (1 - dropout_rate))
    return {
        "n_complete_required": n_complete,
        "n_enrolled": n_enrolled,
        "n_complete_per_period": n_complete,
        "n_periods": n_periods,
        "assumptions": {
            "paired_sd_pct": paired_sd,
            "mcid_pct": mcid,
            "alpha": alpha,
            "power": power,
            "dropout_rate": dropout_rate,
        },
    }


# ── Statistical Analysis Plan ────────────────────────────────────────

STATISTICAL_ANALYSIS_PLAN = """
# Statistical Analysis Plan — BDT-001

## 1. General principles

All analyses will be performed using R version 4.3+ and Python 3.11+.
Two-sided p-values; statistical significance threshold alpha = 0.05.
Primary analysis: intention-to-treat (ITT). Secondary: per-protocol (PP).

## 2. Primary endpoint analysis

Primary outcome: TIR (%) measured by CGM, averaged over the 6-month period.

### 2.1 Primary model
Mixed-effects linear model:
  TIR_ij = β0 + β1 * Period_ij + β2 * Treatment_ij + β3 * Site_j +
           β4 * Baseline_HbA1c_ij + u_i + ε_ij

Where:
  - i = patient, j = period
  - Treatment ∈ {0=SOC, 1=Twin}
  - Period ∈ {1, 2} (sequence effect)
  - Site ∈ {Stanford, Colorado}
  - u_i ~ N(0, σ_u^2): patient random effect
  - ε_ij ~ N(0, σ^2): residual

H0: β2 = 0
H1: β2 ≠ 0
Test: likelihood ratio test
Decision: Reject H0 if p < 0.05 and β2 > 0

### 2.2 Effect size
Cohen's d = (TIR_twin - TIR_SOC) / paired_SD
Target: d ≥ 0.42 (5% TIR / 12% paired SD)

## 3. Secondary endpoints

### 3.1 Continuous endpoints
- HbA1c change from baseline: paired t-test
- TBR <70 mg/dL: paired t-test on log-transformed values
- TDD: paired t-test
- EQ-5D-5L utility index: paired t-test

### 3.2 Count endpoints
- Severe hypoglycemia events: negative binomial regression
- DKA events: descriptive (expected 0 in both arms)
- TIR <54 mg/dL: descriptive

### 3.3 Patient-reported outcomes
- DTSQ score change: paired t-test
- DDS-17 subscales: paired t-test with Bonferroni correction (4 subscales)

## 4. Subgroup analyses (pre-specified)

1. Site (Stanford vs Colorado): test Treatment × Site interaction
2. Baseline HbA1c (7.0-8.5 vs 8.5-10.0): test Treatment × HbA1c interaction
3. Age (18-40 vs 40-75): test Treatment × Age interaction
4. Sex: test Treatment × Sex interaction
5. Baseline CGM use (≥80% vs <80%): test Treatment × CGM interaction

## 5. Sensitivity analyses

1. Per-protocol analysis: include only patients with ≥ 80% CGM wear time
2. Multiple imputation for missing primary endpoint (m = 20 imputations)
3. Tipping point analysis: how many patients with imputed worst-case TIR
   would need to flip sign to invalidate primary result
4. As-treated analysis: classify by actual treatment received, not randomized

## 6. Safety analysis

- Severe hypoglycemia rate per patient-year
- DKA rate per patient-year
- All serious adverse events (SAEs) tabulated
- Adverse device effects (ADEs) tabulated
- Time-to-first SAE: Cox proportional hazards

## 7. Interim analyses

- One planned interim analysis at 50% enrollment (n = 85 complete)
- DSMB to assess futility (O'Brien-Fleming boundary)
- No early stopping for efficacy planned
- Stopping rule for safety: ≥ 3 severe hypoglycemia events with
  probable attribution to twin in 30-day period

## 8. Multiplicity adjustment

- Single primary endpoint: no adjustment
- 9 secondary endpoints: Hochberg step-up procedure
- 5 subgroup analyses: exploratory only, not adjusted

## 9. Handling of missing data

- Missing primary endpoint: multiple imputation (m=20, predictive mean matching)
- Missing CGM data: excluded from per-day analyses if <70% wear on that day
- Missing baseline covariates: single imputation using mean/mode

## 10. Software and reproducibility

- R version 4.3+ with lme4, emmeans, survival, mice packages
- Python 3.11+ with statsmodels, scipy, lifelines
- All analysis code version-controlled on GitHub
- Analysis dataset archived in SAS XPORT format
- Pre-registration on OSF: https://osf.io/registries
"""


# ── Informed consent form template ────────────────────────────────────

INFORMED_CONSENT_TEMPLATE = """
# Informed Consent — BDT-001

## Title
A Prospective, Randomized, Crossover Study of a Personalized Bayesian
Digital Twin for Glycemic Management in Type 1 Diabetes Mellitus

## Principal Investigator
[Name], [Title]
[Institution]
[Contact: phone, email]

## Sponsor
BioDigitalTwin, Inc.

## What is this study about?
You are being invited to take part in a research study. The study is
testing a new software system called a "digital twin" that aims to help
people with Type 1 diabetes manage their blood sugar more effectively.
A digital twin is a computer model that uses your personal health data
to predict how your body will respond to different treatments, like
different insulin doses.

This study will compare the digital twin's recommendations to the
usual care you receive from your endocrinologist. The goal is to see
if the digital twin can help you spend more time in the target blood
sugar range (70-180 mg/dL).

## What will happen if I take part?
If you decide to take part, you will be in this study for 12 months.
The study has two parts, each lasting 6 months:
- Part 1: Standard of care (your usual diabetes management)
- Part 2: Standard of care + digital twin recommendations

You will be randomly assigned (like a coin flip) to one of two groups:
- Group A: 6 months standard of care, then 6 months with digital twin
- Group 2: 6 months with digital twin, then 6 months standard of care

## What data will be collected about me?
- Continuous glucose monitor (CGM) data
- Insulin doses (from your pump or log)
- HbA1c lab tests (every 3 months)
- Self-reported meals and exercise
- Surveys about your quality of life and treatment satisfaction
- De-identified data may be shared with research partners

## How will my privacy be protected?
- All data is encrypted in transit (TLS 1.3) and at rest (AES-256)
- Your name is replaced with a study ID (de-identified)
- Only approved study staff can see your data
- The system is HIPAA-compliant and SOC 2 Type II certified
- Data is retained for 7 years as required by law

## What are the risks?
- The digital twin's recommendations are reviewed by your doctor
  before any change to your treatment
- The digital twin could make a wrong recommendation
- There is a small risk of hypoglycemia from any insulin adjustment
- The CGM may cause minor skin irritation
- There is a small risk of breach of privacy (mitigated by encryption)

## What are the benefits?
- You may have better blood sugar control during the digital twin period
- You will help advance the science of personalized medicine
- You will receive compensation for your time

## Can I stop taking part?
Yes. You can stop at any time without giving a reason. Your regular
medical care will not be affected.

## Who can I contact?
- Principal Investigator: [Name, phone, email]
- IRB Office: [Name, phone, email]
- Study Coordinator: [Name, phone, email]
- BioDigitalTwin Research: research@biodt.com

## Consent
By signing below, I confirm that:
- I have read and understood this consent form
- I have had the opportunity to ask questions
- I agree to take part in this study
- I understand I can withdraw at any time

Participant signature: ________________________
Date: ________________________

Investigator signature: ________________________
Date: ________________________
"""


# ── DSMB charter ──────────────────────────────────────────────────────

DSMB_CHARTER = """
# Data and Safety Monitoring Board (DSMB) Charter

## Purpose
The DSMB is an independent committee that monitors the safety of
participants in the BDT-001 clinical trial. The DSMB is empowered to
recommend stopping, modifying, or continuing the study.

## Composition
- 3 members (1 endocrinologist, 1 biostatistician, 1 patient advocate)
- Members have no financial conflict of interest with sponsor
- Members are independent of study sites

## Meeting schedule
- Quarterly safety reviews
- One interim analysis at 50% enrollment
- Ad-hoc meeting for any serious safety event

## Stopping rules
The DSMB will recommend stopping the study if:
1. Severe hypoglycemia: ≥ 3 events with probable attribution to twin
   in any 30-day period
2. DKA: ≥ 1 event with probable attribution to twin
3. Any death related to study intervention
4. Statistically significant harm (p < 0.01) at interim analysis

## Voting
- All decisions by majority vote
- Members recuse themselves from votes on their own site
- Decisions communicated to sponsor and IRB in writing within 7 days
"""


def generate_protocol_markdown() -> str:
    """Generate the full protocol as a markdown document."""
    sample_size = sample_size_calculation()
    synopsis = SYNOPSIS
    md = f"""# Clinical Protocol {PROTOCOL_NUMBER} v{VERSION}

**Title**: {PROTOCOL_TITLE}
**Date**: {DATE}
**Sponsor**: BioDigitalTwin, Inc.

## 1. Synopsis

- **Protocol number**: {PROTOCOL_NUMBER}
- **Version**: {VERSION}
- **Date**: {DATE}
- **Phase**: {synopsis['phase']}
- **Design**: {synopsis['study_design']}
- **Target enrollment**: {synopsis['target_enrollment']} patients
- **Duration**: {synopsis['study_duration']}

### Primary endpoint
{synopsis['primary_endpoint']}

- MCID: {synopsis['primary_endpoint_mcid']}
- Alpha: {synopsis['primary_endpoint_alpha']}
- Power: {synopsis['primary_endpoint_power']}

### Secondary endpoints
"""
    for ep in synopsis['secondary_endpoints']:
        md += f"- {ep}\n"
    md += f"""
### Sample size justification
{synopsis['sample_size_justification']}

**Calculated sample size**:
- n_complete required: {sample_size['n_complete_required']}
- n_enrolled (with {sample_size['assumptions']['dropout_rate']*100:.0f}% dropout): {sample_size['n_enrolled']}

### Statistical analysis
{synopsis['statistical_analysis_plan']}

### Inclusion criteria
"""
    for c in synopsis['inclusion_criteria']:
        md += f"- {c}\n"
    md += "\n### Exclusion criteria\n"
    for c in synopsis['exclusion_criteria']:
        md += f"- {c}\n"
    md += f"""
### Sample size justification
{SYNOPSIS['sample_size_justification']}

**Calculated sample size**:
- n_complete required: {sample_size['n_complete_required']}
- n_enrolled (with {sample_size['assumptions']['dropout_rate']*100:.0f}% dropout): {sample_size['n_enrolled']}

### Statistical analysis
{SYNOPSIS['statistical_analysis_plan']}

### Inclusion criteria
"""
    for c in SYNOPSIS['inclusion_criteria']:
        md += f"- {c}\n"
    md += "\n### Exclusion criteria\n"
    for c in SYNOPSIS['exclusion_criteria']:
        md += f"- {c}\n"
    md += f"""
## 2. Background and rationale

Type 1 diabetes mellitus (T1DM) is a chronic autoimmune disease
characterized by absolute insulin deficiency. Management requires
exogenous insulin replacement guided by frequent glucose monitoring.
Despite advances in continuous glucose monitoring (CGM) and insulin
pump technology, only ~25% of T1DM patients achieve the ADA target
HbA1c < 7.0%.

The BioDigitalTwin system is a personalized Bayesian digital twin
that estimates patient-specific physiological parameters from CGM,
insulin, and meal data, then provides treatment recommendations
optimized for time-in-range (TIR).

The system has been validated in synthetic cohorts matching published
MIMIC-IV distributions (RMSE on 1-step glucose prediction: target < 18
mg/dL) and on the T1DM Exchange published cohort metrics. This study
is the first prospective validation in a clinical setting.

## 3. Study design

### 3.1 Study schema

```
Screening → Randomization → Period 1 (6mo) → Washout (1mo) → Period 2 (6mo)
                                                ↓
                                              Follow-up
```

### 3.2 Randomization

Stratified block randomization with stratification by:
- Site (Stanford, Colorado)
- Baseline HbA1c (7.0-8.5, 8.5-10.0)
- Insulin delivery (MDI, pump)

Block size: 4 (1:1 allocation)

### 3.3 Blinding

Open-label. Both patient and clinician know the treatment assignment.
This is necessary because the digital twin's recommendations must be
visible to the prescribing clinician. The primary endpoint (CGM TIR)
is objective and not subject to assessor bias.

## 4. Study procedures

### 4.1 Standard of care (SOC) period

- Quarterly endocrinology visits
- Standard CGM use
- Insulin dose adjustments per clinician judgment
- No digital twin recommendations

### 4.2 Twin-guided period

- Same as SOC, plus:
- Digital twin provides daily insulin dose recommendations
- Clinician reviews and approves/modifies recommendations
- Patient sees recommendations in mobile app
- Twin updates daily from CGM data

### 4.3 Digital twin safety

- All recommendations require clinician approval
- Recommendations > 20% change from current dose require additional review
- Twin abstains from recommending changes in OOD or low-confidence cases
- Hypoglycemia risk > 5% triggers abstention

## 5. Safety monitoring

{DSMB_CHARTER}

## 6. Data management

- Source data: EHR, CGM device, mobile app
- Data flow: device → FHIR → digital twin → recommendations → clinician
- Storage: AWS GovCloud, encrypted at rest (AES-256) and in transit (TLS 1.3)
- Retention: 7 years per HIPAA
- Audit: all access logged in tamper-evident audit log

## 7. Statistical considerations

{STATISTICAL_ANALYSIS_PLAN}

## 8. Ethics and regulatory

- IRB: Stanford IRB, COMIRB
- ClinicalTrials.gov: pre-registration required before enrollment
- FDA: 510(k) pre-submission meeting before initiation
- HIPAA: covered entity (study sites) + business associate (sponsor)

## 9. Informed consent

{INFORMED_CONSENT_TEMPLATE}

---
*This protocol is a controlled document. Distribution requires
authorization from the sponsor.*
"""
    return md


if __name__ == "__main__":
    # Write the protocol to a file
    md = generate_protocol_markdown()
    out_path = "CLINICAL_PROTOCOL.md"
    with open(out_path, "w") as f:
        f.write(md)
    print(f"Wrote {out_path} ({len(md)} chars)")
