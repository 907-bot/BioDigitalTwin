"""
FDA Regulatory Pathway Analysis — Bio Digital Twin.

Classification: The whole-body digital twin is a Software as a Medical Device
(SaMD) that generates personalized treatment recommendations from physiological
modeling. Three possible pathways:

PATHWAY A — 510(k) Clearance (12-18 months, $1-3M)
  - Predicate devices: Tidepool Loop, Dexcom G7 predictive alerts
  - Claim: Substantial equivalence to existing predictive glucose monitors
  - Requires: 50-patient prospective study comparing twin predictions to CGM
  - Risk: Medium — predicate devices exist but none with whole-body modeling

PATHWAY B — De Novo Classification (18-24 months, $3-5M)
  - No legally marketed predicate for "whole-body physiological digital twin"
  - Requires: Special controls for validation, clinical correlation
  - Requires: 100-patient study with safety endpoint
  - Benefit: Creates new device class, stronger IP position

PATHWAY C — Breakthrough Device + De Novo (12-18 months, $2-4M)
  - More streamlined than standard De Novo
  - Must show: Meaningful advantage over existing therapy
  - Typical endpoint: TIR improvement ≥ 10% vs standard of care

CLINICAL TRIAL DESIGN (Phase II Pivotal)
  - Design: Multi-site RCT, 200 patients, 2:1 randomization
  - Inclusion: T2DM, A1c 7.5-10.0%, on metformin ± one additional agent
  - Primary endpoint: TIR difference at 12 weeks (non-inferiority margin 5%)
  - Key secondary: A1c change, hypoglycemia rate, treatment satisfaction
  - Duration: 12-week intervention + 4-week follow-up
  - Sites: 3-5 US academic medical centers

CPT CODE STRATEGY
  - Tier 1: Existing RPM codes (99453, 99454, 99457) — covers CGM interpretation
  - Tier 2: New Category III code for "computational digital twin analysis"
  - Tier 3: Category I code after published RCT — typically 3-5 years post-clearance

REIMBURSEMENT ESTIMATE
  - Initial: $50-100/patient/month bundled with RPM
  - Target: $150-200/patient/month with dedicated code
  - Addressable market (US T2DM): ~30M patients × $150/mo × 10% penetration = $5.4B

COMPETITIVE LANDSCAPE
  - Twin Health: Whole-body digital twin, funded ($400M+), published pilot data
  - Virta Health: Nutritional ketosis reversal, published RCT, ~$350M raised
  - Better Therapeutics: Digital therapeutic (FDA-cleared), ~$50M raised
  - Our advantage: Calibrated uncertainty (knows when it's uncertain),
    causal counterfactual reasoning (explains WHY), open ODE model
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RegulatoryMilestone:
    name: str
    timeline_months: int
    cost_estimate_m: float
    risk_level: str
    dependencies: List[str] = field(default_factory=list)


@dataclass
class RegulatoryPathway:
    name: str
    description: str
    milestones: List[RegulatoryMilestone]
    total_timeline_months: int
    total_cost_m: float
    likelihood_of_success: float
    predicate_devices: List[str] = field(default_factory=list)
    special_controls: List[str] = field(default_factory=list)


PATHWAY_510K = RegulatoryPathway(
    name="510(k) Clearance",
    description="Substantial equivalence to predicate glucose management devices",
    milestones=[
        RegulatoryMilestone("Pre-submission to FDA", 3, 0.1, "low",
                           ["Animal/human data package"]),
        RegulatoryMilestone("Internal validation study", 6, 0.5, "low",
                           ["Calibrated twin ready"]),
        RegulatoryMilestone("510(k) submission", 1, 0.2, "low",
                           ["Validation study report"]),
        RegulatoryMilestone("FDA review", 6, 0.0, "medium",
                           ["Complete submission"]),
        RegulatoryMilestone("Post-market study plan", 2, 0.1, "low",
                           ["510(k) clearance"]),
    ],
    total_timeline_months=18,
    total_cost_m=1.2,
    likelihood_of_success=0.7,
    predicate_devices=[
        "Tidepool Loop (K213567)",
        "Dexcom G7 predictive alerts (K220765)",
    ],
    special_controls=[
        "Clinical validation of prediction accuracy",
        "Software verification and validation",
        "Cybersecurity documentation",
    ],
)


@dataclass
class ClinicalTrialDesign:
    phase: str
    design: str
    n_patients: int
    inclusion_criteria: List[str]
    primary_endpoint: str
    secondary_endpoints: List[str]
    duration_weeks: int
    sites: int


PIVOTAL_TRIAL = ClinicalTrialDesign(
    phase="II/III Pivotal",
    design="Multi-site RCT, 2:1 randomization, parallel arm",
    n_patients=200,
    inclusion_criteria=[
        "T2DM diagnosis ≥ 1 year",
        "A1c 7.5-10.0% at screening",
        "On metformin ± one additional agent (stable ≥ 3 months)",
        "Age 30-75 years",
        "CGM-naïve or willing to use study CGM",
    ],
    primary_endpoint="TIR difference (twin-guided vs standard care) at 12 weeks",
    secondary_endpoints=[
        "A1c change from baseline",
        "TBR (time below range) change",
        "Hypoglycemia event rate",
        "Treatment satisfaction (DTSQ)",
        "Number of treatment modifications per patient",
    ],
    duration_weeks=16,
    sites=5,
)
