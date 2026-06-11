"""
Phase 3: Hierarchical Bayesian Priors.

Three layers:
  Layer 1 (Population prior):  Broad priors for the general population
  Layer 2 (Subpopulation):      Adjusted by age/sex/disease group
  Layer 3 (Individual):        Patient-specific posterior from UKF

Phase 3 parameter additions:
  Circadian (4): circadian_period, circadian_amplitude, light_sensitivity, melatonin_rate
  Adipose   (5): lipolysis_rate, lipogenesis_rate, LDL_clearance, HDL_production, FFA_uptake
  Immune    (4): M1_activation, NFkB_sensitivity, vagal_tone_effect, IL6_clearance
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from .core import LogNormalPrior, NormalPrior, TruncatedNormalPrior, PriorDistribution

# ===================================================================
# Layer 1: Population-Level Priors (unchanged Phase 2 + Phase 3)
# ===================================================================

# ── Metabolic (Phase 1, unchanged) ──
SI_PRIOR = LogNormalPrior(mu=-4.0, sigma=0.5)
HGP_BASAL_PRIOR = NormalPrior(mu=2.0, sigma=0.3)
BETA_RESPONSE_PRIOR = LogNormalPrior(mu=-6.0, sigma=0.4)
RT_PRIOR = TruncatedNormalPrior(mu=180, sigma=15, low=140, high=220)

# ── Cardiovascular (Phase 2, unchanged) ──
ARTERIAL_STIFFNESS_PRIOR = LogNormalPrior(mu=2.5, sigma=0.3)
VASCULAR_RESISTANCE_PRIOR = LogNormalPrior(mu=4.5, sigma=0.25)
BAROREFLEX_GAIN_PRIOR = LogNormalPrior(mu=0.5, sigma=0.3)
AUTONOMIC_TONE_PRIOR = NormalPrior(mu=0.5, sigma=0.1)

# ── Renal (Phase 2, unchanged) ──
BASELINE_GFR_PRIOR = TruncatedNormalPrior(mu=100, sigma=15, low=15, high=180)
RENAL_SENSITIVITY_PRIOR = NormalPrior(mu=0.6, sigma=0.1)
SGLT_ACTIVITY_PRIOR = LogNormalPrior(mu=3.5, sigma=0.3)
SODIUM_RETENTION_PRIOR = NormalPrior(mu=0.5, sigma=0.1)

# ── Circadian (Phase 3) ──
CIRCADIAN_PERIOD_PRIOR = NormalPrior(mu=1440.0, sigma=30.0)        # ~24h ± 30min
CIRCADIAN_AMPLITUDE_PRIOR = NormalPrior(mu=0.8, sigma=0.15)        # oscillation amplitude
LIGHT_SENSITIVITY_PRIOR = LogNormalPrior(mu=-1.2, sigma=0.3)       # phase-shift sensitivity
MELATONIN_RATE_PRIOR = NormalPrior(mu=0.5, sigma=0.15)             # melatonin production rate

# ── Adipose-Lipid (Phase 3) ──
LIPOLYSIS_RATE_PRIOR = LogNormalPrior(mu=-3.0, sigma=0.3)          # FFA release rate
LIPOGENESIS_RATE_PRIOR = LogNormalPrior(mu=-3.9, sigma=0.3)        # FFA uptake rate
LDL_CLEARANCE_PRIOR = LogNormalPrior(mu=-4.2, sigma=0.25)          # LDL receptor clearance
HDL_PRODUCTION_PRIOR = LogNormalPrior(mu=-4.6, sigma=0.25)         # HDL production
FFA_UPTAKE_PRIOR = LogNormalPrior(mu=-3.5, sigma=0.3)              # Tissue FFA uptake

# ── Immune-Inflammatory (Phase 3) ──
M1_ACTIVATION_PRIOR = LogNormalPrior(mu=-2.3, sigma=0.3)           # M1 polarization rate
NFKB_SENSITIVITY_PRIOR = NormalPrior(mu=0.5, sigma=0.15)           # NF-κB gain
VAGAL_TONE_EFFECT_PRIOR = NormalPrior(mu=0.3, sigma=0.1)           # ACh anti-inflammatory
IL6_CLEARANCE_PRIOR = LogNormalPrior(mu=-3.9, sigma=0.25)          # IL-6 clearance

# ── Complete parameter list ──
PRIORS: List[PriorDistribution] = [
    # Metabolic (0-3)
    SI_PRIOR, HGP_BASAL_PRIOR, BETA_RESPONSE_PRIOR, RT_PRIOR,
    # Cardiovascular (4-7)
    ARTERIAL_STIFFNESS_PRIOR, VASCULAR_RESISTANCE_PRIOR, BAROREFLEX_GAIN_PRIOR, AUTONOMIC_TONE_PRIOR,
    # Renal (8-11)
    BASELINE_GFR_PRIOR, RENAL_SENSITIVITY_PRIOR, SGLT_ACTIVITY_PRIOR, SODIUM_RETENTION_PRIOR,
    # Circadian (12-15)
    CIRCADIAN_PERIOD_PRIOR, CIRCADIAN_AMPLITUDE_PRIOR, LIGHT_SENSITIVITY_PRIOR, MELATONIN_RATE_PRIOR,
    # Adipose (16-20)
    LIPOLYSIS_RATE_PRIOR, LIPOGENESIS_RATE_PRIOR, LDL_CLEARANCE_PRIOR, HDL_PRODUCTION_PRIOR, FFA_UPTAKE_PRIOR,
    # Immune (21-24)
    M1_ACTIVATION_PRIOR, NFKB_SENSITIVITY_PRIOR, VAGAL_TONE_EFFECT_PRIOR, IL6_CLEARANCE_PRIOR,
]

PARAMETER_NAMES = [
    # Metabolic (0-3)
    "SI", "HGP_basal", "beta_response", "RT",
    # Cardiovascular (4-7)
    "arterial_stiffness", "vascular_resistance", "baroreflex_gain", "autonomic_tone",
    # Renal (8-11)
    "baseline_GFR", "renal_sensitivity", "SGLT_activity", "sodium_retention",
    # Circadian (12-15)
    "circadian_period", "circadian_amplitude", "light_sensitivity", "melatonin_rate",
    # Adipose (16-20)
    "lipolysis_rate", "lipogenesis_rate", "LDL_clearance", "HDL_production", "FFA_uptake",
    # Immune (21-24)
    "M1_activation", "NFkB_sensitivity", "vagal_tone_effect", "IL6_clearance",
]

PARAMETER_RANGES = {
    # Metabolic
    "SI": (0.001, 0.10),
    "HGP_basal": (0.5, 5.0),
    "beta_response": (0.0001, 0.05),
    "RT": (100, 300),
    # Cardiovascular
    "arterial_stiffness": (5, 50),
    "vascular_resistance": (10, 150),
    "baroreflex_gain": (0.5, 5.0),
    "autonomic_tone": (0.1, 0.9),
    # Renal
    "baseline_GFR": (15, 180),
    "renal_sensitivity": (0.2, 0.9),
    "SGLT_activity": (10, 200),
    "sodium_retention": (0.1, 0.9),
    # Circadian
    "circadian_period": (1350, 1530),
    "circadian_amplitude": (0.2, 1.5),
    "light_sensitivity": (0.05, 1.0),
    "melatonin_rate": (0.1, 1.0),
    # Adipose
    "lipolysis_rate": (0.005, 0.5),
    "lipogenesis_rate": (0.002, 0.2),
    "LDL_clearance": (0.005, 0.05),
    "HDL_production": (0.003, 0.03),
    "FFA_uptake": (0.005, 0.1),
    # Immune
    "M1_activation": (0.01, 0.5),
    "NFkB_sensitivity": (0.1, 1.0),
    "vagal_tone_effect": (0.05, 0.6),
    "IL6_clearance": (0.005, 0.1),
}

# ===================================================================
# Layer 2: Subpopulation Prior Adjustments
# ===================================================================

SUBGROUP_ADJUSTMENTS = {
    "age_gt_60": {
        "SI": (0.7, 1.0),          # Lower SI in elderly
        "arterial_stiffness": (1.2, 2.0),  # Higher stiffness
        "baseline_GFR": (0.6, 1.0),        # Lower GFR
        "circadian_amplitude": (0.7, 1.0), # Blunted circadian
        "M1_activation": (1.2, 2.0),       # Higher inflammation
        "lipolysis_rate": (1.1, 1.3),      # Higher lipolysis
    },
    "obese_bmi_gt_30": {
        "SI": (0.3, 0.7),
        "lipolysis_rate": (1.3, 2.0),
        "M1_activation": (1.5, 3.0),
        "NFkB_sensitivity": (1.2, 1.5),
        "LDL_clearance": (0.7, 1.0),
        "sodium_retention": (1.1, 1.3),
    },
    "diabetes_t2": {
        "SI": (0.2, 0.5),
        "beta_response": (0.3, 0.7),
        "HGP_basal": (1.1, 1.5),
        "M1_activation": (1.3, 2.0),
    },
    "hypertension": {
        "vascular_resistance": (1.2, 1.5),
        "arterial_stiffness": (1.1, 1.5),
        "sodium_retention": (1.1, 1.4),
        "baroreflex_gain": (0.7, 1.0),
    },
    "ckd_stage3": {
        "baseline_GFR": (0.3, 0.6),
        "renal_sensitivity": (1.2, 1.5),
    },
    "female": {
        "HDL_production": (1.1, 1.4),
        "lipolysis_rate": (0.8, 1.0),
    },
}


def _get_distribution_mean(prior: PriorDistribution) -> float:
    """Get the actual distribution mean (not the log-space mu for LogNormal)."""
    if isinstance(prior, LogNormalPrior):
        return np.exp(prior.mu + 0.5 * prior.sigma ** 2)
    elif isinstance(prior, NormalPrior):
        return prior.mu
    elif isinstance(prior, TruncatedNormalPrior):
        # Approximate mean of truncated normal
        from scipy.stats import truncnorm
        a, b = (prior.low - prior.mu) / prior.sigma, (prior.high - prior.mu) / prior.sigma
        return float(truncnorm.mean(a, b, loc=prior.mu, scale=prior.sigma))
    return 0.0


def get_subgroup_priors(
    age: float = 35.0,
    sex: str = "male",
    bmi: float = 24.0,
    has_diabetes: bool = False,
    has_hypertension: bool = False,
    has_ckd: bool = False,
) -> List[PriorDistribution]:
    """
    Generate layer-2 subgroup-adjusted priors.

    Adjusts population priors by scaling means for known subgroups.
    Returns a copy of PRIORS with subgroup-appropriate means.
    """
    from copy import deepcopy
    adjusted = deepcopy(PRIORS)

    subgroups = []
    if age > 60:
        subgroups.append("age_gt_60")
    if bmi > 30:
        subgroups.append("obese_bmi_gt_30")
    if has_diabetes:
        subgroups.append("diabetes_t2")
    if has_hypertension:
        subgroups.append("hypertension")
    if has_ckd:
        subgroups.append("ckd_stage3")
    if sex == "female":
        subgroups.append("female")

    for subgroup in subgroups:
        adjustments = SUBGROUP_ADJUSTMENTS.get(subgroup, {})
        for param_name, (scale_lo, scale_hi) in adjustments.items():
            if param_name in PARAMETER_NAMES:
                idx = PARAMETER_NAMES.index(param_name)
                prior = adjusted[idx]
                scale = (scale_lo + scale_hi) / 2.0
                if hasattr(prior, 'mu') and hasattr(prior, 'sigma'):
                    current_mean = _get_distribution_mean(prior)
                    new_mean = current_mean * scale
                    adjusted[idx] = _rebuild_prior(type(prior), new_mean, prior.sigma, prior)
    return adjusted


def _rebuild_prior(prior_type, mu, sigma, original):
    """Rebuild a prior distribution with a new mean (mu is the new distribution mean, not log)."""
    if prior_type == LogNormalPrior:
        # LogNormal: mean = exp(mu_log + sigma^2/2)
        # Solve for new mu_log: new_mu_log = log(new_mean) - sigma^2/2
        new_mu = np.log(max(mu, 1e-10)) - 0.5 * sigma ** 2
        return LogNormalPrior(mu=new_mu, sigma=sigma)
    elif prior_type == NormalPrior:
        return NormalPrior(mu=mu, sigma=sigma)
    elif prior_type == TruncatedNormalPrior:
        return TruncatedNormalPrior(mu=mu, sigma=sigma, low=original.low, high=original.high)
    return original


def validate_parameter(name: str, value: float) -> bool:
    if name not in PARAMETER_RANGES:
        return True
    lo, hi = PARAMETER_RANGES[name]
    return lo <= value <= hi


def validate_all_parameters(params: np.ndarray) -> List[bool]:
    return [
        validate_parameter(PARAMETER_NAMES[i], params[i])
        if i < len(params) else False
        for i in range(len(PARAMETER_NAMES))
    ]


STATE_NAMES = [
    "G","I","HGP","PGU","IR",
    "SBP","DBP","HR","HRV",
    "GFR","Na","K","Osm",
    "CRP",
    "CLOCK_BMAL1","PER_CRY","cortisol","melatonin",
    "circadian_phase","sleep_pressure",
    "fat_mass","FFA","LDL","HDL","TG",
    "IL6_proxy","TNFa_proxy","M1_M2_ratio","NFkB_activity","InflammatoryLoad",
]

__all__ = [
    "PRIORS", "PARAMETER_NAMES", "STATE_NAMES", "PARAMETER_RANGES",
    "validate_parameter", "validate_all_parameters", "PriorDistribution",
    "get_subgroup_priors", "SUBGROUP_ADJUSTMENTS",
]
