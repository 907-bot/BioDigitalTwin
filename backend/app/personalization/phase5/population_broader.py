"""
Broader Population Priors.

Extends hierarchical Bayesian priors to cover:
  - Pediatric (age 2–17)
  - Pregnant (trimester-specific)
  - Geriatric (age 70+)
  - Athletic
  - Multi-ethnicity (Caucasian, African American, Hispanic, East Asian)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class PopulationModule:
    name: str
    age_range: Tuple[float, float]
    parameter_adjustments: Dict[str, Tuple[float, float]]
    description: str = ""


POPULATION_MODULES = {
    "pediatric_child": PopulationModule(
        name="Pediatric (Child 2–12)",
        age_range=(2, 12),
        parameter_adjustments={
            "SI": (1.2, 1.5),
            "HGP_basal": (1.3, 1.6),
            "beta_response": (1.5, 2.0),
            "baseline_GFR": (1.3, 1.8),
            "baroreflex_gain": (1.2, 1.4),
            "arterial_stiffness": (0.3, 0.6),
            "vascular_resistance": (0.5, 0.8),
            "circadian_amplitude": (1.2, 1.5),
            "lipolysis_rate": (1.2, 1.5),
            "melatonin_rate": (1.3, 1.6),
        },
        description="Children have higher insulin sensitivity, higher GFR, lower vascular resistance, more robust circadian rhythms.",
    ),
    "pediatric_adolescent": PopulationModule(
        name="Pediatric (Adolescent 13–17)",
        age_range=(13, 17),
        parameter_adjustments={
            "SI": (0.8, 1.2),
            "HGP_basal": (1.1, 1.3),
            "beta_response": (1.2, 1.5),
            "baseline_GFR": (1.1, 1.4),
            "baroreflex_gain": (1.1, 1.3),
            "lipolysis_rate": (1.3, 1.8),
            "lipogenesis_rate": (1.2, 1.5),
            "LDL_clearance": (0.8, 1.1),
            "HDL_production": (1.1, 1.3),
            "circadian_period": (0.95, 1.0),
            "melatonin_rate": (1.1, 1.3),
        },
        description="Adolescents have pubertal insulin resistance, higher lipolysis, circadian phase delay.",
    ),
    "pregnant_first": PopulationModule(
        name="Pregnant (First Trimester)",
        age_range=(18, 45),
        parameter_adjustments={
            "SI": (0.9, 1.1),
            "HGP_basal": (1.05, 1.15),
            "baseline_GFR": (1.2, 1.4),
            "vascular_resistance": (0.7, 0.9),
            "sodium_retention": (1.3, 1.6),
            "lipolysis_rate": (1.1, 1.3),
            "lipogenesis_rate": (1.2, 1.4),
            "HDL_production": (1.1, 1.3),
            "FFA_uptake": (0.8, 1.0),
        },
        description="First trimester: increased GFR, vasodilation, sodium retention, increased lipogenesis.",
    ),
    "pregnant_second": PopulationModule(
        name="Pregnant (Second Trimester)",
        age_range=(18, 45),
        parameter_adjustments={
            "SI": (0.6, 0.8),
            "HGP_basal": (1.1, 1.3),
            "beta_response": (1.3, 1.7),
            "baseline_GFR": (1.3, 1.6),
            "vascular_resistance": (0.6, 0.8),
            "sodium_retention": (1.4, 1.8),
            "lipolysis_rate": (1.3, 1.6),
            "lipogenesis_rate": (1.3, 1.6),
            "HDL_production": (1.2, 1.4),
            "M1_activation": (0.7, 0.9),
            "NFkB_sensitivity": (0.7, 0.9),
        },
        description="Second trimester: progressive insulin resistance, expanded blood volume, anti-inflammatory shift.",
    ),
    "pregnant_third": PopulationModule(
        name="Pregnant (Third Trimester)",
        age_range=(18, 45),
        parameter_adjustments={
            "SI": (0.4, 0.6),
            "HGP_basal": (1.2, 1.5),
            "beta_response": (1.5, 2.0),
            "baseline_GFR": (1.4, 1.7),
            "vascular_resistance": (0.5, 0.7),
            "sodium_retention": (1.5, 2.0),
            "lipolysis_rate": (1.5, 2.0),
            "lipogenesis_rate": (1.4, 1.8),
            "FFA_uptake": (0.6, 0.8),
            "M1_activation": (0.6, 0.8),
            "vagal_tone_effect": (1.2, 1.5),
        },
        description="Third trimester: maximal insulin resistance, peak GFR, maximal lipolysis for fetal growth.",
    ),
    "geriatric": PopulationModule(
        name="Geriatric (Age 70+)",
        age_range=(70, 110),
        parameter_adjustments={
            "SI": (0.6, 0.9),
            "HGP_basal": (0.9, 1.1),
            "beta_response": (0.5, 0.8),
            "baseline_GFR": (0.4, 0.7),
            "arterial_stiffness": (1.5, 2.5),
            "vascular_resistance": (1.2, 1.5),
            "baroreflex_gain": (0.5, 0.8),
            "autonomic_tone": (0.7, 0.9),
            "circadian_amplitude": (0.5, 0.8),
            "circadian_period": (0.97, 0.99),
            "light_sensitivity": (0.5, 0.8),
            "melatonin_rate": (0.3, 0.6),
            "lipolysis_rate": (0.7, 0.9),
            "IL6_clearance": (0.5, 0.8),
            "M1_activation": (1.3, 2.0),
            "NFkB_sensitivity": (1.2, 1.5),
        },
        description="Geriatric: sarcopenia, immunosenescence, blunted circadian rhythms, reduced baroreflex, increased arterial stiffness.",
    ),
    "athletic": PopulationModule(
        name="Athletic",
        age_range=(18, 50),
        parameter_adjustments={
            "SI": (1.3, 1.8),
            "HGP_basal": (1.1, 1.3),
            "vagal_tone_effect": (1.3, 1.8),
            "baroreflex_gain": (1.2, 1.5),
            "autonomic_tone": (0.6, 0.8),
            "lipolysis_rate": (1.3, 1.8),
            "LDL_clearance": (1.2, 1.5),
            "HDL_production": (1.2, 1.5),
            "FFA_uptake": (1.3, 1.6),
            "circadian_amplitude": (1.1, 1.3),
            "vascular_resistance": (0.7, 0.9),
            "IL6_clearance": (1.2, 1.5),
            "M1_activation": (0.7, 0.9),
        },
        description="Athletes: high insulin sensitivity, enhanced vagal tone, high lipolytic capacity, anti-inflammatory.",
    ),
}

ETHNICITY_ADJUSTMENTS = {
    "caucasian": {},
    "african_american": {
        "SI": (0.7, 0.9),
        "vascular_resistance": (1.1, 1.3),
        "sodium_retention": (1.2, 1.5),
        "arterial_stiffness": (1.1, 1.3),
        "HDL_production": (0.9, 1.1),
    },
    "hispanic": {
        "SI": (0.8, 1.0),
        "lipolysis_rate": (1.1, 1.3),
        "lipogenesis_rate": (1.1, 1.3),
        "M1_activation": (1.1, 1.3),
    },
    "east_asian": {
        "SI": (0.9, 1.1),
        "beta_response": (0.7, 0.9),
        "lipogenesis_rate": (0.8, 1.0),
        "HDL_production": (0.9, 1.0),
    },
}


PREGNANCY_TRIMESTER_MAP = {
    1: "pregnant_first",
    2: "pregnant_second",
    3: "pregnant_third",
}


def get_population_adjustment(age: float, population: Optional[str] = None,
                                ethnicity: Optional[str] = None,
                                trimester: Optional[int] = None,
                                return_ci: bool = False) -> Dict[str, float]:
    adjustments: Dict[str, float] = {}
    adjustment_ranges: Dict[str, Tuple[float, float]] = {}
    resolved_population: Optional[str] = population

    if population == "pregnant" and trimester in PREGNANCY_TRIMESTER_MAP:
        resolved_population = PREGNANCY_TRIMESTER_MAP[trimester]

    if resolved_population and resolved_population in POPULATION_MODULES:
        for k, v in POPULATION_MODULES[resolved_population].parameter_adjustments.items():
            adj = np.random.uniform(*v) if isinstance(v, tuple) else v
            adjustments[k] = adj
            adjustment_ranges[k] = v if isinstance(v, tuple) else (adj, adj)
    elif 2 <= age < 18:
        key = "pediatric_adolescent" if age >= 13 else "pediatric_child"
        for k, v in POPULATION_MODULES[key].parameter_adjustments.items():
            adjustments[k] = np.random.uniform(*v) if isinstance(v, tuple) else v
            adjustment_ranges[k] = v if isinstance(v, tuple) else (adjustments[k], adjustments[k])
    elif age >= 70:
        for k, v in POPULATION_MODULES["geriatric"].parameter_adjustments.items():
            adjustments[k] = np.random.uniform(*v) if isinstance(v, tuple) else v
            adjustment_ranges[k] = v if isinstance(v, tuple) else (adjustments[k], adjustments[k])
    if ethnicity and ethnicity in ETHNICITY_ADJUSTMENTS:
        for k, v in ETHNICITY_ADJUSTMENTS[ethnicity].items():
            existing = adjustments.get(k, 1.0)
            eth_factor = np.random.uniform(*v) if isinstance(v, tuple) else v
            adjustments[k] = existing * eth_factor
            if k in adjustment_ranges:
                lo, hi = adjustment_ranges[k]
                eth_lo, eth_hi = v if isinstance(v, tuple) else (v, v)
                adjustment_ranges[k] = (lo * eth_lo, hi * eth_hi)
            else:
                adjustment_ranges[k] = v if isinstance(v, tuple) else (eth_factor, eth_factor)
    if return_ci:
        result = {}
        for k, v in adjustments.items():
            lo, hi = adjustment_ranges.get(k, (v, v))
            ci_half = (hi - lo) / (2 * 1.96)
            result[k] = {"mean": v, "ci_95": (lo, hi), "ci_half_width": ci_half,
                         "cv": ci_half / max(abs(v), 0.01)}
        return result
    return adjustments


def get_population_adjustment_with_uncertainty(
    age: float, population: Optional[str] = None,
    ethnicity: Optional[str] = None,
    trimester: Optional[int] = None,
    n_bootstrap: int = 1000,
) -> Dict[str, Dict]:
    """
    Compute population adjustment with bootstrap confidence intervals.

    Returns per-parameter: mean, 95% CI, and coefficient of variation.
    The CV quantifies uncertainty: CV < 0.1 = well-characterized,
    CV > 0.3 = poorly characterized adjustment.
    """
    samples: Dict[str, List[float]] = {}
    for _ in range(n_bootstrap):
        adj = get_population_adjustment(age, population, ethnicity, trimester)
        for k, v in adj.items():
            if k not in samples:
                samples[k] = []
            samples[k].append(v)

    result = {}
    for k, values in samples.items():
        arr = np.array(values)
        mean = float(np.mean(arr))
        lo = float(np.percentile(arr, 2.5))
        hi = float(np.percentile(arr, 97.5))
        cv = float(np.std(arr, ddof=1) / max(abs(mean), 0.01))
        result[k] = {
            "mean": mean,
            "ci_95": (lo, hi),
            "ci_half_width": float((hi - lo) / 2),
            "cv": cv,
            "well_characterized": cv < 0.3,
        }
    return result


def adjust_priors_for_population(priors, age: float, population: Optional[str] = None,
                                  ethnicity: Optional[str] = None,
                                  trimester: Optional[int] = None):
    adjustments = get_population_adjustment(age, population, ethnicity, trimester)
    adjusted = []
    for i, prior in enumerate(priors):
        from app.personalization.priors import PARAMETER_NAMES
        if i < len(PARAMETER_NAMES) and PARAMETER_NAMES[i] in adjustments:
            factor = adjustments[PARAMETER_NAMES[i]]
            if hasattr(prior, 'mu'):
                prior.mu *= factor
            adjusted.append(prior)
        else:
            adjusted.append(prior)
    return adjusted
