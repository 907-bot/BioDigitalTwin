"""
Clinical Study Protocol Framework.

Formalizes a clinical trial design for digital twin validation.
Includes power analysis, endpoint definitions, inclusion/exclusion criteria,
statistical analysis plan, and feasibility assessment.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any
from enum import Enum
import numpy as np
from scipy import stats


class StudyPhase(Enum):
    FEASIBILITY = "feasibility"
    PILOT_RCT = "pilot_rct"
    PIVOTAL_RCT = "pivotal_rct"
    POST_MARKET = "post_market"


class EndpointType(Enum):
    CONTINUOUS = "continuous"
    BINARY = "binary"
    TIME_TO_EVENT = "time_to_event"


class Blinding(Enum):
    OPEN_LABEL = "open_label"
    SINGLE_BLIND = "single_blind"
    DOUBLE_BLIND = "double_blind"


@dataclass
class StudyEndpointDef:
    name: str
    description: str
    endpoint_type: EndpointType = EndpointType.CONTINUOUS
    superiority_margin: float = 0.0
    non_inferiority_margin: float = 0.0
    expected_effect_size: float = 0.0
    expected_std: float = 1.0
    clinically_significant_difference: float = 0.5
    measurement_timing: str = "baseline, 3, 6, 12 months"


@dataclass
class StudyPopulation:
    n_patients: int = 500
    n_arms: int = 2
    allocation_ratio: List[float] = field(default_factory=lambda: [1.0, 1.0])
    inclusion_criteria: List[str] = field(default_factory=list)
    exclusion_criteria: List[str] = field(default_factory=list)
    expected_dropout_rate: float = 0.15
    expected_recruitment_rate: float = 10.0
    recruitment_duration_months: float = 12.0
    follow_up_duration_months: float = 12.0


@dataclass
class StatisticalAnalysisPlan:
    primary_analysis: str = "Superiority analysis: two-sample t-test or ANCOVA with baseline adjustment"
    secondary_analysis: str = "Subgroup analysis by age, sex, diabetes status, CKD status"
    safety_analysis: str = "Adverse event rates compared via Fisher's exact test"
    missing_data_handling: str = "Multiple imputation (MICE) with sensitivity analysis"
    interim_analysis: str = "One interim analysis at 50% enrollment for futility (O'Brien-Fleming boundary)"
    significance_level: float = 0.05
    power: float = 0.80
    multiplicity_correction: str = "Bonferroni-Holm for secondary endpoints"


@dataclass
class ClinicalStudyProtocol:
    title: str
    phase: StudyPhase = StudyPhase.PILOT_RCT
    objectives: List[str] = field(default_factory=list)
    endpoints: List[StudyEndpointDef] = field(default_factory=list)
    population: StudyPopulation = field(default_factory=StudyPopulation)
    arms_description: List[str] = field(default_factory=list)
    blinding: Blinding = Blinding.OPEN_LABEL
    statistical_plan: StatisticalAnalysisPlan = field(default_factory=StatisticalAnalysisPlan)
    data_collection: List[str] = field(default_factory=list)
    twin_integration: List[str] = field(default_factory=list)
    ethical_considerations: List[str] = field(default_factory=list)


class PowerAnalyzer:
    """
    Power analysis and sample size calculation for twin validation trials.
    """

    @staticmethod
    def sample_size_continuous(alpha: float = 0.05, power: float = 0.80,
                                effect_size: float = 0.5, std: float = 1.0,
                                n_arms: int = 2) -> int:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        n_per_arm = int(np.ceil(2 * (z_alpha + z_beta) ** 2 * std ** 2 / effect_size ** 2))
        return max(n_per_arm * n_arms, 10)

    @staticmethod
    def sample_size_binary(alpha: float = 0.05, power: float = 0.80,
                            p_control: float = 0.2, p_treatment: float = 0.35,
                            n_arms: int = 2) -> int:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        p_bar = (p_control + p_treatment) / 2
        n_per_arm = int(np.ceil(
            (z_alpha + z_beta) ** 2 * (p_control * (1 - p_control) + p_treatment * (1 - p_treatment)) /
            (p_treatment - p_control) ** 2
        ))
        return max(n_per_arm * n_arms, 10)

    @staticmethod
    def power_curve(n_range: List[int], effect_size: float = 0.5,
                    std: float = 1.0, alpha: float = 0.05) -> List[Dict]:
        return [
            {"n": n, "power": float(1 - stats.norm.cdf(
                stats.norm.ppf(1 - alpha / 2) - effect_size * np.sqrt(n / 2) / std
            ))}
            for n in n_range
        ]

    @staticmethod
    def minimum_detectable_effect(n_per_arm: int, std: float = 1.0,
                                   alpha: float = 0.05, power: float = 0.80) -> float:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        return float((z_alpha + z_beta) * std * np.sqrt(2 / n_per_arm))


def generate_twin_validation_protocol() -> ClinicalStudyProtocol:
    """
    Generates a complete clinical study protocol for twin-assisted
    diabetes management vs. standard of care.
    """
    return ClinicalStudyProtocol(
        title="Twin-Assisted Diabetes Management: A Pilot Randomized Controlled Trial",
        phase=StudyPhase.PILOT_RCT,
        objectives=[
            "Determine if twin-assisted care improves HbA1c at 6 months vs standard care",
            "Assess twin prediction accuracy for hypoglycemia events",
            "Evaluate clinician adoption and usability of twin recommendations",
            "Estimate effect size for a pivotal trial",
        ],
        endpoints=[
            StudyEndpointDef(
                name="HbA1c_change",
                description="Change in HbA1c from baseline to 6 months",
                endpoint_type=EndpointType.CONTINUOUS,
                expected_effect_size=0.5,
                expected_std=1.0,
                clinically_significant_difference=0.3,
            ),
            StudyEndpointDef(
                name="time_in_range",
                description="Percentage of CGM readings in 70-180 mg/dL range",
                endpoint_type=EndpointType.CONTINUOUS,
                expected_effect_size=8.0,
                expected_std=12.0,
                clinically_significant_difference=5.0,
            ),
            StudyEndpointDef(
                name="hypoglycemia_rate",
                description="Number of hypoglycemic events (<54 mg/dL) per week",
                endpoint_type=EndpointType.CONTINUOUS,
                expected_effect_size=0.5,
                expected_std=0.8,
            ),
            StudyEndpointDef(
                name="SBP_change",
                description="Change in office SBP from baseline to 6 months (mmHg)",
                endpoint_type=EndpointType.CONTINUOUS,
                expected_effect_size=5.0,
                expected_std=12.0,
                clinically_significant_difference=3.0,
            ),
        ],
        population=StudyPopulation(
            n_patients=200,
            n_arms=2,
            allocation_ratio=[1.0, 1.0],
            inclusion_criteria=[
                "Age 18-75 years",
                "Type 2 diabetes (HbA1c 7.0-10.0%)",
                "On stable diabetes medication for ≥3 months",
                "Willing to use CGM for 12 months",
                "Access to smartphone for twin app",
            ],
            exclusion_criteria=[
                "Type 1 diabetes",
                "eGFR < 30 mL/min/1.73m²",
                "Pregnancy or planning pregnancy",
                "Severe hypoglycemia unawareness",
                "Active malignancy",
                "Cirrhosis or heart failure (NYHA III-IV)",
            ],
            expected_dropout_rate=0.15,
            expected_recruitment_rate=8.0,
            recruitment_duration_months=10.0,
            follow_up_duration_months=12.0,
        ),
        arms_description=[
            "Standard of care: ADA guideline-based diabetes management by primary care physician",
            "Twin-assisted: standard care + digital twin with weekly CGM review, "
            "counterfactual simulation for treatment optimization, "
            "and personalized lifestyle recommendations from multi-agent system",
        ],
        blinding=Blinding.OPEN_LABEL,
        statistical_plan=StatisticalAnalysisPlan(
            primary_analysis="ANCOVA with baseline HbA1c as covariate, treatment arm as fixed effect",
            secondary_analysis="Subgroup: age (<65 vs ≥65), baseline HbA1c (<8% vs ≥8%), CKD status",
            safety_analysis="Hypoglycemia rate compared via negative binomial regression",
            missing_data_handling="Mixed models for repeated measures (MMRM) under MAR assumption",
            interim_analysis="Futility analysis at 50% enrollment (conditional power <20% → stop)",
            significance_level=0.05,
            power=0.80,
        ),
        data_collection=[
            "CGM (Dexcom G6 or equivalent) — 14-day wear at baseline, 3, 6, 12 months",
            "Office BP at each visit",
            "HbA1c, lipid panel, serum creatinine, eGFR at each visit",
            "Hypoglycemia events (self-report + CGM-confirmed)",
            "Medication adherence (pill count + electronic monitoring)",
            "Quality of life (SF-36, DDS) at baseline, 6, 12 months",
            "Clinician satisfaction survey at 6 months",
        ],
        twin_integration=[
            "CGM data streamed to twin platform daily",
            "Weekly twin state update and drift assessment",
            "Counterfactual simulation of medication adjustment or lifestyle change",
            "Multi-agent consensus recommendation sent to clinician via EHR",
            "Clinician reviews twin recommendation and accepts/modifies/rejects",
            "Monthly twin accuracy report with calibration metrics",
        ],
        ethical_considerations=[
            "IRB approval from each participating site",
            "HIPAA-compliant data storage and transmission",
            "Clinician retains final decision authority (twin is advisory only)",
            "Data Safety Monitoring Board (DSMB) with quarterly reviews",
            "Adverse event reporting within 24 hours",
            "Patient data de-identified for analysis",
            "Optional continuation in open-label extension at 12 months",
            "Twin model validated for the specific patient population before enrollment",
        ],
    )


def compute_power_analysis(n_total: Optional[int] = None) -> Dict:
    endpoints = [
        ("HbA1c change", 0.5, 1.0, 0.3),
        ("Time in range (%)", 8.0, 12.0, 5.0),
        ("SBP change (mmHg)", 5.0, 12.0, 3.0),
    ]
    results = []
    for name, effect, std, mcid in endpoints:
        if n_total:
            n = n_total // 2
            power = float(1 - stats.norm.cdf(
                stats.norm.ppf(1 - 0.05 / 2) - effect * np.sqrt(n / 2) / std
            ))
            mde = PowerAnalyzer.minimum_detectable_effect(n, std)
            results.append({
                "endpoint": name, "n_per_arm": n, "power": power,
                "minimum_detectable_effect": mde,
            })
        else:
            n_required = PowerAnalyzer.sample_size_continuous(
                effect_size=effect, std=std
            )
            results.append({
                "endpoint": name, "n_required": n_required,
                "effect_size": effect, "std": std,
            })
    return {"power_analysis": results}


def generate_study_report() -> Dict:
    protocol = generate_twin_validation_protocol()
    analysis = compute_power_analysis(n_total=200)
    return {
        "protocol": {
            "title": protocol.title,
            "phase": protocol.phase.value,
            "objectives": protocol.objectives,
            "n_patients": protocol.population.n_patients,
            "n_arms": protocol.population.n_arms,
            "blinding": protocol.blinding.value,
            "inclusion_criteria": protocol.population.inclusion_criteria,
            "exclusion_criteria": protocol.population.exclusion_criteria,
            "endpoints": [
                {"name": e.name, "description": e.description, "mcid": e.clinically_significant_difference}
                for e in protocol.endpoints
            ],
        },
        "power_analysis": analysis["power_analysis"],
        "feasibility": {
            "recruitment_feasible": protocol.population.expected_recruitment_rate >= 5,
            "recruitment_duration_months": protocol.population.recruitment_duration_months,
            "follow_up_months": protocol.population.follow_up_duration_months,
            "dropout_adjusted_n": int(protocol.population.n_patients * (1 - protocol.population.expected_dropout_rate)),
        },
    }
