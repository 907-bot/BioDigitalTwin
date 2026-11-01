"""
backend/app/molecular/drug_response/organ_response.py
Organ-Level and Whole-Body Systemic Consequences Evaluator.

Translates tissue responses into:
- Organ-level clinical endpoints (e.g., GFR, Daily Glucosuria, Cardiac Output, Blood Pressure)
- Whole-body systemic consequences (e.g., HbA1c trajectory, MAP, Fluid Volume)
- Potential adverse effects and safety flags (e.g., Euglycemic DKA risk, Hypotension, Bradycardia)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from app.molecular.tissue.cell_types import OrganType
from app.molecular.drug_response.tissue_response import TissueResponse


@dataclass
class OrganResponseReport:
    organ: OrganType
    target_id: str
    primary_organ_endpoints: Dict[str, Any]
    whole_body_consequences: Dict[str, Any]
    potential_adverse_effects: List[Dict[str, str]]
    summary: str


class OrganResponseEvaluator:
    """
    Evaluates whole-organ physiological transformations and systemic bodily responses.
    """
    @staticmethod
    def evaluate(
        organ: OrganType,
        target_id: str,
        tissue_response: TissueResponse,
        current_plasma_glucose_mg_dL: float = 120.0,
        current_sbp_mmHg: float = 130.0,
        current_hr_bpm: float = 75.0,
    ) -> OrganResponseReport:
        tid = target_id.upper()
        fold = tissue_response.modulated_value

        organ_endpoints: Dict[str, Any] = {}
        whole_body: Dict[str, Any] = {}
        adverse_effects: List[Dict[str, str]] = []

        if tid == "SGLT2" and organ == OrganType.KIDNEY:
            tmg_modulated = tissue_response.modulated_value  # e.g. ~37.5 mg/min
            gfr = 100.0 # mL/min
            effective_threshold = (tmg_modulated / gfr) * 100.0  # mg/dL
            filtered_load_mg_min = (gfr * current_plasma_glucose_mg_dL) / 100.0
            reabsorbed_load = min(filtered_load_mg_min, tmg_modulated)
            excretion_rate_mg_min = max(0.0, filtered_load_mg_min - reabsorbed_load)
            daily_glucosuria_g_day = (excretion_rate_mg_min * 1440.0) / 1000.0
            daily_natriuresis_mmol_day = daily_glucosuria_g_day * (1000.0 / 180.16) * 0.15 # ~40-60 mmol/day coupled Na+ excretion

            organ_endpoints = {
                "effective_renal_glucose_threshold_mg_dL": round(effective_threshold, 1),
                "urinary_glucose_excretion_g_day": round(daily_glucosuria_g_day, 1),
                "osmotic_natriuresis_mmol_day": round(daily_natriuresis_mmol_day, 1),
                "tubuloglomerular_feedback": "restored_afferent_arteriolar_constriction_reducing_hyperfiltration",
            }
            whole_body = {
                "plasma_glucose_lowering_mg_dL": round(min(current_plasma_glucose_mg_dL - 80.0, daily_glucosuria_g_day * 0.4), 1),
                "blood_pressure_reduction_sbp_mmHg": 4.5,
                "plasma_volume_contraction_L": 0.25,
                "long_term_nephroprotection": "confirmed_via_reduced_intraglomerular_hypertension",
            }
            if current_plasma_glucose_mg_dL < 140.0:
                adverse_effects.append({
                    "adverse_effect": "Euglycemic Diabetic Ketoacidosis (euDKA) Risk",
                    "mechanism": "Carbohydrate deficit and glucosuria shift metabolism to lipolysis and ketogenesis with normal blood glucose.",
                    "severity": "Moderate to High in insulin-deficient states",
                })
            adverse_effects.append({
                "adverse_effect": "Mycotic Genital Infection",
                "mechanism": "Persistent glucosuria elevates glucose concentration in urinary tract.",
                "severity": "Mild to Moderate",
            })
            summary = f"SGLT2 inhibition lowers renal glucose threshold to {effective_threshold:.1f} mg/dL, excreting {daily_glucosuria_g_day:.1f} g/day of glucose with osmotic natriuresis."

        elif tid == "ADRB1" and organ == OrganType.HEART:
            hr_drop = round(15.0 * (1.0 - min(1.0, fold)), 1)
            new_hr = max(48.0, current_hr_bpm - hr_drop)
            sbp_drop = round(12.0 * (1.0 - min(1.0, fold)), 1)
            new_sbp = max(90.0, current_sbp_mmHg - sbp_drop)

            organ_endpoints = {
                "resting_heart_rate_bpm": new_hr,
                "heart_rate_delta_bpm": -hr_drop,
                "cardiac_contractility_elastance": fold,
                "myocardial_oxygen_consumption_reduction_pct": round(hr_drop * 1.5, 1),
            }
            whole_body = {
                "systolic_blood_pressure_mmHg": new_sbp,
                "sbp_delta_mmHg": -sbp_drop,
                "anti_anginal_effect": "improved_subendocardial_perfusion_via_prolonged_diastole",
            }
            if new_hr < 55.0:
                adverse_effects.append({
                    "adverse_effect": "Symptomatic Bradycardia",
                    "mechanism": "Excessive suppression of sinoatrial node autorhythmicity.",
                    "severity": "Moderate",
                })
            summary = f"ADRB1 blockade slows heart rate by {hr_drop:.1f} bpm and reduces myocardial oxygen demand."

        elif tid in ["AGTR1", "CACNA1C"] and organ == OrganType.VASCULATURE:
            sbp_drop = round(16.0 * (1.0 - min(1.0, fold)), 1)
            dbp_drop = round(sbp_drop * 0.65, 1)
            new_sbp = max(95.0, current_sbp_mmHg - sbp_drop)

            organ_endpoints = {
                "arteriolar_vascular_resistance": fold,
                "vasodilatory_response": "systemic_arteriolar_relaxation",
            }
            whole_body = {
                "systolic_blood_pressure_mmHg": new_sbp,
                "sbp_delta_mmHg": -sbp_drop,
                "diastolic_blood_pressure_delta_mmHg": -dbp_drop,
                "cardiac_afterload_reduction": "reduced_left_ventricular_wall_stress",
            }
            if new_sbp < 100.0:
                adverse_effects.append({
                    "adverse_effect": "Orthostatic Hypotension / Dizziness",
                    "mechanism": "Systemic vasodilation impairs postural baroreflex venous return.",
                    "severity": "Mild to Moderate",
                })
            summary = f"Arteriolar resistance reduced to {fold:.2f}x, lowering systolic blood pressure by {sbp_drop:.1f} mmHg."

        elif tid == "GLP1R" and organ == OrganType.PANCREAS:
            insulin_multiplier = tissue_response.modulated_value
            organ_endpoints = {
                "glucose_dependent_insulin_secretion_multiplier": insulin_multiplier,
                "glucagon_secretion_suppression_pct": 45.0,
            }
            whole_body = {
                "postprandial_glucose_blunting_mg_dL": 35.0,
                "gastric_emptying_delay_pct": 30.0,
                "weight_loss_trajectory": "sustained_satiety_signaling_via_hypothalamic_proopiomelanocortin",
            }
            adverse_effects.append({
                "adverse_effect": "Transient Nausea / Gastrointestinal Distress",
                "mechanism": "Delayed gastric emptying and central area postrema activation.",
                "severity": "Mild (usually wanes after 2-4 weeks)",
            })
            summary = f"GLP-1R activation potentiates insulin secretion by {insulin_multiplier:.2f}x in a glucose-dependent manner with zero intrinsic hypoglycemia risk."

        elif tid == "HMGCR" and organ == OrganType.LIVER:
            inhibition_pct = 100.0 - tissue_response.modulated_value
            ldl_reduction_pct = round(inhibition_pct * 0.55, 1)
            organ_endpoints = {
                "hepatic_cholesterol_synthesis_inhibition_pct": round(inhibition_pct, 1),
                "ldl_receptor_upregulation_fold": round(1.0 + (inhibition_pct / 50.0), 2),
            }
            whole_body = {
                "serum_ldl_cholesterol_reduction_pct": ldl_reduction_pct,
                "atherosclerotic_plaque_stabilization": "demonstrated_anti_inflammatory_pleiotropic_effects",
            }
            adverse_effects.append({
                "adverse_effect": "Statin-Associated Muscle Symptoms (SAMS) / Myalgia",
                "mechanism": "Depletion of downstream isoprenoids (Coenzyme Q10/ubiquinone) in skeletal myocytes.",
                "severity": "Mild to Moderate (5-10% clinical incidence)",
            })
            summary = f"HMG-CoA reductase inhibition suppresses hepatic cholesterol synthesis by {inhibition_pct:.1f}%, driving a {ldl_reduction_pct:.1f}% reduction in circulating LDL-C."

        else:
            organ_endpoints = {"functional_modulation_fold": fold}
            whole_body = {"systemic_adaptation": "physiological_steady_state"}
            summary = f"Target {target_id} modulation produces localized tissue adaptation in {organ.value}."

        return OrganResponseReport(
            organ=organ,
            target_id=target_id,
            primary_organ_endpoints=organ_endpoints,
            whole_body_consequences=whole_body,
            potential_adverse_effects=adverse_effects,
            summary=summary,
        )
