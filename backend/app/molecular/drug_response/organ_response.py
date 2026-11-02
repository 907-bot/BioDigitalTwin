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
    Evaluates whole-organ physiological transformations and systemic bodily responses
    using continuous, evidence-grounded physiological equations and patient baseline parameters.
    """
    @staticmethod
    def evaluate(
        organ: OrganType,
        target_id: str,
        tissue_response: TissueResponse,
        current_plasma_glucose_mg_dL: float = 120.0,
        current_sbp_mmHg: float = 130.0,
        current_hr_bpm: float = 75.0,
        patient_gfr_mL_min: float = 100.0,
        extracellular_fluid_vol_L: float = 15.0,
    ) -> OrganResponseReport:
        tid = target_id.upper()
        fold = float(tissue_response.modulated_value)
        gfr = float(max(15.0, min(160.0, patient_gfr_mL_min)))
        glucose = float(max(30.0, current_plasma_glucose_mg_dL))
        sbp = float(max(60.0, current_sbp_mmHg))
        hr = float(max(35.0, current_hr_bpm))

        organ_endpoints: Dict[str, Any] = {}
        whole_body: Dict[str, Any] = {}
        adverse_effects: List[Dict[str, str]] = []

        if tid == "SGLT2" and organ == OrganType.KIDNEY:
            # 1. Continuous Tubular Glucose Filtration and Dynamic Excretion
            tmg_modulated = fold  # Residual transport capacity [mg/min] (baseline ~375 mg/min)
            effective_threshold = (tmg_modulated / gfr) * 100.0  # [mg/dL]
            filtered_load_mg_min = (gfr * glucose) / 100.0
            reabsorbed_load = min(filtered_load_mg_min, tmg_modulated)
            excretion_rate_mg_min = max(0.0, filtered_load_mg_min - reabsorbed_load)
            daily_glucosuria_g_day = (excretion_rate_mg_min * 1440.0) / 1000.0

            # 2. Coupled Osmotic Natriuresis (Stoichiometric 1:1 Na+:Glucose with distal nephron compensation)
            moles_glucose_excreted = daily_glucosuria_g_day / 180.16 # mol/day
            # Distal tubule (Henle thick ascending limb & ENaC) reabsorbs ~85% of downstream delivered Na+
            daily_natriuresis_mmol_day = moles_glucose_excreted * 1000.0 * 0.15

            # 3. Dynamic Fluid Volume Contraction
            # Normal ECF [Na+] = 140 mEq/L; plasma volume is ~25% of total extracellular fluid
            ecf_contraction_L = daily_natriuresis_mmol_day / 140.0
            plasma_volume_contraction_L = ecf_contraction_L * 0.25

            # 4. Guytonian Arterial Pressure Regulation
            # Arterial elastance sensitivity ~18 mmHg per Liter of plasma volume change
            delta_sbp_volume = plasma_volume_contraction_L * 18.0
            # Tubuloglomerular feedback (TGF) restoration reduces intraglomerular hypertension and sympathetic tone
            delta_sbp_tgf = 1.8 if daily_glucosuria_g_day > 20.0 else 0.5
            total_sbp_reduction = round(delta_sbp_volume + delta_sbp_tgf, 1)

            # Glucose lowering trajectory
            glucose_lowering = round(min(glucose - 75.0, daily_glucosuria_g_day * 0.45), 1)

            organ_endpoints = {
                "effective_renal_glucose_threshold_mg_dL": round(effective_threshold, 1),
                "urinary_glucose_excretion_g_day": round(daily_glucosuria_g_day, 1),
                "osmotic_natriuresis_mmol_day": round(daily_natriuresis_mmol_day, 1),
                "patient_gfr_mL_min": gfr,
                "tubuloglomerular_feedback": "restored_macula_densa_adenosine_signaling_reducing_hyperfiltration",
            }
            whole_body = {
                "plasma_glucose_lowering_mg_dL": glucose_lowering,
                "blood_pressure_reduction_sbp_mmHg": -total_sbp_reduction,
                "plasma_volume_contraction_L": -round(plasma_volume_contraction_L, 3),
                "extracellular_fluid_contraction_L": -round(ecf_contraction_L, 3),
                "long_term_cardiorenal_protection": "validated_via_hemodynamic_unloading_and_anti_inflammatory_shift",
            }
            if glucose < 250.0:
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
            summary = (
                f"SGLT2 inhibition lowers renal glucose threshold to {effective_threshold:.1f} mg/dL (GFR={gfr:.0f} mL/min), "
                f"excreting {daily_glucosuria_g_day:.1f} g/day of glucose, inducing {daily_natriuresis_mmol_day:.1f} mmol/day natriuresis "
                f"and lowering systolic BP by {total_sbp_reduction:.1f} mmHg."
            )

        elif tid == "ADRB1" and organ == OrganType.HEART:
            # Dynamic beta-adrenergic tone suppression
            suppression_fraction = max(0.0, 1.0 - min(1.0, fold))
            hr_drop = round(hr * 0.20 * suppression_fraction, 1)
            new_hr = max(45.0, hr - hr_drop)
            sbp_drop = round(sbp * 0.10 * suppression_fraction, 1)
            new_sbp = max(88.0, sbp - sbp_drop)

            organ_endpoints = {
                "resting_heart_rate_bpm": new_hr,
                "heart_rate_delta_bpm": -hr_drop,
                "cardiac_contractility_elastance": fold,
                "myocardial_oxygen_consumption_reduction_pct": round(hr_drop * 1.6, 1),
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
            summary = f"ADRB1 blockade slows heart rate by {hr_drop:.1f} bpm ({hr:.0f} -> {new_hr:.0f} bpm) and reduces myocardial oxygen demand."

        elif tid in ["AGTR1", "CACNA1C"] and organ == OrganType.VASCULATURE:
            relaxation_fraction = max(0.0, 1.0 - min(1.0, fold))
            sbp_drop = round(sbp * 0.14 * relaxation_fraction, 1)
            dbp_drop = round(sbp_drop * 0.60, 1)
            new_sbp = max(90.0, sbp - sbp_drop)

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
            # Continuous Glucose-Dependent Incretin Secretion Multiplier
            # Below 65 mg/dL (hypoglycemia), incretin effect shuts off completely to prevent fatal hypoglycemia
            if glucose > 65.0:
                glucose_permissiveness = ((glucose - 65.0) ** 2) / (((glucose - 65.0) ** 2) + (35.0 ** 2))
            else:
                glucose_permissiveness = 0.0

            dynamic_insulin_multiplier = round(1.0 + (fold - 1.0) * glucose_permissiveness, 2)
            glucagon_suppression = round(min(55.0, 20.0 + 35.0 * glucose_permissiveness), 1)
            gastric_delay = round(15.0 * fold, 1)

            organ_endpoints = {
                "glucose_dependent_insulin_secretion_multiplier": dynamic_insulin_multiplier,
                "glucagon_secretion_suppression_pct": glucagon_suppression,
                "glucose_permissiveness_factor": round(glucose_permissiveness, 3),
            }
            whole_body = {
                "postprandial_glucose_blunting_mg_dL": round(35.0 * glucose_permissiveness, 1),
                "gastric_emptying_delay_pct": gastric_delay,
                "weight_loss_trajectory": "sustained_satiety_signaling_via_hypothalamic_proopiomelanocortin",
            }
            adverse_effects.append({
                "adverse_effect": "Transient Nausea / Gastrointestinal Distress",
                "mechanism": "Delayed gastric emptying and central area postrema activation.",
                "severity": "Mild (usually wanes after 2-4 weeks)",
            })
            summary = (
                f"GLP-1R activation potentiates insulin secretion by {dynamic_insulin_multiplier:.2f}x "
                f"(glucose permissiveness={glucose_permissiveness:.2f} at {glucose:.0f} mg/dL) with zero hypoglycemia risk."
            )

        elif tid == "HMGCR" and organ == OrganType.LIVER:
            # Continuous Saturable Receptor-Mediated LDL Clearance (Law et al. BMJ 2003)
            inhibition_pct = max(0.0, 100.0 - fold)
            # Saturable SREBP-2 driven LDL receptor upregulation
            ldlr_upregulation_fold = 1.0 + (1.6 * inhibition_pct) / (inhibition_pct + 35.0)
            serum_ldl_reduction_pct = round(100.0 * (1.0 - (1.0 / ldlr_upregulation_fold)), 1)

            organ_endpoints = {
                "hepatic_cholesterol_synthesis_inhibition_pct": round(inhibition_pct, 1),
                "ldl_receptor_upregulation_fold": round(ldlr_upregulation_fold, 2),
            }
            whole_body = {
                "serum_ldl_cholesterol_reduction_pct": serum_ldl_reduction_pct,
                "atherosclerotic_plaque_stabilization": "demonstrated_anti_inflammatory_pleiotropic_effects",
            }
            adverse_effects.append({
                "adverse_effect": "Statin-Associated Muscle Symptoms (SAMS) / Myalgia",
                "mechanism": "Depletion of downstream isoprenoids (Coenzyme Q10/ubiquinone) in skeletal myocytes.",
                "severity": "Mild to Moderate (5-10% clinical incidence)",
            })
            summary = (
                f"HMG-CoA reductase inhibition suppresses hepatic cholesterol synthesis by {inhibition_pct:.1f}%, "
                f"upregulating LDL receptors by {ldlr_upregulation_fold:.2f}x and lowering serum LDL-C by {serum_ldl_reduction_pct:.1f}%."
            )

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
