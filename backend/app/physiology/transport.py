"""
Shared Vascular Transport and Compartmental Circulation Layer.

Models whole-body hemodynamics, closed-loop blood flow distribution,
and substance transport between organs:
- Cardiac Output: CO = HR * SV
- Organ blood flow partitioning: Brain, Coronary, Hepatic, Renal, Gut, Muscle, Adipose
- Arterial & venous blood gas carriage (O2, CO2, HCO3, pH via Henderson-Hasselbalch)
- Arterial nutrient, metabolite, and hormone convection
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional


@dataclass
class HemodynamicState:
    total_blood_volume_L: float = 5.0
    heart_rate_bpm: float = 72.0
    stroke_volume_mL: float = 70.0
    cardiac_output_L_min: float = 5.04
    total_peripheral_resistance_mmHg_min_L: float = 18.0
    mean_arterial_pressure_mmHg: float = 93.0
    systolic_bp_mmHg: float = 120.0
    diastolic_bp_mmHg: float = 80.0
    central_venous_pressure_mmHg: float = 4.0
    hematocrit: float = 0.42
    hemoglobin_g_dL: float = 14.5

    # Arterial blood gas & chemistry
    pH: float = 7.40
    paO2_mmHg: float = 95.0
    paCO2_mmHg: float = 40.0
    hco3_mEq_L: float = 24.0
    saO2_pct: float = 97.5

    # Circulating substrates & electrolytes
    glucose_mg_dL: float = 95.0
    insulin_uU_mL: float = 8.0
    glucagon_pg_mL: float = 70.0
    ffa_mmol_L: float = 0.45
    lactate_mmol_L: float = 1.0
    sodium_mEq_L: float = 140.0
    potassium_mEq_L: float = 4.2
    osmolality_mOsm_kg: float = 290.0

    # Endocrine & signaling
    cortisol_nmol_L: float = 350.0
    angiotensin_ii_pg_mL: float = 15.0
    aldosterone_ng_dL: float = 10.0
    adh_pg_mL: float = 2.5
    glp1_pmol_L: float = 15.0


class BloodTransportLayer:
    """
    Simulates vascular convection, organ perfusion fractions,
    solute mixing, and mass transport balances.
    """

    def __init__(self, patient_weight_kg: float = 70.0):
        self.weight_kg = patient_weight_kg
        self.state = HemodynamicState()
        # Blood volume scales roughly with body weight (~70 mL/kg)
        self.state.total_blood_volume_L = (70.0 * patient_weight_kg) / 1000.0

    def compute_organ_perfusion(
        self,
        exercise_intensity: float = 0.0,
        sympathetic_tone: float = 1.0,
    ) -> Dict[str, float]:
        """
        Calculates organ-specific blood flow rates (L/min) as fractions of Cardiac Output.
        Flow shifts toward active muscle and skin during exercise or SNS surge.
        """
        co = self.state.cardiac_output_L_min

        # Basal fractional flows (Guyton & Hall)
        # Brain ~14%, Coronary ~4%, Renal ~22%, Gut/Splanchnic ~24%, Muscle ~18%, Adipose/Skin ~18%
        if exercise_intensity > 0.05:
            # Exercise vasodilation in skeletal muscle
            muscle_frac = min(0.75, 0.18 + exercise_intensity * 0.55)
            renal_frac = max(0.06, 0.22 * (1.0 - exercise_intensity * 0.6))
            gut_frac = max(0.06, 0.24 * (1.0 - exercise_intensity * 0.6))
            brain_frac = 0.14 / (1.0 + exercise_intensity * 0.8)
            coronary_frac = min(0.06, 0.04 + exercise_intensity * 0.02)
            adipose_frac = max(0.03, 0.18 * (1.0 - exercise_intensity * 0.7))
        else:
            muscle_frac = 0.18
            renal_frac = 0.22
            gut_frac = 0.24
            brain_frac = 0.14
            coronary_frac = 0.04
            adipose_frac = 0.18

        total_frac = muscle_frac + renal_frac + gut_frac + brain_frac + coronary_frac + adipose_frac
        norm = 1.0 / total_frac

        return {
            "cardiac_output": co,
            "flow_brain": co * brain_frac * norm,
            "flow_coronary": co * coronary_frac * norm,
            "flow_renal": co * renal_frac * norm,
            "flow_gut": co * gut_frac * norm,
            "flow_hepatic_portal": co * gut_frac * norm,
            "flow_hepatic_artery": co * 0.06,
            "flow_hepatic_total": co * (gut_frac * norm + 0.06),
            "flow_muscle": co * muscle_frac * norm,
            "flow_adipose": co * adipose_frac * norm,
            "flow_pulmonary": co,  # All CO passes through lungs
        }

    def update_hemodynamics(
        self,
        hr: float,
        sv: float,
        tpr: float,
        dt_min: float = 1.0,
    ) -> HemodynamicState:
        """Update pressure and flows from cardiac performance and vascular resistance."""
        self.state.heart_rate_bpm = float(np.clip(hr, 30.0, 220.0))
        self.state.stroke_volume_mL = float(np.clip(sv, 20.0, 180.0))
        self.state.cardiac_output_L_min = (self.state.heart_rate_bpm * self.state.stroke_volume_mL) / 1000.0
        self.state.total_peripheral_resistance_mmHg_min_L = float(np.clip(tpr, 8.0, 35.0))

        # MAP = CO * TPR + CVP
        self.state.mean_arterial_pressure_mmHg = (
            self.state.cardiac_output_L_min * self.state.total_peripheral_resistance_mmHg_min_L +
            self.state.central_venous_pressure_mmHg
        )
        # Pulse pressure proportional to stroke volume
        pp = 40.0 * (self.state.stroke_volume_mL / 70.0)
        self.state.systolic_bp_mmHg = float(self.state.mean_arterial_pressure_mmHg + (2.0 / 3.0) * pp)
        self.state.diastolic_bp_mmHg = float(self.state.mean_arterial_pressure_mmHg - (1.0 / 3.0) * pp)
        return self.state

    def update_blood_gas_equilibrium(
        self,
        pao2: float,
        paco2: float,
        hco3: float,
    ) -> Tuple[float, float]:
        """
        Computes oxygen saturation via Severinghaus/Hill equation
        and arterial pH via Henderson-Hasselbalch equation.
        """
        self.state.paO2_mmHg = float(np.clip(pao2, 20.0, 300.0))
        self.state.paCO2_mmHg = float(np.clip(paco2, 15.0, 90.0))
        self.state.hco3_mEq_L = float(np.clip(hco3, 10.0, 45.0))

        # Severinghaus equation for Hb-O2 saturation
        p = self.state.paO2_mmHg
        sao2 = 100.0 * (((p ** 3) + 150.0 * p) / ((p ** 3) + 150.0 * p + 23400.0))
        self.state.saO2_pct = float(np.clip(sao2, 10.0, 100.0))

        # Henderson-Hasselbalch: pH = 6.1 + log10([HCO3-] / (0.0306 * paCO2))
        ratio = self.state.hco3_mEq_L / max(0.1, 0.0306 * self.state.paCO2_mmHg)
        self.state.pH = float(np.clip(6.10 + np.log10(max(0.1, ratio)), 6.80, 7.80))

        return self.state.saO2_pct, self.state.pH

    def compute_oxygen_delivery(self) -> float:
        """
        DO2 = Cardiac Output (dL/min) * Arterial O2 Content (CaO2, mL O2/dL).
        Normal ~1000 mL O2/min at rest.
        """
        co_dL = self.state.cardiac_output_L_min * 10.0
        # CaO2 = 1.34 * Hb * (SaO2/100) + 0.0031 * PaO2
        cao2 = 1.34 * self.state.hemoglobin_g_dL * (self.state.saO2_pct / 100.0) + 0.0031 * self.state.paO2_mmHg
        return float(co_dL * cao2)

    def integrate_fluxes(self, organ_fluxes: Dict[str, float], dt_min: float = 1.0) -> None:
        """
        Mixes net organ outputs and metabolic consumption into blood concentrations.
        Fluxes in mass/min or mmol/min divided by distribution volume.
        """
        plasma_vol_dL = (self.state.total_blood_volume_L * (1.0 - self.state.hematocrit)) * 10.0
        ecf_vol_L = self.weight_kg * 0.20  # ~14 L extracellular fluid for glucose/electrolytes

        # Glucose flux (mg/min)
        if "glucose_flux_mg_min" in organ_fluxes:
            dG = (organ_fluxes["glucose_flux_mg_min"] / (ecf_vol_L * 10.0)) * dt_min
            self.state.glucose_mg_dL = float(np.clip(self.state.glucose_mg_dL + dG, 20.0, 600.0))

        # Insulin flux (uU/min)
        if "insulin_flux_uU_min" in organ_fluxes:
            dI = (organ_fluxes["insulin_flux_uU_min"] / (plasma_vol_dL * 100.0)) * dt_min
            # Insulin hepatic and renal degradation ~0.2/min
            dI -= 0.15 * self.state.insulin_uU_mL * dt_min
            self.state.insulin_uU_mL = float(np.clip(self.state.insulin_uU_mL + dI, 1.0, 300.0))

        # Free fatty acids flux (mmol/min)
        if "ffa_flux_mmol_min" in organ_fluxes:
            dFFA = (organ_fluxes["ffa_flux_mmol_min"] / (plasma_vol_dL * 0.1)) * dt_min
            dFFA -= 0.10 * self.state.ffa_mmol_L * dt_min
            self.state.ffa_mmol_L = float(np.clip(self.state.ffa_mmol_L + dFFA, 0.05, 3.0))

        # Lactate flux (mmol/min)
        if "lactate_flux_mmol_min" in organ_fluxes:
            dLact = (organ_fluxes["lactate_flux_mmol_min"] / ecf_vol_L) * dt_min
            dLact -= 0.15 * self.state.lactate_mmol_L * dt_min
            self.state.lactate_mmol_L = float(np.clip(self.state.lactate_mmol_L + dLact, 0.2, 15.0))
