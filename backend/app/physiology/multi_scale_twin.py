"""
Multi-Organ Multi-Scale Physiological Twin Coordinator.

Integrates:
- BloodTransportLayer (Hemodynamics & solute convection)
- RespiratoryOrganModel (Ventilation & gas exchange)
- HepaticOrganModel (Carbohydrate, lipid & clearance metabolism)
- GastrointestinalOrganModel (Gut motility, absorption & incretins)
- AutonomicEndocrineModel (Baroreflex, ANS tone, HPA, RAAS, ADH)

Executes continuous closed-loop bidirectional organ coupling:
Lungs <-> Blood <-> Heart <-> Gut <-> Liver <-> Endocrine & Neuro Control
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

from .transport import BloodTransportLayer, HemodynamicState
from .respiratory import RespiratoryOrganModel
from .hepatic import HepaticOrganModel
from .gastrointestinal import GastrointestinalOrganModel
from .autonomic_endocrine import AutonomicEndocrineModel


@dataclass
class TwinSimulationSnapshot:
    step_minute: float
    glucose_mg_dL: float
    insulin_uU_mL: float
    heart_rate_bpm: float
    mean_arterial_pressure_mmHg: float
    cardiac_output_L_min: float
    respiratory_rate_bpm: float
    paO2_mmHg: float
    paCO2_mmHg: float
    pH: float
    saO2_pct: float
    hepatic_glycogen_g: float
    cortisol_nmol_L: float
    glp1_pmol_L: float
    sympathetic_tone: float
    # Extended metrics for clinical & API consumption
    timestamp_minutes: float = 0.0
    stroke_volume_mL: float = 70.0
    systolic_bp_mmHg: float = 120.0
    diastolic_bp_mmHg: float = 80.0
    organ_blood_flows_L_min: Optional[Dict[str, float]] = None
    tidal_volume_L: float = 0.50
    alveolar_ventilation_L_min: float = 4.9
    glucagon_pg_mL: float = 70.0
    liver_glycogen_g: float = 100.0
    hgp_mg_min: float = 150.0
    stomach_carb_mass_g: float = 0.0
    intestinal_glucose_mass_g: float = 0.0
    parasympathetic_tone: float = 0.70
    renin_activity_ng_mL_h: float = 1.2
    aldosterone_ng_dL: float = 8.5
    cortisol_ug_dL: float = 10.0


TwinWholeBodySnapshot = TwinSimulationSnapshot


class MultiOrganPhysiologicalTwin:
    """
    Coupled multi-organ human physiological simulation engine.
    """

    def __init__(self, weight_kg: float = 70.0, height_cm: float = 175.0, age_yr: float = 35.0):
        self.weight_kg = weight_kg
        self.height_cm = height_cm
        self.age_yr = age_yr
        self.current_time_min = 0.0

        # Subsystems
        self.transport = BloodTransportLayer(patient_weight_kg=weight_kg)
        self.respiratory = RespiratoryOrganModel()
        self.hepatic = HepaticOrganModel()
        self.gastrointestinal = GastrointestinalOrganModel()
        self.autonomic_endocrine = AutonomicEndocrineModel()

    @property
    def organs(self) -> Dict[str, Any]:
        return {
            "lungs": self.respiratory,
            "liver": self.hepatic,
            "gastrointestinal": self.gastrointestinal,
            "autonomic_endocrine": self.autonomic_endocrine,
        }

    def feed_meal(self, carbs_g: float, lipids_g: float = 0.0, proteins_g: float = 0.0) -> None:
        """Introduce meal bolus into the gastrointestinal system."""
        self.gastrointestinal.ingest_meal(carbs_g, lipids_g, proteins_g)

    def step(
        self,
        dt_minutes: float = 1.0,
        inputs: Optional[Dict[str, float]] = None,
        method: str = "rk4",
    ) -> TwinSimulationSnapshot:
        """
        Advance whole-body physiology forward by dt_minutes with synchronous multi-organ coupling.
        """
        if inputs is None:
            inputs = {}

        t_state = self.transport.state

        # Transport context passed to organs
        transport_dict = {
            "mean_arterial_pressure_mmHg": t_state.mean_arterial_pressure_mmHg,
            "cardiac_output_L_min": t_state.cardiac_output_L_min,
            "glucose_mg_dL": t_state.glucose_mg_dL,
            "insulin_uU_mL": t_state.insulin_uU_mL,
            "glucagon_pg_mL": t_state.glucagon_pg_mL,
            "ffa_mmol_L": t_state.ffa_mmol_L,
            "lactate_mmol_L": t_state.lactate_mmol_L,
            "osmolality_mOsm_kg": t_state.osmolality_mOsm_kg,
            "hco3_mEq_L": t_state.hco3_mEq_L,
            "cortisol_nmol_L": t_state.cortisol_nmol_L,
        }

        # 1. Advance Autonomic & Endocrine Axis (Baroreflex, ANS, hormones)
        ae_state = self.autonomic_endocrine.step(dt_minutes, inputs, transport_dict, method=method)
        ae_fluxes = self.autonomic_endocrine.exchange_fluxes(transport_dict)
        hr_mult, sv_mult, tpr_mult = self.autonomic_endocrine.compute_cardiovascular_modulation()

        # Update circulation with neuro-endocrine modulation
        ex = inputs.get("exercise", 0.0)
        base_hr = 72.0 + ex * 65.0
        base_sv = 70.0 + ex * 18.0
        # Active metabolic hyperemia in working muscle drops peripheral resistance
        base_tpr = 18.0 / (1.0 + ex * 1.6)
        self.transport.update_hemodynamics(base_hr * hr_mult, base_sv * sv_mult, base_tpr * tpr_mult, dt_minutes)
        flows = self.transport.compute_organ_perfusion(
            exercise_intensity=ex, sympathetic_tone=ae_fluxes["sympathetic_tone"]
        )

        # 2. Advance Gastrointestinal System
        gi_state = self.gastrointestinal.step(dt_minutes, inputs, transport_dict, method=method)
        gi_fluxes = self.gastrointestinal.exchange_fluxes(transport_dict)

        # 3. Advance Hepatic System
        hep_state = self.hepatic.step(dt_minutes, inputs, transport_dict, method=method)
        hep_fluxes = self.hepatic.exchange_fluxes(transport_dict)

        # 4. Advance Respiratory System
        resp_state = self.respiratory.step(dt_minutes, inputs, transport_dict, method=method)
        resp_fluxes = self.respiratory.exchange_fluxes(transport_dict)

        # Update arterial blood gas equilibrium from lungs
        self.transport.update_blood_gas_equilibrium(
            resp_fluxes["paO2"], resp_fluxes["paCO2"], t_state.hco3_mEq_L
        )

        # 5. Integrate Inter-Organ Mass Fluxes into Systemic Blood
        # Peripheral glucose uptake: basal ~140 mg/min (2 mg/kg/min) balancing basal HGP
        insulin_factor = 1.0 + 0.05 * max(0.0, t_state.insulin_uU_mL - 8.0)
        glucose_factor = max(0.2, t_state.glucose_mg_dL / 95.0)
        exercise_factor = 1.0 + inputs.get("exercise", 0.0) * 3.5
        peripheral_glucose_uptake = 140.0 * glucose_factor * insulin_factor * exercise_factor

        net_glucose_flux = (
            gi_fluxes["glucose_flux_mg_min"] +
            hep_fluxes["glucose_flux_mg_min"] -
            peripheral_glucose_uptake
        )

        # Pancreatic beta-cell insulin secretion stimulated by glucose and incretins
        glp1_boost = 1.0 + 0.06 * max(0.0, gi_fluxes.get("glp1_concentration_pmol_L", 15.0) - 15.0)
        insulin_secretion = (
            (450.0 + 35.0 * max(0.0, t_state.glucose_mg_dL - 85.0)) * glp1_boost
        )

        combined_fluxes = {
            "glucose_flux_mg_min": net_glucose_flux,
            "insulin_flux_uU_min": insulin_secretion,
            "ffa_flux_mmol_min": gi_fluxes["ffa_flux_mmol_min"],
        }
        self.transport.integrate_fluxes(combined_fluxes, dt_minutes)

        # Sync hormone concentrations from endocrine layer into transport
        t_state.cortisol_nmol_L = ae_fluxes["cortisol_nmol_L"]
        t_state.angiotensin_ii_pg_mL = ae_fluxes["angiotensin_ii_pg_mL"]
        t_state.aldosterone_ng_dL = ae_fluxes["aldosterone_ng_dL"]
        t_state.adh_pg_mL = ae_fluxes["adh_pg_mL"]
        t_state.glucagon_pg_mL = ae_fluxes["glucagon_pg_mL"]
        t_state.glp1_pmol_L = gi_fluxes["glp1_concentration_pmol_L"]

        self.current_time_min += dt_minutes

        return TwinSimulationSnapshot(
            step_minute=self.current_time_min,
            glucose_mg_dL=float(round(t_state.glucose_mg_dL, 2)),
            insulin_uU_mL=float(round(t_state.insulin_uU_mL, 2)),
            heart_rate_bpm=float(round(t_state.heart_rate_bpm, 1)),
            mean_arterial_pressure_mmHg=float(round(t_state.mean_arterial_pressure_mmHg, 1)),
            cardiac_output_L_min=float(round(t_state.cardiac_output_L_min, 2)),
            respiratory_rate_bpm=float(round(resp_state["respiratory_rate_bpm"], 1)),
            paO2_mmHg=float(round(resp_state["paO2_mmHg"], 1)),
            paCO2_mmHg=float(round(resp_state["paCO2_mmHg"], 1)),
            pH=float(round(resp_state["pH"], 2)),
            saO2_pct=float(round(resp_state["saO2_pct"], 1)),
            hepatic_glycogen_g=float(round(hep_state["glycogen_mass_g"], 2)),
            cortisol_nmol_L=float(round(ae_fluxes["cortisol_nmol_L"], 1)),
            glp1_pmol_L=float(round(gi_fluxes["glp1_concentration_pmol_L"], 1)),
            sympathetic_tone=float(round(ae_fluxes["sympathetic_tone"], 2)),
            timestamp_minutes=self.current_time_min,
            stroke_volume_mL=float(round(t_state.stroke_volume_mL, 1)),
            systolic_bp_mmHg=float(round(t_state.systolic_bp_mmHg, 1)),
            diastolic_bp_mmHg=float(round(t_state.diastolic_bp_mmHg, 1)),
            organ_blood_flows_L_min=flows,
            tidal_volume_L=float(round(resp_state.get("tidal_volume_L", 0.5), 3)),
            alveolar_ventilation_L_min=float(round(resp_state.get("alveolar_ventilation_L_min", 4.9), 2)),
            glucagon_pg_mL=float(round(t_state.glucagon_pg_mL, 1)),
            liver_glycogen_g=float(round(hep_state["glycogen_mass_g"], 2)),
            hgp_mg_min=float(round(hep_state.get("hgp_rate_mg_min", 150.0), 1)),
            stomach_carb_mass_g=float(round(gi_state.get("stomach_carb_mass_g", 0.0), 2)),
            intestinal_glucose_mass_g=float(round(gi_state.get("intestinal_glucose_mass_g", 0.0), 2)),
            parasympathetic_tone=float(round(ae_fluxes.get("parasympathetic_tone", 0.7), 3)),
            renin_activity_ng_mL_h=float(round(ae_fluxes.get("renin_activity_ng_mL_h", 1.2), 2)),
            aldosterone_ng_dL=float(round(ae_fluxes.get("aldosterone_ng_dL", 8.5), 2)),
            cortisol_ug_dL=float(round(ae_fluxes.get("cortisol_nmol_L", 280.0) / 27.6, 2)),
        )

    def validate_whole_body_state(self) -> Tuple[bool, Dict[str, List[str]]]:
        """Verify physiological bounds across all interconnected organs."""
        all_valid = True
        violations = {}

        for organ in [self.respiratory, self.hepatic, self.gastrointestinal, self.autonomic_endocrine]:
            ok, issues = organ.validate_state()
            if not ok:
                all_valid = False
                violations[organ.name] = issues

        # Transport bounds
        t = self.transport.state
        t_issues = []
        if not (20.0 <= t.glucose_mg_dL <= 600.0):
            t_issues.append(f"Glucose={t.glucose_mg_dL} out of bounds")
        if not (6.80 <= t.pH <= 7.80):
            t_issues.append(f"pH={t.pH} out of bounds")
        if not (40.0 <= t.mean_arterial_pressure_mmHg <= 220.0):
            t_issues.append(f"MAP={t.mean_arterial_pressure_mmHg} out of bounds")
        if t_issues:
            all_valid = False
            violations["BloodTransport"] = t_issues

        return all_valid, violations
