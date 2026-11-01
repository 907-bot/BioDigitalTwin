"""
Dedicated Autonomic Nervous System & Endocrine Axis Regulation Model.

Simulates:
1. Autonomic Nervous System & Baroreflex arc:
   - Carotid and aortic baroreceptor firing dynamics
   - Sympathetic (SNS) vs Parasympathetic/Vagal (PNS) tone balance
   - Closed-loop control of Heart Rate (HR), Stroke Volume (SV), and Vascular Resistance (TPR)
2. Hypothalamic-Pituitary-Adrenal (HPA) axis:
   - CRH -> ACTH -> Cortisol cascade
   - Glucocorticoid negative feedback inhibition on pituitary and hypothalamus
3. Renin-Angiotensin-Aldosterone System (RAAS):
   - Renin release triggered by renal hypoperfusion (low GFR/MAP) and SNS stimulation
   - Angiotensin II vasoconstriction
   - Aldosterone-mediated renal sodium retention
4. Hypothalamic osmoregulation & Vasopressin (ADH):
   - Plasma osmolality and volume sensing -> ADH secretion
   - Free water retention feedback
5. Pancreatic Alpha-cell Glucagon dynamics:
   - Glucose-counterregulatory secretion opposing insulin
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from .base import PhysiologicalOrganModel, OrganStateDescriptor, OrganParamDescriptor


@dataclass
class AutonomicEndocrineState:
    sympathetic_tone: float = 1.0        # normalized (1.0 = resting baseline, 0.2 - 4.0)
    parasympathetic_tone: float = 1.0    # normalized (1.0 = resting baseline, 0.1 - 3.0)
    baroreceptor_firing_rate: float = 50.0  # Hz (0 - 100 Hz)
    acth_pg_mL: float = 25.0             # Pituitary ACTH (10 - 60 pg/mL)
    cortisol_nmol_L: float = 350.0       # Adrenal cortisol (100 - 800 nmol/L)
    renin_activity_ng_mL_h: float = 1.5  # Plasma renin activity (0.5 - 4.0)
    angiotensin_ii_pg_mL: float = 15.0   # Active Ang II (5 - 40 pg/mL)
    aldosterone_ng_dL: float = 10.0      # Adrenal aldosterone (3 - 30 ng/dL)
    adh_pg_mL: float = 2.5               # Vasopressin (1.0 - 10.0 pg/mL)
    glucagon_pg_mL: float = 70.0         # Pancreatic glucagon (40 - 200 pg/mL)


class AutonomicEndocrineModel(PhysiologicalOrganModel):
    """
    Integrative neuro-endocrine reflex and hormonal feedback controller.
    """

    def __init__(self):
        super().__init__(name="AutonomicEndocrine")
        self.declare_descriptors()
        self.initialize()

    def declare_descriptors(self) -> None:
        self.descriptors_state = {
            "sympathetic_tone": OrganStateDescriptor(
                "sympathetic_tone", "normalized", 1.0, 0.1, 5.0, "Sympathetic neural activity"
            ),
            "parasympathetic_tone": OrganStateDescriptor(
                "parasympathetic_tone", "normalized", 1.0, 0.05, 4.0, "Vagal / parasympathetic activity"
            ),
            "cortisol_nmol_L": OrganStateDescriptor(
                "cortisol_nmol_L", "nmol/L", 350.0, 50.0, 1000.0, "Circulating cortisol"
            ),
            "angiotensin_ii_pg_mL": OrganStateDescriptor(
                "angiotensin_ii_pg_mL", "pg/mL", 15.0, 2.0, 80.0, "Angiotensin II vasoconstrictor"
            ),
            "aldosterone_ng_dL": OrganStateDescriptor(
                "aldosterone_ng_dL", "ng/dL", 10.0, 1.0, 60.0, "Aldosterone mineralocorticoid"
            ),
            "adh_pg_mL": OrganStateDescriptor(
                "adh_pg_mL", "pg/mL", 2.5, 0.5, 25.0, "Antidiuretic hormone / vasopressin"
            ),
            "glucagon_pg_mL": OrganStateDescriptor(
                "glucagon_pg_mL", "pg/mL", 70.0, 20.0, 300.0, "Alpha-cell glucagon"
            ),
        }

        self.descriptors_param = {
            "target_map_mmHg": OrganParamDescriptor(
                "target_map_mmHg", "mmHg", 93.0, 75.0, 115.0, "Baroreflex pressure setpoint"
            ),
            "baroreflex_sensitivity": OrganParamDescriptor(
                "baroreflex_sensitivity", "gain", 1.5, 0.3, 3.5, "Baroreceptor gain"
            ),
            "target_osmolality_mOsm": OrganParamDescriptor(
                "target_osmolality_mOsm", "mOsm/kg", 290.0, 275.0, 305.0, "Osmoreceptor setpoint"
            ),
            "hpa_stress_gain": OrganParamDescriptor(
                "hpa_stress_gain", "gain", 1.0, 0.2, 3.0, "Cortisol sensitivity to physiological stress"
            ),
        }

    def initialize(self, patient_parameters: Optional[Dict[str, float]] = None) -> None:
        self._state = {
            "sympathetic_tone": 1.0,
            "parasympathetic_tone": 1.0,
            "baroreceptor_firing_rate": 50.0,
            "acth_pg_mL": 25.0,
            "cortisol_nmol_L": 350.0,
            "renin_activity_ng_mL_h": 1.5,
            "angiotensin_ii_pg_mL": 15.0,
            "aldosterone_ng_dL": 10.0,
            "adh_pg_mL": 2.5,
            "glucagon_pg_mL": 70.0,
        }
        self._params = {k: v.default_value for k, v in self.descriptors_param.items()}
        if patient_parameters:
            self.set_parameters(patient_parameters)

    def compute_derivatives(
        self,
        state: Dict[str, float],
        inputs: Dict[str, float],
        transport: Dict[str, float],
    ) -> Dict[str, float]:
        map_val = transport.get("mean_arterial_pressure_mmHg", 93.0)
        osmolality = transport.get("osmolality_mOsm_kg", 290.0)
        glucose = transport.get("glucose_mg_dL", 95.0)
        insulin = transport.get("insulin_uU_mL", 8.0)
        exercise = inputs.get("exercise", 0.0)
        stress = inputs.get("psychological_stress", 0.0)

        # 1. Baroreceptor reflex arc with central command exercise pressor reset:
        # Exercise resets the pressure setpoint upward
        effective_map_setpoint = self._params["target_map_mmHg"] + exercise * 35.0
        delta_p = map_val - effective_map_setpoint
        gain = self._params["baroreflex_sensitivity"]

        # Vagal withdrawal during exercise is marked; high pressure stimulates vagus
        target_vagus = max(0.05, min(1.0, (1.0 / (1.0 + exercise * 6.0)) + gain * max(0.0, delta_p / 35.0)))
        target_sns = max(0.2, 1.0 - gain * (delta_p / 30.0) + exercise * 2.5 + stress * 1.5)

        d_vagus = 0.3 * (target_vagus - state["parasympathetic_tone"])
        d_sns = 0.3 * (target_sns - state["sympathetic_tone"])

        # 2. HPA axis (Cortisol):
        # Stress & low glucose stimulate ACTH & Cortisol; high cortisol provides negative feedback
        stress_drive = (exercise * 0.5 + stress * 1.0) * self._params["hpa_stress_gain"]
        hypo_drive = max(0.0, (70.0 - glucose) / 20.0)
        target_acth = max(5.0, 25.0 + 20.0 * (stress_drive + hypo_drive) - 0.03 * (state["cortisol_nmol_L"] - 350.0))
        d_acth = 0.2 * (target_acth - state["acth_pg_mL"])

        target_cortisol = max(50.0, 350.0 + 12.0 * (state["acth_pg_mL"] - 25.0))
        d_cortisol = 0.05 * (target_cortisol - state["cortisol_nmol_L"])

        # 3. RAAS (Renin -> Ang II -> Aldosterone):
        # Renin released by hypotension (low MAP) or SNS activation
        renal_perfusion_drop = max(0.0, (90.0 - map_val) / 20.0)
        target_renin = max(0.2, 1.5 + 2.0 * renal_perfusion_drop + 0.8 * max(0.0, state["sympathetic_tone"] - 1.0))
        d_renin = 0.1 * (target_renin - state["renin_activity_ng_mL_h"])

        target_ang2 = max(2.0, 15.0 * (state["renin_activity_ng_mL_h"] / 1.5))
        d_ang2 = 0.2 * (target_ang2 - state["angiotensin_ii_pg_mL"])

        target_aldo = max(2.0, 10.0 + 0.5 * (state["angiotensin_ii_pg_mL"] - 15.0))
        d_aldo = 0.05 * (target_aldo - state["aldosterone_ng_dL"])

        # 4. ADH / Vasopressin (Osmoreceptor & Baroreceptor inputs):
        osm_error = max(0.0, osmolality - self._params["target_osmolality_mOsm"])
        volume_error = max(0.0, (85.0 - map_val) / 15.0)
        target_adh = max(0.5, 2.5 + 1.2 * osm_error + 2.5 * volume_error)
        d_adh = 0.1 * (target_adh - state["adh_pg_mL"])

        # 5. Pancreatic Glucagon (Alpha-cell counter-regulatory secretion):
        # Stimulated by hypoglycemia and exercise, suppressed by insulin
        hypo_stimulus = max(0.0, (85.0 - glucose) / 10.0)
        ins_inhibition = max(0.2, 1.0 - (insulin / (insulin + 20.0)))
        target_glucagon = max(20.0, 70.0 * (1.0 + 2.0 * hypo_stimulus + 0.5 * exercise) * ins_inhibition)
        d_glucagon = 0.3 * (target_glucagon - state["glucagon_pg_mL"])

        return {
            "sympathetic_tone": d_sns,
            "parasympathetic_tone": d_vagus,
            "acth_pg_mL": d_acth,
            "cortisol_nmol_L": d_cortisol,
            "renin_activity_ng_mL_h": d_renin,
            "angiotensin_ii_pg_mL": d_ang2,
            "aldosterone_ng_dL": d_aldo,
            "adh_pg_mL": d_adh,
            "glucagon_pg_mL": d_glucagon,
        }

    def compute_cardiovascular_modulation(self) -> Tuple[float, float, float]:
        """
        Modulates Heart Rate, Stroke Volume, and Total Peripheral Resistance:
        Returns (HR_multiplier, SV_multiplier, TPR_multiplier).
        """
        sns = self._state["sympathetic_tone"]
        pns = self._state["parasympathetic_tone"]
        ang2 = self._state["angiotensin_ii_pg_mL"]

        hr_mult = float(np.clip(1.0 + 0.15 * (sns - 1.0) - 0.15 * (pns - 1.0), 0.7, 1.5))
        sv_mult = float(np.clip(1.0 + 0.10 * (sns - 1.0), 0.8, 1.3))
        ang2_effect = 0.010 * (ang2 - 15.0)
        tpr_mult = float(np.clip(1.0 + 0.15 * (sns - 1.0) + ang2_effect, 0.7, 1.5))

        return hr_mult, sv_mult, tpr_mult

    def exchange_fluxes(self, transport: Dict[str, float]) -> Dict[str, float]:
        return {
            "cortisol_nmol_L": float(self._state["cortisol_nmol_L"]),
            "angiotensin_ii_pg_mL": float(self._state["angiotensin_ii_pg_mL"]),
            "aldosterone_ng_dL": float(self._state["aldosterone_ng_dL"]),
            "adh_pg_mL": float(self._state["adh_pg_mL"]),
            "glucagon_pg_mL": float(self._state["glucagon_pg_mL"]),
            "sympathetic_tone": float(self._state["sympathetic_tone"]),
            "parasympathetic_tone": float(self._state["parasympathetic_tone"]),
        }
