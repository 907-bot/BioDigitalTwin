"""
Dedicated Pulmonary & Respiratory Physiology Model.

Simulates:
1. Pulmonary ventilation mechanics:
   - Tidal volume (V_T) & Respiratory Rate (RR)
   - Anatomical dead space (V_D) & Alveolar ventilation (V_A)
2. Alveolar gas exchange:
   - Alveolar gas equation (pAO2)
   - Alveolar-arterial O2 diffusion & A-a gradient
   - Carbon dioxide elimination
3. Chemical control of breathing:
   - Central chemoreceptor hypercapnic drive (paCO2 feedback)
   - Peripheral carotid chemoreceptor hypoxic drive (paO2 < 60 mmHg)
   - Exercise hyperpnea
4. Acid-Base regulation (Respiratory component):
   - Henderson-Hasselbalch arterial pH balance
   - Compensatory hyperventilation in metabolic acidosis
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from .base import PhysiologicalOrganModel, OrganStateDescriptor, OrganParamDescriptor


@dataclass
class RespiratoryState:
    respiratory_rate_bpm: float = 14.0
    tidal_volume_L: float = 0.50
    minute_ventilation_L_min: float = 7.0
    alveolar_ventilation_L_min: float = 4.9
    pAO2_mmHg: float = 102.0
    paO2_mmHg: float = 95.0
    paCO2_mmHg: float = 40.0
    saO2_pct: float = 97.5
    pH: float = 7.40
    vco2_production_mL_min: float = 200.0
    vo2_consumption_mL_min: float = 250.0


class RespiratoryOrganModel(PhysiologicalOrganModel):
    """
    Continuous differential model of respiratory mechanics and alveolar gas exchange.
    """

    def __init__(self):
        super().__init__(name="Respiratory")
        self.declare_descriptors()
        self.initialize()

    def declare_descriptors(self) -> None:
        self.descriptors_state = {
            "respiratory_rate_bpm": OrganStateDescriptor(
                "respiratory_rate_bpm", "breaths/min", 14.0, 4.0, 60.0, "Breathing frequency"
            ),
            "tidal_volume_L": OrganStateDescriptor(
                "tidal_volume_L", "L", 0.50, 0.15, 3.5, "Volume of single breath"
            ),
            "alveolar_ventilation_L_min": OrganStateDescriptor(
                "alveolar_ventilation_L_min", "L/min", 4.9, 0.5, 80.0, "Effective alveolar gas flow"
            ),
            "pAO2_mmHg": OrganStateDescriptor(
                "pAO2_mmHg", "mmHg", 102.0, 30.0, 150.0, "Alveolar oxygen tension"
            ),
            "paO2_mmHg": OrganStateDescriptor(
                "paO2_mmHg", "mmHg", 95.0, 20.0, 140.0, "Arterial oxygen tension"
            ),
            "paCO2_mmHg": OrganStateDescriptor(
                "paCO2_mmHg", "mmHg", 40.0, 15.0, 90.0, "Arterial carbon dioxide tension"
            ),
            "saO2_pct": OrganStateDescriptor(
                "saO2_pct", "%", 97.5, 40.0, 100.0, "Hemoglobin oxygen saturation"
            ),
            "pH": OrganStateDescriptor(
                "pH", "pH units", 7.40, 6.80, 7.80, "Arterial blood pH"
            ),
        }

        self.descriptors_param = {
            "dead_space_L": OrganParamDescriptor(
                "dead_space_L", "L", 0.15, 0.08, 0.30, "Anatomic dead space volume"
            ),
            "barometric_pressure_mmHg": OrganParamDescriptor(
                "barometric_pressure_mmHg", "mmHg", 760.0, 500.0, 800.0, "Atmospheric pressure"
            ),
            "fiO2": OrganParamDescriptor(
                "fiO2", "fraction", 0.21, 0.21, 1.0, "Fraction of inspired oxygen"
            ),
            "respiratory_quotient": OrganParamDescriptor(
                "respiratory_quotient", "ratio", 0.82, 0.70, 1.0, "VCO2 / VO2 ratio"
            ),
            "chemoreceptor_co2_gain": OrganParamDescriptor(
                "chemoreceptor_co2_gain", "L/min/mmHg", 1.8, 0.5, 5.0, "Ventilatory sensitivity to paCO2"
            ),
            "chemoreceptor_o2_gain": OrganParamDescriptor(
                "chemoreceptor_o2_gain", "L/min/mmHg", 0.4, 0.05, 1.5, "Hypoxic ventilatory drive gain"
            ),
            "diffusing_capacity_o2": OrganParamDescriptor(
                "diffusing_capacity_o2", "mL/min/mmHg", 25.0, 5.0, 50.0, "Pulmonary membrane DL_O2"
            ),
        }

    def initialize(self, patient_parameters: Optional[Dict[str, float]] = None) -> None:
        self._state = {
            "respiratory_rate_bpm": 14.0,
            "tidal_volume_L": 0.50,
            "minute_ventilation_L_min": 7.0,
            "alveolar_ventilation_L_min": 4.9,
            "pAO2_mmHg": 102.0,
            "paO2_mmHg": 95.0,
            "paCO2_mmHg": 40.0,
            "saO2_pct": 97.5,
            "pH": 7.40,
            "vco2_production_mL_min": 200.0,
            "vo2_consumption_mL_min": 250.0,
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
        """
        Computes rates of change for ventilation and arterial blood gases.
        """
        fio2 = inputs.get("fiO2", self._params["fiO2"])
        p_atm = self._params["barometric_pressure_mmHg"]
        rq = self._params["respiratory_quotient"]
        vd = self._params["dead_space_L"]
        gain_co2 = self._params["chemoreceptor_co2_gain"]
        gain_o2 = self._params["chemoreceptor_o2_gain"]
        exercise = inputs.get("exercise", 0.0)

        # Baseline metabolic demands (scale with exercise)
        vo2_target = 250.0 * (1.0 + exercise * 6.0)
        vco2_target = vo2_target * rq

        # Chemoreceptor control of alveolar ventilation
        # Driven by arterial pCO2 deviation above 40 mmHg, hypoxia (pO2 < 60),
        # and proportional exercise feedforward (hyperpnea matched to metabolic VCO2)
        co2_drive = gain_co2 * (state["paCO2_mmHg"] - 40.0)
        o2_drive = gain_o2 * max(0.0, 60.0 - state["paO2_mmHg"])
        exercise_drive = (vco2_target - 200.0) / 55.0

        target_ve = max(4.0, 7.0 + co2_drive + o2_drive + exercise_drive)
        # Partition target VE into RR and VT
        target_vt = min(2.5, max(0.35, 0.50 + 0.04 * (target_ve - 7.0)))
        target_rr = target_ve / target_vt
        target_va = target_rr * max(0.1, target_vt - vd)

        d_rr = np.clip(0.3 * (target_rr - state["respiratory_rate_bpm"]), -6.0, 6.0)
        d_vt = np.clip(0.3 * (target_vt - state["tidal_volume_L"]), -0.3, 0.3)
        d_va = np.clip(0.3 * (target_va - state["alveolar_ventilation_L_min"]), -6.0, 6.0)

        # Alveolar Gas Equation: pAO2 = FiO2 * (Patm - 47) - (paCO2 / RQ)
        target_pAO2 = fio2 * (p_atm - 47.0) - (state["paCO2_mmHg"] / rq)
        d_pAO2 = np.clip(0.5 * (target_pAO2 - state["pAO2_mmHg"]), -10.0, 10.0)

        # Arterial paCO2 kinetics:
        # Effective alveolar ventilation transitioning with exercise
        target_paCO2 = (vco2_target / (max(1.0, state["alveolar_ventilation_L_min"]) * 1000.0)) * (p_atm - 47.0)
        d_paCO2 = np.clip(0.15 * (target_paCO2 - state["paCO2_mmHg"]), -2.5, 2.5)

        # Arterial paO2 kinetics:
        # Equilibration toward alveolar pAO2 minus normal A-a gradient (6-12 mmHg)
        a_a_gradient = 8.0 + 0.05 * exercise * 15.0
        target_paO2 = max(25.0, target_pAO2 - a_a_gradient)
        d_paO2 = np.clip(0.5 * (target_paO2 - state["paO2_mmHg"]), -10.0, 10.0)

        # SaO2 via Severinghaus equation
        p = state["paO2_mmHg"]
        target_sao2 = 100.0 * (((p ** 3) + 150.0 * p) / ((p ** 3) + 150.0 * p + 23400.0))
        d_sao2 = np.clip(0.5 * (target_sao2 - state["saO2_pct"]), -5.0, 5.0)

        # Arterial pH (Henderson-Hasselbalch with renal bicarbonate from transport)
        hco3 = transport.get("hco3_mEq_L", 24.0)
        target_ph = 6.10 + np.log10(max(0.1, hco3 / (0.0306 * max(10.0, state["paCO2_mmHg"]))))
        d_ph = np.clip(0.15 * (target_ph - state["pH"]), -0.05, 0.05)

        return {
            "respiratory_rate_bpm": d_rr,
            "tidal_volume_L": d_vt,
            "alveolar_ventilation_L_min": d_va,
            "pAO2_mmHg": d_pAO2,
            "paO2_mmHg": d_paO2,
            "paCO2_mmHg": d_paCO2,
            "saO2_pct": d_sao2,
            "pH": d_ph,
        }

    def exchange_fluxes(self, transport: Dict[str, float]) -> Dict[str, float]:
        """
        Oxygen uptake into blood and carbon dioxide clearance from blood.
        """
        return {
            "o2_uptake_mL_min": float(self._state.get("vo2_consumption_mL_min", 250.0)),
            "co2_elimination_mL_min": float(self._state.get("vco2_production_mL_min", 200.0)),
            "paO2": float(self._state["paO2_mmHg"]),
            "paCO2": float(self._state["paCO2_mmHg"]),
            "saO2": float(self._state["saO2_pct"]),
            "pH": float(self._state["pH"]),
        }
