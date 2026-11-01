"""
BioDigitalTwin — Endocrine & Neuroendocrine Organ Model.

Mechanisms:
- Arterial baroreflex control modulating Sympathetic (SNS) and Parasympathetic (PNS) tone
- Hypothalamic-Pituitary-Adrenal (HPA) axis controlling cortisol circadian secretion and stress response
- Renin-Angiotensin-Aldosterone System (RAAS) cascade activated by renal hypoperfusion
- Posterior pituitary Antidiuretic Hormone (ADH / Vasopressin) regulating osmolality and fluid retention
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from ..core.organ_interface import OrganModel
from ..core.state_registry import StateDescriptor, ParameterDescriptor, get_global_registry
from ..core.shared_environment import SharedEnvironment
from ..core.flux_registry import Flux


class EndocrineOrganModel(OrganModel):
    """
    Neuroendocrine module coordinating autonomic tone, HPA axis, RAAS cascade, and osmoregulation.
    """

    def __init__(self):
        super().__init__(organ_id="endocrine")

        self.state_schema = {
            "endocrine.sympathetic_tone": StateDescriptor(
                name="endocrine.sympathetic_tone",
                display_name="Sympathetic Nervous System Tone",
                unit="normalized",
                compartment="autonomic_nervous_system",
                min_bound=0.20,
                max_bound=3.50,
                nominal=1.00,
                initial_condition_source="hrv_spectral_analysis",
                is_observable=True,
                owning_module="endocrine",
                description="Efferent postganglionic sympathetic adrenergic drive to heart and vasculature",
            ),
            "endocrine.parasympathetic_tone": StateDescriptor(
                name="endocrine.parasympathetic_tone",
                display_name="Parasympathetic Vagal Tone",
                unit="normalized",
                compartment="autonomic_nervous_system",
                min_bound=0.10,
                max_bound=2.50,
                nominal=1.00,
                initial_condition_source="hrv_rmssd",
                is_observable=True,
                owning_module="endocrine",
                description="Cardiac vagal cholinergic tone modulating sinoatrial node conduction",
            ),
            "endocrine.cortisol_nmol_L": StateDescriptor(
                name="endocrine.cortisol_nmol_L",
                display_name="Plasma Cortisol Concentration",
                unit="nmol/L",
                compartment="plasma",
                min_bound=50.0,
                max_bound=1200.0,
                nominal=300.0,
                initial_condition_source="serum_cortisol_lab",
                is_observable=True,
                owning_module="endocrine",
                description="Glucocorticoid hormone secreted by adrenal cortex under ACTH control",
            ),
            "endocrine.renin_activity_ng_mL_h": StateDescriptor(
                name="endocrine.renin_activity_ng_mL_h",
                display_name="Plasma Renin Activity",
                unit="ng/mL/h",
                compartment="plasma",
                min_bound=0.1,
                max_bound=25.0,
                nominal=1.2,
                initial_condition_source="endocrine_lab",
                is_observable=True,
                owning_module="endocrine",
                description="Juxtaglomerular enzymatic activity catalyzing conversion to angiotensin I",
            ),
            "endocrine.aldosterone_ng_dL": StateDescriptor(
                name="endocrine.aldosterone_ng_dL",
                display_name="Plasma Aldosterone",
                unit="ng/dL",
                compartment="plasma",
                min_bound=2.0,
                max_bound=80.0,
                nominal=10.0,
                initial_condition_source="endocrine_lab",
                is_observable=True,
                owning_module="endocrine",
                description="Mineralocorticoid hormone promoting renal distal tubule sodium retention",
            ),
        }

        self.parameter_schema = {
            "endocrine.baroreflex_sensitivity": ParameterDescriptor(
                name="endocrine.baroreflex_sensitivity",
                display_name="Arterial Baroreflex Gain",
                unit="1/mmHg",
                default_value=0.04,
                min_bound=0.01,
                max_bound=0.10,
                provenance="Karas M et al., J Am Coll Cardiol 2005",
                is_personalized=True,
                description="Fractional change in autonomic tone per mmHg deviation from setpoint MAP (93 mmHg)",
            ),
            "endocrine.target_map_mmHg": ParameterDescriptor(
                name="endocrine.target_map_mmHg",
                display_name="Baroreflex MAP Setpoint",
                unit="mmHg",
                default_value=93.0,
                min_bound=70.0,
                max_bound=120.0,
                provenance="Guyton & Hall Textbook of Medical Physiology, 14th Ed.",
                is_personalized=True,
                description="Homeostatic setpoint around which baroreceptors buffer pressure deviations",
            ),
        }

        reg = get_global_registry()
        for d in self.state_schema.values():
            if d.name not in reg._states:
                reg.register_state(d)
        for p in self.parameter_schema.values():
            if p.name not in reg._parameters:
                reg.register_parameter(p)

    def initialize(
        self,
        patient_profile: Optional[Dict[str, Any]] = None,
        environment: Optional[SharedEnvironment] = None,
    ) -> None:
        self._state = {
            "endocrine.sympathetic_tone": 1.00,
            "endocrine.parasympathetic_tone": 1.00,
            "endocrine.cortisol_nmol_L": 300.0,
            "endocrine.renin_activity_ng_mL_h": 1.2,
            "endocrine.aldosterone_ng_dL": 10.0,
        }
        self._params = {k: v.default_value for k, v in self.parameter_schema.items()}

    def compute_derivatives(
        self,
        time: float,
        state: Dict[str, float],
        environment: SharedEnvironment,
    ) -> Dict[str, float]:
        """
        Calculates baroreflex tone shifts, HPA cortisol regulation, and RAAS cascade activation.
        """
        map_val = environment.circulation.mean_arterial_pressure_mmHg
        target_map = self._params["endocrine.target_map_mmHg"]
        gain = self._params["endocrine.baroreflex_sensitivity"]

        # 1. Baroreceptor afferent firing: modulated by central command during exertion
        ex = getattr(environment, "exercise_intensity", 0.0)
        pressure_deviation = map_val - target_map
        target_sns = max(0.3, min(3.0, 1.0 + 1.8 * ex - gain * pressure_deviation))
        target_pns = max(0.2, min(2.5, (1.0 / (1.0 + 1.5 * ex)) + gain * pressure_deviation))

        d_sns = 0.8 * (target_sns - state["endocrine.sympathetic_tone"])
        d_pns = 0.8 * (target_pns - state["endocrine.parasympathetic_tone"])

        # 2. RAAS Activation: triggered by renal hypoperfusion (MAP < 85) or elevated SNS tone
        if map_val < 85.0:
            renin_stim = 1.0 + 0.10 * (85.0 - map_val)
        else:
            renin_stim = 1.0
        target_renin = 1.2 * renin_stim * state["endocrine.sympathetic_tone"]
        d_renin = 0.3 * (target_renin - state["endocrine.renin_activity_ng_mL_h"])

        # Aldosterone tracks renin/angiotensin II
        target_aldo = 10.0 * (state["endocrine.renin_activity_ng_mL_h"] / 1.2)
        d_aldo = 0.2 * (target_aldo - state["endocrine.aldosterone_ng_dL"])

        # 3. Cortisol Circadian Pacemaker + Stress
        # Circadian peak at 8:00 AM (480 min) ~ 450 nmol/L, nadir at midnight ~ 150 nmol/L
        minute_of_day = time % 1440.0
        circadian_angle = 2.0 * np.pi * (minute_of_day - 480.0) / 1440.0
        baseline_cortisol = 300.0 + 150.0 * np.cos(circadian_angle)
        stress_boost = (state["endocrine.sympathetic_tone"] - 1.0) * 80.0
        target_cortisol = max(60.0, baseline_cortisol + stress_boost)
        d_cortisol = 0.2 * (target_cortisol - state["endocrine.cortisol_nmol_L"])

        return {
            "endocrine.sympathetic_tone": float(np.clip(d_sns, -1.0, 1.0)),
            "endocrine.parasympathetic_tone": float(np.clip(d_pns, -1.0, 1.0)),
            "endocrine.cortisol_nmol_L": float(np.clip(d_cortisol, -20.0, 20.0)),
            "endocrine.renin_activity_ng_mL_h": float(np.clip(d_renin, -2.0, 2.0)),
            "endocrine.aldosterone_ng_dL": float(np.clip(d_aldo, -5.0, 5.0)),
        }

    def exchange_fluxes(
        self,
        time: float,
        state: Dict[str, float],
        environment: SharedEnvironment,
    ) -> List[Flux]:
        """
        Updates circulating hormone levels in the shared environment.
        """
        environment.hormones.plasma_renin_activity_ng_mL_h = float(state["endocrine.renin_activity_ng_mL_h"])
        environment.hormones.aldosterone_ng_dL = float(state["endocrine.aldosterone_ng_dL"])
        environment.hormones.cortisol_nmol_L = float(state["endocrine.cortisol_nmol_L"])
        return []
