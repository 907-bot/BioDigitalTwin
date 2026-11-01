"""
BioDigitalTwin — Cardiovascular & Convective Perfusion Organ Model.

Mechanisms:
- Cardiac pump function (Heart rate, stroke volume via Frank-Starling mechanism)
- Arteriolar total peripheral resistance (TPR) and arterial compliance
- Active metabolic hyperemia in working skeletal muscle during exercise
- Mean Arterial Pressure (MAP = CO * TPR + CVP) and pulse pressure dynamics
- Regional perfusion delivery to brain, heart, kidneys, gut, liver, muscle, adipose
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from ..core.organ_interface import OrganModel
from ..core.state_registry import StateDescriptor, ParameterDescriptor, get_global_registry
from ..core.shared_environment import SharedEnvironment
from ..core.flux_registry import Flux


class CardiovascularOrganModel(OrganModel):
    """
    Cardiovascular organ module governing central hemodynamics and regional perfusion.
    """

    def __init__(self):
        super().__init__(organ_id="cardiovascular")

        # 1. State Schema
        self.state_schema = {
            "cardio.heart_rate_bpm": StateDescriptor(
                name="cardio.heart_rate_bpm",
                display_name="Heart Rate",
                unit="bpm",
                compartment="heart",
                min_bound=30.0,
                max_bound=220.0,
                nominal=72.0,
                initial_condition_source="population_prior",
                is_observable=True,
                owning_module="cardiovascular",
                description="Cardiac chronotropic contraction frequency",
            ),
            "cardio.stroke_volume_mL": StateDescriptor(
                name="cardio.stroke_volume_mL",
                display_name="Stroke Volume",
                unit="mL",
                compartment="heart",
                min_bound=20.0,
                max_bound=180.0,
                nominal=70.0,
                initial_condition_source="population_prior",
                is_observable=False,
                owning_module="cardiovascular",
                description="Blood volume ejected per ventricular systole",
            ),
            "cardio.tpr_mmHg_min_L": StateDescriptor(
                name="cardio.tpr_mmHg_min_L",
                display_name="Total Peripheral Resistance",
                unit="mmHg*min/L",
                compartment="systemic_vasculature",
                min_bound=6.0,
                max_bound=35.0,
                nominal=18.0,
                initial_condition_source="population_prior",
                is_observable=False,
                owning_module="cardiovascular",
                description="Net arteriolar resistance opposing systemic blood flow",
            ),
            "cardio.cardiac_output_L_min": StateDescriptor(
                name="cardio.cardiac_output_L_min",
                display_name="Cardiac Output",
                unit="L/min",
                compartment="heart",
                min_bound=1.5,
                max_bound=25.0,
                nominal=5.04,
                initial_condition_source="population_prior",
                is_observable=True,
                owning_module="cardiovascular",
                description="Total volume of blood pumped per minute (HR * SV / 1000)",
            ),
            "cardio.mean_arterial_pressure_mmHg": StateDescriptor(
                name="cardio.mean_arterial_pressure_mmHg",
                display_name="Mean Arterial Pressure",
                unit="mmHg",
                compartment="arterial_tree",
                min_bound=40.0,
                max_bound=200.0,
                nominal=93.0,
                initial_condition_source="cuff_measurement",
                is_observable=True,
                owning_module="cardiovascular",
                description="Time-weighted average arterial pressure driving tissue perfusion",
            ),
        }

        # 2. Parameter Schema
        self.parameter_schema = {
            "cardio.baseline_hr": ParameterDescriptor(
                name="cardio.baseline_hr",
                display_name="Resting Heart Rate",
                unit="bpm",
                default_value=72.0,
                min_bound=45.0,
                max_bound=100.0,
                provenance="Guyton & Hall Textbook of Medical Physiology, 14th Ed.",
                is_personalized=True,
                description="Intrinsic resting sinus rhythm",
            ),
            "cardio.baseline_sv": ParameterDescriptor(
                name="cardio.baseline_sv",
                display_name="Resting Stroke Volume",
                unit="mL",
                default_value=70.0,
                min_bound=40.0,
                max_bound=120.0,
                provenance="Guyton & Hall Textbook of Medical Physiology, 14th Ed.",
                is_personalized=True,
                description="End-diastolic minus end-systolic ventricular volume",
            ),
            "cardio.baseline_tpr": ParameterDescriptor(
                name="cardio.baseline_tpr",
                display_name="Resting Arteriolar Resistance",
                unit="mmHg*min/L",
                default_value=18.0,
                min_bound=10.0,
                max_bound=28.0,
                provenance="Guyton & Hall Textbook of Medical Physiology, 14th Ed.",
                is_personalized=True,
                description="Systemic vascular resistance at rest",
            ),
        }

        # Auto-register with global registry
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
            "cardio.heart_rate_bpm": 72.0,
            "cardio.stroke_volume_mL": 70.0,
            "cardio.tpr_mmHg_min_L": 18.0,
            "cardio.cardiac_output_L_min": 5.04,
            "cardio.mean_arterial_pressure_mmHg": 93.0,
        }
        self._params = {k: v.default_value for k, v in self.parameter_schema.items()}
        if patient_profile:
            if "baseline_hr" in patient_profile:
                self._params["cardio.baseline_hr"] = float(patient_profile["baseline_hr"])
                self._state["cardio.heart_rate_bpm"] = float(patient_profile["baseline_hr"])

    def compute_derivatives(
        self,
        time: float,
        state: Dict[str, float],
        environment: SharedEnvironment,
    ) -> Dict[str, float]:
        """
        Calculates time derivatives for heart rate, stroke volume, and vascular tone.
        Modulated by sympathetic tone and active exercise vasodilation.
        """
        # Exercise stimulus from environment or hormones
        ex = getattr(environment, "exercise_intensity", 0.0)
        pra = environment.hormones.plasma_renin_activity_ng_mL_h
        # SNS tone proxy (normalized around 1.0)
        sns_tone = max(0.5, min(2.5, pra / 1.2))

        # Target heart rate modulated by SNS tone and exertion
        target_hr = self._params["cardio.baseline_hr"] * (1.0 + 0.95 * ex + 0.35 * (sns_tone - 1.0))
        target_sv = self._params["cardio.baseline_sv"] * (1.0 + 0.30 * ex) * (1.0 + 0.15 * (sns_tone - 1.0))
        target_tpr = self._params["cardio.baseline_tpr"] / ((1.0 + 0.70 * ex) * (1.0 + 0.3 * (sns_tone - 1.0)))

        # Dynamic relaxation toward targets (time constants ~0.5 to 1.0 min)
        d_hr = 1.0 * (target_hr - state["cardio.heart_rate_bpm"])
        d_sv = 1.0 * (target_sv - state["cardio.stroke_volume_mL"])
        d_tpr = 0.8 * (target_tpr - state["cardio.tpr_mmHg_min_L"])

        # Algebraic updates for CO and MAP
        new_hr = state["cardio.heart_rate_bpm"] + d_hr * 0.01
        new_sv = state["cardio.stroke_volume_mL"] + d_sv * 0.01
        new_tpr = state["cardio.tpr_mmHg_min_L"] + d_tpr * 0.01

        target_co = (new_hr * new_sv) / 1000.0
        d_co = 2.0 * (target_co - state["cardio.cardiac_output_L_min"])

        target_map = target_co * new_tpr + 4.0
        d_map = 1.5 * (target_map - state["cardio.mean_arterial_pressure_mmHg"])

        return {
            "cardio.heart_rate_bpm": float(np.clip(d_hr, -20.0, 20.0)),
            "cardio.stroke_volume_mL": float(np.clip(d_sv, -10.0, 10.0)),
            "cardio.tpr_mmHg_min_L": float(np.clip(d_tpr, -5.0, 5.0)),
            "cardio.cardiac_output_L_min": float(np.clip(d_co, -5.0, 5.0)),
            "cardio.mean_arterial_pressure_mmHg": float(np.clip(d_map, -15.0, 15.0)),
        }

    def exchange_fluxes(
        self,
        time: float,
        state: Dict[str, float],
        environment: SharedEnvironment,
    ) -> List[Flux]:
        """
        Updates central circulation state in the shared environment.
        """
        environment.update_hemodynamics(
            hr=state["cardio.heart_rate_bpm"],
            sv=state["cardio.stroke_volume_mL"],
            tpr=state["cardio.tpr_mmHg_min_L"],
        )
        return []
