"""
BioDigitalTwin — Endocrine Pancreas Organ Model.

Mechanisms:
- Pancreatic beta-cell insulin secretion stimulated by glucose and potentiated by incretins (GLP-1/GIP)
- Pancreatic alpha-cell glucagon secretion stimulated by hypoglycemia and inhibited by insulin
- Physiological incretin effect amplifying postprandial insulin release
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from ..core.organ_interface import OrganModel
from ..core.state_registry import StateDescriptor, ParameterDescriptor, get_global_registry
from ..core.shared_environment import SharedEnvironment
from ..core.flux_registry import Flux


class PancreaticOrganModel(OrganModel):
    """
    Endocrine pancreas module governing glucose-stimulated insulin and counter-regulatory glucagon.
    """

    def __init__(self):
        super().__init__(organ_id="pancreas")

        self.state_schema = {
            "pancreas.insulin_secretion_rate_uU_min": StateDescriptor(
                name="pancreas.insulin_secretion_rate_uU_min",
                display_name="Beta-Cell Insulin Secretion Rate",
                unit="uU/min",
                compartment="pancreas_islets",
                min_bound=50.0,
                max_bound=8000.0,
                nominal=450.0,
                initial_condition_source="ivgtt_first_phase",
                is_observable=False,
                owning_module="pancreas",
                description="Biphasic pancreatic beta-cell insulin secretion into portal vein",
            ),
            "pancreas.glucagon_secretion_rate_pg_min": StateDescriptor(
                name="pancreas.glucagon_secretion_rate_pg_min",
                display_name="Alpha-Cell Glucagon Secretion Rate",
                unit="pg/min",
                compartment="pancreas_islets",
                min_bound=20.0,
                max_bound=2500.0,
                nominal=200.0,
                initial_condition_source="endocrine_panel",
                is_observable=False,
                owning_module="pancreas",
                description="Pancreatic alpha-cell glucagon release stimulated during hypoglycemia",
            ),
        }

        self.parameter_schema = {
            "pancreas.basal_insulin_secretion": ParameterDescriptor(
                name="pancreas.basal_insulin_secretion",
                display_name="Basal Insulin Secretion",
                unit="uU/min",
                default_value=3800.0,
                min_bound=1000.0,
                max_bound=8000.0,
                provenance="Polonsky KS et al., J Clin Invest 1988",
                is_personalized=True,
                description="Fasting basal insulin secretion rate (~15-20 mU/h in healthy adults)",
            ),
            "pancreas.beta_cell_glucose_sensitivity": ParameterDescriptor(
                name="pancreas.beta_cell_glucose_sensitivity",
                display_name="Beta-Cell Glucose Sensitivity Gain",
                unit="uU/min per mg/dL",
                default_value=250.0,
                min_bound=50.0,
                max_bound=800.0,
                provenance="Cobelli C et al., Diabetes 2007",
                is_personalized=True,
                description="Slope of insulin secretion response to glucose above 85 mg/dL threshold",
            ),
            "pancreas.incretin_potentiating_factor": ParameterDescriptor(
                name="pancreas.incretin_potentiating_factor",
                display_name="GLP-1 Incretin Potentiation Gain",
                unit="1/(pmol/L)",
                default_value=0.05,
                min_bound=0.01,
                max_bound=0.12,
                provenance="Nauck MA et al., Diabetologia 1986",
                is_personalized=False,
                description="Fractional enhancement of glucose-stimulated insulin release per unit of plasma GLP-1",
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
            "pancreas.insulin_secretion_rate_uU_min": 3800.0,
            "pancreas.glucagon_secretion_rate_pg_min": 200.0,
        }
        self._params = {k: v.default_value for k, v in self.parameter_schema.items()}

    def compute_derivatives(
        self,
        time: float,
        state: Dict[str, float],
        environment: SharedEnvironment,
    ) -> Dict[str, float]:
        """
        Calculates glucose- and incretin-stimulated insulin release and glucagon suppression.
        """
        glucose = environment.chemistry.glucose_mg_dL
        insulin = environment.chemistry.insulin_uU_mL
        glp1 = environment.hormones.glp1_pmol_L

        # 1. Incretin amplification of insulin secretion
        glp1_boost = 1.0 + self._params["pancreas.incretin_potentiating_factor"] * max(0.0, glp1 - 15.0)

        # 2. Glucose-stimulated beta-cell insulin secretion
        basal = self._params["pancreas.basal_insulin_secretion"]
        gain = self._params["pancreas.beta_cell_glucose_sensitivity"]
        glucose_excess = max(0.0, glucose - 93.0)
        target_insulin_secretion = (basal + gain * glucose_excess) * glp1_boost

        # 3. Glucagon secretion: stimulated during hypoglycemia, suppressed by hyperinsulinemia
        # Basal ~ 200 pg/min. High glucose (>95) or high insulin (>10) strongly suppresses alpha-cells
        suppression = 1.0 / (1.0 + 0.05 * max(0.0, glucose - 90.0) + 0.05 * max(0.0, insulin - 8.0))
        hypoglycemia_boost = 1.0 + 0.04 * max(0.0, 85.0 - glucose) ** 1.5
        target_glucagon_secretion = 200.0 * suppression * hypoglycemia_boost

        d_ins = 1.0 * (target_insulin_secretion - state["pancreas.insulin_secretion_rate_uU_min"])
        d_glc = 0.5 * (target_glucagon_secretion - state["pancreas.glucagon_secretion_rate_pg_min"])

        return {
            "pancreas.insulin_secretion_rate_uU_min": float(np.clip(d_ins, -500.0, 500.0)),
            "pancreas.glucagon_secretion_rate_pg_min": float(np.clip(d_glc, -100.0, 100.0)),
        }

    def exchange_fluxes(
        self,
        time: float,
        state: Dict[str, float],
        environment: SharedEnvironment,
    ) -> List[Flux]:
        """
        Secretes insulin and glucagon into circulating blood.
        """
        ins_rate = float(state["pancreas.insulin_secretion_rate_uU_min"])
        glc_rate = float(state["pancreas.glucagon_secretion_rate_pg_min"])

        return [
            Flux(
                source="pancreas",
                destination="blood",
                quantity="insulin",
                value=ins_rate,
                unit="uU/min",
                mechanism="Beta-Cell Glucose & Incretin Stimulated Insulin Secretion",
            ),
            Flux(
                source="pancreas",
                destination="blood",
                quantity="glucagon",
                value=glc_rate,
                unit="pg/min",
                mechanism="Alpha-Cell Glucagon Secretion",
            ),
        ]
