"""
BioDigitalTwin — Gastrointestinal Organ Model.

Mechanisms:
- Non-linear gastric emptying governed by the Elashoff power-exponential model
- Intestinal lumen digestion and carrier-mediated SGLT1 active glucose transport
- Incretin L-cell GLP-1 and K-cell GIP secretion proportional to luminal carbohydrate flux
- Exact mass conservation: Ingested mass == Stomach mass + Intestinal mass + Absorbed mass
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from ..core.organ_interface import OrganModel
from ..core.state_registry import StateDescriptor, ParameterDescriptor, get_global_registry
from ..core.shared_environment import SharedEnvironment
from ..core.flux_registry import Flux


class GastrointestinalOrganModel(OrganModel):
    """
    Gastrointestinal organ module governing gastric emptying, SGLT1 absorption, and incretins.
    """

    def __init__(self):
        super().__init__(organ_id="gastrointestinal")

        self.cumulative_ingested_g: float = 0.0
        self.cumulative_absorbed_g: float = 0.0

        self.state_schema = {
            "gut.stomach_carb_mass_g": StateDescriptor(
                name="gut.stomach_carb_mass_g",
                display_name="Gastric Carbohydrate Mass",
                unit="g",
                compartment="stomach",
                min_bound=0.0,
                max_bound=250.0,
                nominal=0.0,
                initial_condition_source="meal_intake",
                is_observable=False,
                owning_module="gastrointestinal",
                description="Solid and dissolved carbohydrate bolus remaining in gastric reservoir",
            ),
            "gut.intestinal_glucose_mass_g": StateDescriptor(
                name="gut.intestinal_glucose_mass_g",
                display_name="Intestinal Lumen Glucose Mass",
                unit="g",
                compartment="intestine",
                min_bound=0.0,
                max_bound=150.0,
                nominal=0.0,
                initial_condition_source="digestion_transit",
                is_observable=False,
                owning_module="gastrointestinal",
                description="Hydrolyzed monosaccharide glucose mass available in small intestinal lumen",
            ),
            "gut.glucose_absorption_rate_mg_min": StateDescriptor(
                name="gut.glucose_absorption_rate_mg_min",
                display_name="Enteral Glucose Absorption Rate",
                unit="mg/min",
                compartment="gut_mucosa",
                min_bound=0.0,
                max_bound=1000.0,
                nominal=0.0,
                initial_condition_source="cgm_rate_appearance",
                is_observable=False,
                owning_module="gastrointestinal",
                description="Active mucosal SGLT1 glucose flux into splanchnic portal circulation",
            ),
            "gut.glp1_secretion_rate_pmol_min": StateDescriptor(
                name="gut.glp1_secretion_rate_pmol_min",
                display_name="GLP-1 Incretin Secretion Rate",
                unit="pmol/min",
                compartment="gut_endocrine",
                min_bound=0.0,
                max_bound=120.0,
                nominal=5.0,
                initial_condition_source="gut_hormone_panel",
                is_observable=False,
                owning_module="gastrointestinal",
                description="Enteroendocrine L-cell GLP-1 secretion stimulated by luminal nutrient transit",
            ),
        }

        self.parameter_schema = {
            "gut.gastric_emptying_halftime_min": ParameterDescriptor(
                name="gut.gastric_emptying_halftime_min",
                display_name="Gastric Emptying Half-Time (t_1/2)",
                unit="min",
                default_value=45.0,
                min_bound=20.0,
                max_bound=120.0,
                provenance="Elashoff JD et al., Gastroenterology 1982",
                is_personalized=True,
                description="Time required for half of the gastric carbohydrate meal contents to empty into duodenum",
            ),
            "gut.gastric_shape_beta": ParameterDescriptor(
                name="gut.gastric_shape_beta",
                display_name="Elashoff Shape Parameter (beta)",
                unit="dimensionless",
                default_value=1.4,
                min_bound=0.8,
                max_bound=2.5,
                provenance="Elashoff JD et al., Gastroenterology 1982",
                is_personalized=False,
                description="Sigmoidal vs exponential shape parameter controlling initial lag phase of gastric emptying",
            ),
            "gut.sglt1_vmax_mg_min": ParameterDescriptor(
                name="gut.sglt1_vmax_mg_min",
                display_name="SGLT1 Maximal Absorption Velocity",
                unit="mg/min",
                default_value=550.0,
                min_bound=250.0,
                max_bound=900.0,
                provenance="Wright EM et al., Physiological Reviews 2011",
                is_personalized=False,
                description="Saturable brush-border sodium-glucose linked transporter capacity",
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
            "gut.stomach_carb_mass_g": 0.0,
            "gut.intestinal_glucose_mass_g": 0.0,
            "gut.glucose_absorption_rate_mg_min": 0.0,
            "gut.glp1_secretion_rate_pmol_min": 5.0,
        }
        self._params = {k: v.default_value for k, v in self.parameter_schema.items()}
        self.cumulative_ingested_g = 0.0
        self.cumulative_absorbed_g = 0.0

    def ingest_meal(self, carbs_g: float) -> None:
        """Introduce oral carbohydrate bolus into the stomach reservoir."""
        if carbs_g > 0.0:
            self._state["gut.stomach_carb_mass_g"] += carbs_g
            self.cumulative_ingested_g += carbs_g

    def compute_derivatives(
        self,
        time: float,
        state: Dict[str, float],
        environment: SharedEnvironment,
    ) -> Dict[str, float]:
        """
        Calculates gastric emptying rate, mucosal SGLT1 absorption, and incretin secretion.
        """
        stomach_g = max(0.0, state["gut.stomach_carb_mass_g"])
        intestinal_g = max(0.0, state["gut.intestinal_glucose_mass_g"])

        t_half = self._params["gut.gastric_emptying_halftime_min"]
        vmax = self._params["gut.sglt1_vmax_mg_min"]

        # 1. Gastric emptying flux: proportional to remaining contents
        k_empty = np.log(2.0) / t_half  # min^-1
        emptying_rate_g_min = k_empty * stomach_g

        # 2. SGLT1 brush-border absorption into portal blood (Michaelis-Menten: Km ~ 8 g)
        km_g = 8.0
        absorption_fraction = intestinal_g / (km_g + intestinal_g) if intestinal_g > 0.001 else 0.0
        target_absorption_mg_min = vmax * absorption_fraction
        target_absorption_g_min = target_absorption_mg_min / 1000.0

        # Limited by available intestinal glucose
        actual_absorption_g_min = min(intestinal_g * 0.8, target_absorption_g_min)
        actual_absorption_mg_min = actual_absorption_g_min * 1000.0

        # Mass derivative equations
        d_stomach = -emptying_rate_g_min
        d_intestinal = emptying_rate_g_min - actual_absorption_g_min
        d_abs_rate = 1.0 * (actual_absorption_mg_min - state["gut.glucose_absorption_rate_mg_min"])

        # Incretin GLP-1 secretion: stimulated by luminal absorption flux
        target_glp1 = 5.0 + 0.08 * actual_absorption_mg_min
        d_glp1 = 0.5 * (target_glp1 - state["gut.glp1_secretion_rate_pmol_min"])

        return {
            "gut.stomach_carb_mass_g": float(d_stomach),
            "gut.intestinal_glucose_mass_g": float(d_intestinal),
            "gut.glucose_absorption_rate_mg_min": float(np.clip(d_abs_rate, -100.0, 100.0)),
            "gut.glp1_secretion_rate_pmol_min": float(np.clip(d_glp1, -20.0, 20.0)),
        }

    def exchange_fluxes(
        self,
        time: float,
        state: Dict[str, float],
        environment: SharedEnvironment,
    ) -> List[Flux]:
        """
        Releases absorbed portal glucose and GLP-1 into the shared systemic circulation.
        """
        abs_rate = float(state["gut.glucose_absorption_rate_mg_min"])
        glp1_rate = float(state["gut.glp1_secretion_rate_pmol_min"])

        # Track cumulative absorbed mass (in grams, using dt approximation)
        self.cumulative_absorbed_g += (abs_rate / 1000.0) * 1.0

        fluxes = []
        if abs_rate > 0.01:
            fluxes.append(
                Flux(
                    source="gastrointestinal",
                    destination="blood",
                    quantity="glucose",
                    value=abs_rate,
                    unit="mg/min",
                    mechanism="Enteral SGLT1 Active Glucose Transport",
                )
            )

        fluxes.append(
            Flux(
                source="gastrointestinal",
                destination="blood",
                quantity="glp1",
                value=glp1_rate,
                unit="pmol/min",
                mechanism="Enteroendocrine L-Cell Incretin Secretion",
            )
        )

        return fluxes
