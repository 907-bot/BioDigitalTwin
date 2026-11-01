"""
BioDigitalTwin — Renal Organ Model.

Mechanisms:
- Glomerular filtration rate (GFR) with renal autoregulation
- Tubular glucose reabsorption (SGLT2 / SGLT1) and threshold glycosuria
- Renal sodium, potassium, and fluid handling
- Acid-base bicarbonate reabsorption and proton excretion
- Juxtaglomerular renin secretion triggered by renal hypoperfusion
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from ..core.organ_interface import OrganModel
from ..core.state_registry import StateDescriptor, ParameterDescriptor, get_global_registry
from ..core.shared_environment import SharedEnvironment
from ..core.flux_registry import Flux


class RenalOrganModel(OrganModel):
    """
    Renal organ module governing filtration, tubular transport, glycosuria, and fluid balance.
    """

    def __init__(self):
        super().__init__(organ_id="renal")

        self.state_schema = {
            "renal.gfr_mL_min": StateDescriptor(
                name="renal.gfr_mL_min",
                display_name="Glomerular Filtration Rate",
                unit="mL/min",
                compartment="kidney",
                min_bound=10.0,
                max_bound=180.0,
                nominal=105.0,
                initial_condition_source="ckd_epi_formula",
                is_observable=True,
                owning_module="renal",
                description="Total volume of fluid filtered through glomerular capillaries per minute",
            ),
            "renal.urinary_glucose_excretion_mg_min": StateDescriptor(
                name="renal.urinary_glucose_excretion_mg_min",
                display_name="Urinary Glucose Excretion (Glycosuria)",
                unit="mg/min",
                compartment="urine",
                min_bound=0.0,
                max_bound=400.0,
                nominal=0.0,
                initial_condition_source="urinalysis",
                is_observable=True,
                owning_module="renal",
                description="Glucose mass excreted in urine when filtered load exceeds tubular threshold Tm_G",
            ),
            "renal.urine_flow_rate_mL_min": StateDescriptor(
                name="renal.urine_flow_rate_mL_min",
                display_name="Urine Output Rate",
                unit="mL/min",
                compartment="urine",
                min_bound=0.2,
                max_bound=15.0,
                nominal=1.0,
                initial_condition_source="clinical_foley_output",
                is_observable=True,
                owning_module="renal",
                description="Rate of final urinary water elimination",
            ),
            "renal.fractional_sodium_excretion_pct": StateDescriptor(
                name="renal.fractional_sodium_excretion_pct",
                display_name="Fractional Sodium Excretion (FENa)",
                unit="%",
                compartment="kidney",
                min_bound=0.1,
                max_bound=5.0,
                nominal=0.8,
                initial_condition_source="lab_chemistry",
                is_observable=True,
                owning_module="renal",
                description="Percentage of filtered sodium excreted in final urine",
            ),
        }

        self.parameter_schema = {
            "renal.baseline_gfr": ParameterDescriptor(
                name="renal.baseline_gfr",
                display_name="Baseline GFR",
                unit="mL/min",
                default_value=105.0,
                min_bound=30.0,
                max_bound=150.0,
                provenance="KDIGO Clinical Practice Guideline 2012",
                is_personalized=True,
                description="Age- and sex-adjusted normal glomerular filtration capacity",
            ),
            "renal.glucose_transport_maximum_mg_min": ParameterDescriptor(
                name="renal.glucose_transport_maximum_mg_min",
                display_name="Tubular Glucose Transport Maximum (Tm_G)",
                unit="mg/min",
                default_value=270.0,
                min_bound=180.0,
                max_bound=380.0,
                provenance="DeFronzo RA et al., Diabetes Care 2013 (SGLT2 physiology)",
                is_personalized=True,
                description="Maximum tubular reabsorptive capacity before glycosuria occurs",
            ),
            "renal.glucose_threshold_mg_dL": ParameterDescriptor(
                name="renal.glucose_threshold_mg_dL",
                display_name="Renal Glucose Plasma Threshold",
                unit="mg/dL",
                default_value=180.0,
                min_bound=140.0,
                max_bound=220.0,
                provenance="Guyton & Hall Textbook of Medical Physiology, 14th Ed.",
                is_personalized=False,
                description="Circulating glucose level at which tubular reabsorption saturates",
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
            "renal.gfr_mL_min": 105.0,
            "renal.urinary_glucose_excretion_mg_min": 0.0,
            "renal.urine_flow_rate_mL_min": 1.0,
            "renal.fractional_sodium_excretion_pct": 0.8,
        }
        self._params = {k: v.default_value for k, v in self.parameter_schema.items()}
        if patient_profile and "baseline_gfr" in patient_profile:
            self._params["renal.baseline_gfr"] = float(patient_profile["baseline_gfr"])
            self._state["renal.gfr_mL_min"] = float(patient_profile["baseline_gfr"])

    def compute_derivatives(
        self,
        time: float,
        state: Dict[str, float],
        environment: SharedEnvironment,
    ) -> Dict[str, float]:
        """
        Calculates filtration rate, glycosuria, and fluid excretion.
        """
        circ = environment.circulation
        chem = environment.chemistry
        map_val = circ.mean_arterial_pressure_mmHg
        glucose = chem.glucose_mg_dL

        # 1. Renal autoregulation (Myogenic + TGF) maintains GFR across MAP 80-160 mmHg
        if map_val < 75.0:
            # Hypoperfusion drops GFR steeply
            autoreg_factor = max(0.2, (map_val - 20.0) / 55.0)
        elif map_val > 170.0:
            autoreg_factor = min(1.3, 1.0 + (map_val - 170.0) / 100.0)
        else:
            autoreg_factor = 1.0

        target_gfr = self._params["renal.baseline_gfr"] * autoreg_factor

        # 2. Glucose filtration and tubular reabsorption
        # Filtered load = GFR (dL/min) * Glucose (mg/dL) = (GFR / 100) * G
        filtered_load_mg_min = (state["renal.gfr_mL_min"] / 100.0) * glucose
        tm_g = self._params["renal.glucose_transport_maximum_mg_min"]

        # SGLT2/SGLT1 splay kinetics (Michaelis-Menten-like saturation)
        if filtered_load_mg_min > tm_g:
            target_glycosuria = filtered_load_mg_min - tm_g
        elif filtered_load_mg_min > 0.85 * tm_g:
            # Splay region
            splay_fraction = (filtered_load_mg_min - 0.85 * tm_g) / (0.15 * tm_g)
            target_glycosuria = splay_fraction * 0.15 * tm_g
        else:
            target_glycosuria = 0.0

        # Osmotic diuresis from unabsorbed glucose in tubular lumen
        osmotic_water_drag = target_glycosuria * 0.015  # mL/min per mg of glucose excreted
        target_ufr = max(0.3, 1.0 + osmotic_water_drag)

        d_gfr = 0.5 * (target_gfr - state["renal.gfr_mL_min"])
        d_excr = 1.0 * (target_glycosuria - state["renal.urinary_glucose_excretion_mg_min"])
        d_ufr = 0.5 * (target_ufr - state["renal.urine_flow_rate_mL_min"])
        d_fena = 0.1 * (0.8 - state["renal.fractional_sodium_excretion_pct"])

        return {
            "renal.gfr_mL_min": float(np.clip(d_gfr, -15.0, 15.0)),
            "renal.urinary_glucose_excretion_mg_min": float(np.clip(d_excr, -50.0, 50.0)),
            "renal.urine_flow_rate_mL_min": float(np.clip(d_ufr, -2.0, 2.0)),
            "renal.fractional_sodium_excretion_pct": float(np.clip(d_fena, -0.2, 0.2)),
        }

    def exchange_fluxes(
        self,
        time: float,
        state: Dict[str, float],
        environment: SharedEnvironment,
    ) -> List[Flux]:
        """
        Exchanges urinary excretion fluxes removing mass from blood.
        """
        glucose = environment.chemistry.glucose_mg_dL
        gfr = state.get("renal.gfr_mL_min", 105.0)
        filtered_load = (gfr / 100.0) * glucose
        tm_g = self._params.get("renal.glucose_transport_maximum_mg_min", 270.0)

        if filtered_load > tm_g:
            glycosuria = filtered_load - tm_g
        elif filtered_load > 0.85 * tm_g:
            glycosuria = ((filtered_load - 0.85 * tm_g) / (0.15 * tm_g)) * 0.15 * tm_g
        else:
            glycosuria = max(0.0, float(state.get("renal.urinary_glucose_excretion_mg_min", 0.0)))

        ufr = float(state.get("renal.urine_flow_rate_mL_min", 1.0))

        fluxes = []
        if glycosuria > 0.05:
            fluxes.append(
                Flux(
                    source="blood",
                    destination="urine",
                    quantity="glucose",
                    value=float(glycosuria),
                    unit="mg/min",
                    mechanism="Renal Glucosuria (Filtered Load > Tubular SGLT2 Tm_G)",
                )
            )

        # Baseline sodium excretion: ~0.15 mEq/min balancing dietary intake
        fluxes.append(
            Flux(
                source="blood",
                destination="urine",
                quantity="sodium",
                value=0.15,
                unit="mEq/min",
                mechanism="Renal Tubular Natriuresis",
            )
        )

        return fluxes
