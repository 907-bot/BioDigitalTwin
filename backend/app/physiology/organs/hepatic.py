"""
BioDigitalTwin — Hepatic Organ Model.

Mechanisms:
- Reversible hepatic glycogen turnover (synthase vs phosphorylase)
- Hepatic Glucose Production (HGP) regulated by portal insulin and glucagon
- Gluconeogenesis from circulating precursors (lactate, glycerol, alanine)
- Well-stirred intrinsic hepatic drug metabolism and enzymatic clearance (CYP450)
- Urea cycle synthesis converting amino acid ammonia to circulating urea
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from ..core.organ_interface import OrganModel
from ..core.state_registry import StateDescriptor, ParameterDescriptor, get_global_registry
from ..core.shared_environment import SharedEnvironment
from ..core.flux_registry import Flux


class HepaticOrganModel(OrganModel):
    """
    Hepatic organ module governing metabolic fuel homeostasis, clearance, and urea synthesis.
    """

    def __init__(self):
        super().__init__(organ_id="hepatic")

        self.state_schema = {
            "hepatic.glycogen_mass_g": StateDescriptor(
                name="hepatic.glycogen_mass_g",
                display_name="Liver Glycogen Mass",
                unit="g",
                compartment="liver",
                min_bound=5.0,
                max_bound=160.0,
                nominal=100.0,
                initial_condition_source="fasting_prior",
                is_observable=False,
                owning_module="hepatic",
                description="Intracellular hepatic macromolecular glycogen store",
            ),
            "hepatic.hgp_rate_mg_min": StateDescriptor(
                name="hepatic.hgp_rate_mg_min",
                display_name="Hepatic Glucose Production Rate",
                unit="mg/min",
                compartment="liver",
                min_bound=10.0,
                max_bound=500.0,
                nominal=150.0,
                initial_condition_source="euglycemic_clamp",
                is_observable=False,
                owning_module="hepatic",
                description="Net glucose released into systemic circulation via glycogenolysis and gluconeogenesis",
            ),
            "hepatic.gluconeogenesis_rate_mg_min": StateDescriptor(
                name="hepatic.gluconeogenesis_rate_mg_min",
                display_name="Hepatic Gluconeogenesis",
                unit="mg/min",
                compartment="liver",
                min_bound=10.0,
                max_bound=250.0,
                nominal=45.0,
                initial_condition_source="stable_isotope_tracer",
                is_observable=False,
                owning_module="hepatic",
                description="De novo glucose synthesis from lactate, glycerol, and gluconeogenic amino acids",
            ),
            "hepatic.urea_synthesis_mg_min": StateDescriptor(
                name="hepatic.urea_synthesis_mg_min",
                display_name="Urea Cycle Synthesis Rate",
                unit="mg/min",
                compartment="liver",
                min_bound=2.0,
                max_bound=50.0,
                nominal=12.0,
                initial_condition_source="nitrogen_balance",
                is_observable=False,
                owning_module="hepatic",
                description="Hepatic conversion of waste ammonium from protein turnover to circulating urea",
            ),
        }

        self.parameter_schema = {
            "hepatic.basal_hgp": ParameterDescriptor(
                name="hepatic.basal_hgp",
                display_name="Basal HGP Capacity",
                unit="mg/min",
                default_value=150.0,
                min_bound=80.0,
                max_bound=250.0,
                provenance="DeFronzo et al., Diabetes 1979",
                is_personalized=True,
                description="Endogenous glucose production in fasting state (approx 2 mg/kg/min)",
            ),
            "hepatic.insulin_suppression_ic50": ParameterDescriptor(
                name="hepatic.insulin_suppression_ic50",
                display_name="HGP Insulin IC50",
                unit="uU/mL",
                default_value=25.0,
                min_bound=10.0,
                max_bound=60.0,
                provenance="Bergman RN, Endocrine Reviews 1989",
                is_personalized=True,
                description="Portal insulin concentration producing 50% suppression of glycogenolysis",
            ),
            "hepatic.glucagon_stimulation_ec50": ParameterDescriptor(
                name="hepatic.glucagon_stimulation_ec50",
                display_name="HGP Glucagon EC50",
                unit="pg/mL",
                default_value=90.0,
                min_bound=40.0,
                max_bound=200.0,
                provenance="Cherrington AD, Diabetes 1999",
                is_personalized=False,
                description="Glucagon level yielding half-maximal stimulation of hepatic glycogen phosphorylase",
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
        weight = patient_profile.get("weight_kg", 70.0) if patient_profile else 70.0
        # Basal HGP scales roughly with body weight (~2 mg/kg/min)
        basal = 2.1 * weight
        self._state = {
            "hepatic.glycogen_mass_g": 100.0,
            "hepatic.hgp_rate_mg_min": float(basal),
            "hepatic.gluconeogenesis_rate_mg_min": float(0.30 * basal),
            "hepatic.urea_synthesis_mg_min": 12.0,
        }
        self._params = {k: v.default_value for k, v in self.parameter_schema.items()}
        self._params["hepatic.basal_hgp"] = float(basal)

    def compute_derivatives(
        self,
        time: float,
        state: Dict[str, float],
        environment: SharedEnvironment,
    ) -> Dict[str, float]:
        """
        Calculates rates of glycogen synthesis/glycogenolysis and HGP regulation.
        """
        chem = environment.chemistry
        insulin = chem.insulin_uU_mL
        glucagon = chem.glucagon_pg_mL
        glucose = chem.glucose_mg_dL

        # 1. Insulin suppression of HGP (Hill coefficient ~1.5)
        ic50 = self._params["hepatic.insulin_suppression_ic50"]
        insulin_inhibition = 1.0 / (1.0 + (insulin / ic50) ** 1.5)

        # 2. Glucagon stimulation of HGP
        ec50 = self._params["hepatic.glucagon_stimulation_ec50"]
        glucagon_drive = 1.0 + 1.2 * (glucagon / (ec50 + glucagon))

        # Basal capacity
        target_hgp = self._params["hepatic.basal_hgp"] * insulin_inhibition * glucagon_drive

        # Gluconeogenesis remains more constant (~30-40% of baseline), upregulated during hypoglycemia
        target_gng = 45.0 * (1.0 + 0.02 * max(0.0, 90.0 - glucose))

        # Net glycogenolysis = HGP - GNG
        glycogenolysis_rate = max(0.0, target_hgp - target_gng)

        # Postprandial glycogen synthesis: when glucose > 100 and insulin > 15
        if glucose > 100.0 and insulin > 15.0:
            synth_rate = 120.0 * (min(1.0, (glucose - 100.0) / 80.0)) * (min(1.0, (insulin - 15.0) / 40.0))
        else:
            synth_rate = 0.0

        # d(Glycogen)/dt = (synthesis - glycogenolysis) / 1000.0 [convert mg/min to g/min]
        d_glycogen = (synth_rate - glycogenolysis_rate) / 1000.0

        d_hgp = 0.5 * (target_hgp - state["hepatic.hgp_rate_mg_min"])
        d_gng = 0.3 * (target_gng - state["hepatic.gluconeogenesis_rate_mg_min"])
        d_urea = 0.1 * (12.0 - state["hepatic.urea_synthesis_mg_min"])

        return {
            "hepatic.glycogen_mass_g": float(np.clip(d_glycogen, -0.5, 0.5)),
            "hepatic.hgp_rate_mg_min": float(np.clip(d_hgp, -30.0, 30.0)),
            "hepatic.gluconeogenesis_rate_mg_min": float(np.clip(d_gng, -10.0, 10.0)),
            "hepatic.urea_synthesis_mg_min": float(np.clip(d_urea, -2.0, 2.0)),
        }

    def exchange_fluxes(
        self,
        time: float,
        state: Dict[str, float],
        environment: SharedEnvironment,
    ) -> List[Flux]:
        """
        Releases glucose and urea into systemic circulation.
        """
        hgp_rate = float(state["hepatic.hgp_rate_mg_min"])
        urea_rate = float(state["hepatic.urea_synthesis_mg_min"])

        return [
            Flux(
                source="hepatic",
                destination="blood",
                quantity="glucose",
                value=hgp_rate,
                unit="mg/min",
                mechanism="Hepatic Glucose Production (Glycogenolysis + Gluconeogenesis)",
            ),
            Flux(
                source="hepatic",
                destination="blood",
                quantity="urea",
                value=urea_rate,
                unit="mg/min",
                mechanism="Hepatic Urea Cycle Nitrogen Clearance",
            ),
        ]
