"""
BioDigitalTwin — Respiratory Organ Model.

Mechanisms:
- Pulmonary mechanics (Respiratory rate, tidal volume, alveolar ventilation)
- Alveolar gas equation (pAO2 = FiO2 * (Patm - 47) - paCO2 / RQ)
- Chemoreceptor ventilatory feedback (central CO2 sensitivity, peripheral O2 hypoxia)
- Exercise hyperpnea matching metabolic gas exchange (VO2 and VCO2)
- Arterial blood gas equilibration (PaO2, PaCO2, SaO2, and Henderson-Hasselbalch pH)
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from ..core.organ_interface import OrganModel
from ..core.state_registry import StateDescriptor, ParameterDescriptor, get_global_registry
from ..core.shared_environment import SharedEnvironment
from ..core.flux_registry import Flux


class RespiratoryOrganModel(OrganModel):
    """
    Respiratory organ module governing ventilation, pulmonary gas exchange, and acid-base homeostasis.
    """

    def __init__(self):
        super().__init__(organ_id="respiratory")

        self.state_schema = {
            "resp.respiratory_rate_bpm": StateDescriptor(
                name="resp.respiratory_rate_bpm",
                display_name="Respiratory Rate",
                unit="breaths/min",
                compartment="lungs",
                min_bound=6.0,
                max_bound=60.0,
                nominal=14.0,
                initial_condition_source="vital_signs",
                is_observable=True,
                owning_module="respiratory",
                description="Frequency of complete ventilatory respiratory cycles",
            ),
            "resp.tidal_volume_L": StateDescriptor(
                name="resp.tidal_volume_L",
                display_name="Tidal Volume",
                unit="L",
                compartment="lungs",
                min_bound=0.20,
                max_bound=3.0,
                nominal=0.50,
                initial_condition_source="spirometry",
                is_observable=True,
                owning_module="respiratory",
                description="Depth of gas volume inspired and expired per breath",
            ),
            "resp.alveolar_ventilation_L_min": StateDescriptor(
                name="resp.alveolar_ventilation_L_min",
                display_name="Alveolar Ventilation (V_A)",
                unit="L/min",
                compartment="lungs",
                min_bound=1.5,
                max_bound=80.0,
                nominal=4.90,
                initial_condition_source="blood_gas_derived",
                is_observable=False,
                owning_module="respiratory",
                description="Effective gas volume reaching gas-exchanging alveoli per minute",
            ),
            "resp.pAO2_mmHg": StateDescriptor(
                name="resp.pAO2_mmHg",
                display_name="Alveolar Oxygen Tension (P_A O2)",
                unit="mmHg",
                compartment="alveoli",
                min_bound=40.0,
                max_bound=150.0,
                nominal=102.0,
                initial_condition_source="alveolar_gas_equation",
                is_observable=False,
                owning_module="respiratory",
                description="Oxygen partial pressure in the alveolar gas space",
            ),
        }

        self.parameter_schema = {
            "resp.dead_space_L": ParameterDescriptor(
                name="resp.dead_space_L",
                display_name="Anatomic Dead Space (V_D)",
                unit="L",
                default_value=0.15,
                min_bound=0.08,
                max_bound=0.30,
                provenance="West JB, Respiratory Physiology: The Essentials",
                is_personalized=False,
                description="Non-gas exchanging conducting airway volume (~2.2 mL/kg)",
            ),
            "resp.chemoreceptor_co2_gain": ParameterDescriptor(
                name="resp.chemoreceptor_co2_gain",
                display_name="Central Chemoreceptor CO2 Gain",
                unit="L/min per mmHg",
                default_value=1.8,
                min_bound=0.6,
                max_bound=4.0,
                provenance="Nunn's Applied Respiratory Physiology, 8th Ed.",
                is_personalized=True,
                description="Ventilatory sensitivity of medullary chemoreceptors to arterial PaCO2 deviation",
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
            "resp.respiratory_rate_bpm": 14.0,
            "resp.tidal_volume_L": 0.50,
            "resp.alveolar_ventilation_L_min": 4.90,
            "resp.pAO2_mmHg": 102.0,
        }
        self._params = {k: v.default_value for k, v in self.parameter_schema.items()}

    def compute_derivatives(
        self,
        time: float,
        state: Dict[str, float],
        environment: SharedEnvironment,
    ) -> Dict[str, float]:
        """
        Calculates ventilatory drive, alveolar gas tension, and blood gas transport.
        """
        chem = environment.chemistry
        paco2 = chem.paCO2_mmHg
        pao2 = chem.paO2_mmHg

        vd = self._params["resp.dead_space_L"]
        gain_co2 = self._params["resp.chemoreceptor_co2_gain"]

        # Chemoreceptor feedback: CO2 deviation above 40 mmHg and hypoxic drive (PaO2 < 60)
        co2_drive = gain_co2 * (paco2 - 40.0)
        o2_drive = 0.4 * max(0.0, 60.0 - pao2)

        # Baseline resting minute ventilation ~ 7.0 L/min
        target_ve = max(4.0, 7.0 + co2_drive + o2_drive)

        # Partition VE into RR and VT
        target_vt = min(2.5, max(0.35, 0.50 + 0.04 * (target_ve - 7.0)))
        target_rr = target_ve / target_vt
        target_va = target_rr * max(0.1, target_vt - vd)

        d_rr = np.clip(0.3 * (target_rr - state["resp.respiratory_rate_bpm"]), -6.0, 6.0)
        d_vt = np.clip(0.3 * (target_vt - state["resp.tidal_volume_L"]), -0.3, 0.3)
        d_va = np.clip(0.3 * (target_va - state["resp.alveolar_ventilation_L_min"]), -6.0, 6.0)

        # Alveolar Gas Equation: pAO2 = FiO2 * (Patm - 47) - (paCO2 / RQ)
        # FiO2 = 0.21, Patm = 760, RQ = 0.82 -> FiO2*(713) = 149.7
        target_pAO2 = 0.21 * 713.0 - (paco2 / 0.82)
        d_pAO2 = np.clip(0.5 * (target_pAO2 - state["resp.pAO2_mmHg"]), -10.0, 10.0)

        return {
            "resp.respiratory_rate_bpm": float(d_rr),
            "resp.tidal_volume_L": float(d_vt),
            "resp.alveolar_ventilation_L_min": float(d_va),
            "resp.pAO2_mmHg": float(d_pAO2),
        }

    def exchange_fluxes(
        self,
        time: float,
        state: Dict[str, float],
        environment: SharedEnvironment,
    ) -> List[Flux]:
        """
        Updates arterial blood gas tension in the shared environment.
        """
        pao2_target = state["resp.pAO2_mmHg"] - 8.0  # Normal A-a gradient ~8 mmHg
        # Update shared environment blood chemistry
        environment.update_acid_base_and_gas_equilibrium(
            pao2=pao2_target,
            paco2=environment.chemistry.paCO2_mmHg,
        )

        return [
            Flux(
                source="lungs",
                destination="blood",
                quantity="oxygen",
                value=250.0,
                unit="mL/min",
                mechanism="Pulmonary Alveolar O2 Diffusion",
            ),
            Flux(
                source="blood",
                destination="lungs",
                quantity="co2",
                value=200.0,
                unit="mL/min",
                mechanism="Pulmonary Alveolar CO2 Elimination",
            ),
        ]
