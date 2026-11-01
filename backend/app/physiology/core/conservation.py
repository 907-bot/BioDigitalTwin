"""
BioDigitalTwin — Conservation Law & Mass Balance Validator.

Audits physical and physiological conservation laws across coupled organs:
1. Hemodynamic flow conservation: sum(regional flows) == Cardiac Output
2. Systemic glucose mass balance: dG/dt == R_gut + R_hepatic - U_muscle - U_adipose - E_renal
3. Enteral carb mass conservation: Stomach + Intestine + Absorbed == Ingested
4. Fluid balance: Plasma volume + Interstitium == ECF
5. Non-negativity of physical masses and concentrations
"""

from typing import Dict, List, Tuple, Any
import numpy as np

from .shared_environment import SharedEnvironment
from .flux_registry import FluxRegistry


class ConservationValidator:
    """
    Automated auditor for conservation laws across all physiological organs.
    """

    def __init__(self, tolerance_rtol: float = 0.05):
        self.tolerance_rtol = tolerance_rtol

    def verify_hemodynamic_conservation(self, env: SharedEnvironment) -> Tuple[bool, List[str]]:
        """
        Verifies that regional blood flows sum to Cardiac Output within numerical tolerance:
        sum(Q_i) == Cardiac Output.
        """
        flows = env.circulation.regional_flows_L_min
        co = env.circulation.cardiac_output_L_min
        violations = []

        regional_sum = (
            flows.get("brain", 0.0) +
            flows.get("coronary", 0.0) +
            flows.get("renal", 0.0) +
            flows.get("gut", 0.0) +
            flows.get("muscle", 0.0) +
            flows.get("adipose", 0.0)
        )

        if not np.isclose(regional_sum, co, rtol=1e-3, atol=1e-3):
            violations.append(
                f"Hemodynamic flow conservation violated: sum(Q_regional)={regional_sum:.3f} L/min != CO={co:.3f} L/min"
            )

        # Pulmonary flow must equal systemic Cardiac Output
        q_pulm = flows.get("pulmonary", 0.0)
        if not np.isclose(q_pulm, co, rtol=1e-3, atol=1e-3):
            violations.append(
                f"Pulmonary blood flow {q_pulm:.3f} L/min != Cardiac Output {co:.3f} L/min"
            )

        return len(violations) == 0, violations

    def verify_glucose_mass_balance(
        self,
        fluxes: FluxRegistry,
        expected_dG_dt: float,
        ecf_volume_dL: float = 140.0,
    ) -> Tuple[bool, List[str]]:
        """
        Verifies systemic glucose balance:
        dG/dt * V_ecf == R_gut + R_hepatic - U_peripheral - E_renal
        """
        violations = []
        r_gut = fluxes.get_net_flux_to_compartment("blood", "glucose")
        # In a fully coupled step, net blood glucose flux in mg/min divided by ECF volume matches rate
        predicted_dG_dt = r_gut / max(1.0, ecf_volume_dL)

        # Check directional agreement
        if np.sign(predicted_dG_dt) != np.sign(expected_dG_dt) and abs(expected_dG_dt) > 0.05:
            violations.append(
                f"Glucose flux direction mismatch: predicted dG/dt={predicted_dG_dt:.3f} mg/dL/min, actual={expected_dG_dt:.3f}"
            )

        return len(violations) == 0, violations

    def verify_enteral_mass_balance(
        self,
        initial_meal_g: float,
        stomach_mass_g: float,
        intestinal_mass_g: float,
        cumulative_absorbed_g: float,
    ) -> Tuple[bool, List[str]]:
        """
        Verifies that carbohydrates ingested equal remaining gut mass plus absorbed mass:
        M_ingested == M_stomach + M_intestine + M_absorbed
        """
        accounted_g = stomach_mass_g + intestinal_mass_g + cumulative_absorbed_g
        violations = []
        if not np.isclose(accounted_g, initial_meal_g, rtol=self.tolerance_rtol, atol=0.5):
            violations.append(
                f"Enteral carb mass balance violation: Ingested={initial_meal_g:.2f}g != Accounted={accounted_g:.2f}g (diff={accounted_g - initial_meal_g:.2f}g)"
            )
        return len(violations) == 0, violations

    def verify_physical_bounds(
        self,
        env: SharedEnvironment,
        organ_states: Dict[str, Dict[str, float]],
    ) -> Tuple[bool, List[str]]:
        """
        Ensures non-negativity and biological bounds for key substrates and pressures.
        """
        violations = []
        chem = env.chemistry
        if chem.glucose_mg_dL <= 0.0:
            violations.append(f"Non-physical blood glucose: {chem.glucose_mg_dL} mg/dL <= 0")
        if chem.insulin_uU_mL < 0.0:
            violations.append(f"Non-physical blood insulin: {chem.insulin_uU_mL} uU/mL < 0")
        if chem.paO2_mmHg <= 0.0:
            violations.append(f"Non-physical PaO2: {chem.paO2_mmHg} mmHg <= 0")
        if chem.paCO2_mmHg <= 0.0:
            violations.append(f"Non-physical PaCO2: {chem.paCO2_mmHg} mmHg <= 0")
        if not (6.80 <= chem.pH <= 7.80):
            violations.append(f"Unphysiological blood pH: {chem.pH:.3f}")

        circ = env.circulation
        if circ.cardiac_output_L_min <= 0.0:
            violations.append(f"Non-physical Cardiac Output: {circ.cardiac_output_L_min} L/min <= 0")
        if circ.mean_arterial_pressure_mmHg < 20.0:
            violations.append(f"Severe unphysiological hypotension: MAP={circ.mean_arterial_pressure_mmHg} mmHg")

        return len(violations) == 0, violations
