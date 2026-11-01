"""
backend/app/molecular/binding/target_engagement.py
Unified Target Engagement Evaluator across Receptors, Enzymes, Transporters, and Ion Channels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
from app.molecular.targets.target_registry import MolecularTarget, TargetClass, ReceptorFamily
from app.molecular.binding.occupancy import compute_equilibrium_occupancy
from app.molecular.binding.binding_kinetics import DynamicBindingEngine


@dataclass
class TargetEngagementResult:
    target_id: str
    target_class: TargetClass
    local_free_conc_nM: float
    fractional_occupancy: float
    modulation_type: str              # "activation", "inhibition", "pore_blockade"
    functional_modulation_pct: float  # 0% (no effect) to 100% (full inhibition/activation)
    effective_Kd_nM: float
    bound_target_density_pmol_g: float
    free_target_density_pmol_g: float
    residence_time_min: float


class TargetEngagementEvaluator:
    """
    Translates local free drug exposure and target characteristics into quantifiable target engagement.
    """
    @staticmethod
    def evaluate(
        target: MolecularTarget,
        cell_type_id: str,
        local_free_conc_nM: float,
        is_agonist: bool = True,
        competing_endogenous_conc_nM: float = 0.0,
    ) -> TargetEngagementResult:
        total_density = target.get_effective_density_pmol_g(cell_type_id)
        if target.target_class in [TargetClass.ENZYME, TargetClass.TRANSPORTER, TargetClass.ION_CHANNEL] and not is_agonist:
            kd = target.functional_params.IC50_nM or target.binding_params.Kd_nM
        else:
            kd = target.binding_params.Kd_nM
        hill = target.binding_params.hill_coefficient
        residence = target.binding_params.residence_time_min

        # Compute equilibrium occupancy with competitive endogenous ligand consideration
        occupancy = compute_equilibrium_occupancy(
            free_conc_nM=local_free_conc_nM,
            Kd_nM=kd,
            hill_coefficient=hill,
            competing_inhibitor_conc_nM=competing_endogenous_conc_nM,
            competing_inhibitor_Ki_nM=kd,
        )

        bound_density = total_density * occupancy
        free_density = max(0.0, total_density - bound_density)

        # Distinguish functional modulation based on target class and mechanism
        if target.target_class == TargetClass.RECEPTOR:
            # Black-Leff operational model of agonism vs competitive blockade
            if is_agonist:
                tau = target.functional_params.tau
                n = hill
                c = local_free_conc_nM
                if c > 1e-12:
                    num = target.functional_params.E_max * (tau ** n) * (c ** n)
                    den = ((c + kd) ** n) + ((tau ** n) * (c ** n))
                    activity_pct = float(num / max(den, 1e-12)) * 100.0
                else:
                    activity_pct = 0.0
                modulation = "activation"
            else:
                # Antagonist produces inhibition proportional to occupancy
                activity_pct = occupancy * 100.0
                modulation = "inhibition"

        elif target.target_class == TargetClass.ENZYME:
            # Enzyme competitive inhibition: v / v0 = 1 / (1 + [I] / IC50)
            ic50 = target.functional_params.IC50_nM or kd
            inhibition_pct = (local_free_conc_nM / (local_free_conc_nM + ic50)) * 100.0 if local_free_conc_nM > 0 else 0.0
            activity_pct = inhibition_pct
            modulation = "inhibition"

        elif target.target_class == TargetClass.TRANSPORTER:
            # Transporter inhibition (e.g. SGLT2)
            ic50 = target.functional_params.IC50_nM or kd
            inhibition_pct = (local_free_conc_nM / (local_free_conc_nM + ic50)) * 100.0 if local_free_conc_nM > 0 else 0.0
            activity_pct = inhibition_pct
            modulation = "inhibition"

        elif target.target_class == TargetClass.ION_CHANNEL:
            # Pore blockade
            ic50 = target.functional_params.IC50_nM or kd
            blockade_pct = (local_free_conc_nM / (local_free_conc_nM + ic50)) * 100.0 if local_free_conc_nM > 0 else 0.0
            activity_pct = blockade_pct
            modulation = "pore_blockade"

        else:
            activity_pct = occupancy * 100.0
            modulation = "modulation"

        return TargetEngagementResult(
            target_id=target.target_id,
            target_class=target.target_class,
            local_free_conc_nM=local_free_conc_nM,
            fractional_occupancy=occupancy,
            modulation_type=modulation,
            functional_modulation_pct=activity_pct,
            effective_Kd_nM=kd,
            bound_target_density_pmol_g=bound_density,
            free_target_density_pmol_g=free_density,
            residence_time_min=residence,
        )
