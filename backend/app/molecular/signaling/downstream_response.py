"""
backend/app/molecular/signaling/downstream_response.py
Cellular Functional Response Evaluator.

Maps second messenger states and target engagement to concrete cellular phenotypes:
- Beta cell insulin secretion multiplier
- Cardiomyocyte contractile force / stroke work multiplier
- Vascular smooth muscle tone and diameter change
- Hepatocyte gluconeogenesis suppression and cholesterol synthesis rate
- Proximal tubule glucose reabsorption capacity
- Skeletal muscle GLUT4 surface translocation
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
from app.molecular.tissue.cell_types import OrganType
from app.molecular.targets.target_registry import TargetClass
from app.molecular.binding.target_engagement import TargetEngagementResult
from app.molecular.signaling.second_messengers import SecondMessengerState


@dataclass
class CellularResponse:
    cell_type_id: str
    primary_cellular_effect: str
    quantitative_change_fold: float     # e.g. 2.2x increase or 0.15x decrease (relative to 1.0 baseline)
    percent_modulation: float
    description: str
    affected_cellular_pathways: list[str]


class CellularResponseEvaluator:
    """
    Evaluates cell-specific biochemical and biophysical consequences of target engagement.
    """
    @staticmethod
    def evaluate(
        cell_type_id: str,
        target_id: str,
        target_class: TargetClass,
        engagement: TargetEngagementResult,
        second_messengers: SecondMessengerState,
    ) -> CellularResponse:
        tid = target_id.upper()
        mod_pct = engagement.functional_modulation_pct
        mod_frac = mod_pct / 100.0

        if tid == "GLP1R" and "beta_cell" in cell_type_id:
            # GLP-1 stimulates insulin exocytosis via cAMP/Epac2/PKA
            fold = 1.0 + 1.8 * (second_messengers.cAMP_uM / (second_messengers.cAMP_uM + 2.0))
            return CellularResponse(
                cell_type_id=cell_type_id,
                primary_cellular_effect="potentiation_of_glucose_stimulated_insulin_secretion",
                quantitative_change_fold=round(fold, 3),
                percent_modulation=mod_pct,
                description=f"cAMP elevated to {second_messengers.cAMP_uM:.2f} uM, potentiating beta-cell insulin granule exocytosis by {fold:.2f}x.",
                affected_cellular_pathways=["cAMP/PKA pathway", "Epac2-mediated vesicle priming", "K-ATP closure modulation"],
            )

        elif tid == "ADRB1" and "cardiomyocyte" in cell_type_id:
            if engagement.modulation_type == "activation":
                fold = 1.0 + 0.85 * second_messengers.PKA_activity_fraction
                desc = f"PKA phosphorylation of L-type Ca2+ channels (Cav1.2) and phospholamban increases contractility by {fold:.2f}x."
            else:
                fold = max(0.40, 1.0 - 0.50 * mod_frac)
                desc = f"ADRB1 beta-blockade attenuates adrenergic tone; inotropy reduced to {fold:.2f}x of baseline."
            return CellularResponse(
                cell_type_id=cell_type_id,
                primary_cellular_effect="myocardial_contractility_modulation",
                quantitative_change_fold=round(fold, 3),
                percent_modulation=mod_pct,
                description=desc,
                affected_cellular_pathways=["Gs-PKA phosphorylation", "Phospholamban (PLN) disinhibition of SERCA2a", "RyR2 calcium release"],
            )

        elif tid in ["AGTR1", "CACNA1C"] and "vascular_smooth_muscle" in cell_type_id:
            if tid == "AGTR1" and engagement.modulation_type == "activation":
                fold = 1.0 + 1.5 * mod_frac
                desc = f"Ang II AT1R activation elevates IP3/Ca2+, increasing smooth muscle tone by {fold:.2f}x."
            else:
                # ARB or Calcium Channel Blocker (Amlodipine) causes relaxation
                fold = max(0.35, 1.0 - 0.65 * mod_frac)
                desc = f"Target blockade reduces intracellular free Ca2+ and MLCK activity; vascular tone drops to {fold:.2f}x (arteriolar vasodilation)."
            return CellularResponse(
                cell_type_id=cell_type_id,
                primary_cellular_effect="vascular_tone_and_contractility",
                quantitative_change_fold=round(fold, 3),
                percent_modulation=mod_pct,
                description=desc,
                affected_cellular_pathways=["PLC-IP3-Ca2+ cascade", "MLCK myosin cross-bridge phosphorylation", "Arteriolar caliber regulation"],
            )

        elif tid == "INSR":
            # Insulin signaling drives Akt phosphorylation -> GLUT4 translocation & HGP suppression
            fold = 1.0 + 3.0 * second_messengers.Akt_phosphorylation_fraction
            desc = f"Insulin receptor autophosphorylation elevates p-Akt ({second_messengers.Akt_phosphorylation_fraction:.2f}), driving GLUT4 sarcolemmal translocation and glycogen synthesis."
            return CellularResponse(
                cell_type_id=cell_type_id,
                primary_cellular_effect="anabolic_glucose_transport_and_glycogenesis",
                quantitative_change_fold=round(fold, 3),
                percent_modulation=mod_pct,
                description=desc,
                affected_cellular_pathways=["IRS-1/PI3K/Akt pathway", "AS160 phosphorylation", "GSK3beta inhibition"],
            )

        elif tid == "SGLT2" and "proximal_tubule" in cell_type_id:
            # SGLT2 transporter inhibition
            remaining_capacity = max(0.01, 1.0 - mod_frac)
            return CellularResponse(
                cell_type_id=cell_type_id,
                primary_cellular_effect="tubular_glucose_and_sodium_reabsorption_inhibition",
                quantitative_change_fold=round(remaining_capacity, 3),
                percent_modulation=mod_pct,
                description=f"SGLT2 cotransporter capacity inhibited by {mod_pct:.1f}%; residual reabsorption is {remaining_capacity * 100:.1f}%.",
                affected_cellular_pathways=["Apical Na+/glucose secondary active transport", "Tubuloglomerular feedback modulation"],
            )

        elif tid == "HMGCR" and "hepatocyte" in cell_type_id:
            remaining_synth = max(0.05, 1.0 - mod_frac)
            return CellularResponse(
                cell_type_id=cell_type_id,
                primary_cellular_effect="mevalonate_synthesis_inhibition",
                quantitative_change_fold=round(remaining_synth, 3),
                percent_modulation=mod_pct,
                description=f"HMG-CoA reductase competitively inhibited by {mod_pct:.1f}%; triggers SREBP-2 activation and LDLR upregulation.",
                affected_cellular_pathways=["Mevalonate pathway", "SREBP-2 cleavage", "Hepatic LDL receptor (LDLR) recycling"],
            )

        # Generic default
        fold = 1.0 + (mod_frac if engagement.modulation_type == "activation" else -mod_frac * 0.5)
        return CellularResponse(
            cell_type_id=cell_type_id,
            primary_cellular_effect="target_mediated_cellular_modulation",
            quantitative_change_fold=round(fold, 3),
            percent_modulation=mod_pct,
            description=f"Molecular engagement at {target_id} induces {engagement.modulation_type} ({mod_pct:.1f}%).",
            affected_cellular_pathways=["Target-coupled downstream effector network"],
        )
