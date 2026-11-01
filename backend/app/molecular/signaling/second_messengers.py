"""
backend/app/molecular/signaling/second_messengers.py
Dynamic Second Messenger Cascades:
- cAMP (Gs / Gi regulation and PDE breakdown)
- Cytosolic Free Calcium [Ca2+] and IP3 (Gq regulation)
- Phosphorylation Cascades: Akt-P, PKA activity, MLCK activation
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class SecondMessengerState:
    cAMP_uM: float                     # Baseline resting ~0.5 - 1.0 uM
    cytosolic_Ca2_uM: float             # Baseline resting ~0.10 uM (100 nM)
    IP3_uM: float                       # Baseline ~0.15 uM
    PKA_activity_fraction: float        # 0.0 to 1.0
    Akt_phosphorylation_fraction: float # 0.0 to 1.0
    MLCK_activity_fraction: float       # 0.0 to 1.0


class SecondMessengerTransducer:
    """
    Simulates second messenger kinetics in response to receptor and target modulation.
    """
    @staticmethod
    def compute_steady_state(
        receptor_family: Optional[str],
        functional_modulation_pct: float,
        is_agonist: bool = True,
        is_blocker: bool = False,
    ) -> SecondMessengerState:
        mod = max(0.0, min(100.0, float(functional_modulation_pct))) / 100.0

        # Baseline resting conditions
        cAMP = 0.50             # uM
        ca2 = 0.10              # uM (100 nM)
        ip3 = 0.15              # uM
        pka = 0.10
        akt = 0.15
        mlck = 0.20

        if receptor_family == "gpcr_gs":
            if is_agonist:
                # Gs activation stimulates Adenylyl Cyclase (up to 8-fold cAMP elevation)
                cAMP = 0.50 + 4.0 * mod
                pka = min(1.0, 0.10 + 0.85 * (cAMP / (cAMP + 1.2)))
            else:
                # Antagonist blocks basal/agonist tone
                cAMP = max(0.15, 0.50 * (1.0 - 0.70 * mod))
                pka = max(0.02, 0.10 * (1.0 - 0.70 * mod))

        elif receptor_family == "gpcr_gi":
            if is_agonist:
                # Gi inhibits AC
                cAMP = max(0.10, 0.50 * (1.0 - 0.75 * mod))
                pka = max(0.02, 0.10 * (1.0 - 0.75 * mod))

        elif receptor_family == "gpcr_gq":
            if is_agonist:
                # Gq stimulates PLC-beta -> IP3 elevation and Ca2+ release
                ip3 = 0.15 + 1.85 * mod
                ca2 = 0.10 + 0.90 * mod  # Peak cytosolic Ca2+ reaches ~1.0 uM
                mlck = min(1.0, 0.20 + 0.75 * mod)
            else:
                # Gq blockade (e.g. ARB Losartan)
                ip3 = max(0.05, 0.15 * (1.0 - 0.65 * mod))
                ca2 = max(0.06, 0.10 * (1.0 - 0.40 * mod))
                mlck = max(0.05, 0.20 * (1.0 - 0.70 * mod))

        elif receptor_family == "receptor_tyrosine_kinase":
            if is_agonist:
                # RTK autophosphorylation activates PI3K -> Akt phosphorylation
                akt = min(1.0, 0.15 + 0.80 * mod)

        elif receptor_family == "ligand_gated_ion_channel":
            if is_agonist:
                ca2 = 0.10 + 1.5 * mod  # Massive local influx
            else:
                ca2 = max(0.05, 0.10 * (1.0 - 0.80 * mod))

        # Check for direct ion channel blockade (e.g. CACNA1C L-type Ca2+ channel)
        if is_blocker and receptor_family == "ion_channel":
            ca2 = max(0.04, 0.10 * (1.0 - 0.60 * mod))
            mlck = max(0.05, 0.20 * (1.0 - 0.65 * mod))

        return SecondMessengerState(
            cAMP_uM=round(cAMP, 4),
            cytosolic_Ca2_uM=round(ca2, 4),
            IP3_uM=round(ip3, 4),
            PKA_activity_fraction=round(pka, 4),
            Akt_phosphorylation_fraction=round(akt, 4),
            MLCK_activity_fraction=round(mlck, 4),
        )
