"""
backend/app/pkpd/receptors.py
Mechanistic Receptor Pharmacology & Carrier-Mediated Transporter Engine.

Implements:
1. Dynamic Law of Mass Action for ligand-receptor binding with surface trafficking,
   internalization, recycling, and mass-balance conservation.
2. Full Black-Leff Operational Model of Agonism:
   E([L]) = (E_max * tau^n * [L]^n) / (([L] + K_A)^n + tau^n * [L]^n)
3. Gaddum-Schild Operational Antagonism for competitive inhibitors:
   K_A,app = K_A * (1 + [B] / K_B)
4. Carrier-Mediated Transporters (SGLT2 / SLC5A2) with Michaelis-Menten
   saturation kinetics and stoichiometric Na+:glucose cotransport (NOT GPCR).
5. Literature-derived physiological parameter baselines and tissue-specific definitions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.integrate import solve_ivp


class ReceptorFamily(str, Enum):
    GPCR_GS = "gpcr_gs"          # Adenylyl cyclase stimulatory (cAMP / PKA)
    GPCR_GI = "gpcr_gi"          # Adenylyl cyclase inhibitory (cAMP decrease)
    GPCR_GQ = "gpcr_gq"          # Phospholipase C-beta (IP3 / DAG / Ca2+)
    RTK = "receptor_tyrosine_kinase"  # e.g., Insulin Receptor, IGF1R
    NUCLEAR = "nuclear_receptor" # e.g., Glucocorticoid Receptor, PPAR-gamma


@dataclass
class ReceptorParameters:
    """
    Biophysical parameters for a specific receptor type in target tissue.
    All parameters are literature-sourced with explicit SI / pharmacological units.
    """
    symbol: str
    name: str
    family: ReceptorFamily
    target_tissue: str
    R_total_pmol_g: float        # Total receptor pool density at baseline (pmol/g tissue)
    k_on: float                  # Association rate constant [1 / (nM * min)]
    k_off: float                 # Dissociation rate constant [1 / min]
    K_A: float                   # Agonist equilibrium dissociation constant [nM]
    tau: float                   # Black-Leff transducer ratio (R_tot / K_E) [dimensionless]
    n: float = 1.0               # Operational transducer Hill slope [dimensionless]
    E_max: float = 1.0           # Maximal achievable tissue system response [0-1 or custom]
    # Trafficking rate constants
    k_internalize: float = 0.02  # Agonist-induced endocytosis rate [1 / min]
    k_recycle: float = 0.005     # Receptor vesicle recycling to membrane [1 / min]
    k_synth: float = 0.05        # Basal de novo translation/insertion rate [pmol/(g*min)]
    k_deg: float = 0.001         # Basal membrane receptor degradation rate [1 / min]
    citation: str = ""

    @property
    def Kd_nM(self) -> float:
        """Thermodynamic equilibrium dissociation constant (k_off / k_on)."""
        return self.k_off / max(self.k_on, 1e-12)


# =============================================================================
# Curated Literature-Validated Human Receptor Library
# =============================================================================
RECEPTOR_LIBRARY: Dict[str, ReceptorParameters] = {
    # 1. Insulin Receptor (INSR) in Hepatic & Skeletal Muscle Tissue
    "INSR": ReceptorParameters(
        symbol="INSR",
        name="Insulin Receptor",
        family=ReceptorFamily.RTK,
        target_tissue="liver_skeletal_muscle",
        R_total_pmol_g=35.0,     # ~20,000 - 50,000 receptors/cell in human hepatocytes
        k_on=0.040,              # 1/(nM*min)
        k_off=0.028,             # 1/min -> Kd = 0.70 nM (~100 uU/mL)
        K_A=0.70,                # nM
        tau=8.0,                 # High transducer efficiency (significant receptor reserve)
        n=1.0,
        E_max=1.0,
        k_internalize=0.035,     # Receptor-mediated endocytosis of insulin
        k_recycle=0.010,
        k_synth=0.14,
        k_deg=0.004,
        citation="Kahn CR et al. (1981) J Clin Invest; Bevan P. (2001) J Cell Sci 114:1429",
    ),
    # 2. Beta-1 Adrenergic Receptor (ADRB1) in Myocardium
    "ADRB1": ReceptorParameters(
        symbol="ADRB1",
        name="Beta-1 Adrenergic Receptor",
        family=ReceptorFamily.GPCR_GS,
        target_tissue="heart_myocardium",
        R_total_pmol_g=12.0,     # ~40-60 fmol/mg protein in ventricular tissue
        k_on=0.015,
        k_off=0.075,             # Kd = 5.0 nM
        K_A=5.0,                 # Epinephrine / Norepinephrine affinity
        tau=15.0,                # Strong spare receptor reserve for inotropy (Kenakin 2014)
        n=1.0,
        E_max=1.0,
        k_internalize=0.020,
        k_recycle=0.005,
        k_synth=0.06,
        k_deg=0.005,
        citation="Brodde OE. (1991) Pharmacol Rev 43:203; Kenakin T. (2014)",
    ),
    # 3. Angiotensin II Type 1 Receptor (AGTR1) in Vascular Smooth Muscle & Adrenal
    "AGTR1": ReceptorParameters(
        symbol="AGTR1",
        name="Angiotensin II Type 1 Receptor",
        family=ReceptorFamily.GPCR_GQ,
        target_tissue="vascular_smooth_muscle",
        R_total_pmol_g=18.0,
        k_on=0.025,
        k_off=0.050,             # Kd = 2.0 nM
        K_A=2.0,                 # Ang II affinity
        tau=6.0,
        n=1.1,
        E_max=1.0,
        k_internalize=0.030,
        k_recycle=0.008,
        k_synth=0.09,
        k_deg=0.005,
        citation="Timmermans PB et al. (1993) Pharmacol Rev 45:205",
    ),
    # 4. GLP-1 Receptor (GLP1R) in Pancreatic Beta-Cells
    "GLP1R": ReceptorParameters(
        symbol="GLP1R",
        name="Glucagon-Like Peptide 1 Receptor",
        family=ReceptorFamily.GPCR_GS,
        target_tissue="pancreatic_islets",
        R_total_pmol_g=6.0,
        k_on=0.050,
        k_off=0.020,             # Kd = 0.40 nM
        K_A=0.40,
        tau=10.0,
        n=1.0,
        E_max=1.0,
        k_internalize=0.045,     # Rapid internalization upon agonist binding
        k_recycle=0.004,
        k_synth=0.03,
        k_deg=0.005,
        citation="Drucker DJ. (2018) Cell Metab 27:740",
    ),
}


# =============================================================================
# Dynamic Receptor Solver with Mass Balance
# =============================================================================
class ReceptorKineticsEngine:
    """
    Integrates continuous receptor binding, endocytosis, and recovery over time,
    and calculates operational tissue response via the complete Black-Leff model.
    """
    def __init__(self, params: ReceptorParameters):
        self.params = params
        # State variables: [R_surface, LR_complex, R_internalized] (pmol/g)
        self.R_surface: float = params.R_total_pmol_g
        self.LR_complex: float = 0.0
        self.R_internalized: float = 0.0

    @property
    def total_receptors(self) -> float:
        """Total receptor conservation check."""
        return self.R_surface + self.LR_complex + self.R_internalized

    def step(
        self,
        agonist_conc_nM: float,
        dt_min: float = 1.0,
        antagonist_conc_nM: float = 0.0,
        antagonist_Kb_nM: float = 1.0,
    ) -> Dict[str, float]:
        """
        Integrates receptor trafficking over time step dt_min.
        Calculates occupancy and operational tissue response.
        """
        L = max(0.0, float(agonist_conc_nM))
        B = max(0.0, float(antagonist_conc_nM))

        # Effective on-rate with competitive antagonist occupancy competition
        # Under rapid competitive equilibrium: apparent Kd shifts by (1 + [B]/K_B)
        antagonist_shift = 1.0 + (B / max(1e-12, antagonist_Kb_nM)) if B > 0 else 1.0
        effective_k_on = self.params.k_on / antagonist_shift

        def derivatives(t, y):
            R_s, LR, R_int = y
            # 1. Mass action binding and unbinding
            v_bind = effective_k_on * L * max(0.0, R_s)
            v_unbind = self.params.k_off * max(0.0, LR)

            # 2. Trafficking and internalization with balanced baseline turnover
            v_int = self.params.k_internalize * max(0.0, LR)
            v_rec = self.params.k_recycle * max(0.0, R_int)
            v_deg_surf = self.params.k_deg * max(0.0, R_s)
            v_deg_int = self.params.k_deg * max(0.0, R_int)
            # Synthesis matches resting degradation so total pool is in homeostasis
            v_syn = self.params.k_deg * self.params.R_total_pmol_g

            dR_s = -v_bind + v_unbind + v_rec + v_syn - v_deg_surf
            dLR = v_bind - v_unbind - v_int
            dR_int = v_int - v_rec - v_deg_int
            return [dR_s, dLR, dR_int]

        y0 = [self.R_surface, self.LR_complex, self.R_internalized]
        sol = solve_ivp(derivatives, [0.0, dt_min], y0, method="RK45")

        self.R_surface = float(max(0.0, sol.y[0, -1]))
        self.LR_complex = float(max(0.0, sol.y[1, -1]))
        self.R_internalized = float(max(0.0, sol.y[2, -1]))

        # --- Operational Model of Agonism (Black & Leff 1983) ---
        # E([L]) = (E_max * tau^n * [L]^n) / (([L] + K_A,app)^n + tau^n * [L]^n)
        K_A_app = self.params.K_A * antagonist_shift
        tau_dynamic = self.params.tau * (self.R_surface / max(1e-6, self.params.R_total_pmol_g))
        n = self.params.n

        if L > 1e-12:
            num = self.params.E_max * (tau_dynamic ** n) * (L ** n)
            den = ((L + K_A_app) ** n) + ((tau_dynamic ** n) * (L ** n))
            tissue_response = float(num / max(den, 1e-12))
        else:
            tissue_response = 0.0

        fractional_occupancy = float(self.LR_complex / max(1e-6, self.R_surface + self.LR_complex))

        return {
            "receptor": self.params.symbol,
            "R_surface_pmol_g": self.R_surface,
            "LR_complex_pmol_g": self.LR_complex,
            "R_internalized_pmol_g": self.R_internalized,
            "R_total_pmol_g": self.total_receptors,
            "fractional_occupancy": fractional_occupancy,
            "tissue_response": tissue_response,
            "antagonist_shift_factor": antagonist_shift,
        }

    def reset(self) -> None:
        """Reset receptor state to baseline resting equilibrium."""
        self.R_surface = self.params.R_total_pmol_g
        self.LR_complex = 0.0
        self.R_internalized = 0.0


# =============================================================================
# Carrier-Mediated Transporter: Renal SGLT2 (SLC5A2)
# =============================================================================
@dataclass
class SGLT2TransporterParameters:
    """
    Renal tubular SGLT2 parameters with physiological transport maximums.
    """
    TmG_mg_min: float = 375.0      # Normal maximal tubular glucose transport capacity (mg/min)
    Km_mg_dL: float = 108.0        # Glucose affinity constant (~6.0 mM)
    Ki_empagliflozin_nM: float = 1.3  # Empagliflozin Ki (Grempler et al. 2012)
    Ki_dapagliflozin_nM: float = 1.2  # Dapagliflozin Ki
    sglt1_contribution_pct: float = 10.0  # Residual SGLT1 reabsorption capacity (~10%)


class SGLT2TransporterEngine:
    """
    Models carrier-mediated sodium-glucose cotransport in the renal proximal tubule.
    Demonstrates transport saturation, competitive pharmacological inhibition,
    glucosuria induction, and stoichiometric natriuresis.
    """
    def __init__(self, params: Optional[SGLT2TransporterParameters] = None):
        self.params = params or SGLT2TransporterParameters()

    def compute_renal_flux(
        self,
        plasma_glucose_mg_dL: float,
        gfr_mL_min: float = 100.0,
        inhibitor_conc_nM: float = 0.0,
        inhibitor_type: str = "empagliflozin",
    ) -> Dict[str, float]:
        """
        Calculates glucose filtration, reabsorption, and excretion rates.
        Units:
          plasma_glucose: mg/dL
          gfr: mL/min
          filtered_load = (plasma_glucose * gfr) / 100 [mg/min]
        """
        G = max(0.0, float(plasma_glucose_mg_dL))
        GFR = max(5.0, float(gfr_mL_min))
        I = max(0.0, float(inhibitor_conc_nM))

        # Select Ki based on drug
        Ki = self.params.Ki_empagliflozin_nM if inhibitor_type.lower() == "empagliflozin" else self.params.Ki_dapagliflozin_nM

        # Filtered load (mg/min): (mg/dL) * (dL / 100 mL) * (mL / min)
        filtered_glucose_mg_min = (G * GFR) / 100.0

        # Michaelis-Menten Carrier Reabsorption with Competitive Inhibition
        # J_SGLT2 = (TmG_SGLT2 * G) / (Km * (1 + I / Ki) + G)
        tm_sglt2 = self.params.TmG_mg_min * 0.90  # 90% via SGLT2
        km_apparent = self.params.Km_mg_dL * (1.0 + (I / Ki))
        sglt2_reabsorption = (tm_sglt2 * G) / (km_apparent + G)

        # SGLT1 handles remaining ~10% (not inhibited by selective SGLT2i)
        tm_sglt1 = self.params.TmG_mg_min * 0.10
        sglt1_reabsorption = (tm_sglt1 * G) / (self.params.Km_mg_dL + G)

        total_reabsorbed_mg_min = min(filtered_glucose_mg_min, sglt2_reabsorption + sglt1_reabsorption)
        urinary_excretion_mg_min = max(0.0, filtered_glucose_mg_min - total_reabsorbed_mg_min)

        # Physiological Renal Threshold (plasma glucose at which excretion begins)
        # Accounts for nephron splay ~0.75-0.80 (DeFronzo et al. 2012)
        effective_capacity_mg_min = (tm_sglt2 * (self.params.Km_mg_dL / km_apparent) + tm_sglt1) * 0.75
        renal_threshold_mg_dL = (effective_capacity_mg_min / GFR) * 100.0

        # Stoichiometric Natriuresis: 1 Na+ cotransported with 1 glucose in SGLT2
        # Mol wt of Glucose = 180.16 g/mol; 1 mg = 1 / 180.16 mmol
        # Glucose reabsorbed (mmol/min) = reabsorbed_mg_min / 180.16
        # Na+ reabsorbed with glucose = same mmol/min (mEq/min)
        glucose_reabsorbed_mmol_min = total_reabsorbed_mg_min / 180.16
        na_cotransport_mEq_min = glucose_reabsorbed_mmol_min
        
        # When SGLT2 is inhibited, unabsorbed Na+ is lost in urine (osmotic natriuresis)
        basal_sglt2_reabsorb = (tm_sglt2 * G) / (self.params.Km_mg_dL + G)
        na_excretion_increase_mEq_min = max(0.0, (basal_sglt2_reabsorb - sglt2_reabsorption) / 180.16)

        return {
            "plasma_glucose_mg_dL": G,
            "gfr_mL_min": GFR,
            "filtered_glucose_mg_min": float(filtered_glucose_mg_min),
            "reabsorbed_glucose_mg_min": float(total_reabsorbed_mg_min),
            "urinary_glucose_mg_min": float(urinary_excretion_mg_min),
            "effective_renal_threshold_mg_dL": float(renal_threshold_mg_dL),
            "sglt2_inhibition_pct": float(1.0 - (sglt2_reabsorption / max(1e-6, basal_sglt2_reabsorb))) * 100.0,
            "na_cotransport_mEq_min": float(na_cotransport_mEq_min),
            "induced_natriuresis_mEq_min": float(na_excretion_increase_mEq_min),
        }
