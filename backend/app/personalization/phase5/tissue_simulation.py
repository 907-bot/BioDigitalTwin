"""
Phase 5 — Pillar 7: Cellular & Tissue Simulation.

Multi-scale tissue models bridging molecular events → cellular response
→ tissue function → organ-level physiology. Each tissue is modeled as
a coupled PDE-inspired ODE system representing:
  - Liver tissue: hepatic lobule with zonation, gluconeogenesis, lipogenesis
  - Kidney tissue: nephron with filtration, reabsorption, secretion
  - Cardiac tissue: cardiomyocyte electrophysiology and contraction
  - Adipose tissue: adipocyte hypertrophy, hyperplasia, inflammation

Connects drug effects to cell response to tissue function to organ outcomes.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum


# ── Tissue Types ──────────────────────────────────────────────

TISSUE_TYPES = ["liver", "kidney", "cardiac", "adipose"]
N_TISSUE_TYPES = len(TISSUE_TYPES)


@dataclass
class TissueState:
    """
    Unified tissue state representation.

    Each tissue type has type-specific variables.
    """
    tissue_type: str
    # Core physiology (varies by tissue type)
    cell_viability: float = 1.0       # 0-1, fraction of healthy cells
    extracellular_matrix: float = 1.0  # 0-1, ECM integrity
    oxygen_tension: float = 0.5        # 0-1, tissue oxygenation
    metabolic_activity: float = 0.5    # 0-1, overall metabolic function
    inflammation_level: float = 0.0    # 0-1, local inflammation
    fibrosis_level: float = 0.0        # 0-1, tissue fibrosis

    # Type-specific variables stored as dict
    specific_vars: Dict[str, float] = field(default_factory=dict)

    def to_array(self) -> np.ndarray:
        base = np.array([
            self.cell_viability, self.extracellular_matrix,
            self.oxygen_tension, self.metabolic_activity,
            self.inflammation_level, self.fibrosis_level,
        ])
        specific = np.array(list(self.specific_vars.values()))
        return np.concatenate([base, specific])

    @classmethod
    def healthy(cls, tissue_type: str) -> 'TissueState':
        return cls(tissue_type=tissue_type)

    def dim(self) -> int:
        return 6 + len(self.specific_vars)


# ── Liver Tissue Model ────────────────────────────────────────

class LiverTissue:
    """
    Liver tissue model — hepatic lobule with metabolic zonation.

    Models:
      - Periportal gluconeogenesis
      - Perivenous lipogenesis
      - Bile acid synthesis
      - Hepatocyte regeneration
      - Stellate cell activation → fibrosis
    """

    def __init__(self):
        self.state = TissueState.healthy("liver")
        self.state.specific_vars = {
            "glycogen_content": 0.5,       # 0-1, liver glycogen stores
            "gluconeogenesis_rate": 0.3,   # 0-1, HGP activity
            "lipogenesis_rate": 0.2,       # 0-1, de novo lipogenesis
            "bile_acid_pool": 0.5,         # 0-1, bile acid synthesis
            "ammonia_clearance": 0.8,      # 0-1, urea cycle function
            "stellate_cell_activation": 0.0,  # 0-1, fibrosis driver
            "kupffer_cell_activity": 0.1,  # 0-1, immune activity
            "insulin_extraction": 0.5,     # 0-1, hepatic insulin clearance
        }

    def compute_dynamics(
        self,
        dt: float,
        glucose: float = 100.0,
        insulin: float = 10.0,
        ffa: float = 0.5,
        inflammatory_signal: float = 0.1,
        drug_effects: Optional[Dict[str, float]] = None,
    ) -> TissueState:
        """
        Compute liver tissue dynamics over time step dt (minutes).
        """
        s = self.state.specific_vars
        drug_effects = drug_effects or {}

        # Gluconeogenesis: suppressed by insulin, driven by glucagon
        insulin_effect = -0.02 * insulin
        glucagon_effect = 0.01 * max(0, 100.0 - glucose) / 100.0
        d_gluconeogenesis = insulin_effect + glucagon_effect

        # Lipogenesis: driven by insulin and FFA
        d_lipogenesis = 0.02 * insulin/10.0 + 0.01 * ffa - 0.01 * s["lipogenesis_rate"]

        # Glycogen: depleted by fasting, replenished by feeding
        if glucose > 100:
            d_glycogen = 0.01 * (glucose - 100) / 100 - 0.005 * s["glycogen_content"]
        else:
            d_glycogen = -0.01 * (100 - glucose) / 100 - 0.005 * s["glycogen_content"]

        # Kupffer cell activation by inflammatory signals
        d_kupffer = 0.02 * inflammatory_signal - 0.01 * s["kupffer_cell_activity"]

        # Stellate cell activation from chronic inflammation
        d_stellate = 0.005 * s["kupffer_cell_activity"] + 0.001 * inflammatory_signal
        d_stellate -= 0.002 * s["stellate_cell_activation"]

        # Insulin extraction: decreases in insulin resistance
        d_ins_extraction = -0.005 * (s["insulin_extraction"] - 0.5)

        # Metformin effect: suppresses gluconeogenesis
        met_effect = drug_effects.get("metformin", 0.0)
        d_gluconeogenesis -= 0.03 * met_effect

        # Update
        dt_scale = dt / 60.0  # scale to hours for metabolic processes
        for var, delta in [
            ("gluconeogenesis_rate", d_gluconeogenesis),
            ("lipogenesis_rate", d_lipogenesis),
            ("glycogen_content", d_glycogen),
            ("kupffer_cell_activity", d_kupffer),
            ("stellate_cell_activation", d_stellate),
            ("insulin_extraction", d_ins_extraction),
        ]:
            new_val = s[var] + delta * dt_scale
            s[var] = np.clip(new_val, 0.0, 1.0)

        # Compute tissue-level outputs
        fibrosis = s["stellate_cell_activation"] * 0.5
        self.state.fibrosis_level = float(np.clip(fibrosis, 0.0, 1.0))
        self.state.metabolic_activity = float(np.clip(
            0.5 * (1.0 - s["stellate_cell_activation"]) +
            0.5 * (1.0 - s["kupffer_cell_activity"] * 0.3),
            0.0, 1.0,
        ))
        self.state.inflammation_level = float(np.clip(
            s["kupffer_cell_activity"] * 0.6, 0.0, 1.0,
        ))

        return self.state

    def compute_coupling_signals(self) -> Dict[str, float]:
        """Output signals to organ-level model."""
        s = self.state.specific_vars
        return {
            "hepatic_ir": float(1.0 - s["insulin_extraction"]),  # hepatic IR proxy
            "hgp_modulation": float(s["gluconeogenesis_rate"] * 2.0),
            "liver_inflammation": float(self.state.inflammation_level),
            "hepatic_lipogenesis": float(s["lipogenesis_rate"]),
        }


# ── Kidney Tissue Model ───────────────────────────────────────

class KidneyTissue:
    """
    Kidney tissue model — nephron with glomerular and tubular components.

    Models:
      - Glomerular filtration (GFR, pressure, permeability)
      - Tubular reabsorption (Na, K, glucose, water)
      - Tubular secretion (H+, K+, drugs)
      - Tubulointerstitial fibrosis
      - Podocyte health
    """

    def __init__(self):
        self.state = TissueState.healthy("kidney")
        self.state.specific_vars = {
            "gfr": 100.0,                 # mL/min
            "glomerular_pressure": 50.0,  # mmHg
            "podocyte_density": 1.0,      # 0-1, podocyte health
            "tubular_function": 1.0,      # 0-1, reabsorption capacity
            "sodium_reabsorption": 0.5,   # 0-1, fractional Na reabsorption
            "glucose_reabsorption": 0.5,  # 0-1, fractional glucose reabsorption
            "interstitial_fibrosis": 0.0, # 0-1, fibrosis level
            "inflammasome_activity": 0.0, # 0-1, NLRP3 activity
        }

    def compute_dynamics(
        self,
        dt: float,
        sbp: float = 120.0,
        glucose: float = 100.0,
        inflammatory_signal: float = 0.1,
        drug_effects: Optional[Dict[str, float]] = None,
    ) -> TissueState:
        """Compute kidney tissue dynamics."""
        s = self.state.specific_vars
        drug_effects = drug_effects or {}
        dt_scale = dt / 60.0  # hours

        # GFR depends on glomerular pressure and podocyte health
        pressure_factor = 1.0 - 0.005 * max(0, sbp - 120)
        target_gfr = 100.0 * pressure_factor * s["podocyte_density"]
        d_gfr = 0.01 * (target_gfr - s["gfr"])

        # Glomerular pressure tracks SBP
        d_glom_pressure = 0.01 * (0.4 * sbp - s["glomerular_pressure"])

        # Podocyte loss from hypertension and hyperglycemia
        d_podocyte = -0.005 * max(0, (sbp - 130) / 50) - 0.003 * max(0, (glucose - 150) / 200)
        d_podocyte -= 0.002 * inflammatory_signal

        # Glucose toxicity: high glucose impairs tubular function
        gluc_tox = max(0, (glucose - 180) / 300)
        d_tubular = -0.005 * gluc_tox - 0.002 * inflammatory_signal

        # Sodium reabsorption: SGLT2 effect
        sglt2_effect = drug_effects.get("sglt2_inhibition", 0.0)
        d_sodium = -0.01 * sglt2_effect - 0.005 * (s["sodium_reabsorption"] - 0.5)

        # Glucose reabsorption: SGLT2 reduces
        d_glucose_reab = -0.03 * sglt2_effect - 0.005 * (s["glucose_reabsorption"] - 0.5)

        # Inflammasome activation from hyperglycemia + pressure
        d_inflammasome = 0.01 * gluc_tox + 0.005 * max(0, (sbp - 140) / 60)
        d_inflammasome -= 0.01 * s["inflammasome_activity"]

        # Fibrosis from chronic inflammation
        d_fibrosis = 0.003 * s["inflammasome_activity"] + 0.001 * inflammatory_signal
        d_fibrosis -= 0.001 * s["interstitial_fibrosis"]

        # Update
        for var, delta in [
            ("gfr", d_gfr), ("glomerular_pressure", d_glom_pressure),
            ("podocyte_density", d_podocyte), ("tubular_function", d_tubular),
            ("sodium_reabsorption", d_sodium),
            ("glucose_reabsorption", d_glucose_reab),
            ("inflammasome_activity", d_inflammasome),
            ("interstitial_fibrosis", d_fibrosis),
        ]:
            new_val = s[var] + delta * dt_scale
            if var in ("gfr", "glomerular_pressure") and var == "gfr":
                s[var] = max(0.0, new_val)
            elif var == "glomerular_pressure":
                s[var] = max(10.0, min(80.0, new_val))
            else:
                s[var] = np.clip(new_val, 0.0, 1.0)

        self.state.metabolic_activity = float(np.clip(
            s["gfr"] / 100.0 * s["podocyte_density"], 0.0, 1.0,
        ))
        self.state.fibrosis_level = float(s["interstitial_fibrosis"])
        self.state.inflammation_level = float(np.clip(
            s["inflammasome_activity"] + inflammatory_signal * 0.3, 0.0, 1.0,
        ))

        return self.state

    def compute_coupling_signals(self) -> Dict[str, float]:
        s = self.state.specific_vars
        return {
            "gfr_output": float(s["gfr"]),
            "kidney_inflammation": float(self.state.inflammation_level),
            "sodium_retention": float(1.0 - s["sodium_reabsorption"]),
            "glomerular_damage": float(1.0 - s["podocyte_density"]),
        }


# ── Cardiac Tissue Model ──────────────────────────────────────

class CardiacTissue:
    """
    Cardiac tissue model — cardiomyocyte electrophysiology and mechanics.

    Models:
      - Action potential conduction
      - Contractile force generation
      - Calcium handling
      - Electrical remodeling (hypertrophy, fibrosis)
      - Metabolic substrate preference (glucose vs FFA)
    """

    def __init__(self):
        self.state = TissueState.healthy("cardiac")
        self.state.specific_vars = {
            "contractility": 0.7,          # 0-1, cardiac contractility
            "heart_rate": 70.0,            # bpm
            "hr_variability": 50.0,        # ms, HRV
            "calcium_handling": 0.8,       # 0-1, Ca2+ cycling efficiency
            "electrical_stability": 0.9,   # 0-1, arrhythmia resistance
            "substrate_flexibility": 0.7,  # 0-1, metabolic flexibility
            "cardiac_fibrosis": 0.0,       # 0-1, myocardial fibrosis
            "oxidative_stress": 0.1,       # 0-1, ROS level
        }

    def compute_dynamics(
        self,
        dt: float,
        sbp: float = 120.0,
        inflammatory_signal: float = 0.1,
        catecholamines: float = 0.3,
        drug_effects: Optional[Dict[str, float]] = None,
    ) -> TissueState:
        """Compute cardiac tissue dynamics."""
        s = self.state.specific_vars
        drug_effects = drug_effects or {}
        dt_scale = dt / 60.0

        # Contractility: driven by catecholamines, reduced by inflammation
        d_contractility = 0.02 * catecholamines - 0.01 * inflammatory_signal
        d_contractility -= 0.005 * (s["contractility"] - 0.7)

        # Heart rate: autonomic balance
        d_hr = 5.0 * catecholamines - 2.0 * inflammatory_signal
        d_hr -= 0.5 * (s["heart_rate"] - 70.0)

        # HRV: reduced by stress, inflammation, fibrosis
        d_hrv = -5.0 * catecholamines - 10.0 * s["cardiac_fibrosis"]
        d_hrv += 1.0 * (50.0 - s["hr_variability"]) * 0.1

        # Calcium handling: impaired by oxidative stress
        d_calcium = -0.02 * s["oxidative_stress"]
        d_calcium -= 0.005 * (s["calcium_handling"] - 0.8)

        # Electrical stability: reduced by fibrosis
        d_electrical = -0.03 * s["cardiac_fibrosis"] - 0.01 * s["oxidative_stress"]
        d_electrical -= 0.005 * (s["electrical_stability"] - 0.9)

        # Fibrosis: driven by pressure overload and inflammation
        d_fibrosis = 0.003 * max(0, (sbp - 140) / 60) + 0.002 * inflammatory_signal
        d_fibrosis -= 0.001 * s["cardiac_fibrosis"]

        # Oxidative stress: metabolic inflexibility + inflammation
        d_oxidative = 0.01 * (1.0 - s["substrate_flexibility"]) + 0.02 * inflammatory_signal
        d_oxidative -= 0.01 * s["oxidative_stress"]

        # Substrate flexibility: reduced by chronic high FFA
        d_flexibility = -0.005 * s["oxidative_stress"]
        d_flexibility -= 0.003 * (s["substrate_flexibility"] - 0.7)

        # Update HR and HRV directly (non-clamped for HR)
        s["heart_rate"] = np.clip(s["heart_rate"] + d_hr * dt_scale, 30.0, 200.0)
        s["hr_variability"] = np.clip(s["hr_variability"] + d_hrv * dt_scale, 5.0, 150.0)

        for var, delta in [
            ("contractility", d_contractility),
            ("calcium_handling", d_calcium),
            ("electrical_stability", d_electrical),
            ("cardiac_fibrosis", d_fibrosis),
            ("oxidative_stress", d_oxidative),
            ("substrate_flexibility", d_flexibility),
        ]:
            new_val = s[var] + delta * dt_scale
            s[var] = np.clip(new_val, 0.0, 1.0)

        # Tissue-level aggregate
        self.state.cell_viability = float(np.clip(
            1.0 - s["cardiac_fibrosis"] * 0.5 - s["oxidative_stress"] * 0.2, 0.0, 1.0,
        ))
        self.state.metabolic_activity = float(np.clip(
            s["contractility"] * s["calcium_handling"], 0.0, 1.0,
        ))
        self.state.fibrosis_level = float(s["cardiac_fibrosis"])

        return self.state

    def compute_coupling_signals(self) -> Dict[str, float]:
        s = self.state.specific_vars
        return {
            "cardiac_output_mod": float(s["contractility"] * s["calcium_handling"]),
            "hr_output": float(s["heart_rate"]),
            "hrv_output": float(s["hr_variability"]),
            "arrhythmia_risk": float(1.0 - s["electrical_stability"]),
        }


# ── Adipose Tissue Model ──────────────────────────────────────

class AdiposeTissue:
    """
    Adipose tissue model — white adipose tissue with hypertrophy and hyperplasia.

    Models:
      - Adipocyte hypertrophy (lipid storage)
      - Adipocyte hyperplasia (recruitment/differentiation)
      - Lipolysis and FFA release
      - Adipokine secretion (leptin, adiponectin)
      - Macrophage infiltration and inflammation
      - Adipose tissue fibrosis
    """

    def __init__(self):
        self.state = TissueState.healthy("adipose")
        self.state.specific_vars = {
            "adipocyte_size": 0.5,          # 0-1, average adipocyte size
            "adipocyte_number": 1.0,        # 0-1, relative cell number
            "lipid_content": 0.5,           # 0-1, total lipid
            "lipolysis_rate": 0.3,          # 0-1, FFA release rate
            "leptin_secretion": 0.5,        # 0-1, leptin level
            "adiponectin_secretion": 0.7,   # 0-1, adiponectin level
            "macrophage_infiltration": 0.0, # 0-1, crown-like structures
            "adipose_fibrosis": 0.0,        # 0-1, fibrosis level
        }

    def compute_dynamics(
        self,
        dt: float,
        insulin: float = 10.0,
        glucose: float = 100.0,
        ffa: float = 0.5,
        inflammatory_signal: float = 0.1,
        drug_effects: Optional[Dict[str, float]] = None,
    ) -> TissueState:
        """Compute adipose tissue dynamics."""
        s = self.state.specific_vars
        drug_effects = drug_effects or {}
        dt_scale = dt / 60.0

        # Adipocyte size: expands with positive energy balance
        d_size = 0.01 * (ffa - 0.3) + 0.005 * (glucose - 100) / 100
        d_size -= 0.005 * s["lipolysis_rate"]

        # Adipocyte hyperplasia (recruitment) in obesity
        d_number = 0.003 * max(0, s["adipocyte_size"] - 0.6)
        d_number -= 0.002 * (s["adipocyte_number"] - 1.0)

        # Lipid content follows size
        d_lipid = 0.01 * (s["adipocyte_size"] - 0.5) + 0.005 * (ffa - 0.3)
        d_lipid -= 0.005 * s["lipolysis_rate"]

        # Lipolysis: inhibited by insulin, activated by stress
        d_lipolysis = -0.02 * insulin/10.0 + 0.01 * inflammatory_signal
        d_lipolysis += 0.005 * s["adipocyte_size"]
        d_lipolysis -= 0.005 * s["lipolysis_rate"]

        # Leptin: proportional to adipocyte size
        d_leptin = 0.02 * s["adipocyte_size"] - 0.01 * s["leptin_secretion"]

        # Adiponectin: inversely related to adipocyte size and inflammation
        d_adiponectin = -0.02 * s["adipocyte_size"] - 0.01 * inflammatory_signal
        d_adiponectin += 0.005 * (0.7 - s["adiponectin_secretion"])

        # Macrophage infiltration driven by adipocyte size/hypoxia
        d_macrophage = 0.02 * max(0, s["adipocyte_size"] - 0.6)
        d_macrophage += 0.01 * inflammatory_signal
        d_macrophage -= 0.01 * s["macrophage_infiltration"]

        # Fibrosis from chronic inflammation
        d_fibrosis = 0.005 * s["macrophage_infiltration"]
        d_fibrosis -= 0.001 * s["adipose_fibrosis"]

        # Insulin-sensitizer effect (thiazolidinedione)
        tzd_effect = drug_effects.get("tzd", 0.0)
        d_number += 0.02 * tzd_effect  # recruitment of small adipocytes

        for var, delta in [
            ("adipocyte_size", d_size), ("adipocyte_number", d_number),
            ("lipid_content", d_lipid), ("lipolysis_rate", d_lipolysis),
            ("leptin_secretion", d_leptin),
            ("adiponectin_secretion", d_adiponectin),
            ("macrophage_infiltration", d_macrophage),
            ("adipose_fibrosis", d_fibrosis),
        ]:
            new_val = s[var] + delta * dt_scale
            s[var] = np.clip(new_val, 0.0, 1.0)

        self.state.inflammation_level = float(np.clip(
            s["macrophage_infiltration"] * 0.7 + inflammatory_signal * 0.3, 0.0, 1.0,
        ))
        self.state.fibrosis_level = float(s["adipose_fibrosis"])
        self.state.metabolic_activity = float(np.clip(
            s["adiponectin_secretion"] / 0.7, 0.0, 1.0,
        ))

        return self.state

    def compute_coupling_signals(self) -> Dict[str, float]:
        s = self.state.specific_vars
        return {
            "ffa_release": float(s["lipolysis_rate"] * 2.0),
            "adipose_inflammation": float(self.state.inflammation_level),
            "leptin_signal": float(s["leptin_secretion"]),
            "adiponectin_signal": float(s["adiponectin_secretion"]),
            "adipose_ir_contribution": float(
                s["macrophage_infiltration"] * 0.5 +
                (1.0 - s["adiponectin_secretion"]) * 0.3
            ),
        }


# ── Tissue Simulator (Orchestrator) ───────────────────────────

class TissueSimulator:
    """
    Orchestrates all tissue models and computes coupling to organ layer.

    Takes inputs from molecular/cellular layer and produces
    tissue-level outputs that feed into organ-level dynamics.
    """

    def __init__(self):
        self.tissues = {
            "liver": LiverTissue(),
            "kidney": KidneyTissue(),
            "cardiac": CardiacTissue(),
            "adipose": AdiposeTissue(),
        }

    def step(
        self,
        dt: float,
        organ_inputs: Dict[str, float],
        drug_effects: Optional[Dict[str, float]] = None,
        inflammatory_signal: float = 0.1,
    ) -> Dict[str, TissueState]:
        """
        Advance all tissue models by dt (minutes).

        Args:
            dt: Time step in minutes
            organ_inputs: Organ-level signals (glucose, insulin, SBP, etc.)
            drug_effects: Drug effect modifiers
            inflammatory_signal: Systemic inflammation level

        Returns:
            Dict of tissue_type → updated TissueState
        """
        glucose = organ_inputs.get("glucose", 100.0)
        insulin = organ_inputs.get("insulin", 10.0)
        sbp = organ_inputs.get("sbp", 120.0)
        ffa = organ_inputs.get("ffa", 0.5)
        catecholamines = organ_inputs.get("catecholamines", 0.3)

        results = {}

        # Liver
        results["liver"] = self.tissues["liver"].compute_dynamics(
            dt, glucose, insulin, ffa, inflammatory_signal, drug_effects,
        )

        # Kidney
        results["kidney"] = self.tissues["kidney"].compute_dynamics(
            dt, sbp, glucose, inflammatory_signal, drug_effects,
        )

        # Cardiac
        results["cardiac"] = self.tissues["cardiac"].compute_dynamics(
            dt, sbp, inflammatory_signal, catecholamines, drug_effects,
        )

        # Adipose
        results["adipose"] = self.tissues["adipose"].compute_dynamics(
            dt, insulin, glucose, ffa, inflammatory_signal, drug_effects,
        )

        return results

    def get_all_coupling_signals(self) -> Dict[str, float]:
        """Aggregate coupling signals from all tissues to organ layer."""
        signals = {}
        for name, tissue in self.tissues.items():
            signals.update(tissue.compute_coupling_signals())
        return signals

    def get_state_summary(self) -> Dict[str, Dict[str, float]]:
        """Get all tissue states as dict."""
        summary = {}
        for name, tissue in self.tissues.items():
            ts = tissue.state
            summary[name] = {
                "cell_viability": ts.cell_viability,
                "metabolic_activity": ts.metabolic_activity,
                "inflammation_level": ts.inflammation_level,
                "fibrosis_level": ts.fibrosis_level,
                **ts.specific_vars,
            }
        return summary


# ── Convenience ───────────────────────────────────────────────

def simulate_tissue_response(
    drug_name: str = "",
    dose: float = 1.0,
    duration_hours: float = 24.0,
    organ_inputs: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Simulate tissue-level response to a drug or condition.

    Args:
        drug_name: Drug name for effect mapping
        dose: Dose fraction (0-1)
        duration_hours: Simulation duration
        organ_inputs: Base organ conditions

    Returns:
        Dict of tissue → trajectory of states
    """
    sim = TissueSimulator()
    inputs = organ_inputs or {
        "glucose": 140.0, "insulin": 15.0, "sbp": 130.0,
        "ffa": 0.5, "catecholamines": 0.3,
    }

    drug_effects = {}
    if drug_name == "metformin":
        drug_effects["metformin"] = dose
    elif drug_name == "empagliflozin":
        drug_effects["sglt2_inhibition"] = dose
    elif drug_name == "pioglitazone":
        drug_effects["tzd"] = dose

    trajectory = {t: [] for t in TISSUE_TYPES}
    steps = int(duration_hours * 60.0 / 5.0)  # 5-min steps

    for step in range(min(steps, 1000)):
        results = sim.step(5.0, inputs, drug_effects)
        for t in TISSUE_TYPES:
            if t in results:
                trajectory[t].append(results[t].to_array())

    return {
        tissue: np.array(traj)
        for tissue, traj in trajectory.items()
    }
