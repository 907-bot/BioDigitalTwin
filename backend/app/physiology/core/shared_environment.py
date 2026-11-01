"""
BioDigitalTwin — Shared Physiological Environment.

Represents the physical transport networks and circulating pools through which
independent organ models communicate:
1. Circulation (hemodynamics, regional blood flows, blood pressure)
2. Blood chemistry & gases (glucose, O2, CO2, electrolytes, metabolites)
3. Fluid balance (TBW, ECF, ICF, plasma volume, hematocrit)
4. Hormone transport (insulin, glucagon, cortisol, RAAS, incretins)
5. Nutrient & metabolite transport (amino acids, fatty acids, lactate, urea)
6. Drug transport (plasma concentrations, free fraction, exposure)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .state_registry import StateDescriptor, get_global_registry


@dataclass
class CirculationState:
    heart_rate_bpm: float = 72.0
    stroke_volume_mL: float = 70.0
    cardiac_output_L_min: float = 5.04
    total_peripheral_resistance_mmHg_min_L: float = 18.0
    mean_arterial_pressure_mmHg: float = 93.0
    systolic_bp_mmHg: float = 120.0
    diastolic_bp_mmHg: float = 80.0
    central_venous_pressure_mmHg: float = 4.0
    regional_flows_L_min: Dict[str, float] = field(default_factory=lambda: {
        "brain": 0.70,
        "coronary": 0.20,
        "renal": 1.10,
        "gut": 1.20,
        "hepatic_portal": 1.20,
        "hepatic_artery": 0.30,
        "hepatic_total": 1.50,
        "muscle": 0.90,
        "adipose": 0.90,
        "pulmonary": 5.04,
    })


@dataclass
class BloodChemistryState:
    glucose_mg_dL: float = 95.0
    insulin_uU_mL: float = 8.0
    glucagon_pg_mL: float = 70.0
    paO2_mmHg: float = 95.0
    paCO2_mmHg: float = 40.0
    saO2_pct: float = 97.5
    pH: float = 7.40
    hco3_mEq_L: float = 24.0
    sodium_mEq_L: float = 140.0
    potassium_mEq_L: float = 4.2
    chloride_mEq_L: float = 102.0
    osmolality_mOsm_kg: float = 290.0
    lactate_mmol_L: float = 1.0
    free_fatty_acids_mmol_L: float = 0.45
    urea_mg_dL: float = 15.0


@dataclass
class FluidBalanceState:
    total_body_water_L: float = 42.0
    extracellular_fluid_L: float = 14.0
    intracellular_fluid_L: float = 28.0
    plasma_volume_L: float = 3.0
    hematocrit: float = 0.42
    hemoglobin_g_dL: float = 14.5


@dataclass
class HormoneTransportState:
    cortisol_nmol_L: float = 300.0
    aldosterone_ng_dL: float = 10.0
    plasma_renin_activity_ng_mL_h: float = 1.2
    angiotensin_ii_pg_mL: float = 15.0
    adh_pg_mL: float = 2.5
    glp1_pmol_L: float = 15.0
    gip_pmol_L: float = 20.0
    thyroid_tsh_uIU_mL: float = 1.8


@dataclass
class DrugTransportState:
    plasma_concentrations_mg_L: Dict[str, float] = field(default_factory=dict)
    free_fractions: Dict[str, float] = field(default_factory=dict)


class SharedEnvironment:
    """
    Authoritative shared physical medium linking all organ systems.
    """

    def __init__(self, patient_weight_kg: float = 70.0, patient_height_cm: float = 175.0):
        self.patient_weight_kg = patient_weight_kg
        self.patient_height_cm = patient_height_cm
        self.exercise_intensity: float = 0.0

        self.circulation = CirculationState()
        self.chemistry = BloodChemistryState()
        self.fluids = FluidBalanceState()
        self.hormones = HormoneTransportState()
        self.drugs = DrugTransportState()

        # Scale fluid volumes based on weight (approx 60% TBW, 20% ECF, 4% Plasma)
        self.fluids.total_body_water_L = float(round(0.60 * patient_weight_kg, 2))
        self.fluids.extracellular_fluid_L = float(round(0.20 * patient_weight_kg, 2))
        self.fluids.intracellular_fluid_L = float(round(0.40 * patient_weight_kg, 2))
        self.fluids.plasma_volume_L = float(round((0.045 * patient_weight_kg), 2))

        # Initialize balanced regional flows matching baseline cardiac output
        self.compute_regional_perfusion()

        # Register circulating transport states in global registry
        self._register_environment_states()

    def _register_environment_states(self) -> None:
        reg = get_global_registry()
        env_descriptors = [
            StateDescriptor("blood.glucose", "Arterial Plasma Glucose", "mg/dL", "blood", 20.0, 600.0, 95.0, "fasting_baseline", True, "transport", "Circulating blood glucose concentration"),
            StateDescriptor("blood.insulin", "Plasma Insulin", "uU/mL", "blood", 0.5, 300.0, 8.0, "fasting_baseline", True, "transport", "Circulating systemic insulin concentration"),
            StateDescriptor("blood.glucagon", "Plasma Glucagon", "pg/mL", "blood", 20.0, 300.0, 70.0, "fasting_baseline", True, "transport", "Circulating pancreatic glucagon"),
            StateDescriptor("blood.pao2_mmHg", "Arterial PaO2", "mmHg", "blood", 20.0, 200.0, 95.0, "room_air_abg", True, "transport", "Arterial oxygen partial pressure"),
            StateDescriptor("blood.paco2_mmHg", "Arterial PaCO2", "mmHg", "blood", 15.0, 90.0, 40.0, "room_air_abg", True, "transport", "Arterial carbon dioxide tension"),
            StateDescriptor("blood.pH", "Arterial Blood pH", "pH", "blood", 6.80, 7.80, 7.40, "room_air_abg", True, "transport", "Arterial blood pH via Henderson-Hasselbalch"),
            StateDescriptor("blood.sodium_mEq_L", "Serum Sodium", "mEq/L", "blood", 115.0, 165.0, 140.0, "basic_metabolic_panel", True, "transport", "Circulating extracellular sodium"),
            StateDescriptor("blood.potassium_mEq_L", "Serum Potassium", "mEq/L", "blood", 2.5, 7.0, 4.2, "basic_metabolic_panel", True, "transport", "Circulating extracellular potassium"),
            StateDescriptor("blood.glp1_pmol_L", "Plasma GLP-1", "pmol/L", "blood", 5.0, 150.0, 15.0, "fasting_baseline", False, "transport", "Circulating active GLP-1 incretin peptide"),
        ]
        for d in env_descriptors:
            if d.name not in reg._states:
                reg.register_state(d)

    def update_hemodynamics(
        self,
        hr: float,
        sv: float,
        tpr: float,
        cvp: float = 4.0,
    ) -> None:
        """Updates central pressures and flows from cardiac pump performance and vascular tone."""
        self.circulation.heart_rate_bpm = float(np.clip(hr, 30.0, 220.0))
        self.circulation.stroke_volume_mL = float(np.clip(sv, 20.0, 180.0))
        self.circulation.cardiac_output_L_min = float(
            (self.circulation.heart_rate_bpm * self.circulation.stroke_volume_mL) / 1000.0
        )
        self.circulation.total_peripheral_resistance_mmHg_min_L = float(np.clip(tpr, 6.0, 40.0))
        self.circulation.central_venous_pressure_mmHg = float(cvp)

        # Mean Arterial Pressure = CO * TPR + CVP
        co = self.circulation.cardiac_output_L_min
        self.circulation.mean_arterial_pressure_mmHg = float(co * tpr + cvp)

        # Pulse pressure proportional to stroke volume
        pp = 40.0 * (self.circulation.stroke_volume_mL / 70.0)
        self.circulation.systolic_bp_mmHg = float(self.circulation.mean_arterial_pressure_mmHg + (2.0 / 3.0) * pp)
        self.circulation.diastolic_bp_mmHg = float(self.circulation.mean_arterial_pressure_mmHg - (1.0 / 3.0) * pp)

    def compute_regional_perfusion(
        self,
        exercise_intensity: float = 0.0,
        sympathetic_tone: float = 1.0,
    ) -> Dict[str, float]:
        """
        Dynamically distributes Cardiac Output to regional organs based on exertion and ANS tone.
        Guarantees that regional systemic flows strictly sum to Cardiac Output.
        """
        co = self.circulation.cardiac_output_L_min

        if exercise_intensity > 0.05:
            # Active muscular hyperemia (vasodilation) and splanchnic vasoconstriction
            muscle_frac = min(0.75, 0.18 + exercise_intensity * 0.55)
            renal_frac = max(0.06, 0.22 * (1.0 - exercise_intensity * 0.60))
            gut_frac = max(0.06, 0.24 * (1.0 - exercise_intensity * 0.60))
            brain_frac = 0.14 / (1.0 + exercise_intensity * 0.80)
            coronary_frac = min(0.06, 0.04 + exercise_intensity * 0.02)
            adipose_frac = max(0.03, 0.18 * (1.0 - exercise_intensity * 0.70))
        else:
            muscle_frac = 0.18
            renal_frac = 0.22
            gut_frac = 0.24
            brain_frac = 0.14
            coronary_frac = 0.04
            adipose_frac = 0.18

        # Normalize fractions to strictly 1.0
        total_frac = muscle_frac + renal_frac + gut_frac + brain_frac + coronary_frac + adipose_frac
        norm = 1.0 / total_frac

        q_brain = co * brain_frac * norm
        q_coronary = co * coronary_frac * norm
        q_renal = co * renal_frac * norm
        q_gut = co * gut_frac * norm
        q_hepatic_artery = co * 0.06
        q_hepatic_total = q_gut + q_hepatic_artery
        q_muscle = co * muscle_frac * norm
        q_adipose = co * adipose_frac * norm

        self.circulation.regional_flows_L_min = {
            "brain": float(q_brain),
            "coronary": float(q_coronary),
            "renal": float(q_renal),
            "gut": float(q_gut),
            "hepatic_portal": float(q_gut),
            "hepatic_artery": float(q_hepatic_artery),
            "hepatic_total": float(q_hepatic_total),
            "muscle": float(q_muscle),
            "adipose": float(q_adipose),
            "pulmonary": float(co),
        }
        return self.circulation.regional_flows_L_min

    def update_acid_base_and_gas_equilibrium(
        self,
        pao2: Optional[float] = None,
        paco2: Optional[float] = None,
        hco3: Optional[float] = None,
        alveolar_ventilation_L_min: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Updates blood gas tension, Severinghaus O2-Hb saturation, and Henderson-Hasselbalch arterial pH.
        """
        if alveolar_ventilation_L_min is not None and (pao2 is None or paco2 is None):
            va = max(1.0, alveolar_ventilation_L_min)
            # VCO2 ~ 200 mL/min = 0.2 L/min; PaCO2 = (VCO2 * 863) / VA
            calc_paco2 = (0.20 * 863.0) / va
            # Alveolar gas eq: PAO2 = 0.21*(760-47) - PaCO2/0.8 = 149.7 - PaCO2*1.25
            calc_pao2 = max(20.0, 149.7 - calc_paco2 * 1.25 - 8.0)  # 8 mmHg A-a gradient
            if paco2 is None:
                paco2 = calc_paco2
            if pao2 is None:
                pao2 = calc_pao2

        if pao2 is not None:
            self.chemistry.paO2_mmHg = float(np.clip(pao2, 20.0, 200.0))
        if paco2 is not None:
            self.chemistry.paCO2_mmHg = float(np.clip(paco2, 10.0, 100.0))
        if hco3 is not None:
            self.chemistry.hco3_mEq_L = float(np.clip(hco3, 5.0, 50.0))

        # Severinghaus O2-Hb Saturation
        p = self.chemistry.paO2_mmHg
        sao2 = 100.0 * (((p ** 3) + 150.0 * p) / ((p ** 3) + 150.0 * p + 23400.0))
        self.chemistry.saO2_pct = float(np.clip(sao2, 40.0, 100.0))

        # Henderson-Hasselbalch equation: pH = 6.10 + log10([HCO3-] / (0.0306 * PaCO2))
        ratio = max(0.1, self.chemistry.hco3_mEq_L / (0.0306 * self.chemistry.paCO2_mmHg))
        self.chemistry.pH = float(np.clip(6.10 + np.log10(ratio), 6.80, 7.80))

        return self.chemistry.saO2_pct, self.chemistry.pH

    def apply_solute_flux(
        self,
        solute_name: Optional[str] = None,
        net_flux_amount: Optional[float] = None,
        dt_minutes: float = 1.0,
        solute: Optional[str] = None,
        flux_mg_min: Optional[float] = None,
        compartment: str = "blood",
    ) -> None:
        """
        Updates concentration of a circulating substance from net organ influx/efflux:
        dC/dt = net_flux / V_distribution
        """
        name = solute_name or solute or ""
        amount = net_flux_amount if net_flux_amount is not None else (flux_mg_min if flux_mg_min is not None else 0.0)

        # Distribution volume mapping
        ecf_dL = self.fluids.extracellular_fluid_L * 10.0   # in dL (140 dL)
        plasma_L = self.fluids.plasma_volume_L             # in L (~3.0 L)

        if name == "glucose":
            # Net flux in mg/min -> ECF distribution (dL)
            # dG/dt = flux_mg_min / V_ecf_dL
            dG = (amount / ecf_dL) * dt_minutes
            self.chemistry.glucose_mg_dL = float(np.clip(self.chemistry.glucose_mg_dL + dG, 20.0, 600.0))

        elif name == "insulin":
            # Net flux in uU/min -> plasma volume (L)
            # Clearance: t1/2 ~ 4-6 min (k_elim ~ 0.15 min^-1)
            k_elim = 0.15
            dI = ((amount / (plasma_L * 1000.0)) - k_elim * self.chemistry.insulin_uU_mL) * dt_minutes
            self.chemistry.insulin_uU_mL = float(np.clip(self.chemistry.insulin_uU_mL + dI, 0.5, 300.0))

        elif name == "glucagon":
            # Glucagon flux in pg/min -> plasma clearance (t1/2 ~ 5 min)
            k_elim = 0.14
            dG = ((amount / (plasma_L * 1000.0)) - k_elim * self.chemistry.glucagon_pg_mL) * dt_minutes
            self.chemistry.glucagon_pg_mL = float(np.clip(self.chemistry.glucagon_pg_mL + dG, 20.0, 300.0))

        elif name == "sodium":
            # Sodium in mEq/min -> ECF (L)
            dNa = (amount / self.fluids.extracellular_fluid_L) * dt_minutes
            self.chemistry.sodium_mEq_L = float(np.clip(self.chemistry.sodium_mEq_L + dNa, 110.0, 170.0))

        elif name == "potassium":
            # Potassium in mEq/min -> ECF (L)
            dK = (amount / self.fluids.extracellular_fluid_L) * dt_minutes
            self.chemistry.potassium_mEq_L = float(np.clip(self.chemistry.potassium_mEq_L + dK, 2.0, 7.5))

        elif name == "glp1":
            # GLP-1 in pmol/min -> plasma distribution with DPP-4 clearance
            k_dpp4 = 0.35  # t1/2 ~ 2 min
            dGLP = ((amount / plasma_L) - k_dpp4 * (self.hormones.glp1_pmol_L - 15.0)) * dt_minutes
            self.hormones.glp1_pmol_L = float(np.clip(self.hormones.glp1_pmol_L + dGLP, 5.0, 150.0))
