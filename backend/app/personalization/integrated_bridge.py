"""
backend/app/personalization/integrated_bridge.py
Multi-Scale Physiological Integration Bridge.

Implements the central multi-scale causal chain:
  Drug Dose -> 2-Compartment PK -> Free Plasma Concentration [C]
            -> Mechanistic Target Engagement (Receptor / Transporter)
            -> Cellular & Second Messenger Transduction
            -> Organ-Level Functional Parameter Shift
            -> Whole-Body 36-Dimensional Dynamical System (Euler/RK)
            -> Bidirectional Feedback (Renal Volume, Acid-Base, Electrolytes)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np

# Import validated sub-engines
from app.pkpd.compartments import PKParams, PatientCovariates, adjust_params_for_patient
from app.pkpd.receptors import (
    RECEPTOR_LIBRARY,
    ReceptorKineticsEngine,
    SGLT2TransporterEngine,
)
from app.dynamics.respiratory_acid_base import RespiratoryAcidBaseEngine, RespiratoryAcidBaseState
from app.dynamics.raas_model import RAASClosedLoopEngine, RAASState
from app.personalization.dynamics import full_dynamics
from app.personalization.state import (
    PHYSIO_DIM,
    _META_OFF, _CARDIO_OFF, _RENAL_OFF, _INFL_OFF,
    _CIRC_OFF, _ADIP_OFF, _IMMUNE_OFF, _MUSC_OFF,
)


@dataclass
class MultiScaleTwinState:
    physio_state: np.ndarray             # 36-dim whole-body physiological state
    respiratory: RespiratoryAcidBaseState
    raas: RAASState
    drug_plasma_conc_mg_L: Dict[str, float] = field(default_factory=dict)
    receptor_signals: Dict[str, float] = field(default_factory=dict)


class MultiScaleTwinBridge:
    """
    Coordinates and drives multi-scale physiological simulations with bidirectional feedback.
    """
    def __init__(self, initial_physio: Optional[np.ndarray] = None):
        # 1. Whole-body 36-dim state vector
        if initial_physio is not None and len(initial_physio) == PHYSIO_DIM:
            self.physio = initial_physio.copy()
        else:
            self.physio = self._create_healthy_baseline_physio()

        # 2. Mechanistic Sub-Engines
        self.respiratory_engine = RespiratoryAcidBaseEngine()
        self.raas_engine = RAASClosedLoopEngine()
        self.sglt2_engine = SGLT2TransporterEngine()

        # 3. Receptor Engines
        self.receptor_engines: Dict[str, ReceptorKineticsEngine] = {
            "INSR": ReceptorKineticsEngine(RECEPTOR_LIBRARY["INSR"]),
            "ADRB1": ReceptorKineticsEngine(RECEPTOR_LIBRARY["ADRB1"]),
            "AGTR1": ReceptorKineticsEngine(RECEPTOR_LIBRARY["AGTR1"]),
            "GLP1R": ReceptorKineticsEngine(RECEPTOR_LIBRARY["GLP1R"]),
        }

        # 4. PK Compartment States: {drug_name: [A_gut, A_cen, A_per]} (mg)
        self.pk_states: Dict[str, List[float]] = {}
        self.pk_params: Dict[str, PKParams] = {}

    def _create_healthy_baseline_physio(self) -> np.ndarray:
        p = np.zeros(PHYSIO_DIM)
        # Metabolic [0..4]: G, I, HGP, PGU, IR
        p[_META_OFF + 0] = 95.0
        p[_META_OFF + 1] = 2.0   # Fasting resting insulin (uU/mL)
        p[_META_OFF + 2] = 2.0
        p[_META_OFF + 3] = 1.0
        p[_META_OFF + 4] = 0.5
        # Cardio [5..8]: SBP, DBP, HR, HRV
        p[_CARDIO_OFF + 0] = 120.0
        p[_CARDIO_OFF + 1] = 80.0
        p[_CARDIO_OFF + 2] = 72.0
        p[_CARDIO_OFF + 3] = 45.0
        # Renal [9..12]: GFR, Na, K, Osm
        p[_RENAL_OFF + 0] = 105.0
        p[_RENAL_OFF + 1] = 140.0
        p[_RENAL_OFF + 2] = 4.2
        p[_RENAL_OFF + 3] = 285.0
        # Inflammation [13]: CRP
        p[_INFL_OFF + 0] = 0.8
        # Circadian [14..19]
        p[_CIRC_OFF + 0] = 1.0   # CLOCK_BMAL1
        p[_CIRC_OFF + 1] = 1.0   # PER_CRY
        p[_CIRC_OFF + 2] = 250.0 # cortisol (nmol/L)
        p[_CIRC_OFF + 3] = 20.0  # melatonin (pmol/L)
        p[_CIRC_OFF + 4] = 3.14  # phase
        p[_CIRC_OFF + 5] = 0.3   # sleep pressure
        # Adipose [20..24]
        p[_ADIP_OFF + 0] = 15.0  # fat_mass (kg)
        p[_ADIP_OFF + 1] = 0.4   # FFA (mmol/L)
        p[_ADIP_OFF + 2] = 100.0 # LDL
        p[_ADIP_OFF + 3] = 50.0  # HDL
        p[_ADIP_OFF + 4] = 110.0 # TG
        # Immune [25..29]
        p[_IMMUNE_OFF + 0] = 1.0
        p[_IMMUNE_OFF + 1] = 1.0
        p[_IMMUNE_OFF + 2] = 0.5
        p[_IMMUNE_OFF + 3] = 0.15
        p[_IMMUNE_OFF + 4] = 10.0
        # Musculoskeletal [30..35]
        p[_MUSC_OFF + 0] = 32.0
        p[_MUSC_OFF + 1] = 60.0
        p[_MUSC_OFF + 2] = 0.1
        p[_MUSC_OFF + 3] = 1.25
        p[_MUSC_OFF + 4] = 1.0
        p[_MUSC_OFF + 5] = 1.0
        return p

    def administer_drug(self, drug_name: str, dose_mg: float, pk_spec: PKParams) -> None:
        """Administer an oral or IV drug dose into PK compartments."""
        d = drug_name.lower()
        self.pk_params[d] = pk_spec
        if d not in self.pk_states:
            self.pk_states[d] = [0.0, 0.0, 0.0]  # [A_gut, A_cen, A_per]

        if pk_spec.route == "iv":
            self.pk_states[d][1] += dose_mg  # Direct into central compartment
        else:
            self.pk_states[d][0] += dose_mg * pk_spec.F  # Into gut absorption compartment

    def step(
        self,
        dt_min: float = 1.0,
        inputs: Optional[Dict[str, float]] = None,
        params: Optional[np.ndarray] = None,
    ) -> MultiScaleTwinState:
        """
        Executes one integrated multi-scale simulation step across all coupled systems.
        """
        inputs = inputs or {}
        from app.personalization.dynamics import DEFAULT_PARAMS
        p_vec = params.copy() if params is not None else DEFAULT_PARAMS.copy()

        # =====================================================================
        # Step 1: Update Pharmacokinetics (PK) ODEs for all active drugs
        # =====================================================================
        plasma_conc_mg_L: Dict[str, float] = {}
        plasma_conc_nM: Dict[str, float] = {}

        # Molecular weights for unit conversions (g/mol = mg/mmol)
        MW = {
            "empagliflozin": 450.91,
            "metformin": 129.16,
            "lisinopril": 405.49,
            "losartan": 422.91,
            "atenolol": 266.34,
            "insulin_regular": 5808.0,
        }

        dt_h = dt_min / 60.0  # PK parameters are in 1/h and L/h
        for drug, state in list(self.pk_states.items()):
            pk = self.pk_params.get(drug)
            if not pk:
                continue
            A_gut, A_cen, A_per = state

            # 2-compartment ODEs
            dA_gut = -pk.ka * A_gut
            dA_cen = (pk.ka * A_gut - (pk.CL / pk.Vc + pk.Q / pk.Vc) * A_cen + (pk.Q / pk.Vp) * A_per)
            dA_per = ((pk.Q / pk.Vc) * A_cen - (pk.Q / pk.Vp) * A_per)

            # Update amounts
            new_A_gut = max(0.0, A_gut + dA_gut * dt_h)
            new_A_cen = max(0.0, A_cen + dA_cen * dt_h)
            new_A_per = max(0.0, A_per + dA_per * dt_h)
            self.pk_states[drug] = [new_A_gut, new_A_cen, new_A_per]

            # Central concentration C = A_cen / Vc (mg/L)
            conc_mg_L = new_A_cen / max(1.0, pk.Vc)
            plasma_conc_mg_L[drug] = float(conc_mg_L)

            # Convert to nM: (mg/L) / (g/mol) * 10^6
            mw = MW.get(drug, 300.0)
            conc_nM = (conc_mg_L / mw) * 1e6
            plasma_conc_nM[drug] = float(conc_nM)

        # =====================================================================
        # Step 2: Target Receptor & Transporter Engagement
        # =====================================================================
        receptor_signals: Dict[str, float] = {}

        # 2a. Insulin Receptor (INSR)
        # Endogenous insulin (uU/mL -> ~0.00694 nM) + exogenous insulin PK
        endogenous_insulin_nM = self.physio[_META_OFF + 1] * 0.00694
        exogenous_insulin_nM = plasma_conc_nM.get("insulin_regular", 0.0)
        total_insulin_nM = endogenous_insulin_nM + exogenous_insulin_nM
        insr_res = self.receptor_engines["INSR"].step(total_insulin_nM, dt_min=dt_min)
        receptor_signals["INSR_tissue_response"] = insr_res["tissue_response"]

        # 2b. ADRB1 (Beta-1 Adrenergic Receptor) & Beta-Blockade
        atenolol_nM = plasma_conc_nM.get("atenolol", 0.0)
        # Endogenous catecholamine resting drive ~ 0.5 nM
        adrb1_res = self.receptor_engines["ADRB1"].step(
            agonist_conc_nM=0.5,
            dt_min=dt_min,
            antagonist_conc_nM=atenolol_nM,
            antagonist_Kb_nM=100.0,  # Atenolol ADRB1 affinity
        )
        receptor_signals["ADRB1_tissue_response"] = adrb1_res["tissue_response"]

        # 2c. SGLT2 Carrier Transporter (Empagliflozin inhibition)
        empa_nM = plasma_conc_nM.get("empagliflozin", 0.0)
        sglt2_res = self.sglt2_engine.compute_renal_flux(
            plasma_glucose_mg_dL=self.physio[_META_OFF + 0],
            gfr_mL_min=self.physio[_RENAL_OFF + 0],
            inhibitor_conc_nM=empa_nM,
            inhibitor_type="empagliflozin",
        )
        receptor_signals["SGLT2_inhibition_pct"] = sglt2_res["sglt2_inhibition_pct"]

        # =====================================================================
        # Step 3: Closed-Loop RAAS & Renal-Cardio Coupling
        # =====================================================================
        # Compute current MAP
        sbp = self.physio[_CARDIO_OFF + 0]
        dbp = self.physio[_CARDIO_OFF + 1]
        map_current = (sbp + 2.0 * dbp) / 3.0

        # Lisinopril ACE inhibition: IC50 ~ 2.0 nM
        lisinopril_nM = plasma_conc_nM.get("lisinopril", 0.0)
        ace_inhibition = (lisinopril_nM / (2.0 + lisinopril_nM)) * 100.0 if lisinopril_nM > 0 else 0.0

        # Losartan AT1 receptor blockade: IC50 ~ 10.0 nM
        losartan_nM = plasma_conc_nM.get("losartan", 0.0)
        at1_blockade = (losartan_nM / (10.0 + losartan_nM)) * 100.0 if losartan_nM > 0 else 0.0

        raas_state = self.raas_engine.step(
            MAP_current_mmHg=map_current,
            GFR_current_mL_min=self.physio[_RENAL_OFF + 0],
            plasma_K_mEq_L=self.physio[_RENAL_OFF + 2],
            dt_min=dt_min,
            ace_inhibition_pct=ace_inhibition,
            at1_blockade_pct=at1_blockade,
        )

        # =====================================================================
        # Step 4: Modulate Whole-Body ODE Parameters via Receptor Transduction
        # =====================================================================
        # Parameter index mapping:
        # 0: SI, 1: HGP_basal, 2: beta_response, 3: RT, 4: arterial_stiffness, 5: vascular_resistance
        from app.personalization.priors import PARAMETER_NAMES

        # A. Insulin receptor transduction stimulates peripheral uptake and suppresses HGP
        # Baseline insr_response ~ 0.50
        insr_gain = receptor_signals["INSR_tissue_response"] / 0.50
        p_vec[0] *= float(np.clip(insr_gain, 0.4, 2.5))

        # B. SGLT2 inhibition lowers renal glucose threshold RT (normally 180 mg/dL)
        p_vec[3] = float(sglt2_res["effective_renal_threshold_mg_dL"])

        # C. RAAS Ang II SVR modifier directly scales vascular resistance parameter
        p_vec[5] *= raas_state.SVR_multiplier

        # D. ADRB1 inotropy/chronotropy scales heart rate dynamics
        adrb1_ratio = receptor_signals["ADRB1_tissue_response"] / 0.90
        # If beta-blocker is present, reduce HR baseline
        if atenolol_nM > 0:
            inputs["exercise"] = inputs.get("exercise", 0.0) * float(np.clip(adrb1_ratio, 0.6, 1.0))

        # =====================================================================
        # Step 5: Advance 36-Dimensional Physiological Dynamics
        # =====================================================================
        self.physio = full_dynamics(self.physio, p_vec, inputs, dt=dt_min)

        # Apply RAAS volume-pressure shift directly to blood pressure
        self.physio[_CARDIO_OFF + 0] = float(np.clip(self.physio[_CARDIO_OFF + 0] + raas_state.MAP_effect_mmHg * 0.05, 50.0, 240.0))
        self.physio[_CARDIO_OFF + 1] = float(np.clip(self.physio[_CARDIO_OFF + 1] + raas_state.MAP_effect_mmHg * 0.03, 30.0, 140.0))
        # Update renal sodium to match RAAS ECF sodium
        self.physio[_RENAL_OFF + 1] = float(raas_state.plasma_Na_mEq_L)

        # =====================================================================
        # Step 6: Advance Respiratory Gas Exchange & Acid-Base Equilibrium
        # =====================================================================
        # Compute metabolic VCO2 production (scales with glucose disposal and exercise)
        exercise_level = inputs.get("exercise", 0.0)
        vco2_prod = 200.0 + (exercise_level * 1500.0) + (self.physio[_META_OFF + 3] * 20.0)
        
        # In lactic acidosis or high glucosuria, minor bicarbonate consumption occurs
        hco3_delta = 0.0
        if self.physio[_META_OFF + 0] > 300.0:
            hco3_delta = -0.01  # Mild diabetic ketoacidotic drift
        
        resp_state = self.respiratory_engine.step(
            dt_min=dt_min,
            VCO2_prod_mL_min=vco2_prod,
            metabolic_HCO3_delta_mEq=hco3_delta,
        )

        return MultiScaleTwinState(
            physio_state=self.physio.copy(),
            respiratory=resp_state,
            raas=raas_state,
            drug_plasma_conc_mg_L=plasma_conc_mg_L,
            receptor_signals=receptor_signals,
        )
