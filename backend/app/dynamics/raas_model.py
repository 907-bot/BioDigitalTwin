"""
backend/app/dynamics/raas_model.py
Closed-Loop Renin-Angiotensin-Aldosterone System (RAAS) & Renal-Cardiovascular Coupling.

Models bidirectional physiological feedback:
1. Renin Secretion:
   Triggered by decreased renal perfusion pressure (MAP < MAP_setpoint) and
   decreased macula densa sodium load (GFR * [Na]).
2. Angiotensin Cascade:
   Renin cleaves Angiotensinogen -> Angiotensin I.
   ACE (angiotensin-converting enzyme) converts Ang I -> Angiotensin II.
   ACE Inhibitors (e.g. Lisinopril) inhibit this step.
3. Angiotensin II Actions:
   - Vasoconstriction: directly increases systemic vascular resistance (SVR) via AT1 receptors.
   - Adrenal Cortex: stimulates aldosterone secretion.
   - Renal: constricts efferent arteriole, preserving GFR under moderate hypotension.
   ARBs (e.g. Losartan) block AT1-mediated vasoconstriction and aldosterone secretion.
4. Aldosterone Actions:
   - Stimulates distal tubule / collecting duct Na+ reabsorption and K+ excretion.
5. Volume-Pressure Feedback (Guytonian circulatory loop):
   - Intravascular volume expansion -> increases mean circulatory filling pressure (MCFP)
     and cardiac preload -> raises MAP.
   - Pressure Natriuresis: higher MAP drives urinary fluid and sodium excretion, closing the loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class RAASState:
    plasma_renin_activity: float = 1.5   # ng/(mL*h) [normal range: 0.5 - 3.0]
    angiotensin_I: float = 20.0          # pg/mL [normal: 10 - 40]
    angiotensin_II: float = 25.0         # pg/mL [normal: 10 - 45]
    aldosterone: float = 12.0            # ng/dL [normal: 4 - 25]
    plasma_volume_L: float = 3.0         # Liters [normal adult: ~2.8 - 3.2 L]
    ecf_sodium_mEq: float = 1960.0       # Total extracellular sodium (14 L * 140 mEq/L)
    plasma_Na_mEq_L: float = 140.0       # Extracellular [Na+] (mEq/L)
    MAP_effect_mmHg: float = 0.0         # Hemodynamic pressure shift from RAAS + volume (mmHg)
    SVR_multiplier: float = 1.0          # Arterial resistance modifier from Ang II


class RAASClosedLoopEngine:
    """
    Closed-loop differential equation engine for RAAS and long-term volume-pressure control.
    """
    def __init__(self, initial_state: Optional[RAASState] = None):
        self.state = initial_state or RAASState()
        # Setpoints and baselines
        self.MAP_setpoint: float = 93.3      # mmHg (~120/80)
        self.baseline_renin: float = 1.5     # ng/(mL*h)
        self.baseline_volume_L: float = 3.0  # L
        self.vascular_compliance_L_mmHg: float = 0.040  # L/mmHg (Guytonian vascular capacitance)

    def step(
        self,
        MAP_current_mmHg: float,
        GFR_current_mL_min: float = 100.0,
        plasma_K_mEq_L: float = 4.2,
        dt_min: float = 1.0,
        water_intake_mL_min: float = 1.5,       # ~2.1 L/day
        sodium_intake_mEq_min: float = 0.10,    # ~140 mEq/day
        ace_inhibition_pct: float = 0.0,        # 0 - 100% (e.g. Lisinopril)
        at1_blockade_pct: float = 0.0,          # 0 - 100% (e.g. Losartan)
    ) -> RAASState:
        """
        Integrates RAAS cascades and fluid balance forward by dt_min.
        """
        MAP = max(40.0, float(MAP_current_mmHg))
        GFR = max(10.0, float(GFR_current_mL_min))
        K_plasma = max(2.5, min(7.0, float(plasma_K_mEq_L)))
        s = self.state

        # 1. Renin Secretion:
        # Stimulated inversely by MAP and by reduced macula densa sodium delivery
        pressure_error = (self.MAP_setpoint - MAP) / self.MAP_setpoint
        # Baroreceptor/renal afferent arteriole sensing:
        renin_drive = self.baseline_renin * np.exp(np.clip(2.5 * pressure_error, -2.0, 3.0))
        # Macula densa sodium load effect:
        na_load_ratio = (GFR * s.plasma_Na_mEq_L) / (100.0 * 140.0)
        renin_drive *= (1.0 / max(0.5, na_load_ratio))

        d_renin = 0.05 * (renin_drive - s.plasma_renin_activity)
        new_renin = float(np.clip(s.plasma_renin_activity + d_renin * dt_min, 0.1, 30.0))

        # 2. Angiotensin I production:
        # Cleaved from angiotensinogen by renin; cleared by ACE and peptidases
        ang_I_prod = new_renin * 15.0
        d_ang_I = (ang_I_prod - 0.20 * s.angiotensin_I)
        new_ang_I = float(np.clip(s.angiotensin_I + d_ang_I * dt_min, 2.0, 200.0))

        # 3. Angiotensin II production via ACE:
        # Inhibited by ACE inhibitors (Lisinopril)
        ace_activity = max(0.01, 1.0 - (ace_inhibition_pct / 100.0))
        ang_II_prod = s.angiotensin_I * 0.25 * ace_activity
        d_ang_II = (ang_II_prod - 0.35 * s.angiotensin_II)
        new_ang_II = float(np.clip(s.angiotensin_II + d_ang_II * dt_min, 1.0, 250.0))

        # Effective Ang II signal (blunted by AT1 receptor blockers like Losartan):
        at1_activity = max(0.01, 1.0 - (at1_blockade_pct / 100.0))
        effective_ang_II_signal = new_ang_II * at1_activity

        # 4. Aldosterone Secretion from Adrenal Cortex:
        # Driven synergistically by Ang II (AT1) and plasma K+; cleared by liver
        k_stim = max(0.5, K_plasma / 4.0)
        aldo_target = (effective_ang_II_signal / 25.0) * 12.0 * k_stim
        d_aldo = 0.03 * (aldo_target - s.aldosterone)
        new_aldo = float(np.clip(s.aldosterone + d_aldo * dt_min, 1.0, 100.0))

        # 5. Tubular Sodium and Water Excretion (Guytonian Pressure Natriuresis):
        # Normal sodium filtration = (GFR * [Na]) / 1000 [mEq/min]
        na_filtered_mEq_min = (GFR * s.plasma_Na_mEq_L) / 1000.0
        
        # Basal fractional reabsorption ~ 99.2%
        # Aldosterone increases reabsorption via ENaC; Ang II stimulates proximal Na+/H+ exchanger
        fractional_reabsorption = 0.992 + 0.005 * ((new_aldo - 12.0) / 12.0) + 0.002 * ((effective_ang_II_signal - 25.0) / 25.0)
        fractional_reabsorption = float(np.clip(fractional_reabsorption, 0.970, 0.998))

        # Pressure natriuresis: arterial pressure above setpoint drives renal excretion
        pressure_natriuresis_factor = 1.0 + 0.03 * (MAP - self.MAP_setpoint)
        na_excreted_mEq_min = max(0.01, na_filtered_mEq_min * (1.0 - fractional_reabsorption) * max(0.2, pressure_natriuresis_factor))

        # Urine volume coupled to sodium excretion + free water clearance (mL/min)
        urine_output_mL_min = max(0.3, na_excreted_mEq_min * 10.0)

        # 6. Intravascular Volume Integration:
        net_fluid_mL = (water_intake_mL_min - urine_output_mL_min) * dt_min
        new_plasma_vol_L = float(np.clip(s.plasma_volume_L + (net_fluid_mL / 1000.0), 1.8, 5.0))

        # Extracellular sodium mass update
        net_na_mEq = (sodium_intake_mEq_min - na_excreted_mEq_min) * dt_min
        new_ecf_na_mEq = float(np.clip(s.ecf_sodium_mEq + net_na_mEq, 1500.0, 2500.0))
        
        # Extracellular fluid volume (ECF ~ 4.5 * plasma volume)
        ecf_vol_L = new_plasma_vol_L * 4.667
        new_plasma_Na = float(np.clip(new_ecf_na_mEq / ecf_vol_L, 120.0, 160.0))

        # 7. Hemodynamic Feedback Coupling:
        # Pressure shift from volume expansion/depletion:
        delta_V = new_plasma_vol_L - self.baseline_volume_L
        volume_pressure_shift = delta_V / self.vascular_compliance_L_mmHg

        # SVR modifier from direct Ang II vasoconstriction:
        svr_mult = 1.0 + 0.012 * (effective_ang_II_signal - 25.0)
        svr_mult = float(np.clip(svr_mult, 0.70, 1.80))

        self.state = RAASState(
            plasma_renin_activity=new_renin,
            angiotensin_I=new_ang_I,
            angiotensin_II=new_ang_II,
            aldosterone=new_aldo,
            plasma_volume_L=new_plasma_vol_L,
            ecf_sodium_mEq=new_ecf_na_mEq,
            plasma_Na_mEq_L=new_plasma_Na,
            MAP_effect_mmHg=float(volume_pressure_shift),
            SVR_multiplier=svr_mult,
        )
        return self.state

