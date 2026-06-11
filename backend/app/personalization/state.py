"""
Phase 3: Whole-Body Cellular-Level Twin State.

Metabolic      [5]: G, I, HGP, PGU, IR
Cardiovascular [4]: SBP, DBP, HR, HRV
Renal          [4]: GFR, Na, K, Osm
Inflammation   [1]: CRP  (legacy; Phase 3 expands to ImmuneInflam)
Circadian      [6]: CLOCK_BMAL1, PER_CRY, cortisol, melatonin, circadian_phase, sleep_pressure
Adipose-Lipid  [5]: fat_mass, FFA, LDL, HDL, TG
Immune-Inflam  [5]: IL6_proxy, TNFa_proxy, M1_M2_ratio, NFkB_activity, InflammatoryLoad
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

# ── Phase 2 subsystem dims (unchanged) ────────────────────────
METABOLIC_DIM = 5
CARDIO_DIM = 4
RENAL_DIM = 4
INFLAMMATION_DIM = 1  # legacy CRP-only
INFLAM_DIM = INFLAMMATION_DIM

# ── Phase 3 new subsystem dims ────────────────────────────────
CIRCADIAN_DIM = 6
ADIPOSE_DIM = 5
IMMUNE_DIM = 5

# ── Total physiological state dimension ───────────────────────
PHYSIO_DIM = (
    METABOLIC_DIM + CARDIO_DIM + RENAL_DIM + INFLAMMATION_DIM +
    CIRCADIAN_DIM + ADIPOSE_DIM + IMMUNE_DIM
)  # = 30

# ── Parameter dimensions ──────────────────────────────────────
METABOLIC_PARAM_DIM = 4
CARDIO_PARAM_DIM = 4
RENAL_PARAM_DIM = 4
CIRCADIAN_PARAM_DIM = 4   # period, amplitude, light_sensitivity, melatonin_rate
ADIPOSE_PARAM_DIM = 5     # lipolysis_rate, lipogenesis_rate, LDL_clearance, HDL_production, FFA_uptake
IMMUNE_PARAM_DIM = 4      # M1_activation, NFkB_sensitivity, vagal_tone_effect, IL6_clearance

PARAM_DIM = (
    METABOLIC_PARAM_DIM + CARDIO_PARAM_DIM + RENAL_PARAM_DIM +
    CIRCADIAN_PARAM_DIM + ADIPOSE_PARAM_DIM + IMMUNE_PARAM_DIM
)  # = 25

# ── Observation dimension ─────────────────────────────────────
OBS_DIM = 16
# indices: 0=G, 1=SBP, 2=DBP, 3=HR, 4=HRV, 5=GFR, 6=Na, 7=K, 8=Osm,
#           9=FFA, 10=LDL, 11=HDL, 12=TG, 13=cortisol, 14=sleep_pressure

# ── Offset constants for array slicing ────────────────────────
_META_OFF = 0
_CARDIO_OFF = METABOLIC_DIM
_RENAL_OFF = _CARDIO_OFF + CARDIO_DIM
_INFL_OFF = _RENAL_OFF + RENAL_DIM
_CIRC_OFF = _INFL_OFF + INFLAMMATION_DIM
_ADIP_OFF = _CIRC_OFF + CIRCADIAN_DIM
_IMMUNE_OFF = _ADIP_OFF + ADIPOSE_DIM


# ====================================================================
# Phase 2 Subsystem Dataclasses (unchanged, with minor extensions)
# ====================================================================

@dataclass
class MetabolicState:
    G: float       # Plasma glucose (mg/dL)
    I: float       # Plasma insulin (μU/mL)
    HGP: float     # Hepatic glucose production (mg/kg/min)
    PGU: float     # Peripheral glucose uptake (mg/kg/min)
    IR: float      # Insulin resistance state (dimensionless)

    def to_array(self) -> np.ndarray:
        return np.array([self.G, self.I, self.HGP, self.PGU, self.IR])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'MetabolicState':
        return cls(G=float(arr[0]), I=float(arr[1]), HGP=float(arr[2]),
                   PGU=float(arr[3]), IR=float(arr[4]))

    def copy(self) -> 'MetabolicState':
        return MetabolicState(self.G, self.I, self.HGP, self.PGU, self.IR)


@dataclass
class CardioState:
    SBP: float     # Systolic blood pressure (mmHg)
    DBP: float     # Diastolic blood pressure (mmHg)
    HR: float      # Heart rate (bpm)
    HRV: float     # Heart rate variability (RMSSD, ms)

    def to_array(self) -> np.ndarray:
        return np.array([self.SBP, self.DBP, self.HR, self.HRV])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'CardioState':
        return cls(SBP=float(arr[0]), DBP=float(arr[1]),
                   HR=float(arr[2]), HRV=float(arr[3]))

    def copy(self) -> 'CardioState':
        return CardioState(self.SBP, self.DBP, self.HR, self.HRV)


@dataclass
class RenalState:
    GFR: float     # Glomerular filtration rate (mL/min/1.73m²)
    Na: float      # Plasma sodium (mEq/L)
    K: float       # Plasma potassium (mEq/L)
    Osm: float     # Plasma osmolality (mOsm/kg)

    def to_array(self) -> np.ndarray:
        return np.array([self.GFR, self.Na, self.K, self.Osm])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'RenalState':
        return cls(GFR=float(arr[0]), Na=float(arr[1]),
                   K=float(arr[2]), Osm=float(arr[3]))

    def copy(self) -> 'RenalState':
        return RenalState(self.GFR, self.Na, self.K, self.Osm)


@dataclass
class InflammatoryState:
    CRP: float     # C-reactive protein (mg/L)

    def to_array(self) -> np.ndarray:
        return np.array([self.CRP])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'InflammatoryState':
        return cls(CRP=float(arr[0]))

    def copy(self) -> 'InflammatoryState':
        return InflammatoryState(self.CRP)


# ====================================================================
# Phase 3 New Subsystem Dataclasses
# ====================================================================

@dataclass
class CircadianState:
    """
    Circadian oscillator at the core-clock gene level.
    Models the CLOCK/BMAL1 ↔ PER/CRY transcriptional-translational feedback loop
    and its hormonal outputs (cortisol, melatonin).
    """
    CLOCK_BMAL1: float    # Activator complex (normalized 0-2)
    PER_CRY: float        # Repressor complex (normalized 0-2)
    cortisol: float       # Serum cortisol (nmol/L), ~140-690 normal range
    melatonin: float      # Serum melatonin (pmol/L), ~0-200 (night peak)
    circadian_phase: float  # Master oscillator phase (radians, 0-2π)
    sleep_pressure: float   # Adenosine-driven sleep need (0-1, 0=awake, 1=asleep)

    def to_array(self) -> np.ndarray:
        return np.array([self.CLOCK_BMAL1, self.PER_CRY, self.cortisol,
                         self.melatonin, self.circadian_phase, self.sleep_pressure])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'CircadianState':
        return cls(
            CLOCK_BMAL1=float(arr[0]), PER_CRY=float(arr[1]),
            cortisol=float(arr[2]), melatonin=float(arr[3]),
            circadian_phase=float(arr[4]), sleep_pressure=float(arr[5]),
        )

    def copy(self) -> 'CircadianState':
        return CircadianState(
            self.CLOCK_BMAL1, self.PER_CRY, self.cortisol,
            self.melatonin, self.circadian_phase, self.sleep_pressure,
        )


@dataclass
class AdiposeLipidState:
    """
    Adipose tissue and lipoprotein metabolism at cellular level.
    - fat_mass: total body fat (kg), slow timescale (days-weeks)
    - FFA: plasma free fatty acids (mmol/L), fast (minutes-hours)
    - LDL, HDL, TG: lipoprotein cholesterol (mg/dL), medium (hours-days)
    """
    fat_mass: float   # Body fat mass (kg)
    FFA: float        # Plasma free fatty acids (mmol/L)
    LDL: float        # LDL cholesterol (mg/dL)
    HDL: float        # HDL cholesterol (mg/dL)
    TG: float         # Triglycerides (mg/dL)

    def to_array(self) -> np.ndarray:
        return np.array([self.fat_mass, self.FFA, self.LDL, self.HDL, self.TG])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'AdiposeLipidState':
        return cls(
            fat_mass=float(arr[0]), FFA=float(arr[1]),
            LDL=float(arr[2]), HDL=float(arr[3]), TG=float(arr[4]),
        )

    def copy(self) -> 'AdiposeLipidState':
        return AdiposeLipidState(self.fat_mass, self.FFA, self.LDL, self.HDL, self.TG)


@dataclass
class ImmuneInflamState:
    """
    Chronic inflammation at the cellular signaling level.
    - IL6_proxy: IL-6 cytokine activity (normalized 0-10)
    - TNFa_proxy: TNF-α activity (normalized 0-10)
    - M1_M2_ratio: Macrophage polarization balance (0=fully M2, 2=fully M1)
    - NFkB_activity: NF-κB pathway activation (normalized 0-1)
    - InflammatoryLoad: composite chronic burden (0-100)
    """
    IL6_proxy: float       # IL-6 activity (normalized)
    TNFa_proxy: float      # TNF-α activity (normalized)
    M1_M2_ratio: float     # Macrophage polarization (0=M2-dominant, 2=M1-dominant)
    NFkB_activity: float   # NF-κB pathway activation (0-1)
    InflammatoryLoad: float  # Composite chronic inflammation burden (0-100)

    def to_array(self) -> np.ndarray:
        return np.array([self.IL6_proxy, self.TNFa_proxy, self.M1_M2_ratio,
                         self.NFkB_activity, self.InflammatoryLoad])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'ImmuneInflamState':
        return cls(
            IL6_proxy=float(arr[0]), TNFa_proxy=float(arr[1]),
            M1_M2_ratio=float(arr[2]), NFkB_activity=float(arr[3]),
            InflammatoryLoad=float(arr[4]),
        )

    def copy(self) -> 'ImmuneInflamState':
        return ImmuneInflamState(
            self.IL6_proxy, self.TNFa_proxy, self.M1_M2_ratio,
            self.NFkB_activity, self.InflammatoryLoad,
        )


# ====================================================================
# Phase 3: Complete Whole-Body Twin State
# ====================================================================

@dataclass
class Phase3TwinState:
    """Complete Phase 3 state: 8 subsystems, 30 dimensions."""
    metabolic: MetabolicState
    cardio: CardioState
    renal: RenalState
    inflammation: InflammatoryState
    circadian: CircadianState
    adipose: AdiposeLipidState
    immune: ImmuneInflamState

    def to_array(self) -> np.ndarray:
        return np.concatenate([
            self.metabolic.to_array(),
            self.cardio.to_array(),
            self.renal.to_array(),
            self.inflammation.to_array(),
            self.circadian.to_array(),
            self.adipose.to_array(),
            self.immune.to_array(),
        ])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Phase3TwinState':
        if len(arr) != PHYSIO_DIM:
            raise ValueError(f"Phase3TwinState requires {PHYSIO_DIM} elements, got {len(arr)}")
        return cls(
            metabolic=MetabolicState.from_array(arr[_META_OFF:_META_OFF+METABOLIC_DIM]),
            cardio=CardioState.from_array(arr[_CARDIO_OFF:_CARDIO_OFF+CARDIO_DIM]),
            renal=RenalState.from_array(arr[_RENAL_OFF:_RENAL_OFF+RENAL_DIM]),
            inflammation=InflammatoryState.from_array(arr[_INFL_OFF:_INFL_OFF+INFLAMMATION_DIM]),
            circadian=CircadianState.from_array(arr[_CIRC_OFF:_CIRC_OFF+CIRCADIAN_DIM]),
            adipose=AdiposeLipidState.from_array(arr[_ADIP_OFF:_ADIP_OFF+ADIPOSE_DIM]),
            immune=ImmuneInflamState.from_array(arr[_IMMUNE_OFF:_IMMUNE_OFF+IMMUNE_DIM]),
        )

    def is_valid(self) -> bool:
        bounds = {
            "G": (20, 600), "I": (0, 500), "HGP": (-5, 15), "PGU": (0, 25), "IR": (0, 20),
            "SBP": (50, 250), "DBP": (30, 150), "HR": (30, 220), "HRV": (5, 200),
            "GFR": (5, 200), "Na": (120, 160), "K": (2.5, 7.0), "Osm": (260, 340),
            "CRP": (0, 100),
            "CLOCK_BMAL1": (0, 2.5), "PER_CRY": (0, 2.5),
            "cortisol": (10, 1000), "melatonin": (0, 300),
            "circadian_phase": (0, 6.29), "sleep_pressure": (0, 1),
            "fat_mass": (2, 100), "FFA": (0.1, 2.0),
            "LDL": (20, 300), "HDL": (10, 120), "TG": (20, 800),
            "IL6_proxy": (0, 10), "TNFa_proxy": (0, 10),
            "M1_M2_ratio": (0, 2), "NFkB_activity": (0, 1),
            "InflammatoryLoad": (0, 100),
        }
        arr = self.to_array()
        violations = []
        for i, (name, (lo, hi)) in enumerate(bounds.items()):
            if not (lo <= arr[i] <= hi + 1e-8):
                violations.append(f"{name}={arr[i]:.4f} ∉ [{lo}, {hi}]")
        if violations:
            return False
        return True

    def validate_or_raise(self) -> None:
        """Raise ValueError if any state variable is out of physiological bounds."""
        bounds = {
            "G": (20, 600), "I": (0, 500), "HGP": (-5, 15), "PGU": (0, 25), "IR": (0, 20),
            "SBP": (50, 250), "DBP": (30, 150), "HR": (30, 220), "HRV": (5, 200),
            "GFR": (5, 200), "Na": (120, 160), "K": (2.5, 7.0), "Osm": (260, 340),
            "CRP": (0, 100),
            "CLOCK_BMAL1": (0, 2.5), "PER_CRY": (0, 2.5),
            "cortisol": (10, 1000), "melatonin": (0, 300),
            "circadian_phase": (0, 6.29), "sleep_pressure": (0, 1),
            "fat_mass": (2, 100), "FFA": (0.1, 2.0),
            "LDL": (20, 300), "HDL": (10, 120), "TG": (20, 800),
            "IL6_proxy": (0, 10), "TNFa_proxy": (0, 10),
            "M1_M2_ratio": (0, 2), "NFkB_activity": (0, 1),
            "InflammatoryLoad": (0, 100),
        }
        arr = self.to_array()
        for i, (name, (lo, hi)) in enumerate(bounds.items()):
            if not (lo <= arr[i] <= hi + 1e-8):
                raise ValueError(
                    f"Physiological state invalid: {name}={arr[i]:.4f} "
                    f"out of bounds [{lo}, {hi}]"
                )

    @property
    def metabolic_arr(self) -> np.ndarray:
        return self.to_array()[_META_OFF:_META_OFF+METABOLIC_DIM]

    @property
    def cardio_arr(self) -> np.ndarray:
        return self.to_array()[_CARDIO_OFF:_CARDIO_OFF+CARDIO_DIM]

    @property
    def renal_arr(self) -> np.ndarray:
        return self.to_array()[_RENAL_OFF:_RENAL_OFF+RENAL_DIM]

    @property
    def circadian_arr(self) -> np.ndarray:
        return self.to_array()[_CIRC_OFF:_CIRC_OFF+CIRCADIAN_DIM]

    @property
    def adipose_arr(self) -> np.ndarray:
        return self.to_array()[_ADIP_OFF:_ADIP_OFF+ADIPOSE_DIM]

    @property
    def immune_arr(self) -> np.ndarray:
        return self.to_array()[_IMMUNE_OFF:_IMMUNE_OFF+IMMUNE_DIM]


# Legacy alias for backward compat in imports
FullTwinState = Phase3TwinState


# ====================================================================
# Legacy support classes (unchanged)
# ====================================================================

@dataclass
class TwinStateAugmented:
    physiological_state: np.ndarray
    parameters: np.ndarray

    @property
    def state(self) -> np.ndarray:
        return np.concatenate([self.physiological_state, self.parameters])

    @classmethod
    def from_flat(cls, flat_state: np.ndarray, physio_dim: int = PHYSIO_DIM) -> 'TwinStateAugmented':
        physio = flat_state[:physio_dim]
        params = flat_state[physio_dim:]
        return cls(physio, params)


class StateEstimator(ABC):
    @abstractmethod
    def predict(self, u: np.ndarray) -> None:
        pass
    @abstractmethod
    def update(self, y: np.ndarray) -> None:
        pass
    @abstractmethod
    def get_state(self) -> np.ndarray:
        pass
    @abstractmethod
    def get_state_covariance(self) -> np.ndarray:
        pass


class StateDynamics:
    @staticmethod
    def glucose_insulin_dynamics(state, inputs, params, dt=1.0):
        from .dynamics import compute_metabolic_dynamics
        if hasattr(state, 'G'):
            meta = MetabolicState(state.G, state.I, state.HGP, state.PGU, state.IR)
        else:
            meta = MetabolicState.from_array(state)
        result = compute_metabolic_dynamics(meta, inputs, params, dt)
        return result

    @staticmethod
    def observation_model(state):
        return np.array([state.G if hasattr(state, 'G') else state[0]])


# ── State names for serialization ─────────────────────────────
STATE_NAMES_PHASE3 = [
    # Metabolic
    "G", "I", "HGP", "PGU", "IR",
    # Cardiovascular
    "SBP", "DBP", "HR", "HRV",
    # Renal
    "GFR", "Na", "K", "Osm",
    # Inflammation (legacy)
    "CRP",
    # Circadian
    "CLOCK_BMAL1", "PER_CRY", "cortisol", "melatonin",
    "circadian_phase", "sleep_pressure",
    # Adipose-Lipid
    "fat_mass", "FFA", "LDL", "HDL", "TG",
    # Immune-Inflammatory
    "IL6_proxy", "TNFa_proxy", "M1_M2_ratio",
    "NFkB_activity", "InflammatoryLoad",
]
