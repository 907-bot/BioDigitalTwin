"""
Phase 3: Virtual Cohort Engine.

Generates synthetic patient populations from posterior distributions
for in-silico trials, treatment simulation, and population modeling.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
from .core import PersonalizationEngine, PHYSIO_DIM, PARAM_DIM
from .priors import PRIORS, PARAMETER_NAMES


@dataclass
class VirtualPatient:
    """A single synthetic patient with full twin state and parameters."""
    patient_id: str
    state: np.ndarray        # 30-dim physiological state
    parameters: np.ndarray   # 25-dim parameters
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "patient_id": self.patient_id,
            "state": self.state.tolist(),
            "parameters": self.parameters.tolist(),
            "metadata": self.metadata,
        }


class VirtualCohortEngine:
    """
    Generate virtual patient cohorts from prior/posterior distributions.

    Methods:
      - sample_from_priors: sample directly from population priors
      - sample_from_posterior: sample around a learned posterior mean/cov
      - generate_discrete_subgroups: create multiple subgroups with varying params
      - simulate_treatment: apply counterfactual to entire cohort
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self._patients: List[VirtualPatient] = []

    def sample_from_priors(
        self,
        n_patients: int = 1000,
        age_range: Tuple[float, float] = (20, 80),
        bmi_range: Tuple[float, float] = (18, 45),
        diabetes_rate: float = 0.15,
        hypertension_rate: float = 0.30,
    ) -> List[VirtualPatient]:
        """
        Generate n_patients by sampling from population priors
        with demographic heterogeneity.
        """
        patients = []
        for i in range(n_patients):
            age = self.rng.uniform(age_range[0], age_range[1])
            bmi = self.rng.uniform(bmi_range[0], bmi_range[1])
            sex = self.rng.choice(["male", "female"])
            has_dm = self.rng.random() < diabetes_rate
            has_htn = self.rng.random() < hypertension_rate
            has_ckd = self.rng.random() < 0.10

            params = np.array([p.sample() for p in PRIORS])

            self._apply_subgroup_adjustment(params, age, bmi, sex, has_dm, has_htn, has_ckd)

            state = self._sample_initial_state(age, bmi, params)

            patient = VirtualPatient(
                patient_id=f"V{i:06d}",
                state=state,
                parameters=params,
                metadata={
                    "age": float(age),
                    "bmi": float(bmi),
                    "sex": sex,
                    "diabetes": has_dm,
                    "hypertension": has_htn,
                    "ckd": has_ckd,
                },
            )
            patients.append(patient)

        self._patients = patients
        return patients

    def _apply_subgroup_adjustment(
        self, params: np.ndarray,
        age: float, bmi: float, sex: str,
        has_dm: bool, has_htn: bool, has_ckd: bool,
    ) -> None:
        """Apply subgroup adjustments to sampled parameters."""
        from .priors import SUBGROUP_ADJUSTMENTS
        adjustments = {}
        if age > 60:
            adjustments.update(SUBGROUP_ADJUSTMENTS.get("age_gt_60", {}))
        if bmi > 30:
            adjustments.update(SUBGROUP_ADJUSTMENTS.get("obese_bmi_gt_30", {}))
        if has_dm:
            adjustments.update(SUBGROUP_ADJUSTMENTS.get("diabetes_t2", {}))
        if has_htn:
            adjustments.update(SUBGROUP_ADJUSTMENTS.get("hypertension", {}))
        if has_ckd:
            adjustments.update(SUBGROUP_ADJUSTMENTS.get("ckd_stage3", {}))
        if sex == "female":
            adjustments.update(SUBGROUP_ADJUSTMENTS.get("female", {}))

        for name, (scale_lo, scale_hi) in adjustments.items():
            if name in PARAMETER_NAMES:
                idx = PARAMETER_NAMES.index(name)
                scale = self.rng.uniform(scale_lo, scale_hi)
                params[idx] *= scale

    def _sample_initial_state(
        self, age: float, bmi: float, params: np.ndarray,
    ) -> np.ndarray:
        """Generate a physiologically plausible initial state."""
        state = np.zeros(PHYSIO_DIM)
        si = params[0]
        hgp = params[1]

        # Metabolic
        state[0] = self.rng.normal(90, 10)         # G
        state[1] = self.rng.normal(5, 2)           # I
        state[2] = hgp                              # HGP
        state[3] = self.rng.normal(5, 1)           # PGU
        state[4] = max(0, 1.0 / max(si, 0.001) - 5)  # IR

        # Cardiovascular
        bp_base = 110 + age * 0.2 + bmi * 0.3
        state[5] = self.rng.normal(bp_base, 10)     # SBP
        state[6] = self.rng.normal(bp_base * 0.65, 5)  # DBP
        state[7] = self.rng.normal(70, 8)            # HR
        state[8] = self.rng.normal(50 - age * 0.15, 10)  # HRV

        # Renal
        gfr_base = 120 - age * 0.5
        state[9] = max(5, self.rng.normal(gfr_base, 10))   # GFR
        state[10] = self.rng.normal(140, 3)                 # Na
        state[11] = self.rng.normal(4.2, 0.3)               # K
        state[12] = self.rng.normal(290, 5)                 # Osm

        # Inflammation (CRP)
        crp_base = 1.0 + bmi * 0.05 + age * 0.01
        state[13] = max(0.1, self.rng.normal(crp_base, 1))  # CRP

        # Circadian
        phase = self.rng.uniform(0, 2 * np.pi)
        state[14] = 1.0 + 0.5 * np.cos(phase)            # CLOCK_BMAL1
        state[15] = 1.0 + 0.5 * np.sin(phase)            # PER_CRY
        state[16] = 300 + 150 * np.cos(phase - np.pi)     # cortisol
        state[17] = max(0, 80 * (-np.cos(phase + 1.0)))   # melatonin
        state[18] = phase                                 # circadian_phase
        state[19] = self.rng.uniform(0.2, 0.6)            # sleep_pressure

        # Adipose
        fat_mass_base = 15 + (bmi - 22) * 0.8
        state[20] = max(2, self.rng.normal(fat_mass_base, 3))  # fat_mass
        state[21] = self.rng.normal(0.4 + bmi * 0.005, 0.1)    # FFA
        state[22] = self.rng.normal(110 + age * 0.3, 20)       # LDL
        state[23] = self.rng.normal(55 - bmi * 0.2, 8)         # HDL
        state[24] = self.rng.normal(120 + bmi * 2, 30)         # TG

        # Immune
        infl_base = 5 + bmi * 0.2 + age * 0.05
        state[25] = self.rng.exponential(1.0)                   # IL6_proxy
        state[26] = self.rng.exponential(0.5)                   # TNFa_proxy
        state[27] = 0.3 + bmi * 0.01                            # M1_M2_ratio
        state[28] = self.rng.beta(2, 5)                         # NFkB_activity
        state[29] = infl_base + self.rng.normal(0, 5)           # InflammatoryLoad

        return np.clip(state, 0, None)

    def sample_from_posterior(
        self,
        base_state: np.ndarray,
        base_params: np.ndarray,
        param_cov: np.ndarray,
        n_patients: int = 100,
        noise_scale: float = 0.05,
    ) -> List[VirtualPatient]:
        """Generate virtual patients from a learned posterior distribution."""
        patients = []
        for i in range(n_patients):
            param_noise = self.rng.multivariate_normal(
                np.zeros(PARAM_DIM), param_cov * noise_scale
            )
            state_noise = np.random.randn(PHYSIO_DIM) * 0.02 * np.abs(base_state + 1)
            params = base_params + param_noise
            state = base_state + state_noise

            patient = VirtualPatient(
                patient_id=f"P{i:06d}",
                state=np.clip(state, 0, None),
                parameters=np.clip(params, 0, None),
                metadata={"source": "posterior", "noise_scale": noise_scale},
            )
            patients.append(patient)
        return patients

    def simulate_treatment(
        self,
        cohort: List[VirtualPatient],
        treatment_fn: Callable[[VirtualPatient], VirtualPatient],
    ) -> List[VirtualPatient]:
        """Apply a counterfactual treatment function to the entire cohort."""
        return [treatment_fn(p.copy()) for p in cohort]

    @property
    def patients(self) -> List[VirtualPatient]:
        return self._patients

    def summary_stats(self) -> Dict:
        if not self._patients:
            return {}
        states = np.array([p.state for p in self._patients])
        params = np.array([p.parameters for p in self._patients])
        return {
            "n_patients": len(self._patients),
            "state_means": states.mean(axis=0).tolist(),
            "state_stds": states.std(axis=0).tolist(),
            "param_means": params.mean(axis=0).tolist(),
            "param_stds": params.std(axis=0).tolist(),
        }
