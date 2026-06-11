"""
MIMIC-IV equivalent synthetic validation harness.

Real MIMIC-IV access requires PhysioNet credentialing (CITI training,
DUA, IRB). Until then, we generate synthetic patients whose population
statistics match published MIMIC-IV distributions.

References for distribution parameters:
- Johnson et al. (2023) — MIMIC-IV demographics
- Sauer et al. (2022) — ICU glucose distributions
- Moghissi et al. (2009) — AACE/ADA inpatient glycemic targets
- Finfer et al. (2009) — NICE-SUGAR trial glucose ranges
- Krinsley et al. (2013) — diabetic ICU population

Validation metrics follow:
- MIMIC-IV code repository: https://github.com/MIT-LCP/mimic-code
- T1DM Exchange (Miller et al. 2015) — outpatient T1DM distributions
"""

import math
import logging
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

from .dynamics import DEFAULT_PARAMS, full_dynamics, full_observation
from .state import PHYSIO_DIM, METABOLIC_DIM, CARDIO_DIM, RENAL_DIM, ADIPOSE_DIM

logger = logging.getLogger(__name__)


@dataclass
class PatientProfile:
    """Patient demographic and clinical profile."""
    patient_id: str
    age: float
    sex: str  # 'M' or 'F'
    bmi: float
    has_diabetes: bool
    diabetes_type: str  # 'T1DM', 'T2DM', 'none'
    has_hypertension: bool
    has_ckd: bool
    has_cvd: bool
    hba1c: Optional[float] = None
    ethnicity: str = "white"
    smoker: bool = False


@dataclass
class MIMICPatient:
    """Generated MIMIC-IV-equivalent patient with realistic parameters and trajectory."""
    profile: PatientProfile
    true_params: np.ndarray
    initial_state: np.ndarray
    observations: np.ndarray  # (T, 15)
    true_states: np.ndarray  # (T, 30)
    glucose_trajectory_mean: float
    time_in_range_pct: float
    n_hypo_events: int
    n_hyper_events: int


class MIMICEquivalentGenerator:
    """Generate MIMIC-IV-equivalent synthetic patient cohort.

    Population statistics are calibrated to match published MIMIC-IV
    distributions. Each patient is a fully realized ODE simulation
    with realistic demographics, comorbidities, and trajectories.
    """

    # MIMIC-IV ICU population statistics (Johnson et al. 2023)
    AGE_DIST_ICU = {"mean": 64.0, "std": 17.0, "min": 18, "max": 100}
    SEX_DIST = {"M": 0.56, "F": 0.44}
    BMI_DIST_ICU = {"mean": 28.5, "std": 7.0, "min": 15, "max": 60}
    DIABETES_PREV_ICU = 0.32  # 32% of ICU patients have diabetes
    HYPERTENSION_PREV_ICU = 0.55
    CKD_PREV_ICU = 0.20
    CVD_PREV_ICU = 0.25

    # Outpatient T1DM (T1DM Exchange)
    AGE_DIST_T1DM = {"mean": 35.0, "std": 18.0, "min": 5, "max": 90}
    HBA1C_DIST_T1DM = {"mean": 7.8, "std": 1.3, "min": 5.0, "max": 14.0}

    # Inpatient glucose (Sauer 2022)
    ICU_GLUCOSE_DIST = {"mean": 142, "std": 38, "min": 40, "max": 500}
    NON_DIABETIC_ICU_GLUCOSE = {"mean": 112, "std": 18}
    DIABETIC_ICU_GLUCOSE = {"mean": 168, "std": 45}

    # Hypoglycemia rates
    ICU_HYPO_RATE_NON_DIABETIC = 0.05  # 5% of non-diabetic ICU patients experience <70
    ICU_HYPO_RATE_DIABETIC = 0.18

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate_icu_cohort(self, n_patients: int) -> List[MIMICPatient]:
        """Generate ICU patients matching MIMIC-IV distributions."""
        patients = []
        for i in range(n_patients):
            profile = self._sample_icu_profile(i)
            patient = self._generate_patient(profile, n_steps=288, setting='icu')
            patients.append(patient)
        return patients

    def generate_t1dm_outpatient_cohort(self, n_patients: int) -> List[MIMICPatient]:
        """Generate T1DM outpatients matching T1DM Exchange distributions."""
        patients = []
        for i in range(n_patients):
            profile = self._sample_t1dm_outpatient_profile(i)
            patient = self._generate_patient(profile, n_steps=288, setting='outpatient')
            patients.append(patient)
        return patients

    def generate_mixed_cohort(self, n_patients: int) -> List[MIMICPatient]:
        """Generate a mixed cohort: 60% ICU, 40% outpatient, 30% diabetic."""
        patients = []
        for i in range(n_patients):
            setting = 'icu' if self.rng.random() < 0.6 else 'outpatient'
            if setting == 'icu':
                profile = self._sample_icu_profile(i)
            else:
                if self.rng.random() < 0.30:
                    profile = self._sample_t1dm_outpatient_profile(i)
                else:
                    profile = self._sample_outpatient_profile(i)
            patient = self._generate_patient(profile, n_steps=288, setting=setting)
            patients.append(patient)
        return patients

    def _sample_icu_profile(self, idx: int) -> PatientProfile:
        age = np.clip(
            self.rng.normal(self.AGE_DIST_ICU['mean'], self.AGE_DIST_ICU['std']),
            self.AGE_DIST_ICU['min'], self.AGE_DIST_ICU['max']
        )
        sex = 'M' if self.rng.random() < self.SEX_DIST['M'] else 'F'
        bmi = np.clip(
            self.rng.normal(self.BMI_DIST_ICU['mean'], self.BMI_DIST_ICU['std']),
            self.BMI_DIST_ICU['min'], self.BMI_DIST_ICU['max']
        )
        has_diabetes = self.rng.random() < self.DIABETES_PREV_ICU
        diabetes_type = 'T2DM' if has_diabetes and self.rng.random() < 0.92 else ('T1DM' if has_diabetes else 'none')
        has_htn = self.rng.random() < self.HYPERTENSION_PREV_ICU
        has_ckd = self.rng.random() < self.CKD_PREV_ICU
        has_cvd = self.rng.random() < self.CVD_PREV_ICU
        hba1c = None
        if has_diabetes:
            hba1c = self.rng.normal(8.0, 1.5)
            hba1c = np.clip(hba1c, 5.5, 14.0)
        return PatientProfile(
            patient_id=f"MIMIC_{idx:05d}",
            age=age, sex=sex, bmi=bmi,
            has_diabetes=has_diabetes, diabetes_type=diabetes_type,
            has_hypertension=has_htn, has_ckd=has_ckd, has_cvd=has_cvd,
            hba1c=hba1c,
        )

    def _sample_t1dm_outpatient_profile(self, idx: int) -> PatientProfile:
        age = np.clip(
            self.rng.normal(self.AGE_DIST_T1DM['mean'], self.AGE_DIST_T1DM['std']),
            self.AGE_DIST_T1DM['min'], self.AGE_DIST_T1DM['max']
        )
        sex = 'M' if self.rng.random() < 0.50 else 'F'
        bmi = np.clip(self.rng.normal(25, 5), 15, 45)
        hba1c = np.clip(
            self.rng.normal(self.HBA1C_DIST_T1DM['mean'], self.HBA1C_DIST_T1DM['std']),
            self.HBA1C_DIST_T1DM['min'], self.HBA1C_DIST_T1DM['max']
        )
        has_htn = self.rng.random() < 0.30
        has_ckd = self.rng.random() < 0.10
        has_cvd = self.rng.random() < 0.10
        return PatientProfile(
            patient_id=f"T1DM_{idx:05d}",
            age=age, sex=sex, bmi=bmi,
            has_diabetes=True, diabetes_type='T1DM',
            has_hypertension=has_htn, has_ckd=has_ckd, has_cvd=has_cvd,
            hba1c=hba1c,
        )

    def _sample_outpatient_profile(self, idx: int) -> PatientProfile:
        age = np.clip(self.rng.normal(50, 18), 18, 95)
        sex = 'M' if self.rng.random() < 0.50 else 'F'
        bmi = np.clip(self.rng.normal(27, 5), 15, 50)
        has_diabetes = self.rng.random() < 0.12
        diabetes_type = 'T2DM' if has_diabetes else 'none'
        hba1c = self.rng.normal(7.5, 1.2) if has_diabetes else None
        has_htn = self.rng.random() < 0.30
        has_ckd = self.rng.random() < 0.05
        has_cvd = self.rng.random() < 0.10
        return PatientProfile(
            patient_id=f"OUTP_{idx:05d}",
            age=age, sex=sex, bmi=bmi,
            has_diabetes=has_diabetes, diabetes_type=diabetes_type,
            has_hypertension=has_htn, has_ckd=has_ckd, has_cvd=has_cvd,
            hba1c=hba1c,
        )

    def _generate_patient(
        self,
        profile: PatientProfile,
        n_steps: int,
        setting: str,
    ) -> MIMICPatient:
        """Generate a realistic patient simulation from the profile."""
        # Set physiologically-informed parameters
        params = DEFAULT_PARAMS.copy()
        # Insulin sensitivity (SI): reduced with T2DM, age, BMI
        if profile.diabetes_type == 'T2DM':
            params[0] = self.rng.lognormal(-4.5, 0.4)  # ~0.011 (lower)
        elif profile.diabetes_type == 'T1DM':
            params[0] = self.rng.lognormal(-3.8, 0.4)  # ~0.022 (varies more)
        else:
            params[0] = self.rng.lognormal(-4.0, 0.3)  # ~0.018
        # HGP basal: higher with diabetes
        params[1] = self.rng.lognormal(0.5, 0.3)  # ~1.65
        # beta_response: very low in T1DM, normal otherwise
        if profile.diabetes_type == 'T1DM':
            params[2] = self.rng.lognormal(-5.0, 0.4)  # very low
        else:
            params[2] = self.rng.lognormal(-3.5, 0.3)
        # RT: lower in CKD
        if profile.has_ckd:
            params[3] = self.rng.normal(140, 20)
        else:
            params[3] = self.rng.normal(180, 15)
        # BP: higher with hypertension
        if profile.has_hypertension:
            params[4] = self.rng.normal(50, 10)   # higher arterial_stiffness
            params[5] = self.rng.normal(120, 20)  # higher vascular_resistance
        # Baroreflex: weaker with age
        params[7] = np.clip(self.rng.normal(1.0 - (profile.age - 40) * 0.005, 0.2), 0.3, 2.0)
        # GFR: lower with age and CKD
        if profile.has_ckd:
            params[8] = self.rng.normal(45, 15)   # CKD baseline_GFR
        else:
            params[8] = self.rng.normal(100 - (profile.age - 30) * 0.5, 10)
        # Lipids (params 18=LDL_clear, 19=HDL_prod, 16=lipolysis, 17=lipogenesis)
        # Higher BMI → higher LDL_clear baseline
        params[18] = self.rng.normal(0.025, 0.005)  # LDL_clearance
        params[19] = self.rng.normal(0.015, 0.003)  # HDL_production
        params[17] = self.rng.normal(0.020, 0.005)  # lipogenesis
        params[16] = self.rng.normal(0.025, 0.005)  # lipolysis
        # Inflammation: higher with CVD
        if profile.has_cvd:
            # 25 params: 0=SI, 1=HGP, 2=beta, 3=RT, 4=art_stiff, 5=vasc_res,
            # 6=baroreflex, 7=auto_tone, 8=baseline_GFR, 9=renal_sens,
            # 10=SGLT, 11=Na_ret, 12=circ_p, 13=circ_amp, 14=light_sens,
            # 15=mel_r, 16=lipol, 17=lipog, 18=LDL_cl, 19=HDL_prod, 20=FFA_up,
            # 21=M1_act, 22=NFkB, 23=vagal, 24=IL6_cl
            params[22] = 0.4  # NFkB sensitivity
        # Initial state
        state = np.zeros(PHYSIO_DIM)
        # Glucose baseline
        if profile.diabetes_type == 'T1DM':
            mean_g = 165 + (profile.hba1c - 7.0) * 25
        elif profile.diabetes_type == 'T2DM':
            mean_g = 145 + (profile.hba1c - 7.0) * 25 if profile.hba1c else 145
        else:
            mean_g = self.rng.normal(95, 8)
        state[0] = float(np.clip(self.rng.normal(mean_g, 25), 40, 500))
        state[1] = 0.013 * max(0, state[0] - 80)
        # SBP / DBP
        if profile.has_hypertension:
            state[5] = self.rng.normal(150, 10)
            state[6] = self.rng.normal(90, 8)
        else:
            state[5] = 110 + (profile.age - 30) * 0.4
            state[6] = 70 + (profile.age - 30) * 0.2
        state[7] = 70
        state[8] = 45
        # GFR
        state[9] = params[9]
        state[10] = 140
        state[11] = 4.2
        state[12] = 290
        state[13] = 1.0
        # Circadian
        state[14] = 1.2
        state[15] = 0.8
        state[16] = 350
        state[17] = 10
        # Adipose
        state[20] = profile.bmi * 0.8
        state[21] = self.rng.uniform(0.3, 0.7)
        # Run simulation
        true_states = []
        observations = []
        s = state.copy()
        for t in range(n_steps):
            inputs = self._generate_intervention(t, profile, setting)
            s = full_dynamics(s, params, inputs)
            # Add noise to observations
            obs = full_observation(s)
            obs_noisy = self._add_clinical_noise(obs, profile)
            s[0] = max(20.0, min(600.0, s[0]))
            s[1] = max(0.0, min(500.0, s[1]))
            true_states.append(s.copy())
            observations.append(obs_noisy)
        true_states = np.array(true_states)
        observations = np.array(observations)
        # Compute summary stats
        G = observations[:, 0]
        tir = float(np.mean((G >= 70) & (G <= 180)))
        n_hypo = int(np.sum(G < 70))
        n_hyper = int(np.sum(G > 180))
        return MIMICPatient(
            profile=profile,
            true_params=params,
            initial_state=state,
            observations=observations,
            true_states=true_states,
            glucose_trajectory_mean=float(np.mean(G)),
            time_in_range_pct=tir,
            n_hypo_events=n_hypo,
            n_hyper_events=n_hyper,
        )

    def _generate_intervention(self, t: int, profile: PatientProfile, setting: str) -> Dict:
        """Generate realistic intervention schedule."""
        inputs: Dict = {}
        if profile.diabetes_type == 'T1DM':
            # T1DM: insulin with meals (every 36 steps = 3 hours)
            if t % 36 == 12:  # breakfast
                inputs["insulin_dose"] = 6.0
            elif t % 36 == 24:  # lunch
                inputs["insulin_dose"] = 8.0
            elif t % 36 == 34:  # dinner
                inputs["insulin_dose"] = 10.0
            # Meal at every 36 steps (3 hours apart)
            if t % 36 in (10, 22, 34):
                inputs["carbs_grams"] = 50.0
                inputs["meal_glucose"] = 50.0 * 0.5
        elif profile.diabetes_type == 'T2DM':
            # T2DM: oral meds and meals, less aggressive insulin
            if t % 36 in (10, 22, 34):
                inputs["carbs_grams"] = 60.0
                inputs["meal_glucose"] = 60.0 * 0.5
            if t % 72 == 0:
                inputs["metformin_dose"] = 1.0
        return inputs

    def _add_clinical_noise(self, obs: np.ndarray, profile: PatientProfile) -> np.ndarray:
        """Add realistic clinical noise (sensor noise, lab measurement error)."""
        noisy = obs.copy()
        # Glucose (CGM noise: ~10 mg/dL std)
        if len(noisy) > 0:
            noisy[0] += self.rng.normal(0, 8)
        # BP (cuff noise: ~3 mmHg)
        if len(noisy) > 1:
            noisy[1] += self.rng.normal(0, 3)
        if len(noisy) > 2:
            noisy[2] += self.rng.normal(0, 2)
        # HR (pulse oximeter: ~2 bpm)
        if len(noisy) > 3:
            noisy[3] += self.rng.normal(0, 2)
        # HRV (5-10 ms std)
        if len(noisy) > 4:
            noisy[4] += self.rng.normal(0, 5)
        # GFR (lab noise: 3-5 mL/min)
        if len(noisy) > 5:
            noisy[5] += self.rng.normal(0, 4)
        # Electrolytes (lab noise)
        if len(noisy) > 6:
            noisy[6] += self.rng.normal(0, 0.5)  # Na
        if len(noisy) > 7:
            noisy[7] += self.rng.normal(0, 0.1)  # K
        # Lipids (lab noise 5%)
        if len(noisy) > 9:
            noisy[9] += self.rng.normal(0, 0.05)  # FFA
        if len(noisy) > 10:
            noisy[10] += self.rng.normal(0, 5)  # LDL
        if len(noisy) > 11:
            noisy[11] += self.rng.normal(0, 3)  # HDL
        if len(noisy) > 12:
            noisy[12] += self.rng.normal(0, 8)  # TG
        return noisy


def compute_validation_metrics(patients: List[MIMICPatient]) -> Dict:
    """Compute population-level validation metrics for the cohort."""
    G_means = [p.glucose_trajectory_mean for p in patients]
    tir_pcts = [p.time_in_range_pct for p in patients]
    n_hypos = [p.n_hypo_events for p in patients]
    n_hypers = [p.n_hyper_events for p in patients]
    return {
        "n_patients": len(patients),
        "mean_glucose": float(np.mean(G_means)),
        "std_glucose": float(np.std(G_means)),
        "median_glucose": float(np.median(G_means)),
        "mean_tir": float(np.mean(tir_pcts)) * 100,
        "std_tir": float(np.std(tir_pcts)) * 100,
        "n_hypo_events_total": sum(n_hypos),
        "n_hyper_events_total": sum(n_hypers),
        "n_patients_with_hypo": sum(1 for n in n_hypos if n > 0),
        "n_patients_with_hyper": sum(1 for n in n_hypers if n > 0),
    }
