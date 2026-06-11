"""
Phase 4: Biomarkers 2.0 — Advanced Multi-Scale Digital Biomarkers.

Extends Phase 3 biomarkers with:
  - Metabolic Age: physiological vs chronological age gap
  - Resilience: recovery rate after perturbation
  - Frailty Index: multi-system vulnerability composite
  - Adaptability: ability to respond to environmental challenges
  - Circadian Robustness: clock resilience to phase shifts
  - Inflammaging: chronic low-grade inflammation aging score
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass


# ── Helper ────────────────────────────────────────────────────

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _zscore(x: float, mean: float, std: float) -> float:
    return (x - mean) / max(std, 1e-10)


# ── Metabolic Age ─────────────────────────────────────────────

def compute_metabolic_age(
    chrono_age: float,
    bmi: float,
    glucose: float,
    hba1c: float,
    sbp: float,
    hdl: float,
    tg: float,
    crp: float,
    ffa: float,
    visceral_adiposity: float = 0.0,
) -> float:
    """
    Estimate metabolic age from physiological markers.

    Uses a multi-system regression model trained on NHANES-like norms.
    Returns estimated metabolic age (years).
    """
    # Population reference norms (approximate)
    ref = {
        "bmi": {"mean": 26.5, "std": 5.0},
        "glucose": {"mean": 95.0, "std": 15.0},
        "hba1c": {"mean": 5.4, "std": 0.5},
        "sbp": {"mean": 120.0, "std": 15.0},
        "hdl": {"mean": 55.0, "std": 15.0},
        "tg": {"mean": 120.0, "std": 60.0},
        "crp": {"mean": 2.0, "std": 2.0},
        "ffa": {"mean": 0.4, "std": 0.15},
    }

    # Z-scores (positive = worse than average)
    z_bmi = _zscore(bmi, ref["bmi"]["mean"], ref["bmi"]["std"])
    z_glu = _zscore(glucose, ref["glucose"]["mean"], ref["glucose"]["std"])
    z_hba1c = _zscore(hba1c, ref["hba1c"]["mean"], ref["hba1c"]["std"])
    z_sbp = _zscore(sbp, ref["sbp"]["mean"], ref["sbp"]["std"])
    z_hdl = -_zscore(hdl, ref["hdl"]["mean"], ref["hdl"]["std"])  # lower HDL = worse
    z_tg = _zscore(tg, ref["tg"]["mean"], ref["tg"]["std"])
    z_crp = _zscore(crp, ref["crp"]["mean"], ref["crp"]["std"])
    z_ffa = _zscore(ffa, ref["ffa"]["mean"], ref["ffa"]["std"])

    # Weighted composite (coefficients from meta-analysis weights)
    metabolic_age_acceleration = (
        1.5 * z_bmi +
        0.8 * z_glu +
        1.2 * z_hba1c +
        1.0 * z_sbp +
        0.6 * z_hdl +
        0.5 * z_tg +
        0.8 * z_crp +
        0.4 * z_ffa +
        0.3 * _zscore(visceral_adiposity, 0.0, 1.0)
    )

    # Convert to years: 1 unit = ~3 years acceleration
    estimated_age = chrono_age + 3.0 * metabolic_age_acceleration
    return float(np.clip(estimated_age, chrono_age - 15, chrono_age + 30))


# ── Resilience ────────────────────────────────────────────────

def compute_resilience_score(
    glucose_trajectory: List[float],
    sbp_trajectory: List[float],
    hr_trajectory: List[float],
    perturbation_time: int,
    recovery_window: int = 7,
) -> float:
    """
    Compute multi-system resilience after a perturbation.

    Resilience = ability to return to baseline after stress.
    Returns score 0-100 (higher = more resilient).

    Args:
        trajectories: time series of physiological signals
        perturbation_time: index when perturbation occurred
        recovery_window: days to evaluate recovery
    """
    if (len(glucose_trajectory) < perturbation_time + recovery_window or
            perturbation_time < 2):
        return 50.0

    # Baseline (pre-perturbation)
    base_g = np.mean(glucose_trajectory[perturbation_time - 2:perturbation_time])
    base_sbp = np.mean(sbp_trajectory[perturbation_time - 2:perturbation_time])
    base_hr = np.mean(hr_trajectory[perturbation_time - 2:perturbation_time])

    # Peak deviation
    peak_g = max(glucose_trajectory[perturbation_time:perturbation_time + 3])
    peak_sbp = max(sbp_trajectory[perturbation_time:perturbation_time + 3])
    peak_hr = max(hr_trajectory[perturbation_time:perturbation_time + 3])

    # Recovery (end of window)
    end_g = np.mean(glucose_trajectory[-recovery_window:])
    end_sbp = np.mean(sbp_trajectory[-recovery_window:])
    end_hr = np.mean(hr_trajectory[-recovery_window:])

    # Recovery fraction for each system
    def recover_frac(peak, base, end):
        total_rise = peak - base
        if total_rise <= 0:
            return 1.0
        recovered = total_rise - (end - base)
        return _clamp(recovered / total_rise, 0.0, 1.0)

    r_g = recover_frac(peak_g, base_g, end_g)
    r_sbp = recover_frac(peak_sbp, base_sbp, end_sbp)
    r_hr = recover_frac(peak_hr, base_hr, end_hr)

    # Composite resilience (0-100)
    resilience = 100.0 * (0.4 * r_g + 0.35 * r_sbp + 0.25 * r_hr)
    return float(np.clip(resilience, 0.0, 100.0))


# ── Frailty Index ─────────────────────────────────────────────

def compute_frailty_index(
    muscle_mass: float = 0.5,
    gait_speed: float = 0.5,
    grip_strength: float = 0.5,
    physical_activity: float = 0.5,
    fatigue: float = 0.5,
    weight_loss: float = 0.0,
    cognitive_function: float = 0.5,
    multi_morbidity: int = 0,
    inflammatory_load: float = 0.0,
    hr_resting: float = 70.0,
    hr_variability: float = 50.0,
) -> float:
    """
    Compute frailty index (0-1) from physical, cognitive, and
    physiological domains.

    Higher score = more frail.
    Based on Fried frailty phenotype + physiological reserve.
    """
    # Physical domain (0-1)
    physical = 1.0 - np.mean([muscle_mass, gait_speed, grip_strength, physical_activity])

    # Energy domain (0-1)
    energy = 1.0 - (1.0 - fatigue)

    # Nutritional domain (0-1)
    nutrition = weight_loss

    # Cognitive domain (0-1)
    cognitive = 1.0 - cognitive_function

    # Medical domain (0-1)
    max_comorbidities = 10
    medical = min(multi_morbidity / max_comorbidities, 1.0)

    # Physiological domain (0-1)
    # Higher HR + lower HRV = less reserve
    hr_score = _clamp((hr_resting - 60.0) / 60.0, 0.0, 1.0)
    hrv_score = _clamp(1.0 - hr_variability / 80.0, 0.0, 1.0)
    physio_reserve = 0.5 * hr_score + 0.5 * hrv_score

    # Inflammatory domain
    inflam = _clamp(inflammatory_load / 50.0, 0.0, 1.0)

    # Weighted composite (weights from literature)
    frailty = (
        0.20 * physical +
        0.10 * energy +
        0.10 * nutrition +
        0.10 * cognitive +
        0.20 * medical +
        0.15 * physio_reserve +
        0.15 * inflam
    )
    return float(np.clip(frailty, 0.0, 1.0))


# ── Adaptability ──────────────────────────────────────────────

def compute_adaptability_score(
    glucose_response: List[float],
    sbp_response: List[float],
    baseline_glucose: float,
    baseline_sbp: float,
    challenge_duration: int = 14,
) -> float:
    """
    Compute adaptability — the ability to maintain homeostasis
    under environmental or dietary challenges.

    Higher score = better adaptability (0-100).
    """
    if len(glucose_response) < challenge_duration:
        return 50.0

    # Glucose deviation from baseline during challenge
    g_dev = [abs(g - baseline_glucose) / baseline_glucose for g in glucose_response[:challenge_duration]]
    sbp_dev = [abs(s - baseline_sbp) / baseline_sbp for s in sbp_response[:challenge_duration]]

    # Mean deviation
    mean_g_dev = np.mean(g_dev)
    mean_sbp_dev = np.mean(sbp_dev)

    # Return to baseline in second half
    if len(glucose_response) >= challenge_duration * 2:
        g_late = glucose_response[challenge_duration:challenge_duration * 2]
        sbp_late = sbp_response[challenge_duration:challenge_duration * 2]
        g_drift = abs(np.mean(g_late) - baseline_glucose) / baseline_glucose
        sbp_drift = abs(np.mean(sbp_late) - baseline_sbp) / baseline_sbp
    else:
        g_drift = mean_g_dev
        sbp_drift = mean_sbp_dev

    # Score: lower deviation + lower drift = higher adaptability
    g_score = max(0.0, 100.0 - 200.0 * (mean_g_dev + g_drift))
    sbp_score = max(0.0, 100.0 - 150.0 * (mean_sbp_dev + sbp_drift))

    adaptability = 0.5 * g_score + 0.5 * sbp_score
    return float(np.clip(adaptability, 0.0, 100.0))


# ── Circadian Robustness ──────────────────────────────────────

def compute_circadian_robustness(
    cortisol_trajectory: List[float],
    phase_trajectory: List[float],
    shift_time: Optional[int] = None,
) -> float:
    """
    Measure circadian clock robustness.

    Without shift: amplitude + regularity of cortisol rhythm (0-100).
    With shift: re-entrainment speed after phase shift.
    """
    if len(cortisol_trajectory) < 14:
        return 50.0

    if shift_time is not None and shift_time < len(cortisol_trajectory) - 7:
        # Phase shift recovery: how fast does rhythm re-stabilize?
        post_shift = cortisol_trajectory[shift_time:]
        if len(post_shift) < 7:
            return 50.0

        # Phase coherence in 7-day windows
        from scipy.signal import periodogram
        try:
            freqs, psd = periodogram(post_shift)
            # Find circadian peak (24h)
            idx_24h = np.argmin(np.abs(freqs - 1.0 / 1440.0))
            circadian_power = psd[idx_24h] / max(psd)
            # Normalized circadian power as robustness
            robustness = _clamp(circadian_power * 100.0, 0.0, 100.0)
        except Exception:
            robustness = 50.0
    else:
        # Baseline robustness: amplitude + regularity
        cort = np.array(cortisol_trajectory[-28:])

        # Amplitude (peak-trough ratio)
        peak = np.max(cort[-7:])  # daily peak
        trough = np.min(cort[-7:])  # daily trough
        amplitude_ratio = (peak - trough) / max(trough, 1.0)
        amp_score = _clamp(amplitude_ratio * 25.0, 0.0, 50.0)

        # Regularity (autocorrelation at 24h lag)
        if len(cort) >= 48:
            acf = np.correlate(cort - np.mean(cort), cort - np.mean(cort), mode="full")
            acf = acf / acf[len(acf) // 2]
            lag_24h = int(1440.0 / 60.0)  # assuming hourly sampling
            regularity = max(0.0, acf[len(acf) // 2 + lag_24h])
        else:
            regularity = 0.5

        reg_score = regularity * 50.0
        robustness = amp_score + reg_score

    return float(np.clip(robustness, 0.0, 100.0))


# ── Inflammaging Score ────────────────────────────────────────

def compute_inflammaging_score(
    il6: float = 2.0,
    tnfa: float = 2.0,
    crp: float = 2.0,
    immune_cell_senescence: float = 0.5,
    oxidative_stress: float = 0.5,
    dna_damage: float = 0.5,
) -> float:
    """
    Compute inflammaging score — chronic low-grade inflammation
    associated with biological aging.

    Returns score 0-100 (higher = more inflammaging).

    Reference levels:
      IL-6:  1-3 pg/mL healthy, >5 elevated
      TNF-α: 1-3 pg/mL healthy, >5 elevated
      CRP:   0.5-2 mg/L healthy, >3 elevated
    """
    # Inflammatory cytokine score
    il6_score = _clamp((il6 - 1.0) / 8.0, 0.0, 1.0)
    tnfa_score = _clamp((tnfa - 1.0) / 8.0, 0.0, 1.0)
    crp_score = _clamp((crp - 0.5) / 10.0, 0.0, 1.0)

    cytokine = 0.35 * il6_score + 0.35 * tnfa_score + 0.30 * crp_score

    # Cellular senescence
    senescence = immune_cell_senescence

    # Oxidative stress & DNA damage
    oxidative = 0.5 * oxidative_stress + 0.5 * dna_damage

    # Composite
    inflammaging = 100.0 * (0.4 * cytokine + 0.3 * senescence + 0.3 * oxidative)
    return float(np.clip(inflammaging, 0.0, 100.0))


# ── All Biomarkers 2.0 (Single Entry Point) ───────────────────

@dataclass
class Biomarkers20:
    """
    Complete Phase 4 biomarker suite.
    """
    metabolic_age: float
    metabolic_age_acceleration: float   # metabolic_age - chronological_age
    resilience_score: float
    frailty_index: float
    adaptability_score: float
    circadian_robustness: float
    inflammaging_score: float
    overall_health_score: float         # composite (0-100)

    def to_dict(self) -> Dict[str, float]:
        return {
            "metabolic_age": self.metabolic_age,
            "metabolic_age_acceleration": self.metabolic_age_acceleration,
            "resilience_score": self.resilience_score,
            "frailty_index": self.frailty_index,
            "adaptability_score": self.adaptability_score,
            "circadian_robustness": self.circadian_robustness,
            "inflammaging_score": self.inflammaging_score,
            "overall_health_score": self.overall_health_score,
        }


def compute_all_biomarkers_20(
    chrono_age: float,
    physio_state: np.ndarray,
    weight_kg: float,
    height_cm: float,
    trajectories: Optional[Dict[str, List[float]]] = None,
    perturbation_time: Optional[int] = None,
) -> Biomarkers20:
    """
    Compute all Phase 4 biomarkers from twin state and trajectories.

    Args:
        chrono_age: patient chronological age (years)
        physio_state: 30-dim Phase 3 physiological state
        weight_kg: body weight
        height_cm: height for BMI calculation
        trajectories: optional dict of trajectory lists
        perturbation_time: optional index for resilience computation
    """
    # Extract state variables
    G = float(physio_state[0])      # glucose
    I = float(physio_state[1])      # insulin
    sbp = float(physio_state[5])    # SBP
    hr = float(physio_state[7])     # HR
    hrv = float(physio_state[8])    # HRV
    gfr = float(physio_state[9])    # GFR
    crp = float(physio_state[13])   # CRP
    ffa = float(physio_state[21])   # FFA
    ldl = float(physio_state[22])   # LDL
    hdl = float(physio_state[23])   # HDL
    tg = float(physio_state[24])    # TG
    il6 = float(physio_state[25])   # IL-6
    tnfa = float(physio_state[26])  # TNF-α
    m1_m2 = float(physio_state[27]) # M1/M2 ratio
    nfkb = float(physio_state[28])  # NF-κB
    inflam_load = float(physio_state[29])  # InflammatoryLoad

    bmi = weight_kg / ((height_cm / 100.0) ** 2)
    hba1c = (G + 46.7) / 28.7  # ADAG estimate

    # Metabolic Age
    met_age = compute_metabolic_age(
        chrono_age, bmi, G, hba1c, sbp, hdl, tg, crp, ffa,
    )
    met_age_accel = met_age - chrono_age

    # Resilience
    if trajectories and "glucose" in trajectories and perturbation_time is not None:
        resilience = compute_resilience_score(
            trajectories["glucose"],
            trajectories.get("sbp", [sbp] * 30),
            trajectories.get("hr", [hr] * 30),
            perturbation_time,
        )
    else:
        resilience = 50.0

    # Frailty
    frailty = compute_frailty_index(
        physical_activity=0.5,
        inflammatory_load=inflam_load,
        hr_resting=hr,
        hr_variability=hrv,
    )

    # Adaptability
    if trajectories and "glucose" in trajectories and len(trajectories["glucose"]) > 14:
        adaptability = compute_adaptability_score(
            trajectories["glucose"],
            trajectories.get("sbp", [sbp] * len(trajectories["glucose"])),
            G, sbp,
        )
    else:
        adaptability = 50.0

    # Circadian robustness
    if trajectories and "cortisol" in trajectories:
        circ_robust = compute_circadian_robustness(
            trajectories["cortisol"],
            trajectories.get("phase", [0.0] * len(trajectories["cortisol"])),
        )
    else:
        circ_robust = 50.0

    # Inflammaging
    inflam_score = compute_inflammaging_score(
        il6=il6, tnfa=tnfa, crp=crp,
        immune_cell_senescence=m1_m2 / 2.0,
        oxidative_stress=nfkb,
        dna_damage=0.3,
    )

    # Overall health score (0-100, higher = healthier)
    # Invert frailty and inflammaging; normalize others
    health = (
        0.20 * (100.0 - abs(met_age_accel) * 5.0) +   # metabolic age proximity
        0.15 * resilience +
        0.20 * (100.0 - frailty * 100.0) +             # invert frailty
        0.15 * adaptability +
        0.15 * circ_robust +
        0.15 * (100.0 - inflam_score)                  # invert inflammaging
    )
    overall_health = float(np.clip(health, 0.0, 100.0))

    return Biomarkers20(
        metabolic_age=met_age,
        metabolic_age_acceleration=met_age_accel,
        resilience_score=resilience,
        frailty_index=frailty,
        adaptability_score=adaptability,
        circadian_robustness=circ_robust,
        inflammaging_score=inflam_score,
        overall_health_score=overall_health,
    )
