"""
Phase 3 Digital Biomarkers — Whole-Body Cellular Twin.

Metabolic:
  IR_score       — 1/SI (continuous)
  Recovery_score — 0–100 from glucose variability + IR
  Stress_score   — 0–100 from glucose swings + IR + RoC
  Metabolic_Flexibility — ability to switch between glucose/FFA oxidation

Cardiovascular:
  Vascular_Age      — estimated from arterial stiffness + SBP
  Arterial_Stiffness — PWV analog (0.5–5.0)
  CV_Resilience     — HRV recovery proxy (0–100)

Adipose-Lipid:
  Lipid_Stress      — composite of LDL/HDL ratio + TG (0–100)
  Adipose_IR        — adipose-specific insulin resistance

Circadian:
  Circadian_Health  — amplitude + phase stability (0–100)
  Sleep_Quality     — sleep pressure dynamics proxy (0–100)

Immune-Inflammatory:
  Inflammatory_Burden — composite InflammatoryLoad (0–100)
  Immune_Resilience   — anti-inflammatory capacity (0–100)

Systemic:
  Biological_Age    — multi-system physiological age estimate
  Allostatic_Load   — cumulative multi-system burden (0–100)
  Metabolic_Syndrome_Risk — composite risk score (0–1)
"""

import numpy as np
from typing import List, Optional


# ── Phase 1 Metabolic Biomarkers (unchanged) ─────────────────

def compute_recovery_score(
    glucose_values: List[float],
    ir_state: float,
    glucose_target: float = 90.0,
) -> float:
    if len(glucose_values) < 2:
        return 50.0
    recent = glucose_values[-6:]
    variability = float(np.std(recent))
    mean_g = float(np.mean(recent))
    cv = variability / max(mean_g, 1.0)
    var_score = max(0.0, 100.0 - cv * 500.0)
    ir_norm = np.clip(ir_state / 20.0, 0.0, 1.0)
    ir_score = (1.0 - ir_norm) * 100.0
    dist = abs(mean_g - glucose_target) / glucose_target
    dist_score = max(0.0, 100.0 - dist * 200.0)
    score = var_score * 0.4 + ir_score * 0.35 + dist_score * 0.25
    return float(np.clip(score, 0.0, 100.0))


def compute_stress_score(
    glucose_values: List[float],
    ir_state: float,
) -> float:
    if len(glucose_values) < 2:
        return 50.0
    recent = glucose_values[-6:]
    variability = float(np.std(recent))
    mean_g = float(np.mean(recent))
    cv = variability / max(mean_g, 1.0)
    var_score = np.clip(cv * 500.0, 0.0, 100.0)
    ir_norm = np.clip(ir_state / 20.0, 0.0, 1.0)
    ir_score = ir_norm * 100.0
    if len(recent) >= 3:
        diffs = np.diff(recent)
        mean_roc = float(np.mean(np.abs(diffs)))
        roc_score = np.clip(mean_roc * 5.0, 0.0, 100.0)
    else:
        roc_score = 0.0
    score = var_score * 0.35 + ir_score * 0.35 + roc_score * 0.3
    return float(np.clip(score, 0.0, 100.0))


# ── Phase 2 CV Biomarkers (unchanged) ────────────────────────

def compute_vascular_age(
    arterial_stiffness: float,
    sbp: float,
    age_base: float = 45.0,
) -> float:
    stiffness_excess = max(0.0, arterial_stiffness - 1.5)
    sbp_excess = max(0.0, (sbp - 115.0) / 25.0)
    estimated = age_base + stiffness_excess * 8.0 + sbp_excess * 10.0
    return float(np.clip(estimated, 20.0, 100.0))


def compute_arterial_stiffness_index(
    arterial_stiffness: float,
    sbp: float,
    dbp: float,
) -> float:
    pp = max(1.0, sbp - dbp)
    raw = arterial_stiffness * pp / 40.0
    return float(np.clip(raw, 0.5, 5.0))


def compute_salt_sensitivity_index(
    renal_sensitivity: float,
    sodium_retention: float,
    sbp: float,
) -> float:
    base = renal_sensitivity * 2.0 + sodium_retention * 3.0
    bp_mod = max(0.0, (sbp - 120.0) / 30.0)
    return float(np.clip(base + bp_mod * 10.0, 0.0, 100.0))


# ── Phase 3 New Biomarkers ────────────────────────────────────

def compute_metabolic_flexibility(
    ir: float,
    ffa: float,
    ldl: float,
    hdl: float,
) -> float:
    """Ability to switch between glucose and FFA oxidation (0–100)."""
    ir_penalty = np.clip((ir - 3.0) / 10.0, 0.0, 1.0) * 40.0
    ffa_score = np.clip((ffa - 0.3) / 0.8, 0.0, 1.0) * 20.0
    lipid_ratio = ldl / max(hdl, 1.0)
    lipid_penalty = np.clip((lipid_ratio - 2.0) / 3.0, 0.0, 1.0) * 40.0
    return float(np.clip(100.0 - ir_penalty - ffa_score - lipid_penalty, 0.0, 100.0))


def compute_lipid_stress(
    ldl: float,
    hdl: float,
    tg: float,
    ffas: float,
) -> float:
    """Composite lipid-driven metabolic stress (0–100)."""
    ldl_score = np.clip((ldl - 100.0) / 100.0, 0.0, 1.0) * 30.0
    hdl_penalty = np.clip((50.0 - hdl) / 30.0, 0.0, 1.0) * 25.0
    tg_score = np.clip((tg - 150.0) / 300.0, 0.0, 1.0) * 25.0
    ffa_stress = np.clip((ffas - 0.5) / 0.8, 0.0, 1.0) * 20.0
    return float(np.clip(ldl_score + hdl_penalty + tg_score + ffa_stress, 0.0, 100.0))


def compute_cv_resilience(
    hrv: float,
    hrv_recovery: Optional[List[float]] = None,
) -> float:
    """HRV-based cardiovascular resilience (0–100)."""
    base = np.clip(hrv / 60.0, 0.0, 1.0) * 60.0
    if hrv_recovery and len(hrv_recovery) >= 2:
        trend = hrv_recovery[-1] - hrv_recovery[0]
        recovery_score = np.clip(trend / 20.0, 0.0, 1.0) * 40.0
    else:
        recovery_score = 20.0
    return float(np.clip(base + recovery_score, 0.0, 100.0))


def compute_circadian_health(
    clock_bmal1_amp: float,
    per_cry_amp: float,
    cortisol_amp: float,
    melatonin_amp: float,
) -> float:
    """Circadian rhythm robustness (0–100)."""
    clock_norm = np.clip(clock_bmal1_amp / 0.8, 0.0, 1.0) * 25.0
    per_norm = np.clip(per_cry_amp / 0.8, 0.0, 1.0) * 25.0
    cort_norm = np.clip(cortisol_amp / 300.0, 0.0, 1.0) * 25.0
    mel_norm = np.clip(melatonin_amp / 100.0, 0.0, 1.0) * 25.0
    return float(np.clip(clock_norm + per_norm + cort_norm + mel_norm, 0.0, 100.0))


def compute_sleep_quality(
    sleep_pressure_drop: float,
    sleep_efficiency: float = 0.85,
) -> float:
    """Sleep quality from circadian dynamics (0–100)."""
    sp_score = np.clip(sleep_pressure_drop * 100.0, 0.0, 50.0)
    eff_score = np.clip(sleep_efficiency * 50.0, 0.0, 50.0)
    return float(np.clip(sp_score + eff_score, 0.0, 100.0))


def compute_inflammatory_burden(
    il6: float,
    tnfa: float,
    nfkb: float,
    m1m2: float,
    crp: float,
) -> float:
    """Composite chronic inflammation (0–100)."""
    il6_norm = np.clip(il6 / 5.0, 0.0, 1.0) * 25.0
    tnfa_norm = np.clip(tnfa / 5.0, 0.0, 1.0) * 20.0
    nfkb_norm = nfkb * 20.0
    m1m2_norm = np.clip(m1m2 / 1.5, 0.0, 1.0) * 15.0
    crp_norm = np.clip(crp / 10.0, 0.0, 1.0) * 20.0
    return float(np.clip(il6_norm + tnfa_norm + nfkb_norm + m1m2_norm + crp_norm, 0.0, 100.0))


def compute_immune_resilience(
    nfkb: float,
    cortisol: float,
    vagal_tone_effect: float,
    hrv: float,
) -> float:
    """Anti-inflammatory capacity (0–100)."""
    nfkb_resist = (1.0 - nfkb) * 30.0
    cort_resist = np.clip(cortisol / 500.0, 0.0, 1.0) * 25.0
    vagal = vagal_tone_effect * 20.0
    hrv_norm = np.clip(hrv / 60.0, 0.0, 1.0) * 25.0
    return float(np.clip(nfkb_resist + cort_resist + vagal + hrv_norm, 0.0, 100.0))


def compute_biological_age(
    state: np.ndarray,
    params: np.ndarray,
    chronological_age: float = 45.0,
) -> float:
    """
    Multi-system physiological age estimate.
    Considers: vascular stiffness, GFR, IR, inflammation, circadian amplitude.
    """
    art_stiff = params[4] if len(params) > 4 else 15.0
    gfr = state[9] if len(state) > 9 else 100.0
    ir = state[4] if len(state) > 4 else 5.0
    crp = state[13] if len(state) > 13 else 1.0
    il6 = state[25] if len(state) > 25 else 1.0
    infl_load = state[29] if len(state) > 29 else 15.0

    art_stiff = float(max(art_stiff, 1e-6))
    gfr = float(max(gfr, 1e-6))
    ir = float(max(ir, 1e-6))
    infl_load = float(max(infl_load, 1e-6))

    vascular_age_contrib = float(np.clip((art_stiff - 15.0) / 20.0, 0.0, 1.0) * 15.0)
    renal_age_contrib = float(np.clip((100.0 - gfr) / max(50.0, gfr * 0.01), 0.0, 1.0) * 20.0)
    ir_age_contrib = float(np.clip((ir - 3.0) / 7.0, 0.0, 1.0) * 15.0)
    inflam_age_contrib = float(np.clip(infl_load / max(50.0, infl_load * 0.01), 0.0, 1.0) * 15.0)

    total_excess = float(vascular_age_contrib + renal_age_contrib + ir_age_contrib + inflam_age_contrib)
    return float(np.clip(chronological_age + total_excess, chronological_age - 10, 120.0))


def compute_allostatic_load(
    state: np.ndarray,
    params: np.ndarray,
) -> float:
    """
    Cumulative multi-system physiological burden (0–100).
    Higher = worse; captures across metabolic, CV, renal, immune.
    """
    ir = state[4] if len(state) > 4 else 5.0
    sbp = state[5] if len(state) > 5 else 120.0
    gfr = state[9] if len(state) > 9 else 100.0
    crp = state[13] if len(state) > 13 else 1.0
    ffa = state[21] if len(state) > 21 else 0.5
    tg = state[24] if len(state) > 24 else 120.0
    infl_load = state[29] if len(state) > 29 else 15.0
    ldl = state[22] if len(state) > 22 else 100.0
    hdl = state[23] if len(state) > 23 else 50.0

    m_load = np.clip((ir - 2.0) / 10.0, 0.0, 1.0) * 15.0
    cv_load = np.clip((sbp - 120.0) / 60.0, 0.0, 1.0) * 20.0
    r_load = np.clip((100.0 - gfr) / 60.0, 0.0, 1.0) * 20.0
    i_load = np.clip(infl_load / 50.0, 0.0, 1.0) * 20.0
    l_load = (np.clip((ldl - 100.0) / 100.0, 0.0, 1.0) * 10.0
              + np.clip((50.0 - hdl) / 30.0, 0.0, 1.0) * 7.5
              + np.clip((tg - 150.0) / 300.0, 0.0, 1.0) * 7.5)
    return float(np.clip(m_load + cv_load + r_load + i_load + l_load, 0.0, 100.0))


def compute_metabolic_syndrome_risk(
    state: np.ndarray,
    params: np.ndarray,
) -> float:
    """
    Composite metabolic syndrome risk (0–1).
    Based on: central obesity, IR, BP, TG/HDL, glucose.
    """
    ir_val = state[4] if len(state) > 4 else 5.0
    sbp = state[5] if len(state) > 5 else 120.0
    tg = state[24] if len(state) > 24 else 120.0
    hdl = state[23] if len(state) > 23 else 50.0
    g = state[0] if len(state) > 0 else 90.0
    fat = state[20] if len(state) > 20 else 20.0

    ir_criteria = 1.0 if ir_val > 3.0 else 0.0
    bp_criteria = 1.0 if sbp > 130.0 else 0.0
    tg_criteria = 1.0 if tg > 150.0 else 0.0
    hdl_criteria = 1.0 if hdl < 40.0 else 0.0
    g_criteria = 1.0 if g > 100.0 else 0.0

    count = ir_criteria + bp_criteria + tg_criteria + hdl_criteria + g_criteria
    return float(np.clip(count / 5.0, 0.0, 1.0))


# ── Aggregator ────────────────────────────────────────────────

def compute_all_biomarkers(
    physio_state: np.ndarray,
    params: np.ndarray,
    glucose_buffer: List[float],
    hrv_buffer: Optional[List[float]] = None,
) -> dict:
    """Compute all Phase 3 digital biomarkers from full state + params."""
    state = physio_state
    p = params

    # Phase 1 & 2 biomarkers
    ir = state[4] if len(state) > 4 else 5.0
    sbp = state[5] if len(state) > 5 else 120.0
    dbp = state[6] if len(state) > 6 else 80.0
    hrv = state[8] if len(state) > 8 else 45.0
    ldl = state[22] if len(state) > 22 else 100.0
    hdl = state[23] if len(state) > 23 else 50.0
    tg = state[24] if len(state) > 24 else 120.0
    ffa = state[21] if len(state) > 21 else 0.5
    crp = state[13] if len(state) > 13 else 1.0
    il6 = state[25] if len(state) > 25 else 1.0
    tnfa = state[26] if len(state) > 26 else 0.5
    nfkb = state[28] if len(state) > 28 else 0.2
    m1m2 = state[27] if len(state) > 27 else 0.5
    infl_load = state[29] if len(state) > 29 else 15.0
    cortisol = state[16] if len(state) > 16 else 350.0
    cb1 = state[14] if len(state) > 14 else 1.0
    pc = state[15] if len(state) > 15 else 0.8
    mel = state[17] if len(state) > 17 else 10.0
    art_stiff = p[4] if len(p) > 4 else 15.0
    renal_sens = p[9] if len(p) > 9 else 0.6
    na_ret = p[11] if len(p) > 11 else 0.5
    vagal = p[23] if len(p) > 23 else 0.3

    recovery = compute_recovery_score(glucose_buffer, ir)
    stress = compute_stress_score(glucose_buffer, ir)

    return {
        # Metabolic
        "insulin_resistance_score": 1.0 / max(p[0], 1e-6) if len(p) > 0 else 100.0,
        "recovery_score": recovery,
        "stress_score": stress,
        "metabolic_flexibility": compute_metabolic_flexibility(ir, ffa, ldl, hdl),
        # Cardiovascular
        "vascular_age": compute_vascular_age(art_stiff, sbp),
        "arterial_stiffness_index": compute_arterial_stiffness_index(art_stiff, sbp, dbp),
        "salt_sensitivity_index": compute_salt_sensitivity_index(renal_sens, na_ret, sbp),
        "cv_resilience": compute_cv_resilience(hrv, hrv_buffer),
        # Adipose-Lipid
        "lipid_stress": compute_lipid_stress(ldl, hdl, tg, ffa),
        # Circadian
        "circadian_health": compute_circadian_health(abs(cb1 - 1.0), abs(pc - 1.0),
                                                      abs(cortisol - 200.0), mel),
        # Immune-Inflammatory
        "inflammatory_burden": compute_inflammatory_burden(il6, tnfa, nfkb, m1m2, crp),
        "immune_resilience": compute_immune_resilience(nfkb, cortisol, vagal, hrv),
        # Systemic
        "biological_age": compute_biological_age(state, p),
        "allostatic_load": compute_allostatic_load(state, p),
        "metabolic_syndrome_risk": compute_metabolic_syndrome_risk(state, p),
    }
