"""
Phase 3: Whole-body cellular-level physiological dynamics.

Subsystems (7): Metabolic, Cardiovascular, Renal, Inflammation (CRP),
                Circadian (core-clock TTFL), Adipose-Lipid, Immune-Inflammatory

Cellular coupling map:
  Circadian → All: cortisol rhythms modulate insulin sensitivity, BP, immune
  Adipose   → Metabolic: FFA→IR via Randle cycle; →CV: FFA→endothelial dysfunction
  Immune    → Metabolic: TNF-α→IR via IRS-1 serine phosphorylation
  Metabolic → Adipose: insulin→lipogenesis, lipolysis suppression
  Metabolic → Immune: hyperglycemia→NF-κB, AGEs→inflammation
  CV        → Renal:  BP→GFR
  Renal     → CV:     Na retention→volume→BP
  Circadian → Immune: cortisol suppresses NF-κB (anti-inflammatory)

Model References:
  - Glucose-insulin: Bergman RN, et al. Am J Physiol. 1979;236(6):E667-77.
    (Minimal model; extended with HGP, renal excretion, exercise)
  - Cardiovascular: Guyton AC, et al. Annu Rev Physiol. 1972;34:13-46.
    (Mean arterial pressure, baroreflex, exercise hemodynamics)
  - Renal: Guyton AC, et al. Am J Physiol. 1972;222(6):1462-73.
    (GFR autoregulation, Na/K handling, SGLT-mediated glucosuria)
  - Circadian: Relogio A, et al. PLoS Comput Biol. 2011;7(2):e1001078.
    (CLOCK/BMAL1-PER/CRY TTFL oscillator; cortisol/melatonin coupling)
  - Adipose-Lipid: Kim S, et al. J Lipid Res. 2007;48(3):563-77.
    (FFA kinetics, lipoprotein metabolism, insulin-regulated lipolysis)
  - Immune-Inflammatory: Liu T, et al. Signal Transduct Target Ther. 2017;2:17023.
    (NF-κB pathway, M1/M2 polarization, IL-6/TNF-α signaling)
"""

import numpy as np
from typing import Dict, Any

from .state import (
    Phase3TwinState,
    MetabolicState, CardioState, RenalState, InflammatoryState,
    CircadianState, AdiposeLipidState, ImmuneInflamState,
    METABOLIC_DIM, CARDIO_DIM, RENAL_DIM, INFLAMMATION_DIM,
    CIRCADIAN_DIM, ADIPOSE_DIM, IMMUNE_DIM,
    PHYSIO_DIM,
    _META_OFF, _CARDIO_OFF, _RENAL_OFF, _INFL_OFF,
    _CIRC_OFF, _ADIP_OFF, _IMMUNE_OFF,
)

# ── Physical constants ──────────────────────────────────────
TAU_24H = 1440.0  # minutes in 24h
OMEGA_24H = 2.0 * np.pi / TAU_24H  # angular frequency


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ===================================================================
# 1. Circadian System — Core-Clock Transcriptional-Translational Loop
# ===================================================================

def compute_circadian_dynamics(
    circ: CircadianState,
    inputs: Dict[str, float],
    params: Dict[str, float],
    dt: float = 1.0,
) -> CircadianState:
    """
    Cellular-level circadian model using a limit-cycle oscillator
    representing the CLOCK/BMAL1 ↔ PER/CRY feedback loop, with
    light entrainment and hormonal outputs.

    Core mechanism (Goodwin-type oscillator):
      - CLOCK/BMAL1 (activator) drives PER/CRY transcription
      - PER/CRY accumulates, translocates to nucleus, represses CLOCK/BMAL1
      - Light → SCN → phase shift (via PER induction)
    
    Outputs:
      - Cortisol: peaks ~8 AM (phase-shifted from clock)
      - Melatonin: peaks ~2 AM, suppressed by light
      - Sleep pressure: adenosine accumulates while awake, decays while asleep
    """
    period = params.get("circadian_period", TAU_24H)
    amplitude = params.get("circadian_amplitude", 0.8)
    light_sens = params.get("light_sensitivity", 0.3)
    mel_rate = params.get("melatonin_rate", 0.5)

    light = inputs.get("light_level", 0.0)  # 0 (dark) to 1 (bright)
    sleep = inputs.get("sleep", 0.0)        # 0 = awake, 1 = asleep

    omega = 2.0 * np.pi / period

    # Phase evolution: ω + light-induced phase shift (light advances phase)
    light_phase_shift = light_sens * light * 0.05
    d_phase = (omega + light_phase_shift) * dt
    new_phase = (circ.circadian_phase + d_phase) % (2.0 * np.pi)

    # Amplitude dynamics: limit-cycle relaxation
    amp = np.sqrt(
        (circ.CLOCK_BMAL1 - 1.0) ** 2 + (circ.PER_CRY - 1.0) ** 2
    )
    target_amp = amplitude * (1.0 + 0.1 * light)  # light boosts amplitude
    d_amp = 0.01 * (target_amp - amp) * dt
    new_amp = _clamp(amp + d_amp, 0.1, 1.5)

    # CLOCK/BMAL1 and PER/CRY from phase and amplitude
    new_clock = 1.0 + new_amp * np.cos(new_phase)
    new_per = 1.0 + new_amp * np.sin(new_phase)

    # Cortisol: peaks at phase ~π (morning, ~8 AM)
    # phase=0 corresponds to ~midnight; phase=π ~ noon roughly
    cortisol_peak = params.get("cortisol_peak", 550.0)  # nmol/L
    cortisol_nadir = params.get("cortisol_nadir", 100.0)  # nmol/L
    cortisol_mid = (cortisol_peak + cortisol_nadir) / 2.0
    cortisol_amp = (cortisol_peak - cortisol_nadir) / 2.0
    # Cortisol peak ~2h after waking (phase ≈ π/2 from midnight adjusted)
    cortisol_phase_shift = params.get("cortisol_phase_shift", 0.5)  # rad
    new_cortisol = _clamp(
        cortisol_mid + cortisol_amp * np.cos(new_phase - np.pi + cortisol_phase_shift),
        10.0, 1000.0,
    )

    # Melatonin: peaks during darkness, suppressed by light
    # Melatonin rises when phase is in dark zone (phase ≈ 3π/4 to 5π/4)
    dark_signal = max(0.0, -np.cos(new_phase + 1.0))  # ~0 in day, ~1 at night
    light_suppression = 1.0 / (1.0 + light * light_sens * 5.0)
    mel_amp = params.get("melatonin_amp", 120.0)  # pmol/L night peak
    new_melatonin = _clamp(
        mel_amp * dark_signal * light_suppression * mel_rate,
        0.0, 300.0,
    )

    # Sleep pressure: adenosine accumulates while awake, decays while asleep
    wake_increment = 0.005  # per minute of wakefulness
    sleep_decay_rate = 0.02  # per minute of sleep
    if sleep > 0.5:
        d_sleep = -sleep_decay_rate * circ.sleep_pressure * dt
    else:
        d_sleep = wake_increment * (1.0 - circ.sleep_pressure) * dt
    new_sleep_pressure = _clamp(circ.sleep_pressure + d_sleep, 0.0, 1.0)

    return CircadianState(
        CLOCK_BMAL1=_clamp(new_clock, 0.0, 2.5),
        PER_CRY=_clamp(new_per, 0.0, 2.5),
        cortisol=new_cortisol,
        melatonin=new_melatonin,
        circadian_phase=new_phase,
        sleep_pressure=new_sleep_pressure,
    )


# ===================================================================
# 2. Adipose-Lipid System — Cellular Lipolysis & Lipoprotein Metabolism
# ===================================================================

def compute_adipose_dynamics(
    adip: AdiposeLipidState,
    meta_insulin: float,
    inputs: Dict[str, float],
    params: Dict[str, float],
    dt: float = 1.0,
) -> AdiposeLipidState:
    """
    Adipocyte-level lipid metabolism.

    Lipolysis (FFA release):
      - HSL (hormone-sensitive lipase) activated by catecholamines
      - Suppressed by insulin (via PDE3B → cAMP↓)
      - FFA release = basal + (1 - insulin_suppression) * max_rate

    Lipogenesis:
      - Insulin stimulates de novo lipogenesis in liver/adipose
      - Glucose → acetyl-CoA → malonyl-CoA → FAs → TAG

    Lipoprotein metabolism:
      - VLDL secretion from liver (TG-rich)
      - LDL from VLDL catabolism (via LPL)
      - HDL from liver/intestine (ApoA-I)
      - LDL receptor-mediated clearance (downregulated by high LDL/SFA)
    
    Energy balance (fat_mass):
      - Slow timescale: net energy intake - expenditure
    """
    lipolysis_rate = params.get("lipolysis_rate", 0.05)      # FFA release per min
    lipogenesis_rate = params.get("lipogenesis_rate", 0.02)  # FFA uptake per min
    LDL_clearance = params.get("LDL_clearance", 0.015)       # fraction cleared per min
    HDL_production = params.get("HDL_production", 0.01)      # HDL production rate
    FFA_uptake_tissue = params.get("FFA_uptake", 0.03)       # tissue FFA clearance

    dietary_fat = inputs.get("dietary_fat", 0.0)     # g/day equivalent per min
    calorie_intake = inputs.get("calorie_intake", 2000.0) / 1440.0  # kcal/min
    exercise = inputs.get("exercise", 0.0)

    # --- FFA Dynamics ---
    # Insulin suppresses lipolysis (logistic inhibition)
    insulin_suppression = 1.0 / (1.0 + meta_insulin / 10.0)
    basal_lipolysis = lipolysis_rate * 0.1  # basal FFA release
    stimulated_lipolysis = lipolysis_rate * insulin_suppression * (1.0 + exercise * 0.5)
    ffa_release = basal_lipolysis + stimulated_lipolysis

    # FFA uptake by tissues (muscle, liver)
    ffa_uptake = FFA_uptake_tissue * adip.FFA * (1.0 + exercise * 0.3)
    dFFA = (ffa_release - ffa_uptake + dietary_fat * 0.001) * dt
    new_ffa = _clamp(adip.FFA + dFFA, 0.1, 2.0)

    # --- Lipoprotein Dynamics ---
    # VLDL/TG: produced by liver from FFA + dietary fat; cleared by LPL
    tg_production = 0.5 + new_ffa * 15.0 + dietary_fat * 0.5
    tg_clearance = 0.02 * adip.TG * (1.0 + exercise * 0.2)
    dTG = (tg_production - tg_clearance) * dt
    new_tg = _clamp(adip.TG + dTG, 20.0, 800.0)

    # LDL: from VLDL catabolism; cleared by LDL receptors
    ldL_from_vldl = 0.3 * new_tg / 100.0  # VLDL → LDL conversion
    ldL_receptor_uptake = LDL_clearance * adip.LDL
    dLDL = (ldL_from_vldl - ldL_receptor_uptake) * dt
    new_ldl = _clamp(adip.LDL + dLDL, 20.0, 300.0)

    # HDL: produced by liver/intestine; cleared by hepatic uptake
    hdl_prod = HDL_production * (1.0 - exercise * 0.1)  # exercise slightly↑HDL
    hdl_clearance = 0.008 * adip.HDL
    dHDL = (hdl_prod - hdl_clearance) * dt
    new_hdl = _clamp(adip.HDL + dHDL, 10.0, 120.0)

    # --- Fat Mass (slow timescale) ---
    # Energy balance: intake - expenditure
    BMR = 1.0  # kcal/min basal (~1440 kcal/day)
    exercise_ee = exercise * 5.0  # kcal/min during exercise
    total_expenditure = BMR + exercise_ee
    net_energy = (calorie_intake - total_expenditure) * dt  # kcal
    # 1 kg fat ≈ 7700 kcal
    d_fat_mass = net_energy / 7700.0  # kg change per dt
    new_fat_mass = _clamp(adip.fat_mass + d_fat_mass, 2.0, 100.0)

    return AdiposeLipidState(
        fat_mass=new_fat_mass,
        FFA=new_ffa,
        LDL=new_ldl,
        HDL=new_hdl,
        TG=new_tg,
    )


# ===================================================================
# 3. Immune-Inflammatory System — Macrophage & Cytokine Signaling
# ===================================================================

def compute_immune_dynamics(
    immune: ImmuneInflamState,
    meta_ir: float,
    cardio_hrv: float,
    adip_ffa: float,
    circ_cortisol: float,
    inputs: Dict[str, float],
    params: Dict[str, float],
    dt: float = 1.0,
) -> ImmuneInflamState:
    """
    Cellular immune-inflammatory dynamics with NF-κB pathway.

    M1/M2 Macrophage Polarization:
      - M1 (pro-inflammatory): driven by FFAs (TLR4), TNF-α, IFN-γ
      - M2 (anti-inflammatory): driven by IL-4, IL-10, glucocorticoids
      - Adipose tissue: obesity → M1 predominance → chronic inflammation

    NF-κB Pathway:
      - Activated by: TNF-α, IL-1β, TLR ligands (FFA, LPS), ROS
      - Inhibited by: cortisol (via IκBα induction), vagal tone (α7nAChR)
      - Drives transcription of: IL-6, TNF-α, CRP

    Cytokine Cascade:
      - IL-6: produced by M1 macrophages, adipocytes, endothelial cells
      - TNF-α: produced by M1 macrophages, signals via TNFR→NF-κB
      - CRP: liver-derived, induced by IL-6

    Cholinergic Anti-inflammatory Pathway:
      - Vagus nerve → ACh → α7nAChR on macrophages → NF-κB↓
      - HRV is a proxy for vagal tone
    """
    M1_activation = params.get("M1_activation", 0.1)        # M1 polarization rate
    NFkB_sensitivity = params.get("NFkB_sensitivity", 0.5)  # NF-κB response gain
    vagal_tone_effect = params.get("vagal_tone_effect", 0.3) # ACh anti-inflammatory
    IL6_clearance = params.get("IL6_clearance", 0.02)       # IL-6 decay rate

    sleep_quality = inputs.get("sleep_quality", 1.0)  # 0 (poor) to 1 (good)
    alcohol = inputs.get("alcohol", 0.0)              # 0-1 (alcohol intake level)

    # --- M1/M2 Polarization ---
    # M1 drivers: FFA (TLR4), IR, low vagal tone (HRV↓)
    ffa_drive = _clamp((adip_ffa - 0.3) / 1.0, 0.0, 1.0)
    ir_drive = _clamp((meta_ir - 3.0) / 10.0, 0.0, 1.0)
    hrv_drive = _clamp((50.0 - cardio_hrv) / 50.0, 0.0, 1.0)
    # M2 drivers: cortisol (anti-inflammatory), vagal tone, sleep quality
    cortisol_drive = _clamp((circ_cortisol - 100.0) / 500.0, 0.0, 1.0)
    vagal_drive = _clamp(cardio_hrv / 50.0, 0.0, 1.0)

    m1_input = (ffa_drive * 0.3 + ir_drive * 0.3 + hrv_drive * 0.2
                + alcohol * 0.2)
    m2_input = (cortisol_drive * 0.3 + vagal_drive * vagal_tone_effect * 0.4
                + sleep_quality * 0.3)
    target_M1M2 = _clamp(m1_input / (m2_input + 0.1), 0.0, 2.0)
    d_M1M2 = 0.01 * (target_M1M2 - immune.M1_M2_ratio) * dt
    new_m1m2 = _clamp(immune.M1_M2_ratio + d_M1M2, 0.0, 2.0)

    # --- NF-κB Activity ---
    # Activated by: TNF-α (autocrine), FFA (TLR4), IR/ROS
    # Inhibited by: cortisol (IκBα), vagal tone (α7nAChR)
    tnf_stim = immune.TNFa_proxy * NFkB_sensitivity
    ffa_stim = ffa_drive * NFkB_sensitivity * 0.5
    ir_stim = ir_drive * 0.3
    cortisol_inhib = _clamp(circ_cortisol / 500.0, 0.0, 1.0) * 0.5
    vagal_inhib = vagal_drive * vagal_tone_effect * 0.4

    nfkb_input = tnf_stim + ffa_stim + ir_stim
    nfkb_inhibition = cortisol_inhib + vagal_inhib
    target_NFkB = _clamp(nfkb_input * (1.0 - nfkb_inhibition), 0.0, 1.0)
    d_NFkB = 0.02 * (target_NFkB - immune.NFkB_activity) * dt
    new_nfkb = _clamp(immune.NFkB_activity + d_NFkB, 0.0, 1.0)

    # --- IL-6 Proxy ---
    # Produced by NF-κB activation, M1 macrophages; cleared at IL6_clearance
    il6_production = 0.1 + new_nfkb * 2.0 + new_m1m2 * 1.5
    d_IL6 = (il6_production - IL6_clearance * immune.IL6_proxy) * dt
    new_il6 = _clamp(immune.IL6_proxy + d_IL6, 0.0, 10.0)

    # --- TNF-α Proxy ---
    # Produced by M1 macrophages, NF-κB (autocrine loop)
    tnf_production = 0.05 + new_m1m2 * 1.0 + new_nfkb * 1.5
    tnf_decay = 0.03 * immune.TNFa_proxy
    d_TNF = (tnf_production - tnf_decay) * dt
    new_tnf = _clamp(immune.TNFa_proxy + d_TNF, 0.0, 10.0)

    # --- Composite Inflammatory Load ---
    il6_norm = _clamp(new_il6 / 5.0, 0.0, 1.0)
    tnf_norm = _clamp(new_tnf / 5.0, 0.0, 1.0)
    nfkb_norm = new_nfkb
    m1_norm = _clamp(new_m1m2 / 1.5, 0.0, 1.0)
    target_load = (il6_norm * 0.3 + tnf_norm * 0.3
                   + nfkb_norm * 0.2 + m1_norm * 0.2) * 100.0
    d_load = 0.01 * (target_load - immune.InflammatoryLoad) * dt
    new_load = _clamp(immune.InflammatoryLoad + d_load, 0.0, 100.0)

    return ImmuneInflamState(
        IL6_proxy=new_il6,
        TNFa_proxy=new_tnf,
        M1_M2_ratio=new_m1m2,
        NFkB_activity=new_nfkb,
        InflammatoryLoad=new_load,
    )


# ===================================================================
# 4. Updated Legacy Subsystems (with circadian coupling)
# ===================================================================

def compute_metabolic_dynamics(
    meta: MetabolicState,
    inputs: Dict[str, float],
    params: Dict[str, float],
    circ_cortisol: float = 300.0,
    dt: float = 1.0,
) -> MetabolicState:
    """
    Glucose-insulin dynamics with circadian modulation.
    Cortisol → morning insulin resistance (dawn phenomenon).
    """
    SI_base = params.get("SI", 0.018)
    HGP_basal = params.get("HGP_basal", 2.0)
    beta = params.get("beta_response", 0.0025)
    RT = params.get("RT", 180.0)

    # Cortisol modulates insulin sensitivity: high cortisol = lower SI
    # Dawn phenomenon: SI reduced by ~30-50% in early morning
    cortisol_effect = 1.0 - 0.003 * max(0.0, circ_cortisol - 200.0)
    SI = SI_base * _clamp(cortisol_effect, 0.5, 1.0)

    meal_glucose = inputs.get("meal_glucose", 0.0)
    exercise = inputs.get("exercise", 0.0)
    insulin_dose = inputs.get("insulin_dose", 0.0)

    # Insulin-mediated glucose disposal (Bergman minimal model formulation)
    peripheral_uptake = SI * meta.I * meta.G
    # Hepatic glucose production suppressed by insulin (Cherrington model)
    # Total HGP = basal * (1 - fractional suppression by insulin)
    hepatic_prod = HGP_basal * (1 - meta.I / (meta.I + 30))
    # Renal glucose excretion above threshold (SGLT-mediated)
    renal_excr = max(0, (meta.G - RT) * 0.1) if meta.G > RT else 0
    # Basal glucose utilization (insulin-independent, e.g. brain, RBCs)
    basal_util = 0.01 * meta.G

    dG = (-peripheral_uptake - basal_util + hepatic_prod
          + meal_glucose - renal_excr
          - exercise * 0.02 * meta.G)

    glucose_effect = beta * max(0, meta.G - 80)
    dI = -0.2 * meta.I + glucose_effect + insulin_dose

    target_HGP = hepatic_prod
    dHGP = -0.1 * (meta.HGP - target_HGP)
    target_PGU = peripheral_uptake
    dPGU = -0.1 * (meta.PGU - target_PGU)

    target_IR = 1.0 / max(SI, 0.001)
    dIR = -0.05 * (meta.IR - target_IR)

    return MetabolicState(
        G=_clamp(meta.G + dG * dt, 20, 600),
        I=_clamp(meta.I + dI * dt, 0, 500),
        HGP=_clamp(meta.HGP + dHGP * dt, -5, 15),
        PGU=_clamp(meta.PGU + dPGU * dt, 0, 25),
        IR=_clamp(meta.IR + dIR * dt, 0, 20),
    )


def compute_cardio_dynamics(
    cardio: CardioState,
    inputs: Dict[str, float],
    params: Dict[str, float],
    meta_ir: float,
    circ_cortisol: float = 300.0,
    dt: float = 1.0,
) -> CardioState:
    """
    Cardiovascular dynamics with circadian modulation.
    Cortisol → morning BP surge; HRV reflects vagal tone.
    """
    R = params.get("vascular_resistance", 90.0)
    baro_gain = params.get("baroreflex_gain", 1.5)
    auto_tone = params.get("autonomic_tone", 0.5)

    exercise = inputs.get("exercise", 0.0)
    ir_effect = 1.0 + 0.05 * max(0, meta_ir - 5.0)
    R_eff = R * ir_effect * (1.0 - exercise * 0.3)

    # Cortisol raises BP via mineralocorticoid activity
    cortisol_bp_effect = 1.0 + 0.0005 * max(0.0, circ_cortisol - 200.0)

    CO_rest = 5.0
    map_target = CO_rest * R_eff / 10.0 + 50.0
    map_target *= cortisol_bp_effect
    map_now = (cardio.SBP + 2 * cardio.DBP) / 3.0

    dMAP = (map_target - map_now) / 5.0

    dSBP = dMAP * 0.8 + exercise * 8.0
    dDBP = dMAP * 0.6 - exercise * 2.0

    bp_dev = (map_now - 93.0) / 93.0
    baro_correction = -baro_gain * bp_dev * 15.0
    hr_exercise = exercise * 25.0
    dHR = baro_correction + hr_exercise - 0.05 * (cardio.HR - 70.0)

    # Cortisol raises HR (sympathetic effect)
    cortisol_hr_effect = 1.0 + 0.0003 * max(0.0, circ_cortisol - 200.0)
    dHR *= cortisol_hr_effect

    target_hrv = 50.0 * (70.0 / max(cardio.HR, 30))
    dHRV = -0.05 * (cardio.HRV - target_hrv) - exercise * 3.0

    return CardioState(
        SBP=_clamp(cardio.SBP + dSBP * dt, 50, 250),
        DBP=_clamp(cardio.DBP + dDBP * dt, 30, 150),
        HR=_clamp(cardio.HR + dHR * dt, 30, 220),
        HRV=_clamp(cardio.HRV + dHRV * dt, 5, 200),
    )


def compute_renal_dynamics(
    renal: RenalState,
    inputs: Dict[str, float],
    params: Dict[str, float],
    cardio: CardioState,
    meta_g: float,
    dt: float = 1.0,
) -> RenalState:
    """Renal dynamics (unchanged from Phase 2)."""
    baseline_GFR = params.get("baseline_GFR", 100.0)
    renal_sens = params.get("renal_sensitivity", 0.6)
    sglt_act = params.get("SGLT_activity", 30.0)
    na_retention = params.get("sodium_retention", 0.5)

    map_val = (cardio.SBP + 2 * cardio.DBP) / 3.0
    bp_factor = 1.0 + 0.005 * (map_val - 93.0)
    gluc_factor = 1.0 + 0.001 * max(0, meta_g - 100.0)
    target_GFR = baseline_GFR * bp_factor * gluc_factor
    dGFR = -0.02 * (renal.GFR - target_GFR)

    filtered_na = renal.GFR * renal.Na / 1000.0
    na_reabsorbed = filtered_na * na_retention * renal_sens
    na_excretion = filtered_na - na_reabsorbed
    na_input = inputs.get("sodium_intake", 100.0) / 1440.0
    dNa = (na_input - na_excretion) / 50.0

    k_input = inputs.get("potassium_intake", 3.0) / 1440.0
    k_clearance = renal.GFR * renal.K * 0.00001
    dK = (k_input - k_clearance) * 5.0

    water_input = inputs.get("water_intake", 2000.0) / 1440.0
    water_loss = renal.GFR * 0.001 + 0.5
    dOsm = (water_input - water_loss) * 0.01 * (300.0 - renal.Osm) / 300.0

    return RenalState(
        GFR=_clamp(renal.GFR + dGFR * dt, 5, 200),
        Na=_clamp(renal.Na + dNa * dt, 120, 160),
        K=_clamp(renal.K + dK * dt, 2.5, 7.0),
        Osm=_clamp(renal.Osm + dOsm * dt, 260, 340),
    )


def compute_inflammation_dynamics(
    infl: InflammatoryState,
    meta_ir: float,
    cardio_hrv: float,
    immune_il6: float = 1.0,
    dt: float = 1.0,
) -> InflammatoryState:
    """CRP dynamics — now coupled to immune IL-6."""
    # CRP is primarily IL-6 driven
    crp_from_il6 = 1.0 + immune_il6 * 2.0
    ir_stimulus = 1.0 + 0.05 * max(0, meta_ir - 3.0)
    hrv_protection = 1.0 - 0.2 * min(1.0, max(0, 50.0 - cardio_hrv) / 50.0)
    target_crp = 2.0 * crp_from_il6 * ir_stimulus * hrv_protection
    dCRP = -0.01 * (infl.CRP - target_crp)
    return InflammatoryState(CRP=_clamp(infl.CRP + dCRP * dt, 0.1, 100.0))


# ===================================================================
# 5. Full Dynamics Composition
# ===================================================================

def full_dynamics(
    state_arr: np.ndarray,
    params_arr: np.ndarray,
    inputs: Dict[str, float],
    dt: float = 1.0,
) -> np.ndarray:
    """
    Compose all 7 subsystem dynamics with cellular-level coupling.

    Execution order (respects coupling dependencies):
      1. Circadian        ← light, sleep
      2. Adipose-Lipid    ← insulin (from previous step)
      3. Immune-Inflam    ← IR, HRV, FFA, cortisol
      4. Metabolic        ← cortisol, FFA (via IR)
      5. Cardiovascular   ← IR, cortisol
      6. Renal            ← cardio, G
      7. Inflammation     ← IR, HRV, IL-6

    LIMITATIONS:
      - The ODE is a phenomenological model calibrated to population-average
        physiology. Individual parameters require personalization.
      - Causal feedback loops (e.g., G → I → G, SBP → GFR → volume → SBP)
        are resolved sequentially across timesteps, not simultaneously.
        True bidirectional coupling within a single timestep is not modeled.
      - The 30-dim state is only 15-dim observable. Unobserved states
        (HGP, PGU, IR, CLOCK_BMAL1, PER_CRY, CRP, fat_mass, M1_M2_ratio,
        NFkB_activity, InflammatoryLoad) are driven by ODE coupling and
        process noise alone. Structural identifiability of these states
        depends on informative priors.
      - Drug mechanisms are approximated as parameter changes (e.g.,
        SGLT2i → renal threshold reduction). Detailed pharmacokinetics,
        drug-drug interactions, and adverse effects are not modeled.
      - All simulated outputs are from the model itself. External validation
        on independent patient data is required for clinical use.
    """
    ft = Phase3TwinState.from_array(state_arr)
    param_dict = _params_to_dict(params_arr)
    p = param_dict

    # 1. Circadian — needs light, sleep
    new_circ = compute_circadian_dynamics(ft.circadian, inputs, p, dt)

    # 2. Adipose-Lipid — needs insulin (from current state)
    new_adip = compute_adipose_dynamics(ft.adipose, ft.metabolic.I, inputs, p, dt)

    # 3. Immune — needs IR, HRV, FFA, cortisol
    new_immune = compute_immune_dynamics(
        ft.immune, ft.metabolic.IR, ft.cardio.HRV, new_adip.FFA,
        new_circ.cortisol, inputs, p, dt,
    )

    # 4. Metabolic — needs cortisol (circadian coupling)
    new_meta = compute_metabolic_dynamics(ft.metabolic, inputs, p, new_circ.cortisol, dt)

    # 5. Cardiovascular — needs IR, cortisol
    new_cardio = compute_cardio_dynamics(ft.cardio, inputs, p, new_meta.IR, new_circ.cortisol, dt)

    # 6. Renal — needs cardio, G
    new_renal = compute_renal_dynamics(ft.renal, inputs, p, new_cardio, new_meta.G, dt)

    # 7. Inflammation (CRP) — needs IR, HRV, IL-6
    new_infl = compute_inflammation_dynamics(
        ft.inflammation, new_meta.IR, new_cardio.HRV, new_immune.IL6_proxy, dt,
    )

    return Phase3TwinState(new_meta, new_cardio, new_renal, new_infl,
                           new_circ, new_adip, new_immune).to_array()


# ===================================================================
# 6. Observation Model
# ===================================================================

def full_observation(state_arr: np.ndarray) -> np.ndarray:
    """
    Return observable quantities (16-dim).

    Indices:
      0: glucose (CGM)
      1: SBP (BP monitor)
      2: DBP (BP monitor)
      3: HR (wearable)
      4: HRV (wearable)
      5: GFR (lab)
      6: Na (lab)
      7: K (lab)
      8: Osm (lab)
      9: FFA (lab)
      10: LDL (lab)
      11: HDL (lab)
      12: TG (lab)
      13: cortisol (lab/salivary)
      14: sleep_pressure (self-report / actigraphy)
      15: insulin (fasting insulin / HOMA-IR)
    """
    ft = Phase3TwinState.from_array(state_arr)
    return np.array([
        ft.metabolic.G,           # 0
        ft.cardio.SBP,            # 1
        ft.cardio.DBP,            # 2
        ft.cardio.HR,             # 3
        ft.cardio.HRV,            # 4
        ft.renal.GFR,             # 5
        ft.renal.Na,              # 6
        ft.renal.K,               # 7
        ft.renal.Osm,             # 8
        ft.adipose.FFA,           # 9
        ft.adipose.LDL,           # 10
        ft.adipose.HDL,           # 11
        ft.adipose.TG,            # 12
        ft.circadian.cortisol,    # 13
        ft.circadian.sleep_pressure,  # 14
        ft.metabolic.I,           # 15 (insulin — fasting / lab)
    ])


OBS_NAMES = [
    "glucose", "SBP", "DBP", "HR", "HRV",
    "GFR", "Na", "K", "Osm",
    "FFA", "LDL", "HDL", "TG",
    "cortisol", "sleep_pressure", "insulin",
]


# ===================================================================
# 7. Parameter Mapping
# ===================================================================

DEFAULT_PARAMS = np.array([
    0.018, 2.0, 0.0025, 180.0,    # metabolic (0-3)
    0.5, 90.0, 1.0, 1.0,          # cardio (4-7)
    100.0, 1.0, 1.0, 1.0,         # renal (8-11)
    1440.0, 0.8, 0.3, 0.5,        # circadian (12-15)
    0.1, 0.05, 0.03, 0.02, 0.1,   # adipose (16-20)
    0.5, 0.5, 0.3, 0.5,           # immune (21-24)
], dtype=np.float64)


def _params_to_dict(p: np.ndarray) -> Dict[str, float]:
    return {
        # Metabolic (0-3)
        "SI": float(p[0]),
        "HGP_basal": float(p[1]),
        "beta_response": float(p[2]),
        "RT": float(p[3]),
        # Cardiovascular (4-7)
        "arterial_stiffness": float(p[4]),
        "vascular_resistance": float(p[5]),
        "baroreflex_gain": float(p[6]),
        "autonomic_tone": float(p[7]),
        # Renal (8-11)
        "baseline_GFR": float(p[8]),
        "renal_sensitivity": float(p[9]),
        "SGLT_activity": float(p[10]),
        "sodium_retention": float(p[11]),
        # Circadian (12-15)
        "circadian_period": float(p[12]),
        "circadian_amplitude": float(p[13]),
        "light_sensitivity": float(p[14]),
        "melatonin_rate": float(p[15]),
        # Adipose (16-20)
        "lipolysis_rate": float(p[16]),
        "lipogenesis_rate": float(p[17]),
        "LDL_clearance": float(p[18]),
        "HDL_production": float(p[19]),
        "FFA_uptake": float(p[20]),
        # Immune (21-24)
        "M1_activation": float(p[21]),
        "NFkB_sensitivity": float(p[22]),
        "vagal_tone_effect": float(p[23]),
        "IL6_clearance": float(p[24]),
    }
