"""
Clinical Outcome Simulator.

Translates twin trajectories into clinically meaningful endpoints
that regulators, payers, and physicians care about:

  - TIR (Time In Range): % time glucose ∈ [70, 180] mg/dL
  - TBR (Time Below Range): % time glucose < 70 mg/dL
  - TAR (Time Above Range): % time glucose > 180 mg/dL
  - eA1c: estimated HbA1c from mean glucose
  - Hypoglycemia events: rate per day
  - BP control: % time SBP < 130 mmHg
  - Composite endpoints: dual (G + BP) control

This is the translation layer between "ODE states" and "FDA endpoints."
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field

from app.personalization.dynamics import full_dynamics, DEFAULT_PARAMS
from app.personalization.state import PHYSIO_DIM


@dataclass
class ClinicalOutcomes:
    tir: float          # Time In Range (70-180 mg/dL)
    tbr: float          # Time Below Range (< 70 mg/dL)
    tar: float          # Time Above Range (> 180 mg/dL)
    tar_250: float      # Time Above 250 mg/dL
    eA1c: float         # Estimated HbA1c (%)
    mean_glucose: float
    glucose_sd: float
    hypo_events_per_day: float
    sbp_mean: float
    sbp_control_pct: float  # % time SBP < 130
    dbp_mean: float
    hr_mean: float
    tir_70_140: float   # Strict TIR (70-140 mg/dL)
    gmi: float           # Glucose Management Indicator

    def summary(self) -> str:
        return (
            f"TIR={self.tir:.1f}% | TBR={self.tbr:.1f}% | TAR={self.tar:.1f}% | "
            f"eA1c={self.eA1c:.1f}% | GMI={self.gmi:.1f}% | "
            f"mean_G={self.mean_glucose:.0f} mg/dL | "
            f"hypo_events={self.hypo_events_per_day:.1f}/d | "
            f"SBP={self.sbp_mean:.0f} mmHg"
        )

    def composite_score(self) -> float:
        """
        Single score summarizing clinical outcomes.

        Weighted combination: TIR (maximized), TBR (minimized),
        TAR (minimized), SBP control. Range [0, 1].
        """
        tir_score = self.tir / 100.0
        hypo_penalty = max(0, self.tbr - 1.0) / 10.0
        tar_penalty = max(0, self.tar - 10.0) / 50.0
        sbp_score = self.sbp_control_pct / 100.0
        return float(np.clip(
            0.5 * tir_score + 0.2 * sbp_score - 0.2 * hypo_penalty - 0.1 * tar_penalty,
            0.0, 1.0,
        ))


def compute_clinical_outcomes(
    glucose_trace: np.ndarray,
    sbp_trace: Optional[np.ndarray] = None,
    time_step_minutes: float = 5.0,
) -> ClinicalOutcomes:
    """
    Compute clinical endpoints from a glucose trace.

    Args:
        glucose_trace: Array of glucose values in mg/dL
        sbp_trace: Optional array of SBP values
        time_step_minutes: Time between measurements (default 5 min)

    Returns:
        ClinicalOutcomes object with all endpoints
    """
    g = np.asarray(glucose_trace)
    n_hours = len(g) * time_step_minutes / 60.0
    n_days = n_hours / 24.0

    tir = float(np.mean((g >= 70) & (g <= 180)) * 100)
    tbr = float(np.mean(g < 70) * 100)
    tar = float(np.mean(g > 180) * 100)
    tar_250 = float(np.mean(g > 250) * 100)
    tir_70_140 = float(np.mean((g >= 70) & (g <= 140)) * 100)

    mean_g = float(np.mean(g))
    sd_g = float(np.std(g))

    # eA1c from mean glucose (NGSP formula: eA1c = (mean_glucose + 46.7) / 28.7)
    # Ref: Nathan et al. Diabetes Care 2008;31(8):1473-1478
    eA1c = float((mean_g + 46.7) / 28.7) if mean_g > 0 else 0.0
    # GMI (Glucose Management Indicator)
    # Ref: Beck et al. Diabetes Care 2018;41(11):2275-2280
    gmi = float(3.31 + 0.02392 * mean_g) if mean_g > 0 else 0.0

    # Hypoglycemia events: drops below 70 then recovers above 70
    hypo_below = g < 70
    if np.any(hypo_below):
        hypo_starts = np.where(np.diff(hypo_below.astype(int)) == 1)[0]
        hypo_events = len(hypo_starts)
    else:
        hypo_events = 0
    hypo_rate = hypo_events / max(n_days, 0.1)

    sbp_mean = float(np.mean(sbp_trace)) if sbp_trace is not None else 0.0
    sbp_control = float(np.mean(np.asarray(sbp_trace) < 130) * 100) if sbp_trace is not None else 0.0
    dbp_mean = 0.0
    hr_mean = 0.0

    return ClinicalOutcomes(
        tir=tir, tbr=tbr, tar=tar, tar_250=tar_250,
        eA1c=eA1c, mean_glucose=mean_g, glucose_sd=sd_g,
        hypo_events_per_day=hypo_rate,
        sbp_mean=sbp_mean, sbp_control_pct=sbp_control,
        dbp_mean=dbp_mean, hr_mean=hr_mean,
        tir_70_140=tir_70_140, gmi=gmi,
    )


class RCTSimulator:
    """
    Randomized Controlled Trial simulator.

    Simulates a two-arm trial comparing intervention vs control
    on clinical outcomes. Computes power, effect size, and NNT.

    Usage:
        sim = RCTSimulator()
        result = sim.run_trial(
            control_fn=lambda s, p: s,  # standard care
            intervention_fn=my_treatment_protocol,
            n_arm=100,
            n_steps=288,  # 1 day at 5-min intervals
        )
        print(f"TIR improvement: {result.tir_delta:.1f}% (p={result.p_value:.3f})")
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def run_trial(
        self,
        control_fn: Callable,
        intervention_fn: Callable,
        n_arm: int = 50,
        n_steps: int = 288,
        outcome_fn: Optional[Callable] = None,
        params_sampler: Optional[Callable] = None,
    ) -> "TrialResult":
        control_outcomes = []
        intervention_outcomes = []

        for _ in range(n_arm):
            params = DEFAULT_PARAMS.copy()
            params[0] = np.exp(self.rng.normal(-4.0, 0.3))
            params[2] = np.exp(self.rng.normal(-6.0, 0.4))
            params[5] = self.rng.lognormal(4.5, 0.2)

            state = np.zeros(PHYSIO_DIM)
            state[0] = self.rng.normal(160, 20)
            state[1] = 0.013 * max(0, state[0] - 80)
            state[5] = self.rng.normal(135, 10)
            state[6] = self.rng.normal(85, 8)
            state[7] = self.rng.normal(75, 5)
            state[14] = 1.0
            state[15] = 1.0
            state[16] = self.rng.normal(350, 50)

            # Control arm
            s_ctrl = state.copy()
            g_ctrl = []
            sbp_ctrl = []
            for t in range(n_steps):
                s_ctrl = control_fn(s_ctrl, params)
                g_ctrl.append(s_ctrl[0])
                sbp_ctrl.append(s_ctrl[5])
            control_outcomes.append(compute_clinical_outcomes(
                np.array(g_ctrl), np.array(sbp_ctrl)
            ))

            # Intervention arm
            s_int = state.copy()
            g_int = []
            sbp_int = []
            for t in range(n_steps):
                s_int = intervention_fn(s_int, params)
                g_int.append(s_int[0])
                sbp_int.append(s_int[5])
            intervention_outcomes.append(compute_clinical_outcomes(
                np.array(g_int), np.array(sbp_int)
            ))

        return self._analyze(control_outcomes, intervention_outcomes)

    def _analyze(
        self,
        control: List[ClinicalOutcomes],
        intervention: List[ClinicalOutcomes],
    ) -> "TrialResult":
        from scipy import stats as sp_stats

        c_tir = np.array([o.tir for o in control])
        i_tir = np.array([o.tir for o in intervention])
        c_eA1c = np.array([o.eA1c for o in control])
        i_eA1c = np.array([o.eA1c for o in intervention])
        c_hypo = np.array([o.hypo_events_per_day for o in control])
        i_hypo = np.array([o.hypo_events_per_day for o in intervention])

        tir_delta = float(np.mean(i_tir - c_tir))
        a1c_delta = float(np.mean(i_eA1c - c_eA1c))
        hypo_delta = float(np.mean(i_hypo - c_hypo))

        # Two-sample t-test
        if len(c_tir) > 1 and len(i_tir) > 1:
            _, tir_p = sp_stats.ttest_ind(i_tir, c_tir)
            _, a1c_p = sp_stats.ttest_ind(i_eA1c, c_eA1c)
        else:
            tir_p = 1.0; a1c_p = 1.0

        # Number Needed to Treat (NNT) for patients achieving TIR > 70%
        # TIR > 70% is the standard ADA clinical target
        # Ref: Battelino et al. Diabetes Care 2019;42(8):1593-1603
        tir_target = 70.0
        responders_c = np.mean(c_tir > tir_target)
        responders_i = np.mean(i_tir > tir_target)
        arr = responders_i - responders_c
        nnt = 1.0 / max(arr, 0.01)

        return TrialResult(
            control_tir=float(np.mean(c_tir)),
            intervention_tir=float(np.mean(i_tir)),
            tir_delta=tir_delta,
            tir_p_value=float(tir_p),
            a1c_delta=a1c_delta,
            a1c_p_value=float(a1c_p),
            hypo_delta=hypo_delta,
            nnt=int(np.ceil(nnt)),
            n_per_arm=len(control),
        )


@dataclass
class TrialResult:
    control_tir: float
    intervention_tir: float
    tir_delta: float
    tir_p_value: float
    a1c_delta: float
    a1c_p_value: float
    hypo_delta: float
    nnt: int
    n_per_arm: int

    def summary(self) -> str:
        stars = "***" if self.tir_p_value < 0.001 else "**" if self.tir_p_value < 0.01 else "*" if self.tir_p_value < 0.05 else "ns"
        return (
            f"TIR: {self.control_tir:.1f}% → {self.intervention_tir:.1f}% "
            f"(Δ={self.tir_delta:+.1f}%, p={self.tir_p_value:.4f} {stars})\n"
            f"eA1c Δ={self.a1c_delta:+.2f}% (p={self.a1c_p_value:.4f})\n"
            f"Hypo Δ={self.hypo_delta:+.2f} events/d | NNT={self.nnt}\n"
            f"N={self.n_per_arm} per arm"
        )

    def power(self, alpha: float = 0.05, n_sim: int = 100) -> float:
        """Compute statistical power via simulation."""
        from scipy import stats as sp_stats
        n = self.n_per_arm
        control_mu = self.control_tir
        effect = self.tir_delta
        control_std = 15.0

        significant = 0
        for _ in range(n_sim):
            c = np.random.randn(n) * control_std + control_mu
            i = np.random.randn(n) * control_std + control_mu + effect
            _, p = sp_stats.ttest_ind(i, c)
            if p < alpha:
                significant += 1
        return significant / n_sim
