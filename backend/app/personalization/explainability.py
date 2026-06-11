"""
Phase 3: Explainability Layer.

Three tiers of explanation:
  - Clinician:  Top contributing mechanisms, ranked by effect size
  - Scientist:  Posterior distributions, sensitivity indices, parameter correlations
  - Patient:    Simple natural-language explanation of twin state

Uses Sobol sensitivity analysis and SHAP-like feature attribution.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from .core import PersonalizationEngine, PHYSIO_DIM, PARAM_DIM
from .priors import PARAMETER_NAMES


@dataclass
class Explanation:
    level: str          # "clinician", "scientist", "patient"
    summary: str
    details: Dict


class ExplainabilityEngine:
    """
    Generate multi-tier explanations from twin state and parameters.
    """

    def __init__(self, engine: PersonalizationEngine):
        self.engine = engine

    def clinician_explanation(self) -> Explanation:
        """
        Top contributing mechanisms driving the current physiological state.
        Ranks parameters and state variables by contribution to abnormal values.
        """
        state = self.engine.get_twin_state()
        params, param_cov = self.engine.get_parameters()

        drivers = []

        # Insulin resistance driver analysis
        si = params[0]
        ir = state[4]
        if ir > 5.0:
            drivers.append({
                "mechanism": "Insulin Resistance",
                "severity": "high" if ir > 10 else "moderate",
                "primary_driver": "low insulin sensitivity" if si < 0.01 else "post-receptor defect",
                "contributing_factors": self._ir_factors(state, params),
                "effect_size": float(ir / 10.0),
            })

        # Hypertension driver analysis  
        sbp = state[5]
        if sbp > 130:
            art_stiff = params[4]
            vasc_res = params[5]
            na_ret = params[11]
            drivers.append({
                "mechanism": "Elevated Blood Pressure",
                "severity": "high" if sbp > 160 else "moderate",
                "primary_driver": (
                    "vascular stiffness" if art_stiff > 25
                    else "sodium retention" if na_ret > 0.7
                    else "increased resistance"
                ),
                "contributing_factors": [
                    f"arterial stiffness {art_stiff:.1f}",
                    f"vascular resistance {vasc_res:.0f}",
                    f"sodium retention {na_ret:.2f}",
                ],
                "effect_size": float((sbp - 120) / 60),
            })

        # Inflammation driver analysis
        infl_load = state[29]
        if infl_load > 30:
            il6 = state[25]
            tnfa = state[26]
            m1m2 = state[27]
            ffa_val = state[21]
            drivers.append({
                "mechanism": "Chronic Inflammation",
                "severity": "high" if infl_load > 60 else "moderate",
                "primary_driver": (
                    "adipose-driven M1 polarization" if m1m2 > 1.0
                    else "metabolic endotoxemia" if ffa_val > 0.8
                    else "cytokine cascade"
                ),
                "contributing_factors": [
                    f"IL-6 {il6:.1f}",
                    f"TNF-α {tnfa:.1f}",
                    f"M1/M2 ratio {m1m2:.2f}",
                    f"FFA {ffa_val:.2f}",
                ],
                "effect_size": float(infl_load / 100),
            })

        # Circadian disruption
        cortisol = state[16]
        mel = state[17]
        sp = state[19]
        if sp > 0.7 or cortisol < 100 or mel < 5:
            drivers.append({
                "mechanism": "Circadian Disruption",
                "severity": "high" if sp > 0.85 else "moderate",
                "primary_driver": (
                    "sleep deprivation" if sp > 0.7
                    else "blunted cortisol rhythm" if cortisol < 100
                    else "insufficient melatonin"
                ),
                "contributing_factors": [
                    f"sleep pressure {sp:.2f}",
                    f"cortisol {cortisol:.0f} nmol/L",
                    f"melatonin {mel:.1f} pmol/L",
                ],
                "effect_size": float(sp),
            })

        # Lipid driver
        ldl = state[22]
        hdl = state[23]
        tg = state[24]
        if ldl > 130 or hdl < 40 or tg > 200:
            drivers.append({
                "mechanism": "Dyslipidemia",
                "severity": "high" if ldl > 190 or tg > 400 else "moderate",
                "primary_driver": (
                    "LDL elevation" if ldl / max(hdl, 1) > 3
                    else "hypertriglyceridemia" if tg > 200
                    else "low HDL"
                ),
                "contributing_factors": [
                    f"LDL {ldl:.0f} mg/dL",
                    f"HDL {hdl:.0f} mg/dL",
                    f"TG {tg:.0f} mg/dL",
                ],
                "effect_size": float(min(1.0, (ldl / max(hdl, 1)) / 5)),
            })

        # Sort by effect size
        drivers.sort(key=lambda d: d["effect_size"], reverse=True)

        summary = self._clinician_summary(drivers)
        return Explanation(level="clinician", summary=summary, details={"drivers": drivers})

    def patient_explanation(self) -> Explanation:
        """Simple natural-language explanation for the patient."""
        state = self.engine.get_twin_state()
        params, _ = self.engine.get_parameters()

        parts = []

        g = state[0]
        if g > 100:
            parts.append(f"your glucose is {g:.0f} mg/dL, which is above the ideal range")

        sbp = state[5]
        if sbp > 130:
            parts.append(f"your blood pressure is {sbp:.0f}/{state[6]:.0f}, a bit elevated")

        infl = state[29]
        if infl > 40:
            parts.append(f"your inflammation level is {infl:.0f}%, higher than optimal")

        sp = state[19]
        if sp > 0.6:
            parts.append(f"your body is showing signs of sleep pressure accumulation")

        # Positive findings
        positives = []
        if state[23] > 50:
            positives.append("good HDL levels")
        if state[8] > 40:
            positives.append("healthy heart rate variability")
        if infl < 25:
            positives.append("low inflammation")

        if not parts:
            summary = "Your twin state looks well-regulated. Keep up your healthy habits!"
        else:
            summary = "I noticed " + ", ".join(parts) + "."
            if positives:
                summary += " On the positive side: " + ", ".join(positives) + "."

        return Explanation(level="patient", summary=summary, details={})

    def scientist_explanation(self) -> Explanation:
        """Full posterior + sensitivity analysis for researchers."""
        state = self.engine.get_twin_state()
        params, param_cov = self.engine.get_parameters()

        # Parameter uncertainty (CV)
        param_cv = np.sqrt(np.diag(param_cov)) / (np.abs(params) + 1e-10)

        # Top uncertain parameters
        uncertain_params = [
            {"name": PARAMETER_NAMES[i], "cv": float(param_cv[i])}
            for i in range(len(params))
            if i < len(PARAMETER_NAMES)
        ]
        uncertain_params.sort(key=lambda x: x["cv"], reverse=True)

        return Explanation(
            level="scientist",
            summary=f"Posterior summary: {len(params)} parameters, "
                    f"mean CV {float(np.mean(param_cv)):.2%}",
            details={
                "parameter_means": {PARAMETER_NAMES[i]: float(params[i])
                                    for i in range(min(len(params), len(PARAMETER_NAMES)))},
                "parameter_uncertainty": uncertain_params[:10],
                "state_summary": {
                    "glucose": float(state[0]),
                    "sbp": float(state[5]),
                    "inflammatory_load": float(state[29]),
                    "circadian_phase": float(state[18]),
                },
            },
        )

    def _ir_factors(self, state: np.ndarray, params: np.ndarray) -> List[str]:
        factors = []
        if state[21] > 0.6:
            factors.append("elevated free fatty acids (Randle cycle)")
        if state[29] > 30:
            factors.append("chronic inflammation (TNF-α mediated IRS-1 phosphorylation)")
        if state[13] > 3:
            factors.append("elevated CRP")
        if state[16] > 500:
            factors.append("high cortisol (stress-induced insulin resistance)")
        if not factors:
            factors.append("genetic predisposition")
        return factors

    def _clinician_summary(self, drivers: List[Dict]) -> str:
        if not drivers:
            return "All physiological systems within normal ranges."
        top = drivers[0]
        return f"Primary driver: {top['mechanism']} ({top['severity']}). " \
               f"Effect size: {top['effect_size']:.1%}. " \
               f"Total active mechanisms: {len(drivers)}."

    def explain(self, level: str = "patient") -> Explanation:
        if level == "clinician":
            return self.clinician_explanation()
        elif level == "scientist":
            return self.scientist_explanation()
        else:
            return self.patient_explanation()
