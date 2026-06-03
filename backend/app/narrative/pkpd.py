"""PK/PD narrative generators."""
from typing import Dict, Any
from ._utils import risk_from_severity


def narrate_pk(drug: str, dose_mg: float, n_doses: int, interval_h: float,
               patient: Dict[str, Any], metrics: Dict[str, Any],
               adjustments: Dict[str, Any] | None = None,
               validation: list | None = None) -> Dict[str, str]:
    cmax = metrics.get("cmax_ss", metrics.get("cmax", 0))
    t_half = metrics.get("half_life_h", 0)
    accumulation = metrics.get("accumulation_ratio", 1.0)
    time_to_ss = metrics.get("time_to_steady_state_h", 0)
    cl = metrics.get("clearance_L_per_h", 0)
    auc = metrics.get("auc_0_inf", 0)
    ss_reached = time_to_ss <= (n_doses * interval_h)

    if t_half < 4:
        t_half_lay = "very short"
    elif t_half < 12:
        t_half_lay = "short"
    elif t_half < 36:
        t_half_lay = "moderate"
    else:
        t_half_lay = "long"

    age = patient.get("age", 70)
    crcl = adjustments.get("crcl_ml_min", 100) if adjustments else 100
    renal_factor = adjustments.get("renal_factor", 1.0) if adjustments else 1.0

    if crcl < 30:
        renal_lay = "severely reduced kidney function"
        risk = "high"
    elif crcl < 60:
        renal_lay = "moderately reduced kidney function"
        risk = "moderate"
    elif crcl < 90:
        renal_lay = "mildly reduced kidney function"
        risk = "low"
    else:
        renal_lay = "normal kidney function"
        risk = "low"

    lay = (f"After {n_doses} doses of {dose_mg} mg {drug} every {interval_h:.0f} hours, "
           f"the patient's peak blood level reaches {cmax:.2f} mg/L. The drug has a "
           f"{t_half_lay} half-life ({t_half:.1f} hours), so it ")
    if t_half > 24:
        lay += "builds up significantly in the body over time"
    elif t_half > 8:
        lay += "moderately accumulates with repeated dosing"
    else:
        lay += "is mostly cleared between doses"

    if accumulation > 1.5:
        lay += (f" (reaching {accumulation:.1f}× the first-dose level at steady state). ")
    else:
        lay += ". "

    lay += f"The patient has {renal_lay}"
    if renal_factor < 0.8:
        lay += f", which is why the drug clearance is {renal_factor:.0%} of normal — they clear it more slowly"
    elif renal_factor > 1.1:
        lay += ", but their kidney-driven clearance is enhanced"
    else:
        lay += ", and drug clearance is near normal"

    if ss_reached:
        lay += f". Steady state is reached around day {time_to_ss/24:.1f}."
    else:
        lay += (f". Steady state is NOT reached within the {n_doses}-dose window "
                f"(would need ~{time_to_ss/24:.1f} days).")

    sci = (f"Population PK simulation: 2-compartment model with first-order absorption "
           f"(ka from literature, ODE solved by LSODA). Patient covariates: "
           f"{age} y, {patient.get('weight_kg', 70)} kg, "
           f"CrCl {crcl:.0f} mL/min (Cockcroft-Gault). Renal adjustment factor: "
           f"{renal_factor:.2f}. Predicted metrics: Cmax_ss={cmax:.3f} mg/L, "
           f"t½={t_half:.2f} h, CL/F={cl:.3f} L/h, AUC₀-∞={auc:.2f} mg·h/L, "
           f"accumulation ratio={accumulation:.2f}, time-to-steady-state="
           f"{time_to_ss:.1f} h. ")
    if validation:
        passed = sum(1 for v in validation if v.get("status") == "pass")
        sci += f"Validation: {passed}/{len(validation)} checks passed."

    return {"headline": f"PK prediction for {dose_mg}mg {drug} q{interval_h:.0f}h",
            "lay": lay, "scientist": sci, "risk_level": risk}


def narrate_pd(drug: str, biomarker: str, baseline: float, peak_mg_L: float,
               pd_model: str, effect_at_tmax: float, max_effect: float,
               min_effect: float, pd_unit: str) -> Dict[str, str]:
    # In the router: for "decrease" direction min_effect=baseline(highest),
    # max_effect=drug nadir(lowest). For "increase" it's reversed.
    # We treat whichever is the larger absolute value as the baseline and
    # the other as the post-drug value.
    is_decrease = min_effect > max_effect
    start_val = min_effect if is_decrease else max_effect
    end_val = max_effect if is_decrease else min_effect
    delta = start_val - end_val
    pct = (delta / start_val * 100) if start_val != 0 else 0
    direction = "decreased" if is_decrease else "increased"

    if abs(pct) < 5:
        lay = (f"At the predicted blood level of {peak_mg_L:.2f} mg/L, {drug} has "
               f"minimal effect on {biomarker}. The patient's {biomarker} stays "
               f"close to {baseline:.0f} {pd_unit}.")
        risk = "low"
    elif abs(pct) < 20:
        lay = (f"{drug.capitalize()} {direction} the patient's {biomarker} by "
               f"{abs(pct):.0f}% (from {start_val:.0f} to {end_val:.0f} "
               f"{pd_unit}) at the predicted drug level. A modest but noticeable "
               f"effect.")
        risk = "low"
    elif abs(pct) < 50:
        lay = (f"{drug.capitalize()} substantially {direction} {biomarker} by "
               f"{abs(pct):.0f}% (from {start_val:.0f} to {end_val:.0f} "
               f"{pd_unit}) at the predicted drug level. A meaningful clinical "
               f"effect.")
        risk = "moderate"
    else:
        lay = (f"{drug.capitalize()} strongly {direction} {biomarker} by "
               f"{abs(pct):.0f}% (from {start_val:.0f} to {end_val:.0f} "
               f"{pd_unit}) at the predicted drug level. Strong pharmacological "
               f"response.")
        risk = "high"

    sci = (f"PD model: {pd_model}. Baseline {biomarker}={baseline:.1f} {pd_unit} "
           f"(model E0). At peak plasma concentration {peak_mg_L:.3f} mg/L, "
           f"predicted effect ={effect_at_tmax:.2f} {pd_unit} "
           f"(Δ={delta:+.2f} {pd_unit}, {pct:+.1f}% from baseline). "
           f"Time-course trajectory: baseline={start_val:.2f}, nadir={end_val:.2f} "
           f"{pd_unit}.")
    return {"headline": f"PD: {drug} → {biomarker}", "lay": lay,
            "scientist": sci, "risk_level": risk}
