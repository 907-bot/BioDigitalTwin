"""UQ narrative generators."""
from typing import Dict, Any
from ._utils import risk_from_severity


def narrate_counterfactual(patient_id: str, treatment: str, biomarker: str,
                            value: float, outcome: str, effect: Dict[str, Any],
                            direction_stability: float, n_bootstrap: int) -> Dict[str, str]:
    mean = effect.get("mean", 0)
    ci_lo = effect.get("ci_lo", 0)
    ci_hi = effect.get("ci_hi", 0)
    ci_level = effect.get("ci_level", 0.9)
    rel = effect.get("ci_width_relative", 0)
    ci_width = ci_hi - ci_lo

    crosses_zero = (ci_lo < 0 < ci_hi)
    excludes_zero = not crosses_zero
    sign = "+" if mean > 0 else ("−" if mean < 0 else "±")

    if crosses_zero:
        lay = (f"Uncertain: forcing {treatment} to {value} on patient {patient_id}'s "
               f"{biomarker} could change {outcome} by anywhere from {ci_lo:+.1f} to "
               f"{ci_hi:+.1f} units. The estimate is {mean:+.2f} but the range crosses "
               f"zero, so we can't be sure of the direction of effect.")
        risk = "moderate"
    elif rel < 0.3:
        lay = (f"Confident: forcing {treatment} to {value} on patient {patient_id}'s "
               f"{biomarker} is predicted to {('increase' if mean>0 else 'decrease')} "
               f"{outcome} by about {abs(mean):.1f} units. The 90% confidence range "
               f"is narrow ({ci_lo:+.1f} to {ci_hi:+.1f}), so this prediction is reliable.")
        risk = "low"
    elif rel < 1.0:
        lay = (f"Reasonably confident: forcing {treatment} to {value} on patient "
               f"{patient_id}'s {biomarker} should {('increase' if mean>0 else 'decrease')} "
               f"{outcome} by about {abs(mean):.1f} units. There's some uncertainty — "
               f"the 90% confidence range is {ci_lo:+.1f} to {ci_hi:+.1f} — but the "
               f"effect is likely real.")
        risk = "low"
    else:
        lay = (f"Estimate suggests {('increase' if mean>0 else 'decrease')} of "
               f"{abs(mean):.1f} units in {outcome}, but the wide confidence range "
               f"({ci_lo:+.1f} to {ci_hi:+.1f}) means we can't be very precise. "
               f"Use this as a rough guide, not a definitive answer.")
        risk = "moderate"

    sci = (f"Bootstrap CIs ({ci_level*100:.0f}%): effect = {mean:+.3f} [{ci_lo:+.3f}, "
           f"{ci_hi:+.3f}] (n={n_bootstrap} resamples). Direction stability across "
           f"resamples: {direction_stability*100:.0f}%. CI width relative to effect: "
           f"{rel:.2f}. ")
    if excludes_zero:
        sci += (f"CI excludes zero — effect is statistically distinguishable from null "
                f"at the {ci_level*100:.0f}% level. ")
    else:
        sci += "CI crosses zero — null hypothesis cannot be rejected. "
    if direction_stability > 0.95:
        sci += "Direction is highly stable."
    elif direction_stability > 0.8:
        sci += "Direction is moderately stable."
    else:
        sci += "Direction is unstable — result is fragile."

    return {"headline": f"Effect of {treatment}→{outcome}: {sign}{abs(mean):.2f}",
            "lay": lay, "scientist": sci, "risk_level": risk}
