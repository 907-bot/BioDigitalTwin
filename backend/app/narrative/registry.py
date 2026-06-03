"""Registry narrative generators."""
from typing import Dict, Any, List
from ._utils import risk_from_severity


def narrate_summary(summary: Dict[str, Any]) -> Dict[str, str]:
    n = summary.get("n_diseases", 0)
    proteins = summary.get("n_target_proteins", 0)
    trials = summary.get("total_clinical_trials", 0)
    by_need = summary.get("by_unmet_need", {})
    n_crit = by_need.get("critical", 0)
    n_high = by_need.get("high", 0)

    lay = (f"The disease registry contains {n} entries covering {proteins} known "
           f"target proteins and {trials:,} active or completed clinical trials. "
           f"{n_crit} disease(s) are marked as having a critical unmet medical need, "
           f"and {n_high} are classified as high-need. The registry is the source of "
           f"truth for the drug-discovery prioritization pipeline.")

    sci = (f"Disease registry summary: {n} entries indexed. {proteins} unique target "
           f"proteins catalogued. {trials:,} total clinical trials across all entries "
           f"(sourced from ClinicalTrials.gov cross-reference). Unmet-need "
           f"distribution: {by_need}.")

    return {"headline": f"Disease registry: {n} entries",
            "lay": lay, "scientist": sci, "risk_level": "low"}


def narrate_disease(d: Dict[str, Any]) -> Dict[str, str]:
    name = d.get("name", d.get("key", "?"))
    need = d.get("unmet_need", "medium")
    n_proteins = len(d.get("target_proteins", []))
    n_trials = d.get("clinical_trials", 0) or 0
    treatments = d.get("current_treatments", "none")

    risk = "low"
    if need == "critical":
        risk = "high"
    elif need == "high":
        risk = "moderate"

    if need == "critical":
        lay = (f"⛔ {name} is a critical unmet medical need. Currently approved "
               f"treatments ({treatments}) are inadequate. This is a high-priority "
               f"target for new therapeutic discovery.")
    elif need == "high":
        lay = (f"⚠️ {name} has a high unmet medical need. Existing treatments "
               f"({treatments}) work for some patients but there is significant room "
               f"for improvement.")
    else:
        lay = (f"{name} has moderate unmet need. Current standard of care "
               f"({treatments}) is generally effective.")

    sci = (f"Disease entry: {name} (key={d.get('key')}). Unmet need: {need}. "
           f"Target proteins: {n_proteins} ({', '.join(d.get('target_proteins', [])[:5])}"
           f"{'...' if n_proteins > 5 else ''}). Clinical trials: {n_trials}. "
           f"Current treatments: {treatments}. "
           f"Added: {d.get('added_at', '?')}, by {d.get('added_by', '?')}.")

    return {"headline": f"{name} (unmet need: {need})",
            "lay": lay, "scientist": sci, "risk_level": risk}
