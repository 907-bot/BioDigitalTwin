"""Regulatory narrative generators."""
from typing import Dict, Any
from ._utils import risk_from_severity


def narrate_profile(drug: str, profile: Dict[str, Any]) -> Dict[str, str]:
    ob = profile.get("orange_book") or {}
    safety = profile.get("safety") or {}
    faers = profile.get("faers") or {}

    black_box = bool(safety.get("black_box_warnings"))
    n_contra = len(safety.get("contraindications", []))
    n_ae = len(safety.get("common_adverse_events", []))
    pregnancy = safety.get("pregnancy_category", "?")
    n_approval = len(ob.get("entries", []))
    first_approval = ob["entries"][0].get("approval_date", "?") if n_approval else None
    faers_total = faers.get("total_reports", 0)
    n_top_reactions = len(faers.get("top_reactions", []))

    risk = "low"
    if black_box:
        risk = "critical"
    elif pregnancy in ("X", "D"):
        risk = "high"
    elif faers_total > 100000:
        risk = "moderate"

    if black_box:
        lay = (f"⛔ {drug.capitalize()} carries a 'Black Box Warning' from the FDA — "
               f"the strongest warning the agency issues. This means {drug} can cause "
               f"serious or life-threatening side effects. It is approved and effective, "
               f"but must be used with extra care and close medical supervision. "
               f"{safety['black_box_warnings'][0]}")
    elif n_contra > 0 and pregnancy == "X":
        lay = (f"⚠️ {drug.capitalize()} is FDA-approved but has {n_contra} known "
               f"contraindication(s) and must not be used in pregnancy (category X). "
               f"It is only safe for carefully selected patients.")
    else:
        lay = (f"{drug.capitalize()} is FDA-approved")
        if n_approval > 0 and first_approval:
            lay += f" (since {first_approval[:4]})"
        lay += (f", with {n_ae} common side effect(s) on record. "
                f"It has been used by a very large number of patients and is generally "
                f"considered safe when prescribed appropriately.")

    sci = (f"FDA regulatory profile for {drug}. Orange Book: {n_approval} approval "
           f"record(s)")
    if first_approval:
        sci += f", first approved {first_approval}"
    sci += (f". Safety: {n_contra} contraindication(s), {n_ae} common AE(s), "
            f"pregnancy category {pregnancy}. ")
    if black_box:
        sci += f"⚠ Black Box Warning present: {safety['black_box_warnings'][0]}. "
    sci += (f"FAERS (live OpenFDA): {faers_total:,} total adverse event reports, "
            f"top {n_top_reactions} reactions catalogued. ")
    if safety.get("typical_dose"):
        sci += f"Typical dose: {safety['typical_dose']}. "
    if safety.get("notes"):
        sci += f"Clinical note: {safety['notes']}"

    return {"headline": f"Regulatory profile: {drug}", "lay": lay,
            "scientist": sci, "risk_level": risk}
