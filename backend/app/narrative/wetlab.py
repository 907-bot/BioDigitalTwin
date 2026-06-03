"""Wet-Lab narrative generators."""
from typing import Dict, Any
from ._utils import risk_from_severity


def narrate_validation(smiles: str, name: str | None, score: float, verdict: str,
                        properties: Dict[str, Any], drug_likeness: Dict[str, Any],
                        filters: Dict[str, Any], dose_response: Dict[str, Any],
                        targets: list, toxicity: list) -> Dict[str, str]:
    label = name or "this molecule"
    mw = properties.get("mw", 0)
    logp = properties.get("logp", 0)
    tpsa = properties.get("tpsa", 0)
    hbd = properties.get("hbd", 0)
    hba = properties.get("hba", 0)
    rot = properties.get("rotatable_bonds", 0)

    lipinski_v = drug_likeness.get("n_lipinski_violations", 0)
    pains_clean = filters.get("pains_clean", True)
    brenk_clean = filters.get("brenk_clean", True)
    sas = filters.get("sas", 0)
    ic50_nM = dose_response.get("ic50_nM", 0)
    hill = dose_response.get("hill_coefficient", 1)
    top_target = targets[0]["target"] if targets else None
    top_sim = targets[0]["tanimoto_similarity"] if targets else 0

    # Drug-likeness interpretation
    issues = []
    if mw > 500: issues.append("high molecular weight")
    if logp > 5: issues.append("high lipophilicity")
    if hbd + hba > 12: issues.append("too many H-bond donors/acceptors")
    if rot > 10: issues.append("excessive flexibility")
    if tpsa > 140: issues.append("high polar surface area (poor permeability)")

    risk = "low"
    if not pains_clean or not brenk_clean or score < 60:
        risk = "moderate"
    if toxicity:
        risk = "high"
    if "ready" in verdict:
        risk = "low"

    if score >= 80 and not toxicity:
        lay = (f"✅ {label.capitalize()} passes the screening suite with a score of "
               f"{score:.0f}/100. It has drug-like physicochemical properties, no "
               f"structural red flags, and is predicted to have a measurable biological "
               f"effect (IC50 ≈ {ic50_nM/1000:.1f} µM). Recommend advancing to wet-lab "
               f"validation. Estimated synthetic accessibility: {'easy' if sas < 3 else 'moderate' if sas < 5 else 'difficult'}.")
    elif score >= 60:
        issue_text = ", ".join(issues[:2]) if issues else "minor structural concerns"
        lay = (f"⚠️ {label.capitalize()} is borderline (score {score:.0f}/100) due to "
               f"{issue_text}. Wet-lab testing may be warranted but expect variable "
               f"results. Consider structural optimization if pursuing further.")
    else:
        issue_text = ", ".join(issues[:3]) if issues else "significant structural liabilities"
        lay = (f"❌ {label.capitalize()} fails the screen (score {score:.0f}/100). "
               f"Multiple concerns: {issue_text}. Not recommended for advancement "
               f"without revision.")

    sci = (f"RDKit validation for {label} (SMILES: {smiles[:40]}{'...' if len(smiles)>40 else ''}). "
           f"Physicochemical: MW={mw:.1f} Da, LogP={logp:.2f}, TPSA={tpsa:.1f} Å², "
           f"HBD={hbd}, HBA={hba}, RotB={rot}. Drug-likeness: Lipinski violations="
           f"{lipinski_v}, Veber violations={drug_likeness.get('n_veber_violations', 0)}. "
           f"Filters: PAINS={'clean' if pains_clean else 'hits '+str(len(filters.get('pains_matches',[])))}, "
           f"Brenk={'clean' if brenk_clean else 'alerts: '+', '.join(filters.get('brenk_matches',[]))}, "
           f"SAS={sas:.2f} (1=easy, 10=hard). Dose-response: IC50={ic50_nM:.0f} nM, "
           f"Hill coefficient={hill:.2f}. Top target prediction: {top_target} "
           f"(Tanimoto similarity {top_sim:.3f}). Toxicity alerts: "
           f"{len(toxicity)} (rule-based hepatotoxicity, hERG, PAINS-derived). "
           f"Overall score: {score:.0f}/100, verdict: {verdict}.")

    return {"headline": f"Validation: {label} (score {score:.0f}/100)",
            "lay": lay, "scientist": sci, "risk_level": risk}
