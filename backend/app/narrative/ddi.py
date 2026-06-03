"""DDI narrative generators."""
from typing import List, Dict, Any
from ._utils import risk_from_severity


def narrate_check(drugs: List[str], interactions: List[Dict[str, Any]],
                  overall_severity: str, n_direct: int, n_inferred: int) -> Dict[str, str]:
    n = len(interactions)
    if n == 0:
        return {
            "headline": f"No drug interactions found among {len(drugs)} medications",
            "lay": (f"None of the {len(drugs)} medications you entered are known to "
                    f"interact with each other in a clinically significant way. "
                    f"They should be safe to take together."),
            "scientist": (f"DDI check across {len(drugs)} drug(s) found no interactions "
                          f"in the curated knowledge base ({n_direct} direct, "
                          f"{n_inferred} inferred). Combinatorial CYP/transporter graph "
                          f"was searched for transitive interactions — none found."),
            "risk_level": "low",
        }
    n_contra = sum(1 for i in interactions if i.get("severity") == "contraindicated")
    n_major = sum(1 for i in interactions if i.get("severity") == "major")
    n_mod = sum(1 for i in interactions if i.get("severity") == "moderate")
    pairs = [f"{i['drug_a']}+{i['drug_b']}" for i in interactions[:5]]
    pairs_str = ", ".join(pairs)
    if len(interactions) > 5:
        pairs_str += f" (and {n-5} more)"

    if n_contra > 0:
        lay = (f"⛔ {n_contra} drug combination(s) are contraindicated — these "
               f"should NOT be taken together: {pairs_str}. They can cause "
               f"serious or life-threatening reactions. Please review this "
               f"regimen with your prescriber.")
    elif n_major > 0:
        lay = (f"⚠️ {n_major} drug combination(s) have a major interaction risk: "
               f"{pairs_str}. These combinations can cause significant side "
               f"effects or reduce how well the drugs work. Your doctor may "
               f"want to adjust doses, monitor lab values, or switch one of "
               f"the medications.")
    elif n_mod > 0:
        lay = (f"ℹ️ {n} drug combination(s) have moderate interactions: "
               f"{pairs_str}. These usually require monitoring but are "
               f"manageable with dose adjustments.")
    else:
        lay = (f"Minor drug interactions found among {n} pair(s): {pairs_str}. "
               f"These are unlikely to cause problems but worth being aware of.")

    mechs = list({i.get("mechanism", "") for i in interactions if i.get("mechanism")})[:3]
    sci = (f"DDI check: {n} interaction(s) in {len(drugs)}-drug regimen "
           f"({n_direct} direct, {n_inferred} inferred via CYP/transporter "
           f"transitive graph). Severity breakdown: {n_contra} contraindicated, "
           f"{n_major} major, {n_mod} moderate, {n - n_contra - n_major - n_mod} minor. ")
    if mechs:
        sci += f"Representative mechanisms: {'; '.join(mechs)}. "
    sci += "Source: curated FDA/CPIC/Lexicomp/Micromedex table."

    return {
        "headline": f"{n} drug interaction(s) in {len(drugs)}-drug regimen",
        "lay": lay, "scientist": sci, "risk_level": risk_from_severity(overall_severity),
    }


def narrate_pair(drug_a: str, drug_b: str, pair: Dict[str, Any] | None,
                 path: List[str] | None) -> Dict[str, str]:
    if pair:
        risk = risk_from_severity(pair.get("severity", "minor"))
        sev = pair.get("severity", "minor")
        if sev == "contraindicated":
            lay = (f"⛔ Do NOT take {drug_a} and {drug_b} together. They "
                   f"interact in a way that can cause serious harm.")
        elif sev == "major":
            lay = (f"⚠️ Taking {drug_a} with {drug_b} carries a major risk. "
                   f"This combination often requires close monitoring or "
                   f"dose adjustment.")
        else:
            lay = (f"{drug_a.capitalize()} and {drug_b} have a "
                   f"{sev} interaction. Usually manageable with monitoring.")
        sci = (f"Curated interaction: {drug_a} + {drug_b} = {sev}. "
               f"Mechanism: {pair.get('mechanism', 'unknown')}. Clinical effect: "
               f"{pair.get('clinical_effect', 'unknown')}. "
               f"Onset: {pair.get('on_set', '?')}. "
               f"Source: {pair.get('cpic_or_fda', 'curated')}. "
               f"Reference: clinical pharmacology databases.")
        headline = f"{sev.upper()}: {drug_a} + {drug_b}"
    elif path and len(path) > 2:
        path_str = " → ".join(path)
        lay = (f"{drug_a.capitalize()} and {drug_b} aren't in our direct "
               f"interaction table, but they may interact indirectly via the "
               f"{path[1]} metabolic pathway. This is a theoretical risk that "
               f"may or may not be clinically significant.")
        sci = (f"No direct interaction found in curated KB. Inferred interaction "
               f"via {path[1]} pathway: {path_str}. Mechanism: {drug_a} may "
               f"modulate {path[1]}, which is involved in {drug_b} metabolism. "
               f"Clinical significance uncertain — monitor if co-administered.")
        headline = f"Inferred: {drug_a} + {drug_b}"
        risk = "moderate"
    else:
        lay = (f"No known interaction between {drug_a} and {drug_b} in our "
               f"curated knowledge base or CYP/transporter graph.")
        sci = (f"No direct or inferred interaction identified. Searched: "
               f"(1) curated DDI table, (2) CYP/transporter graph for "
               f"transitive paths. Both negative.")
        headline = f"No interaction: {drug_a} + {drug_b}"
        risk = "low"

    return {"headline": headline, "lay": lay, "scientist": sci, "risk_level": risk}
