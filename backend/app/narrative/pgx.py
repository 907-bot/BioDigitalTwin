"""PGx narrative generators."""
from typing import List, Dict, Any
from ._utils import risk_from_severity


def narrate_profile(patient_id: str, genes: List[Dict[str, Any]]) -> Dict[str, str]:
    if not genes:
        return {
            "headline": f"No PGx data for {patient_id}",
            "lay": f"We don't have pharmacogenomic data for {patient_id}.",
            "scientist": f"PGx panel not populated for {patient_id}. Call /phase8/attach-pgx to generate.",
            "risk_level": "low",
        }
    n_impaired = sum(1 for g in genes if g["activity"] < 1.0)
    n_enhanced = sum(1 for g in genes if g["activity"] > 1.5)
    impaired_genes = [g["gene"] for g in genes if g["activity"] < 1.0]
    enhanced_genes = [g["gene"] for g in genes if g["activity"] > 1.5]

    if n_impaired == 0 and n_enhanced == 0:
        lay = (f"Patient {patient_id} has a normal pharmacogenomic profile — all "
               f"{len(genes)} drug-metabolizing enzymes are working as expected. "
               f"Standard dosing is appropriate for most medications.")
        sci = (f"PGx panel for {patient_id}: all {len(genes)} CYP/Phase-II genes show "
               f"extensive-metabolizer (EM) phenotype (activity 1.0). No dose "
               f"adjustments warranted on pharmacogenomic grounds.")
        risk = "low"
    elif n_impaired > 0 and n_enhanced == 0:
        genes_str = ", ".join(impaired_genes)
        lay = (f"Patient {patient_id} has reduced activity in {n_impaired} of "
               f"{len(genes)} drug-metabolizing genes ({genes_str}). Their body "
               f"will break down certain drugs more slowly than normal, which can "
               f"lead to higher drug levels and side effects. Reduced doses or "
               f"alternative medications may be needed.")
        sci = (f"PGx panel for {patient_id}: {n_impaired}/{len(genes)} gene(s) show "
               f"reduced activity — {genes_str}. Intermediate or poor metabolizer "
               f"phenotypes for these CYP/Phase-II enzymes predict altered drug "
               f"exposure (typically 1.5-3× higher). CPIC guidelines recommend "
               f"dose reduction or alternative agents for substrates of these enzymes.")
        risk = "moderate" if n_impaired <= 2 else "high"
    elif n_enhanced > 0 and n_impaired == 0:
        genes_str = ", ".join(enhanced_genes)
        lay = (f"Patient {patient_id} is an ultra-rapid metabolizer for "
               f"{n_enhanced} gene(s) ({genes_str}). Their body will process certain "
               f"drugs much faster than normal — standard doses may be ineffective. "
               f"Higher doses or alternative medications may be needed.")
        sci = (f"PGx panel for {patient_id}: ultra-rapid metabolizer (UM) phenotype "
               f"detected for {genes_str}. Activity >1.5× predicts increased drug "
               f"clearance and reduced plasma exposure. CPIC guidelines recommend "
               f"alternative agents for prodrugs (reduced activation) and dose "
               f"increases for parent drugs.")
        risk = "moderate"
    else:
        lay = (f"Patient {patient_id} has a mixed pharmacogenomic profile — some "
               f"enzymes work faster and some slower than normal. Drug choices "
               f"and doses need to be tailored to the specific medication.")
        sci = (f"PGx panel for {patient_id}: heterogeneous phenotype. Reduced "
               f"activity in {', '.join(impaired_genes)} (impact 1.5-3×) and enhanced "
               f"activity in {', '.join(enhanced_genes)} (impact 0.4-0.7×). "
               f"Per-drug guidance required; consult CPIC tables for each affected gene.")
        risk = "high"

    return {"headline": f"PGx profile for {patient_id}", "lay": lay,
            "scientist": sci, "risk_level": risk}


def narrate_check(patient_id: str, drugs: List[str], warnings: List[Dict[str, Any]],
                  highest_severity: str | None) -> Dict[str, str]:
    if not warnings:
        return {
            "headline": f"No pharmacogenomic risks for {patient_id} on {', '.join(drugs) or 'this regimen'}",
            "lay": (f"Based on {patient_id}'s genetic profile, none of the "
                    f"{len(drugs) if drugs else 'checked'} medication(s) are expected to "
                    f"cause problems related to how the body processes them. Standard "
                    f"dosing should be safe."),
            "scientist": (f"PGx check for {patient_id} against {len(drugs) if drugs else 0} "
                          f"drug(s): no drug-gene interactions triggered. All checked "
                          f"substrates fall within normal activity range for this patient's "
                          f"genotype."),
            "risk_level": "low",
        }

    n_critical = sum(1 for w in warnings if w.get("severity") == "critical")
    n_major = sum(1 for w in warnings if w.get("severity") == "major")
    risk = risk_from_severity(highest_severity or "minor")

    drug_list = ", ".join(sorted({w["drug"] for w in warnings}))
    gene_list = ", ".join(sorted({w["gene"] for w in warnings}))

    if n_critical > 0:
        lay = (f"⛔ Serious genetic risk: {patient_id} cannot safely take "
               f"{drug_list}. Their genes ({gene_list}) mean their body will "
               f"process these drugs dangerously — either the drug won't work, "
               f"or it will build up to toxic levels. Please consult a "
               f"pharmacist or doctor for alternative medications before taking them.")
    elif n_major > 0:
        lay = (f"⚠️ Important genetic warning: {patient_id} should use "
               f"{drug_list} with extra caution. Their genes ({gene_list}) "
               f"mean these drugs may work differently than expected. A dose "
               f"adjustment or closer monitoring is recommended.")
    else:
        lay = (f"Patient {patient_id} has a minor genetic consideration with "
               f"{drug_list}. Their genes ({gene_list}) may slightly affect how "
               f"these drugs work, but standard dosing is usually fine.")

    critical_details = "; ".join(
        f"{w['drug']} via {w['gene']} ({w.get('clinical_note', w.get('severity', ''))})"
        for w in warnings if w.get("severity") == "critical"
    )[:300]
    sci = (f"PGx check for {patient_id} on {len(drugs)} drug(s) yielded {len(warnings)} "
           f"warning(s): {n_critical} critical, {n_major} major. Affected genes: "
           f"{gene_list}. ")
    if critical_details:
        sci += f"Critical interactions: {critical_details}. "
    sci += (f"CPIC levels and severity classifications per curated drug-gene "
            f"registry. Recommend clinical review before dispensing.")

    return {"headline": f"{len(warnings)} pharmacogenomic warning(s) for {patient_id}",
            "lay": lay, "scientist": sci, "risk_level": risk}


def narrate_warning(drug: str, gene: str, patient_status: str,
                    severity: str, is_prodrug: bool, impact: float,
                    clinical_note: str) -> Dict[str, str]:
    risk = risk_from_severity(severity)
    if is_prodrug:
        if patient_status in ("PM", "IM"):
            lay = (f"⛔ {drug} is risky for this patient. {drug} needs to be "
                   f"converted into its active form by the {gene} enzyme, but "
                   f"this patient is a {patient_status} (slow converter). The drug "
                   f"may not work properly and could cause side effects. "
                   f"Alternatives: ask your doctor for a different medication.")
            sci = (f"{drug} is a prodrug requiring bioactivation by {gene}. "
                   f"Patient is {patient_status} (activity {impact:.2f}×). "
                   f"Predicted morphine-equivalent exposure is reduced "
                   f"({impact:.2f}× baseline) → inadequate analgesia. "
                   f"CPIC recommends alternative non-{gene} analgesic.")
        else:
            lay = (f"⚠️ {drug} may be processed too quickly by this patient. "
                   f"Their {gene} enzyme works {impact:.1f}× faster than normal, "
                   f"which can cause too much active drug to be produced. "
                   f"Monitor for side effects.")
            sci = (f"{drug} is a prodrug; patient is {patient_status} (activity "
                   f"{impact:.2f}×). Enhanced bioactivation increases exposure "
                   f"to active metabolite. CPIC recommends dose reduction or "
                   f"alternative.")
    else:
        if patient_status in ("PM", "IM"):
            lay = (f"⛔ {drug} will build up in this patient's body. Their "
                   f"{gene} enzyme (which clears {drug}) is too slow, so the "
                   f"drug will reach higher-than-normal levels. This increases "
                   f"the risk of side effects. A lower dose is usually needed.")
            sci = (f"{drug} is cleared by {gene}. Patient is {patient_status} "
                   f"(activity {impact:.2f}×). Impaired clearance produces "
                   f"elevated plasma concentrations ({1/impact:.1f}× accumulation "
                   f"at steady state). CPIC recommends 30-50% dose reduction "
                   f"with therapeutic drug monitoring.")
        else:
            lay = (f"{drug} may be cleared too quickly by this patient. Their "
                   f"{gene} enzyme works faster than normal, so standard doses "
                   f"might be too low. Watch for reduced effectiveness.")
            sci = (f"{drug} is cleared by {gene}. Patient is {patient_status} "
                   f"(activity {impact:.2f}×). Enhanced clearance reduces plasma "
                   f"exposure. Standard dose may be sub-therapeutic; consider "
                   f"plasma-level monitoring.")

    return {
        "headline": f"{severity.upper()}: {drug} × {gene}",
        "lay": lay, "scientist": sci, "risk_level": risk,
    }
