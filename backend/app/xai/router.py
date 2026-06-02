"""Phase 16 — Explainable AI router.

Composes the existing Phase 8-15 endpoints into a unified explanation layer.
Each endpoint takes a structured request, gathers evidence from the relevant
phase, runs feature attribution, and returns a reasoning chain.
"""
from __future__ import annotations
import os
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from .explainers import (
    shap_like_attribution,
    build_reasoning_chain,
    confidence_from_ci,
    topk_features,
)

router = APIRouter(prefix="/phase16", tags=["phase16-xai"])

PATIENTS_CSV = "data/synthetic_patients.csv"


# --------------------------------------------------------------------- #
#  Methods catalogue                                                     #
# --------------------------------------------------------------------- #
@router.get("/methods")
def methods() -> Dict[str, Any]:
    return {
        "methods": [
            {
                "id": "shap_lite",
                "name": "SHAP-lite (leave-one-out marginal contribution)",
                "scope": "any outcome that depends on a feature vector",
                "complexity": "O(n_features) outcome evaluations",
            },
            {
                "id": "reasoning_chain",
                "name": "Structured evidence chain",
                "scope": "decision explanation with question → evidence → conclusion",
                "complexity": "O(n_evidence)",
            },
            {
                "id": "bootstrap_ci",
                "name": "Bootstrap confidence interval → confidence label",
                "scope": "any scalar prediction",
                "complexity": "O(n_bootstrap) outcome evaluations",
            },
            {
                "id": "counterfactual_diff",
                "name": "Counterfactual diff (do-calculus trace)",
                "scope": "SCM / causal models",
                "complexity": "O(graph_size)",
            },
        ],
        "endpoints": [
            "POST /phase16/explain/counterfactual",
            "POST /phase16/explain/ddi",
            "POST /phase16/explain/pk",
            "POST /phase16/explain/pgx",
            "POST /phase16/explain/patient",
        ],
    }


# --------------------------------------------------------------------- #
#  helpers                                                               #
# --------------------------------------------------------------------- #
def _cohort_df() -> pd.DataFrame:
    if not os.path.exists(PATIENTS_CSV):
        raise HTTPException(404, "no cohort — call POST /generate-patients first")
    return pd.read_csv(PATIENTS_CSV)


def _patient_observed(pid: str) -> Dict[str, float]:
    df = _cohort_df()
    row = df[df["patient_id"] == pid]
    if row.empty:
        raise HTTPException(404, f"patient '{pid}' not found")
    r = row.iloc[0]
    from app.graph.ontology import BIOMARKERS
    obs = {b.id: float(r[b.id]) for b in BIOMARKERS}
    obs["bmi"] = float(r["bmi"])
    obs["age"] = float(r["age"])
    return obs


# --------------------------------------------------------------------- #
#  /explain/counterfactual                                               #
# --------------------------------------------------------------------- #
@router.post("/explain/counterfactual")
def explain_counterfactual(body: Dict[str, Any]) -> Dict[str, Any]:
    pid = body.get("patient_id")
    treatment = body.get("treatment")
    biomarker = body.get("biomarker")
    value = float(body.get("value", 1.0))
    outcome = body.get("outcome")
    n_boot = int(body.get("n_bootstrap", 50))
    if not all([pid, treatment, biomarker, outcome]):
        raise HTTPException(400, "patient_id, treatment, biomarker, outcome required")

    t0 = time.time()
    from app.uq.bootstrap import bootstrap_patient_counterfactual
    observed = _patient_observed(pid)
    df = _cohort_df()

    uq = bootstrap_patient_counterfactual(
        df, observed=observed, treatment=treatment, value=value,
        outcome=outcome, n_bootstrap=n_boot, confidence=0.9, seed=42,
    )
    effect = uq["effect"]["mean"]
    ci_lo, ci_hi = uq["effect"]["ci_lo"], uq["effect"]["ci_hi"]
    dir_stab = uq["direction_stability"]

    # Build a feature list and approximate the outcome with a linear model
    base_features = {
        "age": 50.0, "bmi": 25.0, "hr": 70.0, "hrv": 60.0, "spo2": 96.0,
        "glucose": 100.0, "systolic_bp": 120.0, "diastolic_bp": 80.0,
        biomarker: 100.0,
    }
    perturbed = {**base_features, biomarker: value}

    def outcome_fn(feats: Dict[str, float]) -> float:
        weights = {
            "age": 0.0, "bmi": 0.5, "hr": 0.1, "hrv": -0.1, "spo2": 0.05,
            "glucose": 0.3, "systolic_bp": 0.2, "diastolic_bp": 0.1,
        }
        return sum(weights.get(k, 0.0) * v for k, v in feats.items() if k != "__baseline") \
               + 0.4 * feats.get(biomarker, 100.0)

    contribs = shap_like_attribution(base_features, perturbed, outcome_fn)
    top = topk_features(contribs, k=5)
    conf = confidence_from_ci(ci_lo, ci_hi, effect, dir_stab)

    evidence = [
        {"fact": f"Bootstrap effect: {effect:.2f} (CI [{ci_lo:.2f}, {ci_hi:.2f}], n={n_boot})",
         "source": "phase11/uq/bootstrap", "weight": abs(effect) + 0.1},
        {"fact": f"Direction stability across resamples: {dir_stab*100:.0f}%",
         "source": "phase11/uq/bootstrap", "weight": dir_stab},
        *[{"fact": f"{c['feature']}: {c['base_value']:.1f} → {c['perturbed_value']:.1f} "
                  f"contributed {c['contribution']:+.2f} to outcome",
           "source": "phase16/shap_lite", "weight": abs(c["contribution"])}
          for c in top if c.get("feature") != "__total"],
    ]
    top_feat = next((c for c in top if c.get("feature") not in ("__total", None)), None)
    chain = build_reasoning_chain(
        question=f"Why does forcing {treatment}={value} on {biomarker} of patient {pid} change {outcome}?",
        evidence=evidence,
        conclusion=(
            f"The treatment is expected to change {outcome} by {effect:+.2f} units, "
            f"with {conf['label']} confidence. "
            + (f"Top driver: {top_feat['feature']} (contribution {top_feat['contribution']:+.2f})."
               if top_feat else "")
        ),
        confidence=conf["label"],
        confidence_score=conf["score"],
        alternative_hypotheses=(
            ["Effect could be confounded by unobserved covariates.",
             "Linear surrogate may miss nonlinear interactions."]
            if conf["label"] == "low" else
            ["Linear surrogate may miss higher-order interactions."]
            if conf["label"] == "medium" else []
        ),
    )
    return {
        "method": "shap_lite + bootstrap_ci + reasoning_chain",
        "latency_ms": int((time.time() - t0) * 1000),
        "input": {"patient_id": pid, "treatment": treatment, "biomarker": biomarker,
                  "value": value, "outcome": outcome, "n_bootstrap": n_boot},
        "point_estimate": effect,
        "ci": {"lo": ci_lo, "hi": ci_hi, "level": uq["effect"]["ci_level"]},
        "direction_stability": dir_stab,
        "feature_attribution": top,
        "confidence": conf,
        "reasoning_chain": chain,
    }


# --------------------------------------------------------------------- #
#  /explain/ddi                                                          #
# --------------------------------------------------------------------- #
@router.post("/explain/ddi")
def explain_ddi(body: Dict[str, Any]) -> Dict[str, Any]:
    drug_a = body.get("drug_a")
    drug_b = body.get("drug_b")
    if not (drug_a and drug_b):
        raise HTTPException(400, "drug_a and drug_b required")
    t0 = time.time()
    from app.ddi import database as ddi_db
    from app.ddi import graph as ddi_graph
    pair_list = ddi_db.find_pair(drug_a, drug_b)
    pair = pair_list[0] if pair_list else None
    # Derive a "transitive path" by walking CYP graph: find common enzymes
    path = None
    try:
        from app.ddi.graph import find_inhibitors, find_substrates
        for enzyme in ["CYP3A4", "CYP2C9", "CYP2D6", "CYP2C19", "CYP1A2"]:
            subs = set(find_substrates(enzyme))
            inhibs = set(find_inhibitors(enzyme))
            if (drug_a.lower() in inhibs and drug_b.lower() in subs) or \
               (drug_b.lower() in inhibs and drug_a.lower() in subs):
                path = [drug_a, enzyme, drug_b]
                break
    except Exception:
        path = None

    evidence: List[Dict[str, Any]] = []
    if pair:
        evidence.append({
            "fact": f"Curated interaction: {pair.mechanism} → {pair.clinical_effect}",
            "source": "phase9/ddi/database",
            "weight": {"contraindicated": 3.0, "major": 2.0, "moderate": 1.5, "minor": 0.5}.get(pair.severity, 1.0),
        })
    if path and len(path) > 2:
        evidence.append({
            "fact": f"Transitive path via CYP/transporter: {' → '.join(path)}",
            "source": "phase9/ddi/graph",
            "weight": 1.5,
        })
        for i in range(len(path) - 1):
            evidence.append({
                "fact": f"  hop {i+1}: {path[i]} modulates {path[i+1]}",
                "source": "phase9/ddi/graph",
                "weight": 0.5,
            })
    if not evidence:
        evidence.append({"fact": "No curated or graph-derived interaction found.",
                         "source": "phase9/ddi", "weight": 1.0})

    chain = build_reasoning_chain(
        question=f"Why do {drug_a} and {drug_b} interact?",
        evidence=evidence,
        conclusion=(
            f"Severity: {pair.severity}. Mechanism: {pair.mechanism}. Clinical effect: {pair.clinical_effect}."
            if pair else
            (f"Inferred interaction via {len(path)-1} metabolic hop(s)."
             if path and len(path) > 2 else
             "No mechanistic interaction identified in the curated knowledge base.")
        ),
        confidence="high" if pair else ("medium" if path and len(path) > 2 else "low"),
        confidence_score=1.0 if pair else (0.6 if path and len(path) > 2 else 0.2),
    )
    return {
        "method": "curated_kb + cyp_graph + reasoning_chain",
        "latency_ms": int((time.time() - t0) * 1000),
        "input": {"drug_a": drug_a, "drug_b": drug_b},
        "direct_interaction": (
            {"drug_a": pair.drug_a, "drug_b": pair.drug_b, "severity": pair.severity,
             "mechanism": pair.mechanism, "clinical_effect": pair.clinical_effect,
             "source": pair.cpic_or_fda}
            if pair else None
        ),
        "transitive_path": path,
        "reasoning_chain": chain,
    }


# --------------------------------------------------------------------- #
#  /explain/pk                                                           #
# --------------------------------------------------------------------- #
@router.post("/explain/pk")
def explain_pk(body: Dict[str, Any]) -> Dict[str, Any]:
    drug = body.get("drug")
    dose_mg = float(body.get("dose_mg", 100))
    patient = body.get("patient", {})
    t0 = time.time()
    from app.pkpd import compartments as pkpd_c
    from app.pkpd import registry as pkpd_reg

    rec = pkpd_reg.get_drug(drug)
    if rec is None:
        raise HTTPException(404, f"Drug '{drug}' not in PK/PD registry")

    weight = float(patient.get("weight_kg", 70))
    age = float(patient.get("age", 50))
    sex = patient.get("sex", "male")
    crcl = pkpd_c.cockcroft_gault_egfr(
        age=age, weight=weight, sex=sex,
        serum_cr=float(patient.get("serum_creatinine_mg_dl", 1.0)),
    )
    vd_per_kg = (rec.pk.Vc + rec.pk.Vp) / 70.0  # Vd per kg at 70kg
    allom = pkpd_c.allometric_scale(vd_per_kg, ref_weight=70, actual_weight=weight)
    frac_renal = pkpd_reg._frac_renal(drug)
    renal_factor = pkpd_c.adjust_for_renal(
        CL=rec.pk.CL, crcl=crcl, frac_renal=frac_renal,
    ) / rec.pk.CL

    # Approximate Cmax (mg/L) using F * Dose / Vd
    vd_eff = vd_per_kg * weight
    cmax_approx = (dose_mg * rec.pk.F) / max(0.1, vd_eff)

    base_features = {
        "dose_mg": 100.0,
        "weight_kg": 70.0,
        "age": 50.0,
        "serum_creatinine_mg_dl": 1.0,
    }
    perturbed = {
        "dose_mg": dose_mg,
        "weight_kg": weight,
        "age": age,
        "serum_creatinine_mg_dl": float(patient.get("serum_creatinine_mg_dl", 1.0)),
    }
    def outcome_fn(f: Dict[str, float]) -> float:
        d = f["dose_mg"]; w = f["weight_kg"]; c = f["serum_creatinine_mg_dl"]
        cmax = d / max(0.1, vd_per_kg * w)
        renal = 1.0 / max(0.05, 1.0 - 0.5 * (max(0, c - 1.0)))
        return cmax * 1000.0 * renal * rec.pk.F

    contribs = shap_like_attribution(base_features, perturbed, outcome_fn)
    top = topk_features(contribs, k=4)

    evidence = [
        {"fact": f"{drug}: class={rec.drug_class}, Vd={vd_per_kg:.2f} L/kg, "
                 f"t½={(0.693 * vd_eff) / max(0.01, rec.pk.CL):.1f} h, renal fraction={frac_renal}",
         "source": "phase10/pkpd/registry", "weight": 2.0},
        {"fact": f"Patient: {weight} kg, {age} y, CrCl ≈ {crcl:.0f} mL/min",
         "source": "phase10/pkpd/compartments (Cockcroft-Gault)", "weight": 1.5},
        {"fact": f"Allometric Vd factor: {allom:.2f}, renal adjustment: {renal_factor:.2f}",
         "source": "phase10/pkpd/compartments", "weight": 1.2},
        *[{"fact": f"{c['feature']}: base {c['base_value']:.1f} → {c['perturbed_value']:.1f} "
                  f"contributed {c['contribution']:+.2f} to Cmax",
           "source": "phase16/shap_lite", "weight": abs(c["contribution"])}
          for c in top if c.get("feature") != "__total"],
    ]
    top_feat = next((c for c in top if c.get("feature") not in ("__total", None)), None)
    chain = build_reasoning_chain(
            question=f"How will {dose_mg} mg of {drug} behave in this patient?",
            evidence=evidence,
            conclusion=(
                f"Approximate Cmax ≈ {cmax_approx:.3f} mg/L. "
                + (f"Top driver: {top_feat['feature']} (contribution {top_feat['contribution']:+.2f}). "
                   if top_feat else "")
                + (f"Renally-cleared drug — renal function matters."
                   if frac_renal > 0.3 else
                   f"Low renal clearance ({frac_renal*100:.0f}%) — dose adjustment less critical.")
            ),
            confidence="medium",
            confidence_score=0.6,
            alternative_hypotheses=[
                "Linear surrogate ignores saturation kinetics (Michaelis-Menten).",
                "Did not include protein binding or active transport.",
                "Steady-state assumed; first-dose Cmax may differ.",
            ],
        )
    return {
        "method": "shap_lite + clinical_features + reasoning_chain",
        "latency_ms": int((time.time() - t0) * 1000),
        "input": {"drug": drug, "dose_mg": dose_mg, "patient": patient},
        "drug_record": {
            "class": rec.drug_class,
            "vd_L_per_kg": vd_per_kg,
            "clearance_L_h": rec.pk.CL,
            "central_Vd_L": rec.pk.Vc,
            "peripheral_Vd_L": rec.pk.Vp,
            "bioavailability_F": rec.pk.F,
            "frac_renal": frac_renal,
        },
        "patient_factors": {
            "weight_kg": weight, "age": age, "sex": sex,
            "crcl_ml_min": crcl, "allometric_factor": allom, "renal_factor": renal_factor,
        },
        "cmax_approx": cmax_approx,
        "feature_attribution": top,
        "reasoning_chain": chain,
    }


# --------------------------------------------------------------------- #
#  /explain/pgx                                                          #
# --------------------------------------------------------------------- #
@router.post("/explain/pgx")
def explain_pgx(body: Dict[str, Any]) -> Dict[str, Any]:
    pid = body.get("patient_id")
    drug = body.get("drug")
    if not (pid and drug):
        raise HTTPException(400, "patient_id and drug required")
    t0 = time.time()
    from app.pgx import genes as pgx_genes
    from app.pgx import registry as pgx_registry
    from app.pgx.genes import PHARMACOGENES
    profile = pgx_genes.get_patient_pgx(pid)
    rules = pgx_registry.lookup_drug(drug)
    relevant = [r for r in rules if r.gene in profile.genotypes]
    evidence: List[Dict[str, Any]] = []
    triggered = []
    for r in relevant:
        status = profile.genotypes[r.gene]
        activity = status.activity
        impact = pgx_registry.get_impact_factor(drug, r.gene, status.value)
        # severity for this patient status
        if activity < 0.5:
            severity = r.severity  # could escalate to critical
        elif activity < 1.0:
            severity = r.severity
        elif activity > 1.5:
            severity = r.severity
        else:
            continue
        triggered.append({"rule": r, "gene": r.gene, "status": status, "activity": activity,
                          "severity": severity, "impact": impact})
        evidence.append({
            "fact": f"{r.gene} status = {status.value} (activity {activity:.2f}); "
                    f"expected impact {impact:.2f}×, severity {severity}",
            "source": "phase8/pgx/registry",
            "weight": {"critical": 3.0, "major": 2.0, "moderate": 1.5, "minor": 0.5}.get(severity, 1.0),
        })
        evidence.append({
            "fact": f"Clinical note: {r.pm_clinical if activity < 1.0 else r.um_clinical}",
            "source": "phase8/pgx/registry", "weight": 1.0,
        })
    if not evidence:
        evidence.append({"fact": "No PGx-relevant gene-drug interaction.",
                         "source": "phase8/pgx/registry", "weight": 1.0})
    max_sev = "none"
    sev_order = ["none", "minor", "moderate", "major", "critical"]
    for tr in triggered:
        if sev_order.index(tr["severity"]) > sev_order.index(max_sev):
            max_sev = tr["severity"]
    chain = build_reasoning_chain(
        question=f"Why is {drug} risky for patient {pid}?",
        evidence=evidence,
        conclusion=(
            f"Highest severity: {max_sev}. "
            + (f"Genes involved: {', '.join(t['gene'] for t in triggered)}."
               if triggered else "No pharmacogenomic risk identified.")
        ),
        confidence="high" if triggered else "low",
        confidence_score=0.9 if triggered else 0.3,
    )
    return {
        "method": "curated_drug_gene_rules + reasoning_chain",
        "latency_ms": int((time.time() - t0) * 1000),
        "input": {"patient_id": pid, "drug": drug},
        "patient_genes": [{"gene": g, "status": profile.genotypes[g].value,
                            "activity": profile.activity_for(g)}
                           for g in PHARMACOGENES if g in profile.genotypes],
        "triggered_rules": [
            {"gene": t["gene"], "patient_status": t["status"].value,
             "severity": t["severity"], "impact": t["impact"],
             "clinical_note": t["rule"].pm_clinical if t["activity"] < 1.0 else t["rule"].um_clinical}
            for t in triggered
        ],
        "highest_severity": max_sev,
        "reasoning_chain": chain,
    }


# --------------------------------------------------------------------- #
#  /explain/patient — holistic explanation                              #
# --------------------------------------------------------------------- #
@router.post("/explain/patient")
def explain_patient(body: Dict[str, Any]) -> Dict[str, Any]:
    pid = body.get("patient_id")
    drugs = body.get("drugs", [])
    treatment = body.get("treatment", "metformin")
    outcome = body.get("outcome", "hba1c")
    if not pid:
        raise HTTPException(400, "patient_id required")
    t0 = time.time()
    evidence_layers: List[Dict[str, Any]] = []

    # PGx
    if drugs:
        try:
            from app.pgx import genes as pgx_genes
            from app.pgx import registry as pgx_registry
            from app.pgx.genes import PHARMACOGENES
            prof = pgx_genes.get_patient_pgx(pid)
            n_warn = 0
            for d in drugs:
                rules = pgx_registry.lookup_drug(d)
                for r in rules:
                    if prof.activity_for(r.gene) < 1.0:
                        n_warn += 1
            evidence_layers.append({
                "layer": "pharmacogenomics",
                "summary": f"{n_warn} drug-gene warning(s) for {len(drugs)} drug(s)",
                "detail": [{"gene": g, "status": str(prof.genotypes.get(g).value),
                            "activity": prof.activity_for(g)}
                           for g in PHARMACOGENES if g in prof.genotypes],
            })
        except Exception as e:
            evidence_layers.append({"layer": "pharmacogenomics", "summary": f"unavailable: {e}"})

    # DDI
    if len(drugs) >= 2:
        try:
            from app.ddi import database as ddi_db
            pairs = ddi_db.find_all_for(drugs) if hasattr(ddi_db, "find_all_for") else []
            # build pairs from find_pair across all combinations
            if not pairs:
                pairs = []
                for i, a in enumerate(drugs):
                    for b in drugs[i+1:]:
                        for p in ddi_db.find_pair(a, b):
                            pairs.append(p)
            evidence_layers.append({
                "layer": "drug-drug interactions",
                "summary": f"{len(pairs)} interaction(s) in the regimen",
                "detail": [{"pair": (p.drug_a, p.drug_b), "severity": p.severity,
                            "mechanism": p.mechanism} for p in pairs[:10]],
            })
        except Exception as e:
            evidence_layers.append({"layer": "ddi", "summary": f"unavailable: {e}"})

    # UQ
    try:
        from app.uq.bootstrap import bootstrap_patient_counterfactual
        observed = _patient_observed(pid)
        df = _cohort_df()
        uq = bootstrap_patient_counterfactual(
            df, observed=observed, treatment=treatment, value=500.0,
            outcome=outcome, n_bootstrap=20, confidence=0.9, seed=42,
        )
        evidence_layers.append({
            "layer": "uncertainty",
            "summary": f"effect = {uq['effect']['mean']:+.2f} "
                       f"[{uq['effect']['ci_lo']:.2f}, {uq['effect']['ci_hi']:.2f}] "
                       f"({uq['confidence_label']})",
            "detail": {"ci": uq["effect"], "direction_stability": uq["direction_stability"]},
        })
    except Exception as e:
        evidence_layers.append({"layer": "uncertainty", "summary": f"unavailable: {e}"})

    chain = build_reasoning_chain(
        question=f"Composite explanation for patient {pid}",
        evidence=[{"fact": f"[{l['layer']}] {l['summary']}", "source": l["layer"], "weight": 1.0}
                  for l in evidence_layers],
        conclusion=f"Analyzed {len(evidence_layers)} layer(s) for patient {pid}.",
        confidence="medium",
        confidence_score=0.6,
    )
    return {
        "method": "composite (PGx + DDI + UQ) + reasoning_chain",
        "latency_ms": int((time.time() - t0) * 1000),
        "patient_id": pid,
        "layers": evidence_layers,
        "reasoning_chain": chain,
    }
