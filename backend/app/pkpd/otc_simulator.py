"""
OTC Cold vs Fever Drug Mis-match Simulator.

Simulates physiological outcomes and toxicity/side-effect profiles when an OTC drug
(Paracetamol, Ibuprofen, Pseudoephedrine, Diphenhydramine) is administered under matching
vs. mismatched clinical indications (e.g. fever vs viral cold without fever, cold in hypertension).
"""

from typing import Dict, Any, List
from pydantic import BaseModel, Field


class OTCSimulationRequest(BaseModel):
    patient_condition: str = Field("fever_39C", pattern="^(fever_39C|viral_cold_no_fever|cold_with_hypertension|fever_with_ckd)$")
    administered_drug: str = Field("paracetamol", pattern="^(paracetamol|ibuprofen|pseudoephedrine|diphenhydramine)$")
    dose_mg: float = Field(500.0, ge=100.0, le=2000.0)
    repeat_doses_per_day: int = Field(3, ge=1, le=6)
    duration_days: int = Field(3, ge=1, le=14)


def simulate_otc_mismatch(
    patient_condition: str = "viral_cold_no_fever",
    administered_drug: str = "paracetamol",
    dose_mg: float = 500.0,
    repeat_doses_per_day: int = 3,
    duration_days: int = 3,
) -> Dict[str, Any]:
    """
    Simulates physiological response to OTC drugs under clinical indication matching vs mismatching.
    """
    total_daily_dose_mg = dose_mg * repeat_doses_per_day

    # Determine baseline temperature and BP based on condition
    condition_profiles = {
        "fever_39C": {"base_temp": 39.1, "base_sbp": 122.0, "base_hr": 98.0, "has_fever": True, "has_hypertension": False},
        "viral_cold_no_fever": {"base_temp": 36.8, "base_sbp": 118.0, "base_hr": 74.0, "has_fever": False, "has_hypertension": False},
        "cold_with_hypertension": {"base_temp": 37.0, "base_sbp": 152.0, "base_hr": 82.0, "has_fever": False, "has_hypertension": True},
        "fever_with_ckd": {"base_temp": 38.8, "base_sbp": 138.0, "base_hr": 92.0, "has_fever": True, "has_hypertension": True, "has_ckd": True},
    }
    prof = condition_profiles.get(patient_condition, condition_profiles["viral_cold_no_fever"])

    # Drug appropriateness matrix
    matched_conditions = {
        "paracetamol": ["fever_39C", "fever_with_ckd"],
        "ibuprofen": ["fever_39C"],
        "pseudoephedrine": ["viral_cold_no_fever"],
        "diphenhydramine": ["viral_cold_no_fever"],
    }
    is_indicated = patient_condition in matched_conditions.get(administered_drug, [])

    # Simulate drug response
    temp_trajectory = []
    sbp_trajectory = []
    napqi_load = []
    gsh_depletion_pct = []
    warnings = []

    cur_temp = prof["base_temp"]
    cur_sbp = prof["base_sbp"]
    cur_hr = prof["base_hr"]
    cur_gsh = 100.0  # % Glutathione reserve

    # Overdose check
    if administered_drug == "paracetamol" and total_daily_dose_mg > 4000:
        warnings.append("CRITICAL: Paracetamol daily dose exceeds maximum 4,000 mg/day limit! Severe risk of acute liver failure.")

    if administered_drug == "pseudoephedrine" and prof["has_hypertension"]:
        warnings.append("WARNING: Pseudoephedrine alpha-1 vasoconstriction exacerbates baseline hypertension (+12-18 mmHg SBP). Risk of hypertensive crisis.")

    if administered_drug == "ibuprofen" and prof.get("has_ckd"):
        warnings.append("WARNING: Ibuprofen COX-2 inhibition reduces renal perfusion in CKD, escalating acute kidney injury (AKI) risk.")

    if administered_drug == "paracetamol" and not prof["has_fever"]:
        warnings.append("MISMATCH: Paracetamol given for cold without fever provides no antipyretic benefit, subjecting liver to unnecessary hepatic CYP2E1 NAPQI load.")

    # Time series simulation over duration_days * 24 hours
    hours = list(range(1, duration_days * 24 + 1))
    for h in hours:
        # Check if dose taken at this hour
        is_dose_hour = (h - 1) % (24 // repeat_doses_per_day) == 0

        # Antipyretic effect
        if prof["has_fever"] and administered_drug in ["paracetamol", "ibuprofen"]:
            target_temp = 36.8
            drop_rate = 0.4 if administered_drug == "ibuprofen" else 0.3
            cur_temp = max(target_temp, cur_temp - drop_rate * (1 if is_dose_hour else 0.05))
        else:
            cur_temp = prof["base_temp"] + (0.1 if not prof["has_fever"] else -0.05 * (h / 24))

        # Vasopressor effect
        if administered_drug == "pseudoephedrine":
            sbp_bump = 14.0 if prof["has_hypertension"] else 8.0
            cur_sbp = prof["base_sbp"] + sbp_bump * (1.0 if is_dose_hour else 0.5)
        elif administered_drug == "ibuprofen" and prof["has_hypertension"]:
            cur_sbp = prof["base_sbp"] + 6.0
        else:
            cur_sbp = prof["base_sbp"]

        # Paracetamol hepatic metabolism & Glutathione kinetics
        if administered_drug == "paracetamol":
            # 5-10% converted by CYP2E1 to NAPQI
            hourly_napqi = (total_daily_dose_mg / 24.0) * 0.08
            cur_gsh = max(10.0, cur_gsh - (hourly_napqi / 15.0) + 0.5)  # regeneration
        else:
            hourly_napqi = 0.0
            cur_gsh = min(100.0, cur_gsh + 0.2)

        if h % 6 == 0:
            temp_trajectory.append({"hour": h, "body_temp_C": round(cur_temp, 2)})
            sbp_trajectory.append({"hour": h, "systolic_bp_mmHg": round(cur_sbp, 1)})
            napqi_load.append({"hour": h, "napqi_umol": round(hourly_napqi, 2)})
            gsh_depletion_pct.append({"hour": h, "glutathione_reserve_pct": round(cur_gsh, 1)})

    verdict = "CLINICALLY_APPROPRIATE" if is_indicated and len(warnings) == 0 else "CAUTION_MISMATCH"
    if any("CRITICAL" in w for w in warnings):
        verdict = "CONTRAINDICATED_HAZARD"

    return {
        "patient_condition": patient_condition,
        "administered_drug": administered_drug,
        "dose_mg": dose_mg,
        "repeat_doses_per_day": repeat_doses_per_day,
        "total_daily_dose_mg": total_daily_dose_mg,
        "duration_days": duration_days,
        "is_indicated": is_indicated,
        "clinical_verdict": verdict,
        "warnings": warnings,
        "outcomes": {
            "initial_temp_C": prof["base_temp"],
            "final_temp_C": round(cur_temp, 2),
            "peak_sbp_mmHg": round(max(s["systolic_bp_mmHg"] for s in sbp_trajectory), 1),
            "lowest_gsh_reserve_pct": round(min(g["glutathione_reserve_pct"] for g in gsh_depletion_pct), 1),
        },
        "trajectories": {
            "temperature": temp_trajectory,
            "blood_pressure": sbp_trajectory,
            "glutathione_reserve": gsh_depletion_pct,
        },
    }
