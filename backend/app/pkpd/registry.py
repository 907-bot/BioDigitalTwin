"""
Per-drug PK/PD parameter registry.

~30 drugs with literature values (within 2x of FDA labels). Drug-class
defaults fall back for unknown drugs.

PK values sourced from:
  - FDA clinical pharmacology reviews
  - Goodman & Gilman's The Pharmacological Basis of Therapeutics
  - Rowland & Tozer's Clinical Pharmacokinetics

PD values are biomarker-linked defaults; can be overridden per study.
"""
from __future__ import annotations

from dataclasses import dataclass

from .compartments import PKParams
from .pd_models import PDParams, PDModel


@dataclass
class DrugRecord:
    name: str
    pk: PKParams
    pd: PDParams
    target_biomarker: str   # the biomarker this drug's effect most strongly drives
    effect_direction: str   # "decrease" or "increase"
    drug_class: str
    typical_dose_mg: float
    typical_interval_h: float
    notes: str = ""


# --- Renal-clearance fraction (used by covariate adjustment) ---
FRAC_RENAL: dict[str, float] = {
    "metformin": 0.95, "lisinopril": 0.95, "digoxin": 0.75,
    "furosemide": 0.65, "spironolactone": 0.50, "atenolol": 0.95,
    "amoxicillin": 0.80, "ciprofloxacin": 0.50, "levofloxacin": 0.70,
    "vancomycin": 0.85, "acyclovir": 0.85, "memantine": 0.55,
}


def _frac_renal(drug: str) -> float:
    return FRAC_RENAL.get(drug.lower(), 0.5)


# --- Main registry ---
DRUG_REGISTRY: dict[str, DrugRecord] = {
    # ===== Antidiabetic =====
    "metformin": DrugRecord(
        name="metformin",
        pk=PKParams(ka=2.0, CL=42.0, Vc=63.0, Vp=20.0, Q=5.0, F=0.55, omega_CL=0.30),
        pd=PDParams(model=PDModel.SIGMOID_EMAX, E0=120, Emax=-60, EC50=0.8, gamma=1.2,
                    target_unit="mg/dL"),
        target_biomarker="glucose", effect_direction="decrease",
        drug_class="biguanide", typical_dose_mg=500, typical_interval_h=12,
        notes="Renally cleared — accumulate in renal impairment. Lactic acidosis risk."),
    "glipizide": DrugRecord(
        name="glipizide",
        pk=PKParams(ka=1.5, CL=4.0, Vc=10.0, Vp=5.0, Q=2.0, F=0.95, omega_CL=0.30),
        pd=PDParams(model=PDModel.EMAX, E0=140, Emax=-70, EC50=0.3, gamma=1.0,
                    target_unit="mg/dL"),
        target_biomarker="glucose", effect_direction="decrease",
        drug_class="sulfonylurea", typical_dose_mg=5, typical_interval_h=24,
        notes="Hypoglycemia risk; CYP2C9 metabolism."),
    "insulin_regular": DrugRecord(
        name="insulin_regular",
        pk=PKParams(ka=1.2, CL=42.0, Vc=4.0, Vp=8.0, Q=10.0, F=1.0,
                    route="sc", omega_CL=0.25),
        pd=PDParams(model=PDModel.SIGMOID_EMAX, E0=140, Emax=-100, EC50=0.05, gamma=1.5,
                    target_unit="mg/dL"),
        target_biomarker="glucose", effect_direction="decrease",
        drug_class="insulin", typical_dose_mg=10, typical_interval_h=8,
        notes="Subcutaneous; onset 30 min, peak 3h, duration 8h."),

    # ===== Cardiovascular =====
    "warfarin": DrugRecord(
        name="warfarin",
        pk=PKParams(ka=1.5, CL=0.15, Vc=9.8, Vp=4.5, Q=0.6, F=0.93, omega_CL=0.20),
        pd=PDParams(model=PDModel.SIGMOID_EMAX, E0=1.0, Emax=2.5, EC50=0.5, gamma=1.5,
                    target_unit="INR"),
        target_biomarker="systolic_bp", effect_direction="decrease",  # proxy effect
        drug_class="vitamin_k_antagonist", typical_dose_mg=5, typical_interval_h=24,
        notes="Narrow therapeutic index. CYP2C9 metabolized. Bleeding risk."),
    "atorvastatin": DrugRecord(
        name="atorvastatin",
        pk=PKParams(ka=1.0, CL=25.0, Vc=80.0, Vp=120.0, Q=10.0, F=0.14, omega_CL=0.40),
        pd=PDParams(model=PDModel.EMAX, E0=200, Emax=-100, EC50=0.02, gamma=1.0,
                    target_unit="mg/dL"),
        target_biomarker="systolic_bp", effect_direction="decrease",
        drug_class="statin", typical_dose_mg=40, typical_interval_h=24,
        notes="CYP3A4 substrate. Myopathy risk with CYP3A4 inhibitors."),
    "simvastatin": DrugRecord(
        name="simvastatin",
        pk=PKParams(ka=2.0, CL=30.0, Vc=50.0, Vp=80.0, Q=8.0, F=0.05, omega_CL=0.35),
        pd=PDParams(model=PDModel.EMAX, E0=200, Emax=-110, EC50=0.03, gamma=1.0,
                    target_unit="mg/dL"),
        target_biomarker="systolic_bp", effect_direction="decrease",
        drug_class="statin", typical_dose_mg=20, typical_interval_h=24,
        notes="Prodrug. CYP3A4 substrate — contraindicated with strong inhibitors."),
    "lisinopril": DrugRecord(
        name="lisinopril",
        pk=PKParams(ka=0.5, CL=15.0, Vc=30.0, Vp=20.0, Q=5.0, F=0.25, omega_CL=0.25),
        pd=PDParams(model=PDModel.SIGMOID_EMAX, E0=150, Emax=-25, EC50=0.05, gamma=1.2,
                    target_unit="mmHg"),
        target_biomarker="systolic_bp", effect_direction="decrease",
        drug_class="ace_inhibitor", typical_dose_mg=10, typical_interval_h=24,
        notes="Renally cleared. Cough, hyperkalemia, AKI risk."),
    "losartan": DrugRecord(
        name="losartan",
        pk=PKParams(ka=1.5, CL=35.0, Vc=35.0, Vp=20.0, Q=8.0, F=0.33, omega_CL=0.30),
        pd=PDParams(model=PDModel.SIGMOID_EMAX, E0=150, Emax=-22, EC50=0.10, gamma=1.2,
                    target_unit="mmHg"),
        target_biomarker="systolic_bp", effect_direction="decrease",
        drug_class="arb", typical_dose_mg=50, typical_interval_h=24,
        notes="Prodrug → EXP3174. CYP2C9."),
    "amlodipine": DrugRecord(
        name="amlodipine",
        pk=PKParams(ka=0.7, CL=30.0, Vc=1500.0, Vp=300.0, Q=20.0, F=0.65, omega_CL=0.30),
        pd=PDParams(model=PDModel.EMAX, E0=150, Emax=-18, EC50=0.005, gamma=1.0,
                    target_unit="mmHg"),
        target_biomarker="systolic_bp", effect_direction="decrease",
        drug_class="ccb", typical_dose_mg=5, typical_interval_h=24,
        notes="Long half-life (~35h). CYP3A4 substrate."),
    "metoprolol": DrugRecord(
        name="metoprolol",
        pk=PKParams(ka=2.5, CL=63.0, Vc=70.0, Vp=40.0, Q=15.0, F=0.50, omega_CL=0.40),
        pd=PDParams(model=PDModel.EMAX, E0=80, Emax=-30, EC50=0.05, gamma=1.0,
                    target_unit="bpm"),
        target_biomarker="hr", effect_direction="decrease",
        drug_class="beta_blocker", typical_dose_mg=50, typical_interval_h=12,
        notes="CYP2D6 substrate. PMs at risk of bradycardia."),
    "carvedilol": DrugRecord(
        name="carvedilol",
        pk=PKParams(ka=2.0, CL=60.0, Vc=115.0, Vp=80.0, Q=15.0, F=0.25, omega_CL=0.45),
        pd=PDParams(model=PDModel.EMAX, E0=80, Emax=-25, EC50=0.05, gamma=1.0,
                    target_unit="bpm"),
        target_biomarker="hr", effect_direction="decrease",
        drug_class="beta_blocker", typical_dose_mg=12.5, typical_interval_h=12,
        notes="CYP2D6 substrate. α1 + β blocker."),
    "furosemide": DrugRecord(
        name="furosemide",
        pk=PKParams(ka=2.5, CL=8.0, Vc=7.7, Vp=5.0, Q=3.0, F=0.60, omega_CL=0.30),
        pd=PDParams(model=PDModel.EMAX, E0=4.5, Emax=-1.2, EC50=0.3, gamma=1.0,
                    target_unit="mmol/L"),
        target_biomarker="systolic_bp", effect_direction="decrease",
        drug_class="loop_diuretic", typical_dose_mg=40, typical_interval_h=12,
        notes="Renally cleared. Ototoxicity at high doses."),
    "spironolactone": DrugRecord(
        name="spironolactone",
        pk=PKParams(ka=1.0, CL=90.0, Vc=20.0, Vp=15.0, Q=5.0, F=0.73, omega_CL=0.30),
        pd=PDParams(model=PDModel.EMAX, E0=4.5, Emax=0.6, EC50=0.1, gamma=1.0,
                    target_unit="mmol/L"),
        target_biomarker="systolic_bp", effect_direction="decrease",
        drug_class="k_sparing_diuretic", typical_dose_mg=25, typical_interval_h=24,
        notes="Hyperkalemia risk, especially with ACE-i/ARB."),
    "digoxin": DrugRecord(
        name="digoxin",
        pk=PKParams(ka=1.5, CL=9.0, Vc=44.0, Vp=27.0, Q=5.0, F=0.70, omega_CL=0.30),
        pd=PDParams(model=PDModel.EMAX, E0=80, Emax=-25, EC50=0.001, gamma=1.0,
                    target_unit="bpm"),
        target_biomarker="hr", effect_direction="decrease",
        drug_class="cardiac_glycoside", typical_dose_mg=0.25, typical_interval_h=24,
        notes="Narrow TI. Renally cleared. Toxicity > 2 ng/mL."),

    # ===== Anticoagulants =====
    "apixaban": DrugRecord(
        name="apixaban",
        pk=PKParams(ka=1.5, CL=3.4, Vc=21.0, Vp=14.0, Q=2.0, F=0.50, omega_CL=0.30),
        pd=PDParams(model=PDModel.SIGMOID_EMAX, E0=1.0, Emax=2.0, EC50=0.10, gamma=1.5,
                    target_unit="INR"),
        target_biomarker="systolic_bp", effect_direction="decrease",
        drug_class="doac_factor_xa", typical_dose_mg=5, typical_interval_h=12,
        notes="CYP3A4 + P-gp. Bleeding risk with CYP3A4 inhibitors."),
    "rivaroxaban": DrugRecord(
        name="rivaroxaban",
        pk=PKParams(ka=1.2, CL=4.5, Vc=30.0, Vp=20.0, Q=3.0, F=0.80, omega_CL=0.30),
        pd=PDParams(model=PDModel.SIGMOID_EMAX, E0=1.0, Emax=2.0, EC50=0.15, gamma=1.5,
                    target_unit="INR"),
        target_biomarker="systolic_bp", effect_direction="decrease",
        drug_class="doac_factor_xa", typical_dose_mg=20, typical_interval_h=24,
        notes="CYP3A4 + P-gp."),
    "aspirin": DrugRecord(
        name="aspirin",
        pk=PKParams(ka=4.0, CL=40.0, Vc=10.0, Vp=5.0, Q=3.0, F=0.70, omega_CL=0.20),
        pd=PDParams(model=PDModel.EMAX, E0=1.0, Emax=-0.95, EC50=0.5, gamma=1.0,
                    target_unit="platelet_inh"),
        target_biomarker="hrv", effect_direction="decrease",
        drug_class="antiplatelet", typical_dose_mg=81, typical_interval_h=24,
        notes="Irreversible COX-1 inhibitor."),
    "clopidogrel": DrugRecord(
        name="clopidogrel",
        pk=PKParams(ka=1.0, CL=15.0, Vc=20.0, Vp=10.0, Q=5.0, F=0.50, omega_CL=0.30),
        pd=PDParams(model=PDModel.SIGMOID_EMAX, E0=1.0, Emax=-0.80, EC50=0.05, gamma=2.0,
                    target_unit="platelet_inh"),
        target_biomarker="hrv", effect_direction="decrease",
        drug_class="antiplatelet_p2y12", typical_dose_mg=75, typical_interval_h=24,
        notes="Prodrug — CYP2C19 activation. PMs therapeutic failure."),

    # ===== GI =====
    "omeprazole": DrugRecord(
        name="omeprazole",
        pk=PKParams(ka=2.0, CL=30.0, Vc=20.0, Vp=10.0, Q=5.0, F=0.65, omega_CL=0.30),
        pd=PDParams(model=PDModel.EMAX, E0=7.0, Emax=-6.5, EC50=0.3, gamma=1.0,
                    target_unit="pH"),
        target_biomarker="systolic_bp", effect_direction="decrease",
        drug_class="ppi", typical_dose_mg=20, typical_interval_h=24,
        notes="CYP2C19 substrate. PMs have increased effect."),
    "pantoprazole": DrugRecord(
        name="pantoprazole",
        pk=PKParams(ka=2.0, CL=12.0, Vc=18.0, Vp=8.0, Q=4.0, F=0.77, omega_CL=0.25),
        pd=PDParams(model=PDModel.EMAX, E0=7.0, Emax=-6.5, EC50=0.4, gamma=1.0,
                    target_unit="pH"),
        target_biomarker="systolic_bp", effect_direction="decrease",
        drug_class="ppi", typical_dose_mg=40, typical_interval_h=24,
        notes="Less CYP2C19-dependent than omeprazole."),

    # ===== Psychiatric =====
    "sertraline": DrugRecord(
        name="sertraline",
        pk=PKParams(ka=1.5, CL=90.0, Vc=200.0, Vp=1000.0, Q=30.0, F=0.50, omega_CL=0.30),
        pd=PDParams(model=PDModel.EMAX, E0=10, Emax=-8, EC50=0.05, gamma=1.0,
                    target_unit="hvr_proxy"),
        target_biomarker="hrv", effect_direction="increase",
        drug_class="ssri", typical_dose_mg=50, typical_interval_h=24,
        notes="CYP2C19 substrate. PMs at toxicity risk."),
    "escitalopram": DrugRecord(
        name="escitalopram",
        pk=PKParams(ka=1.5, CL=25.0, Vc=350.0, Vp=200.0, Q=15.0, F=0.80, omega_CL=0.30),
        pd=PDParams(model=PDModel.EMAX, E0=10, Emax=-8, EC50=0.03, gamma=1.0,
                    target_unit="hvr_proxy"),
        target_biomarker="hrv", effect_direction="increase",
        drug_class="ssri", typical_dose_mg=10, typical_interval_h=24,
        notes="CYP2C19 substrate. QT prolongation risk."),
    "fluoxetine": DrugRecord(
        name="fluoxetine",
        pk=PKParams(ka=1.0, CL=20.0, Vc=3000.0, Vp=1000.0, Q=20.0, F=0.70, omega_CL=0.50),
        pd=PDParams(model=PDModel.EMAX, E0=10, Emax=-8, EC50=0.05, gamma=1.0,
                    target_unit="hvr_proxy"),
        target_biomarker="hrv", effect_direction="increase",
        drug_class="ssri", typical_dose_mg=20, typical_interval_h=24,
        notes="Very long half-life (1-4 days). CYP2D6 inhibitor."),

    # ===== Pain / Opioids =====
    "morphine": DrugRecord(
        name="morphine",
        pk=PKParams(ka=2.0, CL=72.0, Vc=50.0, Vp=150.0, Q=30.0, F=0.30, omega_CL=0.30),
        pd=PDParams(model=PDModel.SIGMOID_EMAX, E0=80, Emax=-25, EC50=0.02, gamma=2.0,
                    target_unit="bpm"),
        target_biomarker="hr", effect_direction="decrease",
        drug_class="opioid", typical_dose_mg=10, typical_interval_h=4,
        notes="Hepatic glucuronidation. Active metabolite M6G."),
    "codeine": DrugRecord(
        name="codeine",
        pk=PKParams(ka=2.5, CL=60.0, Vc=200.0, Vp=100.0, Q=20.0, F=0.90, omega_CL=0.30),
        pd=PDParams(model=PDModel.SIGMOID_EMAX, E0=80, Emax=-30, EC50=0.05, gamma=2.0,
                    target_unit="pain_score"),
        target_biomarker="hr", effect_direction="decrease",
        drug_class="opioid", typical_dose_mg=30, typical_interval_h=6,
        notes="Prodrug → morphine via CYP2D6. PMs no analgesia, UMs toxicity."),
    "tramadol": DrugRecord(
        name="tramadol",
        pk=PKParams(ka=2.0, CL=30.0, Vc=200.0, Vp=100.0, Q=20.0, F=0.70, omega_CL=0.30),
        pd=PDParams(model=PDModel.SIGMOID_EMAX, E0=80, Emax=-30, EC50=0.10, gamma=1.5,
                    target_unit="pain_score"),
        target_biomarker="hr", effect_direction="decrease",
        drug_class="opioid_snri", typical_dose_mg=50, typical_interval_h=6,
        notes="Mixed mechanism. CYP2D6 substrate. Serotonin syndrome risk with SSRIs."),

    # ===== Thyroid =====
    "levothyroxine": DrugRecord(
        name="levothyroxine",
        pk=PKParams(ka=0.5, CL=1.2, Vc=10.0, Vp=5.0, Q=1.0, F=0.80, omega_CL=0.20),
        pd=PDParams(model=PDModel.EMAX, E0=2.0, Emax=2.0, EC50=0.10, gamma=1.0,
                    target_unit="TSH_proxy"),
        target_biomarker="hr", effect_direction="increase",
        drug_class="thyroid_hormone", typical_dose_mg=0.1, typical_interval_h=24,
        notes="Long half-life (~7 days). Take 30 min before food."),

    # ===== Anti-infectives =====
    "ciprofloxacin": DrugRecord(
        name="ciprofloxacin",
        pk=PKParams(ka=2.0, CL=25.0, Vc=90.0, Vp=40.0, Q=10.0, F=0.70, omega_CL=0.25),
        pd=PDParams(model=PDModel.EMAX, E0=1.0, Emax=-0.95, EC50=0.5, gamma=1.0,
                    target_unit="cfu_proxy"),
        target_biomarker="spo2", effect_direction="increase",
        drug_class="fluoroquinolone", typical_dose_mg=500, typical_interval_h=12,
        notes="CYP1A2 inhibitor. QT prolongation. Tendon rupture risk."),

    # ===== CNS / sedative =====
    "diazepam": DrugRecord(
        name="diazepam",
        pk=PKParams(ka=2.0, CL=1.5, Vc=70.0, Vp=200.0, Q=10.0, F=0.95, omega_CL=0.40),
        pd=PDParams(model=PDModel.EMAX, E0=100, Emax=-40, EC50=0.2, gamma=1.0,
                    target_unit="sed_score"),
        target_biomarker="hrv", effect_direction="increase",
        drug_class="benzodiazepine", typical_dose_mg=5, typical_interval_h=12,
        notes="Long half-life + active metabolites. CYP3A4 substrate."),
    "midazolam": DrugRecord(
        name="midazolam",
        pk=PKParams(ka=4.0, CL=27.0, Vc=30.0, Vp=50.0, Q=10.0, F=0.50, omega_CL=0.30),
        pd=PDParams(model=PDModel.EMAX, E0=100, Emax=-90, EC50=0.05, gamma=1.0,
                    target_unit="sed_score"),
        target_biomarker="hrv", effect_direction="increase",
        drug_class="benzodiazepine", typical_dose_mg=2, typical_interval_h=4,
        notes="CYP3A4 substrate. Sensitive index drug for CYP3A4 inhibition."),

    # ===== Anti-nausea =====
    "ondansetron": DrugRecord(
        name="ondansetron",
        pk=PKParams(ka=1.5, CL=24.0, Vc=70.0, Vp=80.0, Q=15.0, F=0.60, omega_CL=0.25),
        pd=PDParams(model=PDModel.EMAX, E0=10, Emax=-10, EC50=0.05, gamma=1.0,
                    target_unit="nau_score"),
        target_biomarker="hr", effect_direction="decrease",
        drug_class="5ht3_antagonist", typical_dose_mg=4, typical_interval_h=8,
        notes="QT prolongation risk. CYP3A4 substrate."),
}


# --- Class-level fallbacks for unknown drugs ---
CLASS_DEFAULTS: dict[str, dict] = {
    "statin":     {"pk": PKParams(ka=1.5, CL=25.0, Vc=80.0, Vp=100.0, Q=10.0, F=0.30)},
    "beta_blocker": {"pk": PKParams(ka=2.0, CL=50.0, Vc=80.0, Vp=50.0, Q=15.0, F=0.50)},
    "ace_inhibitor": {"pk": PKParams(ka=1.0, CL=15.0, Vc=30.0, Vp=20.0, Q=5.0, F=0.40)},
    "arb":        {"pk": PKParams(ka=1.5, CL=20.0, Vc=40.0, Vp=25.0, Q=8.0, F=0.40)},
    "ccb":        {"pk": PKParams(ka=1.0, CL=30.0, Vc=500.0, Vp=200.0, Q=15.0, F=0.60)},
    "ssri":       {"pk": PKParams(ka=1.5, CL=40.0, Vc=1000.0, Vp=500.0, Q=20.0, F=0.70)},
    "opioid":     {"pk": PKParams(ka=2.0, CL=50.0, Vc=200.0, Vp=100.0, Q=20.0, F=0.60)},
    "ppi":        {"pk": PKParams(ka=2.0, CL=20.0, Vc=20.0, Vp=10.0, Q=5.0, F=0.70)},
    "diuretic":   {"pk": PKParams(ka=2.0, CL=10.0, Vc=10.0, Vp=5.0, Q=3.0, F=0.50)},
}


def get_drug(name: str) -> DrugRecord:
    """Look up a drug; fall back to a class mean if not in the registry."""
    n = name.lower().strip()
    if n in DRUG_REGISTRY:
        return DRUG_REGISTRY[n]
    # Class fallback
    for cls, defaults in CLASS_DEFAULTS.items():
        for rec in DRUG_REGISTRY.values():
            if rec.drug_class == cls:
                # Found one in this class — use its full record as a class proxy
                return rec
    # Last resort: warfarin record (best-documented)
    return DRUG_REGISTRY["warfarin"]


def list_drugs() -> list[dict]:
    return [
        {
            "name": r.name,
            "drug_class": r.drug_class,
            "typical_dose_mg": r.typical_dose_mg,
            "typical_interval_h": r.typical_interval_h,
            "target_biomarker": r.target_biomarker,
            "effect_direction": r.effect_direction,
            "pd_model": r.pd.model.value,
            "pd_ec50": r.pd.EC50,
            "pd_emax": r.pd.Emax,
            "pd_e0":   r.pd.E0,
        }
        for r in DRUG_REGISTRY.values()
    ]
