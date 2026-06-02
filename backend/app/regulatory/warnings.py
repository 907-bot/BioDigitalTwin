"""
Curated black-box warnings, contraindications, and notable adverse effects
for ~80 commonly-prescribed drugs.

This complements the live FAERS data (real-world counts) with the curated
clinical knowledge that the FDA labels carry but isn't easily extracted
from event counts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class DrugSafety:
    drug: str
    black_box: list[str] = field(default_factory=list)
    contraindications: list[str] = field(default_factory=list)
    common_aes: list[dict] = field(default_factory=list)
    notes: str = ""
    pregnancy: str = "unknown"   # A, B, C, D, X
    typical_dose: str = ""


SAFETY_REGISTRY: dict[str, DrugSafety] = {
    "warfarin": DrugSafety(
        drug="warfarin",
        black_box=[
            "Fatal bleeding risk — monitor INR closely; avoid in patients with high bleeding risk.",
        ],
        contraindications=[
            "Pregnancy (teratogenic — fetal warfarin syndrome)",
            "Active bleeding or within 48h of major surgery",
            "Severe hepatic or renal impairment",
        ],
        common_aes=[
            {"ae": "Hemorrhage",     "frequency": "1-10%"},
            {"ae": "Bruising",       "frequency": "5-15%"},
            {"ae": "GI upset",       "frequency": "1-5%"},
        ],
        notes="Narrow therapeutic index. INR goal 2-3 (2.5-3.5 for mechanical valves).",
        pregnancy="X",
        typical_dose="2-10 mg PO q24h, titrated to INR",
    ),
    "apixaban": DrugSafety(
        drug="apixaban",
        black_box=[
            "Premature discontinuation increases risk of thrombotic events.",
            "Epidural/spinal hematoma risk with neuraxial anesthesia.",
        ],
        contraindications=["Active bleeding", "Severe hepatic impairment"],
        common_aes=[
            {"ae": "Bleeding",       "frequency": "1-10%"},
            {"ae": "Bruising",       "frequency": "5-15%"},
        ],
        notes="DOAC. Avoid combination with strong CYP3A4 + P-gp inhibitors.",
        pregnancy="B",
        typical_dose="5 mg PO q12h (2.5 mg q12h if 2 of 3: age>=80, weight<=60kg, Cr>=1.5)",
    ),
    "metformin": DrugSafety(
        drug="metformin",
        black_box=[
            "Lactic acidosis (rare, but high mortality) — particularly in renal impairment.",
        ],
        contraindications=[
            "eGFR < 30 mL/min",
            "Acute or chronic metabolic acidosis",
            "Hepatic impairment",
        ],
        common_aes=[
            {"ae": "Diarrhea",       "frequency": "10-25%"},
            {"ae": "Nausea",         "frequency": "5-15%"},
            {"ae": "Abdominal pain", "frequency": "3-8%"},
        ],
        notes="Hold for iodinated contrast in eGFR < 60.",
        pregnancy="B",
        typical_dose="500 mg PO q12h, titrate to 2000 mg/day",
    ),
    "atorvastatin": DrugSafety(
        drug="atorvastatin",
        black_box=[],
        contraindications=[
            "Active liver disease",
            "Pregnancy (teratogenic)",
            "Concurrent strong CYP3A4 inhibitors at high doses",
        ],
        common_aes=[
            {"ae": "Myalgia",        "frequency": "3-8%"},
            {"ae": "Headache",       "frequency": "5-10%"},
            {"ae": "Elevated LFTs",  "frequency": "1-2%"},
        ],
        notes="Rhabdomyolysis risk with CYP3A4 inhibitors.",
        pregnancy="X",
        typical_dose="10-80 mg PO q24h",
    ),
    "simvastatin": DrugSafety(
        drug="simvastatin",
        black_box=[],
        contraindications=[
            "Concurrent strong CYP3A4 inhibitors (clarithromycin, itraconazole, ritonavir)",
            "Active liver disease",
            "Pregnancy",
        ],
        common_aes=[
            {"ae": "Myalgia",        "frequency": "3-8%"},
            {"ae": "Rhabdomyolysis", "frequency": "rare, dose-dependent"},
        ],
        notes="Max 20 mg with amiodarone, max 10 mg with gemfibrozil.",
        pregnancy="X",
        typical_dose="5-40 mg PO q24h",
    ),
    "lisinopril": DrugSafety(
        drug="lisinopril",
        black_box=[
            "Fetal toxicity — discontinue as soon as pregnancy is detected.",
        ],
        contraindications=[
            "Pregnancy",
            "History of ACE-i angioedema",
            "Bilateral renal artery stenosis",
        ],
        common_aes=[
            {"ae": "Dry cough",      "frequency": "10-20%"},
            {"ae": "Hyperkalemia",   "frequency": "1-5%"},
            {"ae": "Hypotension",    "frequency": "1-10%"},
            {"ae": "Angioedema",     "frequency": "0.1-0.7%"},
        ],
        notes="Renally cleared; reduce dose if eGFR < 30.",
        pregnancy="D",
        typical_dose="10-40 mg PO q24h",
    ),
    "losartan": DrugSafety(
        drug="losartan",
        black_box=[
            "Fetal toxicity — discontinue as soon as pregnancy is detected.",
        ],
        contraindications=[
            "Pregnancy",
            "Bilateral renal artery stenosis",
        ],
        common_aes=[
            {"ae": "Hyperkalemia",   "frequency": "1-5%"},
            {"ae": "Hypotension",    "frequency": "1-5%"},
        ],
        notes="Prodrug; CYP2C9 metabolized.",
        pregnancy="D",
        typical_dose="25-100 mg PO q24h",
    ),
    "amlodipine": DrugSafety(
        drug="amlodipine",
        black_box=[],
        contraindications=["Hypersensitivity"],
        common_aes=[
            {"ae": "Peripheral edema",  "frequency": "5-10%"},
            {"ae": "Headache",          "frequency": "5-10%"},
            {"ae": "Flushing",          "frequency": "1-5%"},
        ],
        notes="CYP3A4 substrate. Long half-life.",
        pregnancy="C",
        typical_dose="2.5-10 mg PO q24h",
    ),
    "metoprolol": DrugSafety(
        drug="metoprolol",
        black_box=[],
        contraindications=[
            "Severe bradycardia",
            "Heart block (>1st degree)",
            "Decompensated heart failure",
            "Asthma (for non-selective)",
        ],
        common_aes=[
            {"ae": "Fatigue",       "frequency": "5-15%"},
            {"ae": "Bradycardia",   "frequency": "1-5%"},
            {"ae": "Dizziness",     "frequency": "5-10%"},
        ],
        notes="CYP2D6 — PMs at risk.",
        pregnancy="C",
        typical_dose="25-200 mg PO q12h (or q24h for succinate)",
    ),
    "furosemide": DrugSafety(
        drug="furosemide",
        black_box=[],
        contraindications=[
            "Anuria",
            "Severe hypokalemia/hyponatremia",
            "Sulfonamide allergy (cross-reactivity)",
        ],
        common_aes=[
            {"ae": "Hypokalemia",   "frequency": "10-30%"},
            {"ae": "Hyponatremia",  "frequency": "1-5%"},
            {"ae": "Ototoxicity",   "frequency": "rare, dose-dependent"},
        ],
        notes="Renally cleared; loop diuretic.",
        pregnancy="C",
        typical_dose="20-80 mg PO q12-24h",
    ),
    "spironolactone": DrugSafety(
        drug="spironolactone",
        black_box=[
            "Tumor risk in animal studies — use lowest effective dose.",
        ],
        contraindications=[
            "Hyperkalemia",
            "Addison's disease",
            "Concurrent eplerenone",
        ],
        common_aes=[
            {"ae": "Hyperkalemia",   "frequency": "5-15%"},
            {"ae": "Gynecomastia",   "frequency": "5-10% (dose-dependent)"},
        ],
        notes="Monitor K+ closely, especially with ACE-i/ARB.",
        pregnancy="C",
        typical_dose="25-200 mg PO q24h",
    ),
    "digoxin": DrugSafety(
        drug="digoxin",
        black_box=[],
        contraindications=[
            "Ventricular fibrillation",
            "Second/third degree heart block",
        ],
        common_aes=[
            {"ae": "Nausea",         "frequency": "1-10%"},
            {"ae": "Visual disturbance (yellow halo)", "frequency": "toxicity"},
            {"ae": "Arrhythmias",    "frequency": "toxicity > 2 ng/mL"},
        ],
        notes="Narrow therapeutic index. Renally cleared.",
        pregnancy="C",
        typical_dose="0.125-0.25 mg PO q24h",
    ),
    "aspirin": DrugSafety(
        drug="aspirin",
        black_box=[],
        contraindications=[
            "Active bleeding",
            "Children with viral infection (Reye syndrome)",
            "Severe asthma with nasal polyps",
        ],
        common_aes=[
            {"ae": "GI upset",       "frequency": "5-15%"},
            {"ae": "GI bleeding",    "frequency": "1-2%"},
            {"ae": "Tinnitus",       "frequency": "toxicity"},
        ],
        notes="Irreversible COX-1.",
        pregnancy="D",
        typical_dose="81-325 mg PO q24h",
    ),
    "clopidogrel": DrugSafety(
        drug="clopidogrel",
        black_box=[
            "Reduced effectiveness in CYP2C19 poor metabolizers (~30% of population).",
        ],
        contraindications=[
            "Active bleeding",
            "Severe hepatic impairment",
        ],
        common_aes=[
            {"ae": "Bleeding",       "frequency": "1-10%"},
            {"ae": "Bruising",       "frequency": "5-15%"},
        ],
        notes="Prodrug; CYP2C19 activation.",
        pregnancy="B",
        typical_dose="75 mg PO q24h (300-600 mg loading)",
    ),
    "omeprazole": DrugSafety(
        drug="omeprazole",
        black_box=[],
        contraindications=["Hypersensitivity", "Concurrent rilpivirine"],
        common_aes=[
            {"ae": "Headache",       "frequency": "3-10%"},
            {"ae": "Diarrhea",       "frequency": "1-5%"},
            {"ae": "Hypomagnesemia", "frequency": "rare, long-term"},
        ],
        notes="CYP2C19 substrate. Long-term: B12, Mg, fracture risk.",
        pregnancy="C",
        typical_dose="20-40 mg PO q24h",
    ),
    "sertraline": DrugSafety(
        drug="sertraline",
        black_box=[
            "Suicidality in children, adolescents, and young adults (24 and under).",
        ],
        contraindications=[
            "Concurrent MAOI (or within 14 days)",
            "Concurrent pimozide",
        ],
        common_aes=[
            {"ae": "Nausea",         "frequency": "10-25%"},
            {"ae": "Sexual dysfunction", "frequency": "10-20%"},
            {"ae": "Insomnia",       "frequency": "10-20%"},
        ],
        notes="CYP2C19 — PMs at toxicity risk.",
        pregnancy="C",
        typical_dose="50-200 mg PO q24h",
    ),
    "codeine": DrugSafety(
        drug="codeine",
        black_box=[
            "Respiratory depression and death have occurred in children who are CYP2D6 ultra-rapid metabolizers.",
            "Post-operative use in children undergoing tonsillectomy/adenoidectomy is contraindicated.",
        ],
        contraindications=[
            "Children < 12 years",
            "Post-op tonsillectomy/adenoidectomy patients < 18 years",
            "Lactation (neonatal toxicity)",
            "Severe asthma",
        ],
        common_aes=[
            {"ae": "Sedation",       "frequency": "10-30%"},
            {"ae": "Nausea",         "frequency": "10-25%"},
            {"ae": "Constipation",   "frequency": "10-25%"},
        ],
        notes="Prodrug; CYP2D6 activation to morphine.",
        pregnancy="C",
        typical_dose="15-60 mg PO q4-6h PRN",
    ),
    "tramadol": DrugSafety(
        drug="tramadol",
        black_box=[
            "Respiratory depression risk — especially in children and CYP2D6 ultra-rapid metabolizers.",
            "Concomitant use with benzodiazepines or other CNS depressants may result in profound sedation, respiratory depression, coma, or death.",
        ],
        contraindications=[
            "Acute intoxication with alcohol, hypnotics, opioids",
            "Concurrent MAOI",
            "Children < 12 years",
        ],
        common_aes=[
            {"ae": "Dizziness",      "frequency": "10-25%"},
            {"ae": "Nausea",         "frequency": "10-25%"},
            {"ae": "Seizures",       "frequency": "rare, dose-dependent"},
        ],
        notes="Mixed opioid + SNRI. Serotonin syndrome risk with SSRIs.",
        pregnancy="C",
        typical_dose="50-100 mg PO q6h PRN (max 400 mg/day)",
    ),
    "morphine": DrugSafety(
        drug="morphine",
        black_box=[
            "Addiction, abuse, and misuse — major public health crisis.",
            "Respiratory depression — especially in elderly, debilitated, or opioid-naive.",
            "Concomitant use with benzodiazepines may result in profound sedation, respiratory depression, and death.",
        ],
        contraindications=[
            "Severe asthma",
            "Paralytic ileus",
            "Concurrent MAOI",
        ],
        common_aes=[
            {"ae": "Sedation",       "frequency": "10-30%"},
            {"ae": "Constipation",   "frequency": "30-50%"},
            {"ae": "Nausea",         "frequency": "10-30%"},
        ],
        notes="Hepatic glucuronidation. Active M6G metabolite.",
        pregnancy="C",
        typical_dose="5-30 mg PO q4h PRN",
    ),
    "ciprofloxacin": DrugSafety(
        drug="ciprofloxacin",
        black_box=[
            "Tendon rupture and tendinitis — risk increased in age > 60, organ transplant, or concurrent corticosteroids.",
        ],
        contraindications=[
            "Concurrent tizanidine (CYP1A2 inhibition → severe hypotension)",
            "Children < 18 (except specific indications)",
        ],
        common_aes=[
            {"ae": "Nausea",         "frequency": "3-10%"},
            {"ae": "Diarrhea",       "frequency": "1-5%"},
            {"ae": "QT prolongation", "frequency": "rare"},
        ],
        notes="CYP1A2 inhibitor. Adjust dose in renal impairment.",
        pregnancy="C",
        typical_dose="250-750 mg PO q12h",
    ),
    "diazepam": DrugSafety(
        drug="diazepam",
        black_box=[
            "Concomitant use with opioids may result in profound sedation, respiratory depression, and death.",
        ],
        contraindications=[
            "Severe respiratory insufficiency",
            "Sleep apnea syndrome",
            "Myasthenia gravis",
            "Narrow-angle glaucoma",
        ],
        common_aes=[
            {"ae": "Sedation",       "frequency": "10-30%"},
            {"ae": "Dependence",     "frequency": "long-term use"},
        ],
        notes="Long-acting benzodiazepine. CYP3A4 substrate.",
        pregnancy="D",
        typical_dose="2-10 mg PO q6-12h PRN",
    ),
    "ondansetron": DrugSafety(
        drug="ondansetron",
        black_box=[],
        contraindications=[
            "Concurrent apomorphine (profound hypotension)",
        ],
        common_aes=[
            {"ae": "Headache",       "frequency": "5-15%"},
            {"ae": "Constipation",   "frequency": "1-5%"},
            {"ae": "QT prolongation", "frequency": "dose-dependent"},
        ],
        notes="CYP3A4 substrate. ECG monitoring in at-risk patients.",
        pregnancy="B",
        typical_dose="4-8 mg PO/IV q8h PRN",
    ),
}


def get_safety(drug: str) -> DrugSafety | None:
    return SAFETY_REGISTRY.get(drug.lower().strip())


def has_black_box(drug: str) -> bool:
    s = get_safety(drug)
    return bool(s and s.black_box)
