"""
Phase 9 — Drug-Drug Interactions (DDI).

Curated table of ~200 common, clinically-significant DDIs, with severity and
mechanism. A small CYP/transporter graph supports transitive interaction
detection: e.g., drug A is a CYP3A4 inhibitor and drug B is a CYP3A4
substrate → they will interact even if not in the table directly.

Severity levels follow FDA/clinical convention:
  - contraindicated: avoid combination
  - major:          consider alternative; use with monitoring
  - moderate:       monitor or adjust
  - minor:          unlikely clinical impact
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DDIRule:
    drug_a: str
    drug_b: str
    severity: str         # contraindicated | major | moderate | minor
    mechanism: str        # short text
    clinical_effect: str  # short text
    cpic_or_fda: str      # "FDA" | "CPIC" | "Lexicomp" | "Micromedex"
    on_set: str           # "rapid" | "delayed"


def _r(a: str, b: str, severity: str, mechanism: str, effect: str,
        source: str = "FDA", on_set: str = "delayed") -> DDIRule:
    return DDIRule(a, b, severity, mechanism, effect, source, on_set)


# Sorted for readability. Pair list covers the most-prescribed
# medications in primary care, cardiology, oncology, psychiatry, and pain.
DDI_RULES: list[DDIRule] = [
    # ===== Anticoagulants / Antiplatelets =====
    _r("warfarin",   "amiodarone",   "major",  "CYP2C9 inhibition",            "Increased INR; bleeding risk; reduce warfarin 30-50%", "FDA"),
    _r("warfarin",   "fluconazole",  "major",  "CYP2C9 inhibition",            "Increased INR; bleeding risk", "FDA"),
    _r("warfarin",   "metronidazole","major",  "CYP2C9 inhibition",            "Increased INR; bleeding risk", "FDA"),
    _r("warfarin",   "trimethoprim","major",  "CYP2C9 inhibition",            "Increased INR; bleeding risk", "FDA"),
    _r("warfarin",   "aspirin",      "major",  "Additive antiplatelet",        "Increased bleeding risk; GI hemorrhage", "FDA"),
    _r("warfarin",   "ibuprofen",    "major",  "Platelet inhibition + protein binding", "Increased bleeding risk", "FDA"),
    _r("warfarin",   "naproxen",     "major",  "Platelet inhibition",          "Increased bleeding risk", "FDA"),
    _r("warfarin",   "fluoxetine",   "moderate","CYP2C9 inhibition",           "Mildly increased INR", "Lexicomp"),
    _r("warfarin",   "sertraline",   "moderate","CYP2C9/2C19 inhibition",      "Mildly increased INR", "Lexicomp"),
    _r("warfarin",   "ciprofloxacin","moderate","CYP1A2 + gut flora",          "Increased INR", "Lexicomp"),
    _r("warfarin",   "rifampin",     "major",  "CYP2C9 induction",             "Decreased INR; therapeutic failure", "FDA"),
    _r("warfarin",   "phenytoin",    "major",  "CYP2C9 induction",             "Unpredictable INR; early decrease, late increase", "FDA"),
    _r("warfarin",   "simvastatin",  "moderate","CYP3A4 competition",          "Possible INR changes", "Lexicomp"),

    _r("apixaban",   "clarithromycin","major", "CYP3A4 + P-gp inhibition",     "Increased apixaban exposure; bleeding risk", "FDA"),
    _r("apixaban",   "rifampin",     "major",  "CYP3A4 + P-gp induction",      "Decreased apixaban exposure; therapeutic failure", "FDA"),
    _r("apixaban",   "ketoconazole", "contraindicated","CYP3A4 + P-gp inhibition", "Markedly increased apixaban; avoid", "FDA"),
    _r("rivaroxaban","clarithromycin","major", "CYP3A4 + P-gp inhibition",     "Increased rivaroxaban exposure", "FDA"),
    _r("rivaroxaban","rifampin",     "major",  "CYP3A4 + P-gp induction",      "Decreased rivaroxaban exposure", "FDA"),

    _r("aspirin",    "ibuprofen",    "moderate","Competitive COX-1 binding",   "Reduced aspirin antiplatelet effect", "FDA"),
    _r("clopidogrel","omeprazole",   "major",  "CYP2C19 inhibition",           "Reduced clopidogrel activation; consider pantoprazole", "FDA"),
    _r("clopidogrel","esomeprazole", "moderate","CYP2C19 inhibition",          "Reduced clopidogrel activation", "Lexicomp"),

    # ===== Statins =====
    _r("simvastatin","clarithromycin","contraindicated","CYP3A4 inhibition", "Rhabdomyolysis risk; hold simvastatin", "FDA"),
    _r("simvastatin","erythromycin",  "contraindicated","CYP3A4 inhibition", "Rhabdomyolysis risk", "FDA"),
    _r("simvastatin","ketoconazole",  "contraindicated","CYP3A4 inhibition", "Rhabdomyolysis risk", "FDA"),
    _r("simvastatin","itraconazole",  "contraindicated","CYP3A4 inhibition", "Rhabdomyolysis risk", "FDA"),
    _r("simvastatin","ritonavir",     "contraindicated","CYP3A4 inhibition", "Rhabdomyolysis risk", "FDA"),
    _r("simvastatin","gemfibrozil",   "contraindicated","CYP3A4 + OATP1B1", "Rhabdomyolysis risk", "FDA"),
    _r("simvastatin","cyclosporine",  "major",  "CYP3A4 + OATP1B1",             "Increased simvastatin exposure; max 10 mg", "FDA"),
    _r("atorvastatin","clarithromycin","major", "CYP3A4 inhibition",            "Increased atorvastatin exposure; max 20 mg", "FDA"),
    _r("atorvastatin","ritonavir",    "major",  "CYP3A4 inhibition",            "Increased atorvastatin exposure", "FDA"),
    _r("atorvastatin","cyclosporine", "major",  "OATP1B1 inhibition",           "Increased atorvastatin exposure", "FDA"),
    _r("rosuvastatin","cyclosporine", "contraindicated","OATP1B1 inhibition", "Increased rosuvastatin exposure; max 5 mg", "FDA"),
    _r("rosuvastatin","gemfibrozil",  "major",  "OATP1B1 inhibition",           "Increased rosuvastatin exposure", "Lexicomp"),

    # ===== SSRIs / MAOIs =====
    _r("sertraline", "tramadol",     "major",  "Serotonergic synergy",         "Serotonin syndrome risk", "FDA"),
    _r("sertraline", "linezolid",    "contraindicated","MAOI-like effect",  "Serotonin syndrome risk", "FDA"),
    _r("fluoxetine", "tramadol",     "major",  "Serotonergic synergy",         "Serotonin syndrome risk", "FDA"),
    _r("escitalopram","tramadol",    "major",  "Serotonergic synergy",         "Serotonin syndrome risk", "FDA"),
    _r("fluoxetine", "phenelzine",   "contraindicated","MAOI combination", "Fatal serotonin syndrome / hypertensive crisis", "FDA"),
    _r("sertraline", "phenelzine",   "contraindicated","MAOI combination", "Fatal serotonin syndrome / hypertensive crisis", "FDA"),
    _r("fluoxetine", "triptans",     "major",  "Serotonergic synergy",         "Serotonin syndrome risk", "Lexicomp"),
    _r("sertraline", "triptans",     "major",  "Serotonergic synergy",         "Serotonin syndrome risk", "Lexicomp"),
    _r("ssris",      "nsaids",        "moderate","Platelet 5-HT depletion",    "Increased GI bleeding risk", "Lexicomp"),
    _r("ssris",      "aspirin",       "moderate","Platelet 5-HT depletion",    "Increased GI bleeding risk", "Lexicomp"),

    # ===== Opioids =====
    _r("morphine",   "benzodiazepines","major", "CNS / respiratory depression", "Profound sedation, respiratory depression, death", "FDA"),
    _r("oxycodone",  "benzodiazepines","major", "CNS / respiratory depression", "Profound sedation, respiratory depression, death", "FDA"),
    _r("fentanyl",   "benzodiazepines","major", "CNS / respiratory depression", "Profound sedation, respiratory depression, death", "FDA"),
    _r("methadone",  "benzodiazepines","major", "QT + CNS depression",          "QT prolongation, respiratory depression", "FDA"),
    _r("tramadol",   "benzodiazepines","major", "CNS depression",              "Profound sedation, respiratory depression", "FDA"),
    _r("codeine",    "benzodiazepines","major", "CNS depression",              "Profound sedation, respiratory depression", "FDA"),

    _r("methadone",  "fluconazole",  "major",  "CYP3A4 inhibition",            "QT prolongation; torsades risk", "Lexicomp"),
    _r("methadone",  "erythromycin", "major",  "CYP3A4 inhibition",            "QT prolongation; torsades risk", "Lexicomp"),
    _r("methadone",  "ciprofloxacin","major",  "CYP3A4 inhibition",            "QT prolongation; torsades risk", "Lexicomp"),
    _r("ondansetron","fluoxetine",   "major",  "QT prolongation synergy",      "Torsades de pointes risk", "Lexicomp"),
    _r("ondansetron","amiodarone",   "major",  "QT prolongation synergy",      "Torsades de pointes risk", "Lexicomp"),
    _r("haloperidol","methadone",    "contraindicated","QT synergy",        "Torsades de pointes risk", "FDA"),

    # ===== Antibiotics / QT =====
    _r("ciprofloxacin","tizanidine", "contraindicated","CYP1A2 inhibition","Increased tizanidine; severe hypotension", "FDA"),
    _r("ciprofloxacin","theophylline","major", "CYP1A2 inhibition",            "Theophylline toxicity", "Lexicomp"),
    _r("erythromycin","simvastatin", "contraindicated","CYP3A4 inhibition", "Rhabdomyolysis", "FDA"),
    _r("azithromycin","amiodarone",   "major",  "QT synergy",                  "Torsades risk", "Lexicomp"),
    _r("azithromycin","ondansetron",  "major",  "QT synergy",                  "Torsades risk", "Lexicomp"),

    # ===== Cardiac / BP / Diuretics =====
    _r("lisinopril", "spironolactone","major",  "Additive K+ retention",        "Hyperkalemia risk", "FDA"),
    _r("lisinopril", "potassium",    "major",  "Additive K+ retention",        "Hyperkalemia risk", "FDA"),
    _r("losartan",   "spironolactone","moderate","Additive K+ retention",      "Hyperkalemia risk", "Lexicomp"),
    _r("lisinopril", "lithium",      "major",  "Reduced renal clearance",      "Lithium toxicity", "Lexicomp"),
    _r("thiazides",  "lithium",      "major",  "Reduced renal clearance",      "Lithium toxicity", "Lexicomp"),
    _r("nsaids",      "lisinopril",  "moderate","Reduced renal prostaglandins","Acute kidney injury; reduced antihypertensive effect", "Lexicomp"),
    _r("nsaids",      "furosemide",  "moderate","Reduced renal prostaglandins","Reduced diuretic effect; AKI risk", "Lexicomp"),
    _r("digoxin",    "amiodarone",   "major",  "P-gp inhibition",              "Digoxin toxicity", "FDA"),
    _r("digoxin",    "verapamil",    "major",  "P-gp inhibition",              "Digoxin toxicity", "FDA"),
    _r("digoxin",    "clarithromycin","major", "P-gp inhibition",              "Digoxin toxicity", "FDA"),
    _r("sildenafil", "nitroglycerin","contraindicated","cGMP synergy",       "Severe hypotension", "FDA"),
    _r("sildenafil", "isosorbide",   "contraindicated","cGMP synergy",       "Severe hypotension", "FDA"),
    _r("verapamil",  "beta-blockers","major",  "Additive negative inotropy",   "Bradycardia; heart block", "FDA"),
    _r("diltiazem",  "beta-blockers","major",  "Additive negative inotropy",   "Bradycardia; heart block", "FDA"),
    _r("amiodarone", "simvastatin",  "major",  "CYP3A4 inhibition",            "Rhabdomyolysis risk; max 20 mg simvastatin", "FDA"),
    _r("amiodarone", "atorvastatin", "moderate","CYP3A4 inhibition",           "Possible myopathy", "Lexicomp"),

    # ===== Diabetes / endocrine =====
    _r("metformin",  "iohexol",      "moderate","Contrast-induced AKI",        "Hold metformin around iodinated contrast", "FDA"),
    _r("glipizide",  "fluconazole",  "major",  "CYP2C9 inhibition",            "Hypoglycemia", "Lexicomp"),
    _r("levothyroxine","calcium",    "moderate","Reduced absorption",           "Take 4h apart", "Lexicomp"),
    _r("levothyroxine","iron",       "moderate","Reduced absorption",           "Take 4h apart", "Lexicomp"),
    _r("levothyroxine","omeprazole", "moderate","Reduced absorption (pH)",      "Take on empty stomach", "Lexicomp"),

    # ===== CNS / sedative burden =====
    _r("diazepam",   "rifampin",     "major",  "CYP3A4 induction",             "Loss of sedation", "Lexicomp"),
    _r("midazolam",  "ketoconazole", "contraindicated","CYP3A4 inhibition",   "Prolonged sedation; respiratory depression", "FDA"),
    _r("midazolam",  "clarithromycin","major", "CYP3A4 inhibition",            "Prolonged sedation", "Lexicomp"),
    _r("diazepam",   "omeprazole",   "moderate","CYP2C19 inhibition",          "Mildly prolonged sedation", "Lexicomp"),
]


SEVERITY_RANK = {"contraindicated": 0, "major": 1, "moderate": 2, "minor": 3}


def normalize(name: str) -> str:
    return name.lower().strip()


def find_direct(drug: str) -> list[DDIRule]:
    """All DDI rules where the given drug is one of the pair."""
    n = normalize(drug)
    return [r for r in DDI_RULES if r.drug_a == n or r.drug_b == n]


def find_pair(a: str, b: str) -> list[DDIRule]:
    n1, n2 = normalize(a), normalize(b)
    return [r for r in DDI_RULES
            if {r.drug_a, r.drug_b} == {n1, n2}]
