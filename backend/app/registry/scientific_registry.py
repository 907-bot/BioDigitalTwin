"""
backend/app/registry/scientific_registry.py
Scientific Model Registry for BioDigitalTwin.

Every mathematical and physiological model in the platform is formally registered here with:
- Formal model name and biological domain
- Governing mathematical equations (in LaTeX and plaintext)
- State variables with explicit SI / clinical units
- Parameters with literature citations (Bergman, Guyton, Black & Leff, etc.)
- Physiological assumptions and boundary conditions
- Input/Output contracts and coupling dependencies
- Verification and acceptance test criteria
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any


class ModelDomain(str, Enum):
    PHARMACOKINETICS = "pharmacokinetics"
    RECEPTOR_PHARMACOLOGY = "receptor_pharmacology"
    TRANSPORTER_KINETICS = "transporter_kinetics"
    METABOLIC_ENDOCRINE = "metabolic_endocrine"
    CARDIOVASCULAR = "cardiovascular"
    RENAL_ELECTROLYTE = "renal_electrolyte"
    RESPIRATORY_ACID_BASE = "respiratory_acid_base"
    CIRCADIAN_OSCILLATOR = "circadian_oscillator"
    IMMUNE_INFLAMMATORY = "immune_inflammatory"


@dataclass
class VariableContract:
    symbol: str
    name: str
    unit: str
    physio_min: float
    physio_max: float
    baseline_healthy: float
    description: str


@dataclass
class ParameterSpec:
    name: str
    symbol: str
    unit: str
    default_value: float
    valid_range: tuple[float, float]
    citation: str
    description: str


@dataclass
class ModelContract:
    model_id: str
    name: str
    domain: ModelDomain
    governing_equations: List[str]
    state_variables: List[VariableContract]
    parameters: List[ParameterSpec]
    input_contract: List[str]
    output_contract: List[str]
    coupling_dependencies: List[str]
    assumptions: List[str]
    validation_tests: List[str]


# -----------------------------------------------------------------------------
# Global Scientific Model Registry
# -----------------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, ModelContract] = {}


def register_model(contract: ModelContract) -> None:
    MODEL_REGISTRY[contract.model_id] = contract


def get_registered_model(model_id: str) -> Optional[ModelContract]:
    return MODEL_REGISTRY.get(model_id)


def list_registered_models() -> List[Dict[str, Any]]:
    return [
        {
            "model_id": m.model_id,
            "name": m.name,
            "domain": m.domain.value,
            "equations_count": len(m.governing_equations),
            "state_variables": [v.symbol for v in m.state_variables],
            "dependencies": m.coupling_dependencies,
        }
        for m in MODEL_REGISTRY.values()
    ]


# -----------------------------------------------------------------------------
# 1. Black-Leff Operational Model of Agonism
# -----------------------------------------------------------------------------
register_model(ModelContract(
    model_id="black_leff_operational_agonism",
    name="Black-Leff Operational Model of Receptor Agonism & Antagonism",
    domain=ModelDomain.RECEPTOR_PHARMACOLOGY,
    governing_equations=[
        "rho = [L] / ([L] + K_A)",
        "E([L]) = (E_max * tau^n * [L]^n) / (([L] + K_A)^n + tau^n * [L]^n)",
        "tau = [R_total] / K_E",
        "K_A_apparent = K_A * (1 + [B] / K_B)",
        "d[LR]/dt = k_on * [L] * [R_free] - k_off * [LR] - k_int * [LR]"
    ],
    state_variables=[
        VariableContract("R_free", "Free Cell Surface Receptors", "pmol/g", 0.0, 500.0, 50.0, "Unbound functional receptor density"),
        VariableContract("LR", "Ligand-Receptor Complexes", "pmol/g", 0.0, 500.0, 0.0, "Bound agonist-receptor complex"),
        VariableContract("R_int", "Internalized Receptors", "pmol/g", 0.0, 500.0, 5.0, "Endocytosed internalized receptors awaiting recycling/degradation"),
    ],
    parameters=[
        ParameterSpec("k_on", "k_on", "1/(nM*min)", 0.02, (0.0001, 10.0), "Kenakin T. (2014) A Pharmacology Primer", "Bimolecular association rate"),
        ParameterSpec("k_off", "k_off", "1/min", 0.04, (0.0001, 10.0), "Kenakin T. (2014) A Pharmacology Primer", "Unbinding rate constant"),
        ParameterSpec("K_A", "K_A", "nM", 2.0, (0.01, 10000.0), "Black JW, Leff P. (1983) Proc R Soc Lond B 220:141-162", "Equilibrium dissociation constant"),
        ParameterSpec("tau", "tau", "dimensionless", 5.0, (0.01, 100.0), "Black JW, Leff P. (1983)", "Transducer ratio (R_tot / K_E)"),
        ParameterSpec("n", "n", "dimensionless", 1.0, (0.5, 4.0), "Black JW, Leff P. (1983)", "Transducer Hill slope"),
    ],
    input_contract=["ligand_conc_nM", "antagonist_conc_nM", "dt_min"],
    output_contract=["fractional_occupancy", "tissue_response_fraction", "surface_receptor_density"],
    coupling_dependencies=["two_compartment_pk"],
    assumptions=[
        "Homogeneous cell surface distribution of receptors",
        "Receptor-G protein coupling is non-rate limiting relative to ligand unbinding",
        "Conservation of total receptor mass (R_total = R_free + LR + R_int) in absence of de novo synthesis",
    ],
    validation_tests=[
        "test_mass_balance_conservation",
        "test_equilibrium_hyperbolic_occupancy",
        "test_operational_concentration_response_curve",
        "test_competitive_antagonist_rightward_shift"
    ]
))

# -----------------------------------------------------------------------------
# 2. SGLT2 (SLC5A2) Transporter Kinetics
# -----------------------------------------------------------------------------
register_model(ModelContract(
    model_id="sglt2_carrier_kinetics",
    name="SGLT2 (SLC5A2) Renal Glucose-Sodium Cotransport Kinetics",
    domain=ModelDomain.TRANSPORTER_KINETICS,
    governing_equations=[
        "J_glucose = (T_mG * G_plasma) / (K_m * (1 + [I] / K_i) + G_plasma)",
        "J_sodium = J_glucose  (1:1 stoichiometry in SGLT2)",
        "Glucosuria = max(0.0, Filtered_Glucose - J_glucose)",
        "Filtered_Glucose = GFR * G_plasma"
    ],
    state_variables=[
        VariableContract("Filtered_Glucose", "Glomerular Filtered Glucose Load", "mg/min", 50.0, 1000.0, 120.0, "GFR * Plasma Glucose"),
        VariableContract("Reabsorbed_Glucose", "Tubular Glucose Reabsorbed", "mg/min", 0.0, 450.0, 120.0, "SGLT2 + SGLT1 mediated reabsorption"),
        VariableContract("Urinary_Glucose", "Glucosuric Excretion Rate", "mg/min", 0.0, 600.0, 0.0, "Glucose escaping tubular reabsorption"),
    ],
    parameters=[
        ParameterSpec("T_mG", "T_mG", "mg/min", 375.0, (200.0, 500.0), "DeFronzo RA et al. (2012) Diabetes Care 35:2095", "Maximal tubular reabsorptive capacity"),
        ParameterSpec("K_m", "K_m", "mg/dL", 108.0, (50.0, 200.0), "Hummel CS et al. (2011) Am J Physiol Cell Physiol 300:C14", "Glucose affinity constant (6.0 mM)"),
        ParameterSpec("K_i", "K_i", "nM", 1.3, (0.1, 50.0), "Grempler R et al. (2012) Diabetes Obes Metab 14:83", "Empagliflozin SGLT2 inhibition constant"),
    ],
    input_contract=["plasma_glucose_mg_dL", "gfr_mL_min", "inhibitor_conc_nM"],
    output_contract=["urinary_glucose_excretion_mg_min", "reabsorbed_sodium_mEq_min"],
    coupling_dependencies=["metabolic_bergman", "renal_guyton"],
    assumptions=[
        "SGLT2 accounts for ~90% of renal glucose reabsorption; SGLT1 accounts for remaining ~10%",
        "1:1 Na:glucose stoichiometric coupling in proximal convoluted tubule S1/S2 segments",
    ],
    validation_tests=[
        "test_renal_threshold_normal_180mg_dL",
        "test_sglt2_inhibitor_glucosuria_induction",
        "test_natriuresis_coupling"
    ]
))

# -----------------------------------------------------------------------------
# 3. Respiratory Mechanics, Gas Exchange & Acid-Base Balance
# -----------------------------------------------------------------------------
register_model(ModelContract(
    model_id="respiratory_acid_base",
    name="Respiratory Gas Exchange, Bohr-Shifted Hemoglobin & Acid-Base Buffering",
    domain=ModelDomain.RESPIRATORY_ACID_BASE,
    governing_equations=[
        "pH = 6.10 + log10([HCO3_minus] / (0.0307 * PaCO2))",
        "P_50(pH, PaCO2) = 26.8 * 10^(-0.48 * (pH - 7.4) + 0.06 * log10(PaCO2 / 40.0))",
        "SaO2 = (PaO2^2.7) / (P_50^2.7 + PaO2^2.7)",
        "dPaCO2/dt = (V_CO2_prod - 0.863 * V_A * PaCO2) / V_dist_CO2",
        "dPaO2/dt = (V_A * (P_I_O2 - PaO2) - Q_cardiac * 1.34 * Hb * (SaO2 - SvO2)) / V_dist_O2"
    ],
    state_variables=[
        VariableContract("PaO2", "Arterial Oxygen Partial Pressure", "mmHg", 40.0, 120.0, 95.0, "Arterial partial pressure of dissolved O2"),
        VariableContract("PaCO2", "Arterial CO2 Partial Pressure", "mmHg", 20.0, 80.0, 40.0, "Arterial partial pressure of CO2"),
        VariableContract("HCO3", "Plasma Bicarbonate", "mEq/L", 10.0, 40.0, 24.0, "Plasma bicarbonate ion concentration"),
        VariableContract("pH", "Arterial Blood pH", "pH units", 6.8, 7.8, 7.40, "-log10[H+]"),
        VariableContract("SaO2", "Arterial Oxygen Saturation", "percent", 70.0, 100.0, 98.0, "Fractional hemoglobin oxygen saturation"),
    ],
    parameters=[
        ParameterSpec("P_50_base", "P_50_base", "mmHg", 26.8, (24.0, 30.0), "Severinghaus JW. (1979) J Appl Physiol 46:599", "Standard P50 at pH 7.40, PaCO2 40"),
        ParameterSpec("solubility_CO2", "alpha_CO2", "mmol/(L*mmHg)", 0.0307, (0.030, 0.032), "Henderson LJ. (1908) Am J Physiol", "CO2 plasma solubility coefficient"),
        ParameterSpec("pK_a", "pK_a", "dimensionless", 6.10, (6.08, 6.12), "Hasselbalch KA. (1916) Biochem Z", "Carbonic acid first dissociation constant"),
        ParameterSpec("Hb", "Hemoglobin", "g/dL", 15.0, (8.0, 20.0), "Clinical Reference", "Blood hemoglobin concentration"),
    ],
    input_contract=["alveolar_ventilation_L_min", "metabolic_CO2_prod_mL_min", "cardiac_output_L_min"],
    output_contract=["arterial_pH", "SaO2_pct", "PaO2_mmHg", "PaCO2_mmHg"],
    coupling_dependencies=["cardiovascular_guyton", "metabolic_bergman"],
    assumptions=[
        "Immediate open-system equilibrium between dissolved CO2 and carbonic acid",
        "Respiratory compensation: alveolar ventilation responds to chemoreceptors sensing pH and PaCO2",
        "Bohr effect: Acidosis (lower pH) shifts P50 rightwards, facilitating O2 unloading",
    ],
    validation_tests=[
        "test_henderson_hasselbalch_normal_values",
        "test_bohr_effect_acidosis_rightward_shift",
        "test_hyperventilation_respiratory_alkalosis"
    ]
))

# -----------------------------------------------------------------------------
# 4. Renin-Angiotensin-Aldosterone System (RAAS)
# -----------------------------------------------------------------------------
register_model(ModelContract(
    model_id="raas_hemodynamics",
    name="Renin-Angiotensin-Aldosterone (RAAS) & Long-Term Volume Regulation",
    domain=ModelDomain.RENAL_ELECTROLYTE,
    governing_equations=[
        "Renin_secretion = R_0 * (MAP_0 / MAP)^2 * (GFR_0 / GFR)",
        "d[AngII]/dt = k_ACE * Renin - k_deg_AngII * [AngII]",
        "d[Aldo]/dt = k_aldo * ([AngII] / ([AngII] + K_m_AngII)) * ([K] / 4.0) - k_clear_aldo * [Aldo]",
        "Na_reabsorption_fraction = Na_base * (1 + beta_aldo * [Aldo]) * (1 + beta_AngII * [AngII])",
        "d[V_plasma]/dt = Fluid_Intake - Urine_Volume(GFR, Na_reabsorption)"
    ],
    state_variables=[
        VariableContract("Plasma_Renin", "Plasma Renin Activity", "ng/(mL*h)", 0.1, 20.0, 1.5, "Juxtaglomerular renin activity"),
        VariableContract("Angiotensin_II", "Plasma Angiotensin II", "pg/mL", 5.0, 150.0, 25.0, "Vasoactive Ang II peptide"),
        VariableContract("Aldosterone", "Plasma Aldosterone", "ng/dL", 2.0, 80.0, 12.0, "Mineralocorticoid hormone"),
        VariableContract("Plasma_Volume", "Effective Circulating Plasma Volume", "L", 2.0, 5.0, 3.0, "Intravascular fluid volume"),
    ],
    parameters=[
        ParameterSpec("k_ACE", "k_ACE", "1/min", 0.08, (0.01, 0.5), "Guyton AC et al. (1972) Annu Rev Physiol", "Angiotensin I to II conversion rate"),
        ParameterSpec("k_deg_AngII", "k_deg_AngII", "1/min", 0.35, (0.1, 1.0), "Hall JE. (2015) Guyton Textbook of Medical Physiology", "Ang II enzymatic degradation"),
        ParameterSpec("k_aldo", "k_aldo", "ng/(dL*min)", 0.20, (0.05, 1.0), "Guyton AC. (1972)", "Maximal aldosterone synthesis rate"),
        ParameterSpec("MAP_0", "MAP_0", "mmHg", 93.3, (70.0, 110.0), "Physiological Setpoint", "Mean arterial pressure setpoint"),
    ],
    input_contract=["MAP_mmHg", "GFR_mL_min", "plasma_K_mEq_L", "fluid_intake_mL_min"],
    output_contract=["plasma_volume_L", "vascular_resistance_mod", "urinary_sodium_excretion"],
    coupling_dependencies=["cardiovascular_guyton", "renal_electrolyte"],
    assumptions=[
        "Negative feedback: Elevated MAP suppresses juxtaglomerular renin release",
        "Angiotensin II acts on vascular smooth muscle AT1 receptors and adrenal zona glomerulosa",
        "Aldosterone increases ENaC expression in renal collecting duct to reabsorb Na+",
    ],
    validation_tests=[
        "test_hypotension_stimulates_renin",
        "test_ace_inhibitor_blocks_ang_ii",
        "test_aldosterone_enhances_sodium_retention"
    ]
))

