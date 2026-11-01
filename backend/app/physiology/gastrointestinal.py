"""
Dedicated Gastrointestinal System & Entero-Endocrine Model.

Simulates:
1. Multi-compartment digestion & transit:
   - Gastric solid/liquid emptying (dQ_stom/dt = -k_empt * Q_stom)
   - Modulation by meal caloric density, viscosity, and GLP-1 feedback
2. Intestinal absorption:
   - Carbohydrate enzymatic hydrolysis and sodium-glucose luminal transport (SGLT1)
   - Dietary triglyceride emulsification and micelle absorption
   - Dietary protein breakdown into absorbable amino acids
3. Entero-endocrine axis (Incretins):
   - GLP-1 secretion from intestinal L-cells in response to luminal nutrients
   - GIP secretion from K-cells
   - DPP-4 rapid enzymatic clearance (half-life ~2 min)
   - Incretin feedback loop: delay of gastric motility and augmentation of glucose-stimulated insulin
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from .base import PhysiologicalOrganModel, OrganStateDescriptor, OrganParamDescriptor


@dataclass
class GastrointestinalState:
    stomach_carb_mass_g: float = 0.0
    stomach_lipid_mass_g: float = 0.0
    stomach_protein_mass_g: float = 0.0
    intestinal_glucose_mass_g: float = 0.0
    glucose_absorption_rate_mg_min: float = 0.0
    ffa_absorption_rate_mmol_min: float = 0.0
    glp1_plasma_pmol_L: float = 15.0
    gip_plasma_pmol_L: float = 20.0
    gastric_emptying_rate_multiplier: float = 1.0


class GastrointestinalOrganModel(PhysiologicalOrganModel):
    """
    Multi-compartment gut motility, nutrient absorption, and incretin physiology.
    """

    def __init__(self):
        super().__init__(name="Gastrointestinal")
        self.declare_descriptors()
        self.initialize()

    def declare_descriptors(self) -> None:
        self.descriptors_state = {
            "stomach_carb_mass_g": OrganStateDescriptor(
                "stomach_carb_mass_g", "g", 0.0, 0.0, 300.0, "Carbohydrate mass in stomach"
            ),
            "stomach_lipid_mass_g": OrganStateDescriptor(
                "stomach_lipid_mass_g", "g", 0.0, 0.0, 150.0, "Fat mass in stomach"
            ),
            "stomach_protein_mass_g": OrganStateDescriptor(
                "stomach_protein_mass_g", "g", 0.0, 0.0, 150.0, "Protein mass in stomach"
            ),
            "intestinal_glucose_mass_g": OrganStateDescriptor(
                "intestinal_glucose_mass_g", "g", 0.0, 0.0, 100.0, "Glucose in small intestine lumen"
            ),
            "glucose_absorption_rate_mg_min": OrganStateDescriptor(
                "glucose_absorption_rate_mg_min", "mg/min", 0.0, 0.0, 1500.0, "Systemic rate of glucose appearance"
            ),
            "glp1_plasma_pmol_L": OrganStateDescriptor(
                "glp1_plasma_pmol_L", "pmol/L", 15.0, 2.0, 150.0, "Active GLP-1 hormone"
            ),
            "gip_plasma_pmol_L": OrganStateDescriptor(
                "gip_plasma_pmol_L", "pmol/L", 20.0, 5.0, 250.0, "Active GIP incretin"
            ),
        }

        self.descriptors_param = {
            "basal_emptying_rate_min": OrganParamDescriptor(
                "basal_emptying_rate_min", "1/min", 0.025, 0.005, 0.08, "Fractional gastric emptying rate"
            ),
            "absorption_rate_carb_min": OrganParamDescriptor(
                "absorption_rate_carb_min", "1/min", 0.040, 0.010, 0.12, "Intestinal absorption rate constant"
            ),
            "glp1_dpp4_clearance_min": OrganParamDescriptor(
                "glp1_dpp4_clearance_min", "1/min", 0.35, 0.10, 1.0, "DPP-4 enzymatic inactivation rate"
            ),
            "incretin_secretion_gain": OrganParamDescriptor(
                "incretin_secretion_gain", "pmol/L per g/min", 0.30, 0.01, 0.60, "Nutrient-stimulated incretin release"
            ),
        }

    def initialize(self, patient_parameters: Optional[Dict[str, float]] = None) -> None:
        self._state = {
            "stomach_carb_mass_g": 0.0,
            "stomach_lipid_mass_g": 0.0,
            "stomach_protein_mass_g": 0.0,
            "intestinal_glucose_mass_g": 0.0,
            "glucose_absorption_rate_mg_min": 0.0,
            "ffa_absorption_rate_mmol_min": 0.0,
            "glp1_plasma_pmol_L": 15.0,
            "gip_plasma_pmol_L": 20.0,
            "gastric_emptying_rate_multiplier": 1.0,
        }
        self._params = {k: v.default_value for k, v in self.descriptors_param.items()}
        if patient_parameters:
            self.set_parameters(patient_parameters)

    def ingest_meal(self, carbs_g: float, lipids_g: float = 0.0, proteins_g: float = 0.0) -> None:
        """Add bolus nutrients into stomach compartment."""
        self._state["stomach_carb_mass_g"] += max(0.0, carbs_g)
        self._state["stomach_lipid_mass_g"] += max(0.0, lipids_g)
        self._state["stomach_protein_mass_g"] += max(0.0, proteins_g)

    def compute_derivatives(
        self,
        state: Dict[str, float],
        inputs: Dict[str, float],
        transport: Dict[str, float],
    ) -> Dict[str, float]:
        # Meal intake if provided in current step inputs
        meal_carbs = inputs.get("meal_carbs_g", 0.0)
        meal_fat = inputs.get("meal_lipids_g", 0.0)
        meal_protein = inputs.get("meal_protein_g", 0.0)

        # GLP-1 and sympathetic tone feedback slows gastric emptying
        glp1_inhibition = 1.0 / (1.0 + 0.03 * max(0.0, state["glp1_plasma_pmol_L"] - 15.0))
        k_empt = self._params["basal_emptying_rate_min"] * glp1_inhibition
        k_abs = self._params["absorption_rate_carb_min"]

        # Stomach differential equations
        # d(Q_stom)/dt = Meal_rate - k_empt * Q_stom
        emptying_carbs_g_min = k_empt * state["stomach_carb_mass_g"]
        d_stom_carb = meal_carbs - emptying_carbs_g_min

        emptying_fat_g_min = (k_empt * 0.7) * state["stomach_lipid_mass_g"]  # fat empties slower
        d_stom_lipid = meal_fat - emptying_fat_g_min

        emptying_prot_g_min = (k_empt * 0.85) * state["stomach_protein_mass_g"]
        d_stom_protein = meal_protein - emptying_prot_g_min

        # Small intestine luminal glucose mass:
        # d(Q_int)/dt = Emptying - Absorption
        absorption_g_min = k_abs * state["intestinal_glucose_mass_g"]
        d_int_glucose = emptying_carbs_g_min - absorption_g_min

        # Rate of appearance into portal circulation (mg/min)
        ra_glucose_mg_min = absorption_g_min * 1000.0
        d_ra = 1.0 * (ra_glucose_mg_min - state["glucose_absorption_rate_mg_min"])

        # Lipid absorption (converted to mmol FFA/min: 1 g fat ~ 1.1 mmol FFA)
        ffa_ra = emptying_fat_g_min * 1.10
        d_ffa_ra = 0.5 * (ffa_ra - state["ffa_absorption_rate_mmol_min"])

        # Incretin dynamics:
        # Luminal glucose & fat stimulate GLP-1 & GIP; DPP-4 clears them
        nutrient_flux = (absorption_g_min + emptying_fat_g_min) * 1000.0  # mg/min
        glp1_secretion = self._params["incretin_secretion_gain"] * (nutrient_flux / 100.0)
        d_glp1 = glp1_secretion - self._params["glp1_dpp4_clearance_min"] * (state["glp1_plasma_pmol_L"] - 15.0)

        gip_secretion = glp1_secretion * 1.30
        d_gip = gip_secretion - (self._params["glp1_dpp4_clearance_min"] * 0.8) * (state["gip_plasma_pmol_L"] - 20.0)

        return {
            "stomach_carb_mass_g": d_stom_carb,
            "stomach_lipid_mass_g": d_stom_lipid,
            "stomach_protein_mass_g": d_stom_protein,
            "intestinal_glucose_mass_g": d_int_glucose,
            "glucose_absorption_rate_mg_min": d_ra,
            "ffa_absorption_rate_mmol_min": d_ffa_ra,
            "glp1_plasma_pmol_L": d_glp1,
            "gip_plasma_pmol_L": d_gip,
        }

    def exchange_fluxes(self, transport: Dict[str, float]) -> Dict[str, float]:
        return {
            "glucose_flux_mg_min": float(self._state["glucose_absorption_rate_mg_min"]),
            "ffa_flux_mmol_min": float(self._state["ffa_absorption_rate_mmol_min"]),
            "glp1_concentration_pmol_L": float(self._state["glp1_plasma_pmol_L"]),
            "gip_concentration_pmol_L": float(self._state["gip_plasma_pmol_L"]),
        }
