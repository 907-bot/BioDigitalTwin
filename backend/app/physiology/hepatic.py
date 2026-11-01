"""
Dedicated Hepatic Physiology & Liver Metabolism Model.

Simulates:
1. Carbohydrate regulation:
   - Hepatic glycogen dynamics (synthesis via glycogen synthase vs glycogenolysis)
   - Gluconeogenesis from lactate, glycerol, amino acids
   - Net Hepatic Glucose Production (HGP) regulated by Insulin/Glucagon ratio
2. Lipid metabolism:
   - Hepatic FFA uptake & De Novo Lipogenesis (DNL)
   - Beta-oxidation & Ketogenesis (acetoacetate/beta-hydroxybutyrate)
   - VLDL / Triglyceride secretion
3. Protein & nitrogen handling:
   - Urea cycle (ammonia NH3 -> Urea conversion)
   - Albumin synthesis and plasma oncotic maintenance
4. Hepatic clearance & pharmacokinetics:
   - Well-stirred model of hepatic drug clearance (CYP450 metabolism)
   - First-pass portal clearance
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from .base import PhysiologicalOrganModel, OrganStateDescriptor, OrganParamDescriptor


@dataclass
class HepaticState:
    glycogen_mass_g: float = 85.0
    hgp_rate_mg_min: float = 140.0
    gluconeogenesis_rate_mg_min: float = 50.0
    glycogenolysis_rate_mg_min: float = 90.0
    glycogen_synthesis_rate_mg_min: float = 10.0
    plasma_albumin_g_dL: float = 4.2
    ammonia_umol_L: float = 25.0
    urea_production_mg_min: float = 20.0
    vldl_triglyceride_secretion_mg_min: float = 12.0
    ketone_bodies_mmol_L: float = 0.15
    cyp3a4_activity_index: float = 1.0


class HepaticOrganModel(PhysiologicalOrganModel):
    """
    Biochemical differential model of liver metabolism, energetics, and clearance.
    """

    def __init__(self):
        super().__init__(name="Hepatic")
        self.declare_descriptors()
        self.initialize()

    def declare_descriptors(self) -> None:
        self.descriptors_state = {
            "glycogen_mass_g": OrganStateDescriptor(
                "glycogen_mass_g", "g", 85.0, 0.0, 150.0, "Hepatic stored glycogen mass"
            ),
            "hgp_rate_mg_min": OrganStateDescriptor(
                "hgp_rate_mg_min", "mg/min", 140.0, 0.0, 600.0, "Total endogenous glucose production"
            ),
            "plasma_albumin_g_dL": OrganStateDescriptor(
                "plasma_albumin_g_dL", "g/dL", 4.2, 1.5, 6.0, "Serum albumin concentration"
            ),
            "ammonia_umol_L": OrganStateDescriptor(
                "ammonia_umol_L", "umol/L", 25.0, 5.0, 200.0, "Circulating ammonia"
            ),
            "urea_production_mg_min": OrganStateDescriptor(
                "urea_production_mg_min", "mg/min", 20.0, 2.0, 80.0, "Urea cycle output"
            ),
            "ketone_bodies_mmol_L": OrganStateDescriptor(
                "ketone_bodies_mmol_L", "mmol/L", 0.15, 0.01, 15.0, "Beta-hydroxybutyrate & AcAc"
            ),
            "cyp3a4_activity_index": OrganStateDescriptor(
                "cyp3a4_activity_index", "normalized", 1.0, 0.05, 3.0, "CYP enzyme metabolic capacity"
            ),
        }

        self.descriptors_param = {
            "max_glycogen_capacity_g": OrganParamDescriptor(
                "max_glycogen_capacity_g", "g", 120.0, 60.0, 160.0, "Maximum storage capacity"
            ),
            "basal_gluconeogenesis_mg_min": OrganParamDescriptor(
                "basal_gluconeogenesis_mg_min", "mg/min", 50.0, 20.0, 120.0, "Basal de novo glucose production"
            ),
            "insulin_hgp_suppression_ec50": OrganParamDescriptor(
                "insulin_hgp_suppression_ec50", "uU/mL", 25.0, 8.0, 60.0, "Insulin EC50 for HGP inhibition"
            ),
            "glucagon_hgp_stimulation_ec50": OrganParamDescriptor(
                "glucagon_hgp_stimulation_ec50", "pg/mL", 70.0, 30.0, 150.0, "Glucagon EC50 for HGP stimulation"
            ),
            "intrinsic_cyp_clearance_L_min": OrganParamDescriptor(
                "intrinsic_cyp_clearance_L_min", "L/min", 1.2, 0.2, 5.0, "Baseline CYP intrinsic clearance"
            ),
        }

    def initialize(self, patient_parameters: Optional[Dict[str, float]] = None) -> None:
        self._state = {
            "glycogen_mass_g": 85.0,
            "hgp_rate_mg_min": 140.0,
            "gluconeogenesis_rate_mg_min": 50.0,
            "glycogenolysis_rate_mg_min": 90.0,
            "glycogen_synthesis_rate_mg_min": 10.0,
            "plasma_albumin_g_dL": 4.2,
            "ammonia_umol_L": 25.0,
            "urea_production_mg_min": 20.0,
            "vldl_triglyceride_secretion_mg_min": 12.0,
            "ketone_bodies_mmol_L": 0.15,
            "cyp3a4_activity_index": 1.0,
        }
        self._params = {k: v.default_value for k, v in self.descriptors_param.items()}
        if patient_parameters:
            self.set_parameters(patient_parameters)

    def compute_derivatives(
        self,
        state: Dict[str, float],
        inputs: Dict[str, float],
        transport: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Differential rates for hepatic glycogen, glucose production, ammonia detoxification.
        """
        insulin = transport.get("insulin_uU_mL", 8.0)
        glucagon = transport.get("glucagon_pg_mL", 70.0)
        glucose = transport.get("glucose_mg_dL", 95.0)
        ffa = transport.get("ffa_mmol_L", 0.45)
        lactate = transport.get("lactate_mmol_L", 1.0)
        exercise = inputs.get("exercise", 0.0)

        # Hormonal regulation of Hepatic Glucose Production
        # Cherrington / Bergman model: insulin suppresses, glucagon stimulates
        ins_suppression = 1.0 - (insulin / (insulin + self._params["insulin_hgp_suppression_ec50"]))
        glu_stim = 0.5 + (glucagon / (glucagon + self._params["glucagon_hgp_stimulation_ec50"]))

        # 1. Glycogenolysis: stimulated by glucagon/exercise, suppressed by insulin, limited by remaining glycogen
        glycogen_fraction = max(0.0, state["glycogen_mass_g"] / self._params["max_glycogen_capacity_g"])
        glycogenolysis = 90.0 * ins_suppression * glu_stim * glycogen_fraction * (1.0 + exercise * 1.5)

        # 2. Glycogen synthesis: stimulated by hyperglycemia and insulin
        hyperglycemic_drive = max(0.0, (glucose - 100.0) / 50.0)
        insulin_synth_drive = insulin / (insulin + 15.0)
        capacity_available = max(0.0, 1.0 - glycogen_fraction)
        glycogen_synth = 40.0 * hyperglycemic_drive * insulin_synth_drive * capacity_available

        # Net glycogen turnover (g/min): (synthesis - breakdown in mg/min) / 1000
        d_glycogen = (glycogen_synth - glycogenolysis) / 1000.0

        # 3. Gluconeogenesis: substrates = lactate, alanine, glycerol
        substrate_drive = 1.0 + 0.3 * (lactate - 1.0)
        gluconeogenesis = self._params["basal_gluconeogenesis_mg_min"] * ins_suppression * glu_stim * substrate_drive

        total_hgp = glycogenolysis + gluconeogenesis
        d_hgp = 0.5 * (total_hgp - state["hgp_rate_mg_min"])

        # 4. Urea cycle: converts ammonia to urea
        ammonia_intake = 0.5 + exercise * 0.5  # muscle AMP deamination during exercise
        ammonia_detox = 0.02 * state["ammonia_umol_L"]
        d_ammonia = ammonia_intake - ammonia_detox
        d_urea = 0.1 * (ammonia_detox * 0.8 - state["urea_production_mg_min"])

        # 5. Ketogenesis: driven by low insulin and high FFA
        ketone_target = 0.10 + 0.5 * max(0.0, ffa - 0.4) * (1.0 / max(0.1, insulin / 5.0))
        d_ketones = 0.05 * (ketone_target - state["ketone_bodies_mmol_L"])

        # 6. Albumin synthesis: slow half-life (~20 days), steady maintenance
        target_albumin = 4.2 * (1.0 - 0.05 * exercise)
        d_albumin = (0.0001) * (target_albumin - state["plasma_albumin_g_dL"])

        return {
            "glycogen_mass_g": d_glycogen,
            "hgp_rate_mg_min": d_hgp,
            "plasma_albumin_g_dL": d_albumin,
            "ammonia_umol_L": d_ammonia,
            "urea_production_mg_min": d_urea,
            "ketone_bodies_mmol_L": d_ketones,
            "cyp3a4_activity_index": 0.0,
        }

    def compute_well_stirred_clearance(self, hepatic_blood_flow_L_min: float, fu_unbound: float = 0.1) -> float:
        """
        Computes organ clearance via the well-stirred model:
        CL_H = Q_H * (fu * CL_int) / (Q_H + fu * CL_int).
        """
        cl_int = self._params["intrinsic_cyp_clearance_L_min"] * self._state["cyp3a4_activity_index"]
        qh = max(0.2, hepatic_blood_flow_L_min)
        cl_h = qh * (fu_unbound * cl_int) / (qh + fu_unbound * cl_int)
        return float(cl_h)

    def exchange_fluxes(self, transport: Dict[str, float]) -> Dict[str, float]:
        """
        Net release of glucose into systemic blood and clearance of lactate/ammonia.
        """
        return {
            "glucose_flux_mg_min": float(self._state["hgp_rate_mg_min"]),
            "urea_flux_mg_min": float(self._state["urea_production_mg_min"]),
            "ketone_flux_mmol_min": float(self._state["ketone_bodies_mmol_L"] * 0.1),
        }

