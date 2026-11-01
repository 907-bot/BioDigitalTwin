"""
Cellular & Sub-Organ Level Simulator Module.

Models sub-organ cellular dynamics:
1. Cytokine Signaling Cascade (M1/M2 macrophages, TNF-α, IL-1β, IL-6, IL-10)
2. Hepatocyte Metabolism & Toxicity (CYP2E1, NAPQI, Glutathione GSH, ALT/AST)
3. Hypothalamic Thermoregulation (Pyrogens → PGE2 → T_set → T_core dynamics)
"""

import numpy as np
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class CellularSimulationRequest(BaseModel):
    simulation_type: str = Field("fever_thermoregulation", pattern="^(fever_thermoregulation|hepatocyte_toxicity|cytokine_cascade)$")
    pyrogen_load: float = Field(0.8, ge=0.0, le=3.0, description="Pyrogen/infection intensity (0=none, 1=moderate, 3=severe)")
    drug_dose_mg: float = Field(1000.0, ge=0.0, le=4000.0)
    drug_name: str = Field("paracetamol", pattern="^(paracetamol|ibuprofen|none)$")
    duration_hours: int = Field(24, ge=4, le=72)


def simulate_cellular_suborgan(
    simulation_type: str = "fever_thermoregulation",
    pyrogen_load: float = 0.8,
    drug_dose_mg: float = 1000.0,
    drug_name: str = "paracetamol",
    duration_hours: int = 24,
) -> Dict[str, Any]:
    """
    Simulate cellular level cytokine cascades, hepatocyte toxicity, and hypothalamic thermoregulation.
    """
    hours = list(range(1, duration_hours + 1))
    trajectory = []

    # State variables
    tnf_alpha = 0.1
    il_1beta = 0.1
    il_6 = 0.2
    il_10 = 0.5  # anti-inflammatory
    m1_ratio = 0.2  # M1 polarization (0-1)

    pge2 = 0.05
    t_set = 37.0  # Hypothalamic set point (°C)
    t_core = 37.0 # Core body temp (°C)

    gsh_percent = 100.0
    napqi_umol = 0.0
    alt_ast_u_l = 25.0
    cell_viability_pct = 100.0

    # COX inhibition factor from drug
    cox_inhibition = 0.0
    if drug_name == "ibuprofen" and drug_dose_mg > 0:
        cox_inhibition = min(0.95, (drug_dose_mg / 400.0) * 0.8)
    elif drug_name == "paracetamol" and drug_dose_mg > 0:
        cox_inhibition = min(0.90, (drug_dose_mg / 1000.0) * 0.75)

    for h in hours:
        # 1. Cytokine cascade
        if pyrogen_load > 0 and h <= 12:
            m1_ratio = min(0.95, m1_ratio + 0.08 * pyrogen_load)
            tnf_alpha = min(10.0, tnf_alpha + 0.8 * pyrogen_load * m1_ratio)
            il_1beta = min(10.0, il_1beta + 0.6 * pyrogen_load * m1_ratio)
            il_6 = min(10.0, il_6 + 0.7 * tnf_alpha - 0.2 * il_10)
            il_10 = min(5.0, il_10 + 0.1 * il_6)
        else:
            m1_ratio = max(0.2, m1_ratio - 0.03)
            tnf_alpha = max(0.1, tnf_alpha - 0.3)
            il_1beta = max(0.1, il_1beta - 0.25)
            il_6 = max(0.2, il_6 - 0.3)
            il_10 = max(0.5, il_10 - 0.05)

        # 2. Hypothalamic PGE2 & set point
        pyrogen_signal = 0.4 * tnf_alpha + 0.4 * il_1beta + 0.2 * il_6
        uninhibited_pge2 = 0.05 + 0.5 * pyrogen_signal
        pge2 = max(0.05, uninhibited_pge2 * (1.0 - cox_inhibition))
        
        # Hypothalamic thermostat set point
        t_set = 37.0 + 2.5 * (pge2 / (pge2 + 0.5))
        
        # Core body temperature lags set point
        d_temp = 0.35 * (t_set - t_core)
        t_core += d_temp

        # 3. Hepatocyte metabolism (CYP2E1 & NAPQI under paracetamol)
        if drug_name == "paracetamol" and drug_dose_mg > 0:
            napqi_rate = (drug_dose_mg / 1000.0) * 2.5
            napqi_umol = max(0.0, napqi_rate - (gsh_percent / 100.0) * 1.8)
            gsh_depletion = (drug_dose_mg / 1000.0) * 3.5
            gsh_percent = max(5.0, gsh_percent - gsh_depletion + 0.8) # regeneration
            
            if gsh_percent < 30.0:
                cell_viability_pct = max(10.0, cell_viability_pct - 2.5)
                alt_ast_u_l += 35.0
            else:
                alt_ast_u_l = max(25.0, alt_ast_u_l - 0.5)
        else:
            gsh_percent = min(100.0, gsh_percent + 1.0)
            napqi_umol = max(0.0, napqi_umol - 0.5)
            alt_ast_u_l = max(25.0, alt_ast_u_l - 0.5)

        trajectory.append({
            "hour": h,
            "core_temperature_C": round(float(t_core), 2),
            "hypothalamic_setpoint_C": round(float(t_set), 2),
            "pge2_level": round(float(pge2), 3),
            "tnf_alpha": round(float(tnf_alpha), 2),
            "il_1beta": round(float(il_1beta), 2),
            "il_6": round(float(il_6), 2),
            "il_10": round(float(il_10), 2),
            "m1_macrophage_ratio": round(float(m1_ratio), 2),
            "glutathione_gsh_pct": round(float(gsh_percent), 1),
            "napqi_umol": round(float(napqi_umol), 2),
            "hepatocyte_viability_pct": round(float(cell_viability_pct), 1),
            "alt_ast_u_l": round(float(alt_ast_u_l), 1),
        })

    return {
        "simulation_type": simulation_type,
        "pyrogen_load": pyrogen_load,
        "drug_name": drug_name,
        "drug_dose_mg": drug_dose_mg,
        "cox_inhibition_pct": round(cox_inhibition * 100.0, 1),
        "summary": {
            "peak_core_temp_C": round(max(t["core_temperature_C"] for t in trajectory), 2),
            "final_core_temp_C": round(trajectory[-1]["core_temperature_C"], 2),
            "peak_il6": round(max(t["il_6"] for t in trajectory), 2),
            "min_gsh_pct": round(min(t["glutathione_gsh_pct"] for t in trajectory), 1),
            "final_hepatocyte_viability_pct": round(trajectory[-1]["hepatocyte_viability_pct"], 1),
            "peak_alt_ast_u_l": round(max(t["alt_ast_u_l"] for t in trajectory), 1),
        },
        "trajectory": trajectory,
    }
