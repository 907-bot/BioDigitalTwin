"""
backend/app/dynamics/respiratory_acid_base.py
Biophysically Validated Respiratory Gas Exchange, Bohr-Shifted Hemoglobin,
and Henderson-Hasselbalch Acid-Base Equilibrium Engine.

Governing Equations & Physical Units:
1. Henderson-Hasselbalch Acid-Base Equation:
   pH = pKa + log10([HCO3-] / (alpha * PaCO2))
   where pKa = 6.10, alpha = 0.0307 mmol/(L*mmHg) [plasma CO2 solubility at 37°C]
   [HCO3-] in mEq/L (= mmol/L), PaCO2 in mmHg.
2. Severinghaus-Bohr Hemoglobin-Oxygen Dissociation Equation:
   P50(pH, PaCO2) = 26.8 * 10^(-0.48 * (pH - 7.40) + 0.06 * log10(PaCO2 / 40.0))
   BOHR EFFECT VERIFICATION: Lower pH (acidosis) yields higher P50 (rightward shift),
   promoting O2 dissociation and tissue delivery.
3. Alveolar Gas Equation (BTPS-corrected):
   PAO2 = FiO2 * (P_atm - P_H2O) - (PaCO2 / RER)
   P_atm = 760 mmHg, P_H2O = 47 mmHg, FiO2 = 0.209, RER = 0.80
4. Carbon Dioxide Mass Balance:
   d(PaCO2)/dt = (0.863 * VCO2 - V_A * PaCO2) / V_dist_CO2
   VCO2 in mL/min, V_A in L/min, 0.863 mmHg*L/mL BTPS conversion factor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class RespiratoryAcidBaseState:
    PaO2_mmHg: float = 95.0        # Arterial dissolved oxygen partial pressure (mmHg)
    PaCO2_mmHg: float = 40.0       # Arterial carbon dioxide partial pressure (mmHg)
    HCO3_mEq_L: float = 24.0       # Plasma bicarbonate concentration (mEq/L)
    pH: float = 7.40               # Arterial blood pH
    SaO2_pct: float = 97.5         # Hemoglobin arterial oxygen saturation (%)
    P50_mmHg: float = 26.8         # Current oxygen affinity P50 (mmHg)
    V_A_L_min: float = 4.2         # Alveolar ventilation (L/min)
    PAO2_mmHg: float = 99.0        # Alveolar oxygen partial pressure (mmHg)


class RespiratoryAcidBaseEngine:
    """
    Simulates respiratory ventilation, pulmonary gas exchange,
    arterial blood gas (ABG) parameters, and systemic acid-base balance.
    """
    def __init__(self, initial_state: Optional[RespiratoryAcidBaseState] = None):
        self.state = initial_state or RespiratoryAcidBaseState()
        # Physical constants
        self.pKa: float = 6.10
        self.alpha_CO2: float = 0.0307     # mmol / (L * mmHg)
        self.P_atm: float = 760.0          # mmHg at sea level
        self.P_H2O: float = 47.0           # mmHg vapor pressure at 37°C
        self.RER: float = 0.80             # Respiratory Exchange Ratio (VCO2 / VO2)
        self.V_dist_CO2_L: float = 15.0    # CO2 distribution volume (L)
        self.BTPS_factor: float = 0.863    # mmHg * L / mL
        self.hill_n: float = 2.7           # Hill cooperativity coefficient
        self.P50_standard: float = 26.8    # Standard P50 at pH 7.40, PaCO2 40 mmHg

    def compute_pH(self, PaCO2_mmHg: float, HCO3_mEq_L: float) -> float:
        """
        Henderson-Hasselbalch equation:
        pH = 6.10 + log10([HCO3-] / (0.0307 * PaCO2))
        """
        PaCO2 = max(10.0, float(PaCO2_mmHg))
        HCO3 = max(2.0, float(HCO3_mEq_L))
        dissolved_CO2 = self.alpha_CO2 * PaCO2
        ratio = HCO3 / max(1e-6, dissolved_CO2)
        return float(self.pKa + np.log10(ratio))

    def compute_P50(self, pH: float, PaCO2_mmHg: float) -> float:
        """
        Calculates P50 with physiologically verified Bohr effect:
        - Acidosis (pH < 7.40) -> exponent is positive -> P50 increases (rightward shift).
        - Alkalosis (pH > 7.40) -> exponent is negative -> P50 decreases (leftward shift).
        """
        PaCO2 = max(10.0, float(PaCO2_mmHg))
        pH_delta = pH - 7.40
        log_co2_ratio = np.log10(PaCO2 / 40.0)
        # Exponent: -0.48 * (pH - 7.40) + 0.06 * log10(PaCO2 / 40.0)
        exponent = -0.48 * pH_delta + 0.06 * log_co2_ratio
        return float(self.P50_standard * (10.0 ** exponent))

    def compute_SaO2(self, PaO2_mmHg: float, P50_mmHg: float) -> float:
        """
        Calculates fractional oxygen saturation (%) using Hill equation:
        SaO2 = 100 * (PaO2^n) / (P50^n + PaO2^n)
        """
        PaO2 = max(0.0, float(PaO2_mmHg))
        P50 = max(10.0, float(P50_mmHg))
        p_ratio = (PaO2 / P50) ** self.hill_n
        sat_fraction = p_ratio / (1.0 + p_ratio)
        return float(sat_fraction * 100.0)

    def compute_PAO2(self, PaCO2_mmHg: float, FiO2: float = 0.209) -> float:
        """
        Alveolar gas equation:
        PAO2 = FiO2 * (P_atm - P_H2O) - (PaCO2 / RER)
        """
        PaCO2 = max(10.0, float(PaCO2_mmHg))
        P_inspired = FiO2 * (self.P_atm - self.P_H2O)
        return float(P_inspired - (PaCO2 / self.RER))

    def step(
        self,
        dt_min: float = 1.0,
        alveolar_ventilation_L_min: Optional[float] = None,
        VCO2_prod_mL_min: float = 200.0,
        metabolic_HCO3_delta_mEq: float = 0.0,
        FiO2: float = 0.209,
        Aa_gradient_mmHg: float = 7.0,
    ) -> RespiratoryAcidBaseState:
        """
        Integrates ventilation, gas exchange, and acid-base status forward by dt_min.
        """
        current_pH = self.compute_pH(self.state.PaCO2_mmHg, self.state.HCO3_mEq_L)

        # Chemoreceptor reflex: ventilation increases under hypercapnia or acidosis
        if alveolar_ventilation_L_min is None:
            # Baseline V_A ~ 4.2 L/min
            co2_drive = max(0.0, (self.state.PaCO2_mmHg - 40.0) * 0.25)
            h_drive = max(0.0, (7.40 - current_pH) * 15.0)
            V_A = float(np.clip(4.2 + co2_drive + h_drive, 1.5, 30.0))
        else:
            V_A = float(max(1.0, alveolar_ventilation_L_min))

        # 1. Update PaCO2 via mass balance
        # dPaCO2/dt = (0.863 * VCO2 - V_A * PaCO2) / V_dist_CO2
        dPaCO2 = (self.BTPS_factor * VCO2_prod_mL_min - V_A * self.state.PaCO2_mmHg) / self.V_dist_CO2_L
        new_PaCO2 = float(np.clip(self.state.PaCO2_mmHg + dPaCO2 * dt_min, 15.0, 90.0))

        # 2. Update HCO3- (slow metabolic renal handling or acute buffer change)
        new_HCO3 = float(np.clip(self.state.HCO3_mEq_L + metabolic_HCO3_delta_mEq * dt_min, 8.0, 42.0))

        # 3. New pH
        new_pH = self.compute_pH(new_PaCO2, new_HCO3)

        # 4. Alveolar PAO2 and Arterial PaO2
        new_PAO2 = self.compute_PAO2(new_PaCO2, FiO2=FiO2)
        new_PaO2 = float(np.clip(new_PAO2 - Aa_gradient_mmHg, 30.0, 130.0))

        # 5. P50 with Bohr shift
        new_P50 = self.compute_P50(new_pH, new_PaCO2)

        # 6. SaO2
        new_SaO2 = self.compute_SaO2(new_PaO2, new_P50)

        self.state = RespiratoryAcidBaseState(
            PaO2_mmHg=new_PaO2,
            PaCO2_mmHg=new_PaCO2,
            HCO3_mEq_L=new_HCO3,
            pH=new_pH,
            SaO2_pct=new_SaO2,
            P50_mmHg=new_P50,
            V_A_L_min=V_A,
            PAO2_mmHg=new_PAO2,
        )
        return self.state

