"""
Clinical Endpoints for Study BDT-001
Primary and secondary endpoint calculations and analysis
"""
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List, Tuple
from pydantic import BaseModel, Field
import logging
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PrimaryEndpointResult(BaseModel):
    """Primary endpoint (TIR) calculation result"""
    patient_id: str
    intervention: str  # A or B
    tir_percent: float
    time_below_range_percent: float
    time_above_range_percent: float
    total_readings: int
    days_analyzed: int
    target_range: Tuple[float, float] = (70, 180)
    target_met: bool = False


class SecondaryEndpointResult(BaseModel):
    """Secondary endpoint calculation result"""
    patient_id: str
    intervention: str
    endpoint_name: str
    value: float
    unit: str
    baseline_value: Optional[float] = None
    change_from_baseline: Optional[float] = None
    percent_change: Optional[float] = None


class ClinicalEndpoints:
    """
    Clinical endpoint calculations for study BDT-001
    Primary: Time in Range (TIR)
    Secondary: HbA1c, glucose variability, QoL, etc.
    """
    
    def __init__(self):
        self.tir_target_range = (70, 180)  # mg/dL
        self.tir_target_percent = 70.0  # Target: 70% time in range
        self.hypoglycemia_threshold = 70  # mg/dL
        self.severe_hypoglycemia_threshold = 54  # mg/dL
    
    def calculate_tir(
        self,
        glucose_values: List[float],
        timestamps: List[datetime],
        target_range: Optional[Tuple[float, float]] = None
    ) -> PrimaryEndpointResult:
        """
        Calculate Time in Range (TIR) - primary endpoint
        TIR = percentage of glucose readings within target range (70-180 mg/dL)
        """
        if target_range is None:
            target_range = self.tir_target_range
        
        if not glucose_values:
            raise ValueError("No glucose values provided")
        
        glucose_array = np.array(glucose_values)
        min_glucose, max_glucose = target_range
        
        # Calculate TIR
        in_range = np.sum((glucose_array >= min_glucose) & (glucose_array <= max_glucose))
        below_range = np.sum(glucose_array < min_glucose)
        above_range = np.sum(glucose_array > max_glucose)
        total = len(glucose_array)
        
        tir_percent = (in_range / total) * 100
        below_range_percent = (below_range / total) * 100
        above_range_percent = (above_range / total) * 100
        
        # Calculate days analyzed
        if timestamps:
            time_span = (max(timestamps) - min(timestamps)).days
        else:
            time_span = 0
        
        return PrimaryEndpointResult(
            patient_id="",  # To be filled by caller
            intervention="",  # To be filled by caller
            tir_percent=round(tir_percent, 2),
            time_below_range_percent=round(below_range_percent, 2),
            time_above_range_percent=round(above_range_percent, 2),
            total_readings=total,
            days_analyzed=time_span,
            target_range=target_range,
            target_met=tir_percent >= self.tir_target_percent
        )
    
    def calculate_glucose_variability(
        self,
        glucose_values: List[float]
    ) -> Dict[str, float]:
        """
        Calculate glucose variability metrics
        Includes CV, SD, and other variability measures
        """
        if not glucose_values:
            return {}
        
        glucose_array = np.array(glucose_values)
        
        mean = np.mean(glucose_array)
        std = np.std(glucose_array)
        cv = (std / mean) * 100 if mean > 0 else 0
        
        # Calculate interquartile range
        q25, q75 = np.percentile(glucose_array, [25, 75])
        iqr = q75 - q25
        
        # Calculate MAGE (Mean Amplitude of Glycemic Excursions)
        # Simplified version
        if len(glucose_array) > 1:
            differences = np.abs(np.diff(glucose_array))
            mage = np.mean(differences)
        else:
            mage = 0
        
        return {
            "mean": round(mean, 2),
            "std": round(std, 2),
            "cv_percent": round(cv, 2),
            "min": round(np.min(glucose_array), 2),
            "max": round(np.max(glucose_array), 2),
            "iqr": round(iqr, 2),
            "mage": round(mage, 2)
        }
    
    def calculate_hba1c_reduction(
        self,
        baseline_hba1c: float,
        followup_hba1c: float
    ) -> SecondaryEndpointResult:
        """
        Calculate HbA1c reduction
        """
        reduction = baseline_hba1c - followup_hba1c
        percent_reduction = (reduction / baseline_hba1c) * 100 if baseline_hba1c > 0 else 0
        
        return SecondaryEndpointResult(
            patient_id="",
            intervention="",
            endpoint_name="HbA1c reduction",
            value=round(reduction, 2),
            unit="% (absolute)",
            baseline_value=round(baseline_hba1c, 2),
            change_from_baseline=round(reduction, 2),
            percent_change=round(percent_reduction, 2)
        )
    
    def calculate_ukf_rmse(
        self,
        predicted_glucose: List[float],
        actual_glucose: List[float]
    ) -> float:
        """
        Calculate UKF estimation accuracy (RMSE)
        Secondary endpoint for twin model performance
        """
        if len(predicted_glucose) != len(actual_glucose):
            raise ValueError("Predicted and actual glucose arrays must have same length")
        
        if not predicted_glucose:
            return 0.0
        
        predicted = np.array(predicted_glucose)
        actual = np.array(actual_glucose)
        
        rmse = np.sqrt(np.mean((predicted - actual) ** 2))
        return round(rmse, 2)
    
    def analyze_crossover_results(
        self,
        results_arm_a: List[PrimaryEndpointResult],
        results_arm_b: List[PrimaryEndpointResult]
    ) -> Dict[str, Any]:
        """
        Analyze crossover study results
        Performs paired statistical tests
        """
        # Extract TIR values by patient
        tir_a = [r.tir_percent for r in results_arm_a]
        tir_b = [r.tir_percent for r in results_arm_b]
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(tir_a, tir_b)
        
        # Mean difference
        mean_diff = np.mean(np.array(tir_b) - np.array(tir_a))
        
        # Effect size (Cohen's d for paired samples)
        diff_array = np.array(tir_b) - np.array(tir_a)
        std_diff = np.std(diff_array)
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0
        
        # Confidence interval
        n = len(tir_a)
        se = std_diff / np.sqrt(n)
        ci_lower = mean_diff - 1.96 * se
        ci_upper = mean_diff + 1.96 * se
        
        # Proportion meeting target
        target_met_a = sum(1 for r in results_arm_a if r.target_met) / len(results_arm_a) * 100
        target_met_b = sum(1 for r in results_arm_b if r.target_met) / len(results_arm_b) * 100
        
        return {
            "sample_size": n,
            "mean_tir_arm_a": round(np.mean(tir_a), 2),
            "mean_tir_arm_b": round(np.mean(tir_b), 2),
            "mean_difference": round(mean_diff, 2),
            "std_difference": round(std_diff, 2),
            "paired_t_statistic": round(t_stat, 4),
            "paired_p_value": round(p_value, 4),
            "significant": p_value < 0.05,
            "cohens_d": round(cohens_d, 3),
            "confidence_interval": {
                "lower": round(ci_lower, 2),
                "upper": round(ci_upper, 2)
            },
            "target_met_arm_a_percent": round(target_met_a, 2),
            "target_met_arm_b_percent": round(target_met_b, 2),
            "interpretation": self._interpret_results(p_value, mean_diff, cohens_d)
        }
    
    def _interpret_results(self, p_value: float, mean_diff: float, cohens_d: float) -> str:
        """Interpret statistical results"""
        if p_value >= 0.05:
            return "No statistically significant difference between interventions"
        
        if mean_diff > 0:
            direction = "Intervention B superior to Intervention A"
        else:
            direction = "Intervention A superior to Intervention B"
        
        if abs(cohens_d) < 0.2:
            magnitude = "small effect size"
        elif abs(cohens_d) < 0.5:
            magnitude = "medium effect size"
        elif abs(cohens_d) < 0.8:
            magnitude = "large effect size"
        else:
            magnitude = "very large effect size"
        
        return f"{direction} with {magnitude} (p={p_value:.4f}, d={cohens_d:.3f})"
    
    def calculate_quality_of_life_score(
        self,
        dqoly_scores: List[int]
    ) -> Dict[str, float]:
        """
        Calculate Diabetes Quality of Life (DQOLY) score
        Secondary endpoint for patient-reported outcomes
        """
        if not dqoly_scores:
            return {}
        
        scores_array = np.array(dqoly_scores)
        
        return {
            "mean": round(np.mean(scores_array), 2),
            "std": round(np.std(scores_array), 2),
            "min": int(np.min(scores_array)),
            "max": int(np.max(scores_array)),
            "median": round(np.median(scores_array), 2)
        }
    
    def calculate_treatment_satisfaction(
        self,
        satisfaction_scores: List[int]
    ) -> Dict[str, float]:
        """
        Calculate treatment satisfaction score
        Secondary endpoint for patient satisfaction
        """
        if not satisfaction_scores:
            return {}
        
        scores_array = np.array(satisfaction_scores)
        
        return {
            "mean": round(np.mean(scores_array), 2),
            "std": round(np.std(scores_array), 2),
            "satisfied_percent": round(np.sum(scores_array >= 4) / len(scores_array) * 100, 2)  # Assuming 1-5 scale
        }


# Singleton instance
_clinical_endpoints = None

def get_clinical_endpoints() -> ClinicalEndpoints:
    """Get singleton clinical endpoints instance"""
    global _clinical_endpoints
    if _clinical_endpoints is None:
        _clinical_endpoints = ClinicalEndpoints()
    return _clinical_endpoints
