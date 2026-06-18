"""
Clinical Study BDT-001 Design
50-200 patient crossover study for T1DM Time in Range (TIR) endpoint
"""
import json
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StudyPhase(str, Enum):
    """Study phases for crossover design"""
    SCREENING = "screening"
    BASELINE = "baseline"
    INTERVENTION_A = "intervention_a"
    WASHOUT = "washout"
    INTERVENTION_B = "intervention_b"
    FOLLOWUP = "followup"
    COMPLETED = "completed"


class RandomizationArm(str, Enum):
    """Randomization arms for crossover study"""
    AB = "AB"  # Intervention A first, then B
    BA = "BA"  # Intervention B first, then A


class PatientStatus(str, Enum):
    """Patient enrollment status"""
    SCREENING = "screening"
    ENROLLED = "enrolled"
    ACTIVE = "active"
    COMPLETED = "completed"
    WITHDRAWN = "withdrawn"
    LOST_TO_FOLLOWUP = "lost_to_followup"


class StudyDesign(BaseModel):
    """Clinical study design parameters"""
    study_id: str = "BDT-001"
    study_title: str = "Bio-Digital Twin for Type 1 Diabetes Management"
    study_type: str = "crossover"
    target_enrollment: int = Field(default=200, ge=50, le=200)
    min_enrollment: int = 50
    
    # Study duration
    screening_duration_weeks: int = 4
    baseline_duration_weeks: int = 2
    intervention_duration_weeks: int = 12
    washout_duration_weeks: int = 4
    followup_duration_weeks: int = 4
    
    # Inclusion criteria
    age_min: int = 18
    age_max: int = 65
    t1dm_duration_min_years: int = 1
    hba1c_min: float = 7.0
    hba1c_max: float = 10.0
    
    # Exclusion criteria
    exclude_pregnant: bool = True
    exclude_severe_hypoglycemia: bool = True
    exclude_diabetic_ketoacidosis: bool = True
    exclude_other_chronic_conditions: bool = True
    
    # Primary endpoint
    primary_endpoint: str = "Time in Range (TIR)"
    tir_target_range: tuple = (70, 180)  # mg/dL
    tir_target_percent: float = 70.0  # Target: 70% time in range
    
    # Secondary endpoints
    secondary_endpoints: List[str] = [
        "HbA1c reduction",
        "Glucose variability (CV)",
        "Time below range (<70 mg/dL)",
        "Time above range (>180 mg/dL)",
        "Quality of life (DQOLY)",
        "Treatment satisfaction",
        "UKF estimation accuracy (RMSE)"
    ]
    
    # Interventions
    intervention_a: str = "Standard of care (SMBG + basal-bolus insulin)"
    intervention_b: str = "Bio-Digital Twin + CGM + predictive insulin dosing"
    
    # Statistical parameters
    alpha: float = 0.05
    power: float = 0.80
    effect_size: float = 0.5  # Cohen's d
    
    # Safety monitoring
    dsmb_required: bool = True  # Data Safety Monitoring Board
    adverse_event_reporting: bool = True
    stopping_rules: List[str] = Field(default_factory=lambda: [
        "Severe hypoglycemia rate > 5%",
        "HbA1c increase > 1.0%",
        "Serious adverse event related to intervention"
    ])


class ClinicalStudyBDT001:
    """
    Clinical Study BDT-001 implementation
    Manages study design, enrollment, and tracking
    """
    
    def __init__(self, design: StudyDesign = None):
        self.design = design or StudyDesign()
        self.enrolled_patients: Dict[str, Dict[str, Any]] = {}
        self.study_start_date: Optional[date] = None
        self.study_end_date: Optional[date] = None
        
    def initialize_study(self, start_date: date = None) -> Dict[str, Any]:
        """Initialize the clinical study"""
        if start_date is None:
            start_date = date.today()
        
        self.study_start_date = start_date
        
        # Calculate study end date
        total_weeks = (
            self.design.screening_duration_weeks +
            self.design.baseline_duration_weeks +
            (2 * self.design.intervention_duration_weeks) +
            self.design.washout_duration_weeks +
            self.design.followup_duration_weeks
        )
        
        self.study_end_date = start_date + timedelta(weeks=total_weeks)
        
        return {
            "study_id": self.design.study_id,
            "start_date": self.study_start_date.isoformat(),
            "end_date": self.study_end_date.isoformat(),
            "duration_weeks": total_weeks,
            "target_enrollment": self.design.target_enrollment,
            "primary_endpoint": self.design.primary_endpoint,
            "status": "initialized"
        }
    
    def calculate_sample_size(self) -> Dict[str, Any]:
        """
        Calculate required sample size for crossover study
        Using paired t-test formula for crossover design
        """
        from scipy import stats
        
        alpha = self.design.alpha
        power = self.design.power
        effect_size = self.design.effect_size
        
        # Z-scores for alpha and power
        z_alpha = stats.norm.ppf(1 - alpha/2)  # Two-tailed
        z_beta = stats.norm.ppf(power)
        
        # Sample size formula for crossover design
        # n = 2 * (z_alpha + z_beta)^2 / effect_size^2
        n_per_group = 2 * (z_alpha + z_beta)**2 / effect_size**2
        
        # Add 20% for attrition
        n_with_attrition = n_per_group * 1.2
        
        return {
            "effect_size": effect_size,
            "alpha": alpha,
            "power": power,
            "z_alpha": z_alpha,
            "z_beta": z_beta,
            "n_per_group": round(n_per_group),
            "n_total": round(n_per_group * 2),
            "n_with_attrition": round(n_with_attrition * 2),
            "target_enrollment": self.design.target_enrollment,
            "adequate": round(n_with_attrition * 2) <= self.design.target_enrollment
        }
    
    def get_study_timeline(self, patient_id: str = None) -> Dict[str, Any]:
        """
        Get study timeline for a patient or overall study
        """
        if not self.study_start_date:
            return {"error": "Study not initialized"}
        
        timeline = {
            "screening": {
                "start": self.study_start_date.isoformat(),
                "end": (self.study_start_date + timedelta(weeks=self.design.screening_duration_weeks)).isoformat(),
                "duration_weeks": self.design.screening_duration_weeks
            },
            "baseline": {
                "start": (self.study_start_date + timedelta(weeks=self.design.screening_duration_weeks)).isoformat(),
                "end": (self.study_start_date + timedelta(weeks=self.design.screening_duration_weeks + self.design.baseline_duration_weeks)).isoformat(),
                "duration_weeks": self.design.baseline_duration_weeks
            },
            "intervention_a": {
                "start": (self.study_start_date + timedelta(weeks=self.design.screening_duration_weeks + self.design.baseline_duration_weeks)).isoformat(),
                "end": (self.study_start_date + timedelta(weeks=self.design.screening_duration_weeks + self.design.baseline_duration_weeks + self.design.intervention_duration_weeks)).isoformat(),
                "duration_weeks": self.design.intervention_duration_weeks
            },
            "washout": {
                "start": (self.study_start_date + timedelta(weeks=self.design.screening_duration_weeks + self.design.baseline_duration_weeks + self.design.intervention_duration_weeks)).isoformat(),
                "end": (self.study_start_date + timedelta(weeks=self.design.screening_duration_weeks + self.design.baseline_duration_weeks + self.design.intervention_duration_weeks + self.design.washout_duration_weeks)).isoformat(),
                "duration_weeks": self.design.washout_duration_weeks
            },
            "intervention_b": {
                "start": (self.study_start_date + timedelta(weeks=self.design.screening_duration_weeks + self.design.baseline_duration_weeks + self.design.intervention_duration_weeks + self.design.washout_duration_weeks)).isoformat(),
                "end": (self.study_start_date + timedelta(weeks=self.design.screening_duration_weeks + self.design.baseline_duration_weeks + 2*self.design.intervention_duration_weeks + self.design.washout_duration_weeks)).isoformat(),
                "duration_weeks": self.design.intervention_duration_weeks
            },
            "followup": {
                "start": (self.study_start_date + timedelta(weeks=self.design.screening_duration_weeks + self.design.baseline_duration_weeks + 2*self.design.intervention_duration_weeks + self.design.washout_duration_weeks)).isoformat(),
                "end": self.study_end_date.isoformat(),
                "duration_weeks": self.design.followup_duration_weeks
            }
        }
        
        if patient_id and patient_id in self.enrolled_patients:
            patient_data = self.enrolled_patients[patient_id]
            timeline["patient_enrollment_date"] = patient_data.get("enrollment_date")
            timeline["patient_randomization_arm"] = patient_data.get("randomization_arm")
            timeline["patient_current_phase"] = patient_data.get("current_phase")
        
        return timeline
    
    def get_inclusion_criteria(self) -> List[str]:
        """Get study inclusion criteria"""
        return [
            f"Age {self.design.age_min}-{self.design.age_max} years",
            f"Type 1 diabetes duration ≥ {self.design.t1dm_duration_min_years} years",
            f"HbA1c {self.design.hba1c_min}-{self.design.hba1c_max}%",
            "On continuous subcutaneous insulin infusion (CSII) or multiple daily injections (MDI)",
            "Willing to wear CGM device for study duration",
            "Access to smartphone for data collection",
            "Able to provide informed consent"
        ]
    
    def get_exclusion_criteria(self) -> List[str]:
        """Get study exclusion criteria"""
        criteria = []
        
        if self.design.exclude_pregnant:
            criteria.append("Pregnancy or lactation")
        if self.design.exclude_severe_hypoglycemia:
            criteria.append("History of severe hypoglycemia in past 6 months")
        if self.design.exclude_diabetic_ketoacidosis:
            criteria.append("History of diabetic ketoacidosis in past 12 months")
        if self.design.exclude_other_chronic_conditions:
            criteria.append("Other significant chronic conditions (e.g., renal failure, liver disease)")
        
        criteria.extend([
            "Use of other investigational devices",
            "Inability to comply with study procedures",
            "Cognitive impairment affecting consent"
        ])
        
        return criteria
    
    def get_study_summary(self) -> Dict[str, Any]:
        """Get comprehensive study summary"""
        return {
            "study_id": self.design.study_id,
            "study_title": self.design.study_title,
            "study_type": self.design.study_type,
            "target_enrollment": self.design.target_enrollment,
            "current_enrollment": len(self.enrolled_patients),
            "enrollment_progress": f"{len(self.enrolled_patients)}/{self.design.target_enrollment}",
            "primary_endpoint": self.design.primary_endpoint,
            "secondary_endpoints": self.design.secondary_endpoints,
            "intervention_a": self.design.intervention_a,
            "intervention_b": self.design.intervention_b,
            "start_date": self.study_start_date.isoformat() if self.study_start_date else None,
            "end_date": self.study_end_date.isoformat() if self.study_end_date else None,
            "status": "active" if self.study_start_date else "not_initialized",
            "sample_size_calculation": self.calculate_sample_size()
        }


# Singleton instance
_clinical_study = None

def get_clinical_study() -> ClinicalStudyBDT001:
    """Get singleton clinical study instance"""
    global _clinical_study
    if _clinical_study is None:
        _clinical_study = ClinicalStudyBDT001()
    return _clinical_study
