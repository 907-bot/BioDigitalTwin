"""
Patient Enrollment for Clinical Study BDT-001
Manages patient screening, enrollment, and randomization
"""
import json
import random
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from enum import Enum
import logging

from .study_design import PatientStatus, RandomizationArm, StudyPhase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScreeningData(BaseModel):
    """Patient screening data"""
    patient_id: str
    age: int
    gender: str
    t1dm_duration_years: float
    hba1c: float
    insulin_delivery_method: str  # CSII or MDI
    cgm_experience: bool
    smartphone_access: bool
    severe_hypoglycemia_history: bool
    dka_history: bool
    pregnancy_status: str
    chronic_conditions: List[str] = Field(default_factory=list)
    consent_signed: bool = False
    screening_date: date = Field(default_factory=date.today)


class EnrolledPatient(BaseModel):
    """Enrolled patient in clinical study"""
    patient_id: str
    enrollment_date: date
    randomization_arm: RandomizationArm
    current_phase: StudyPhase = StudyPhase.SCREENING
    phase_start_date: Optional[date] = None
    status: PatientStatus = PatientStatus.SCREENING
    
    # Screening data
    screening_data: Optional[ScreeningData] = None
    
    # Baseline data
    baseline_hba1c: Optional[float] = None
    baseline_tir: Optional[float] = None
    baseline_weight: Optional[float] = None
    
    # Intervention data
    intervention_a_start_date: Optional[date] = None
    intervention_a_end_date: Optional[date] = None
    intervention_b_start_date: Optional[date] = None
    intervention_b_end_date: Optional[date] = None
    
    # Outcomes
    primary_endpoint_value: Optional[float] = None
    secondary_endpoints: Dict[str, float] = Field(default_factory=dict)
    
    # Adverse events
    adverse_events: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Withdrawal
    withdrawal_date: Optional[date] = None
    withdrawal_reason: Optional[str] = None


class PatientEnrollment:
    """
    Patient enrollment management for clinical study
    Handles screening, eligibility, randomization, and tracking
    """
    
    def __init__(self):
        self.screened_patients: Dict[str, ScreeningData] = {}
        self.enrolled_patients: Dict[str, EnrolledPatient] = {}
        self.randomization_sequence: List[RandomizationArm] = []
        self._generate_randomization_sequence()
    
    def _generate_randomization_sequence(self, block_size: int = 4):
        """
        Generate randomization sequence using block randomization
        Ensures balanced allocation across arms
        """
        sequence = []
        for i in range(0, 200, block_size):  # Up to 200 patients
            block = [RandomizationArm.AB, RandomizationArm.BA] * (block_size // 2)
            random.shuffle(block)
            sequence.extend(block)
        self.randomization_sequence = sequence
    
    def screen_patient(self, screening_data: ScreeningData) -> Dict[str, Any]:
        """
        Screen patient for eligibility
        Returns eligibility determination
        """
        from .study_design import get_clinical_study
        
        study = get_clinical_study()
        design = study.design
        
        eligibility_issues = []
        
        # Check age
        if screening_data.age < design.age_min or screening_data.age > design.age_max:
            eligibility_issues.append(f"Age {screening_data.age} outside range {design.age_min}-{design.age_max}")
        
        # Check T1DM duration
        if screening_data.t1dm_duration_years < design.t1dm_duration_min_years:
            eligibility_issues.append(f"T1DM duration {screening_data.t1dm_duration_years} < {design.t1dm_duration_min_years} years")
        
        # Check HbA1c
        if screening_data.hba1c < design.hba1c_min or screening_data.hba1c > design.hba1c_max:
            eligibility_issues.append(f"HbA1c {screening_data.hba1c} outside range {design.hba1c_min}-{design.hba1c_max}")
        
        # Check severe hypoglycemia
        if design.exclude_severe_hypoglycemia and screening_data.severe_hypoglycemia_history:
            eligibility_issues.append("History of severe hypoglycemia")
        
        # Check DKA history
        if design.exclude_diabetic_ketoacidosis and screening_data.dka_history:
            eligibility_issues.append("History of diabetic ketoacidosis")
        
        # Check pregnancy
        if design.exclude_pregnant and screening_data.pregnancy_status.lower() in ["pregnant", "yes"]:
            eligibility_issues.append("Pregnant")
        
        # Check chronic conditions
        if design.exclude_other_chronic_conditions and screening_data.chronic_conditions:
            eligibility_issues.append(f"Chronic conditions: {', '.join(screening_data.chronic_conditions)}")
        
        # Check consent
        if not screening_data.consent_signed:
            eligibility_issues.append("Consent not signed")
        
        # Check CGM experience
        if not screening_data.cgm_experience:
            eligibility_issues.append("No CGM experience")
        
        # Check smartphone access
        if not screening_data.smartphone_access:
            eligibility_issues.append("No smartphone access")
        
        is_eligible = len(eligibility_issues) == 0
        
        # Store screening data
        self.screened_patients[screening_data.patient_id] = screening_data
        
        return {
            "patient_id": screening_data.patient_id,
            "eligible": is_eligible,
            "eligibility_issues": eligibility_issues,
            "screening_date": screening_data.screening_date.isoformat(),
            "consent_signed": screening_data.consent_signed
        }
    
    def enroll_patient(self, patient_id: str) -> Optional[EnrolledPatient]:
        """
        Enroll eligible patient and randomize to study arm
        Returns enrolled patient data or None if ineligible
        """
        # Check if patient was screened
        if patient_id not in self.screened_patients:
            logger.error(f"Patient {patient_id} not screened")
            return None
        
        screening_data = self.screened_patients[patient_id]
        
        # Check eligibility
        eligibility_result = self.screen_patient(screening_data)
        if not eligibility_result["eligible"]:
            logger.error(f"Patient {patient_id} not eligible")
            return None
        
        # Check if already enrolled
        if patient_id in self.enrolled_patients:
            logger.error(f"Patient {patient_id} already enrolled")
            return None
        
        # Check enrollment cap
        from .study_design import get_clinical_study
        study = get_clinical_study()
        if len(self.enrolled_patients) >= study.design.target_enrollment:
            logger.error("Enrollment cap reached")
            return None
        
        # Randomize
        randomization_index = len(self.enrolled_patients)
        if randomization_index >= len(self.randomization_sequence):
            logger.error("Randomization sequence exhausted")
            return None
        
        randomization_arm = self.randomization_sequence[randomization_index]
        
        # Create enrolled patient
        enrolled_patient = EnrolledPatient(
            patient_id=patient_id,
            enrollment_date=date.today(),
            randomization_arm=randomization_arm,
            current_phase=StudyPhase.BASELINE,
            phase_start_date=date.today(),
            status=PatientStatus.ENROLLED,
            screening_data=screening_data
        )
        
        self.enrolled_patients[patient_id] = enrolled_patient
        
        logger.info(f"Enrolled patient {patient_id} to arm {randomization_arm}")
        return enrolled_patient
    
    def update_patient_phase(
        self,
        patient_id: str,
        new_phase: StudyPhase
    ) -> bool:
        """Update patient's current study phase"""
        if patient_id not in self.enrolled_patients:
            logger.error(f"Patient {patient_id} not enrolled")
            return False
        
        patient = self.enrolled_patients[patient_id]
        patient.current_phase = new_phase
        patient.phase_start_date = date.today()
        
        # Update status based on phase
        if new_phase == StudyPhase.COMPLETED:
            patient.status = PatientStatus.COMPLETED
        elif new_phase in [StudyPhase.INTERVENTION_A, StudyPhase.INTERVENTION_B]:
            patient.status = PatientStatus.ACTIVE
        
        logger.info(f"Updated patient {patient_id} to phase {new_phase}")
        return True
    
    def record_adverse_event(
        self,
        patient_id: str,
        event_type: str,
        severity: str,
        description: str,
        related_to_intervention: bool = False
    ) -> bool:
        """Record adverse event for patient"""
        if patient_id not in self.enrolled_patients:
            logger.error(f"Patient {patient_id} not enrolled")
            return False
        
        patient = self.enrolled_patients[patient_id]
        patient.adverse_events.append({
            "event_type": event_type,
            "severity": severity,
            "description": description,
            "related_to_intervention": related_to_intervention,
            "date": date.today().isoformat()
        })
        
        logger.info(f"Recorded adverse event for patient {patient_id}: {event_type}")
        return True
    
    def withdraw_patient(
        self,
        patient_id: str,
        reason: str
    ) -> bool:
        """Withdraw patient from study"""
        if patient_id not in self.enrolled_patients:
            logger.error(f"Patient {patient_id} not enrolled")
            return False
        
        patient = self.enrolled_patients[patient_id]
        patient.status = PatientStatus.WITHDRAWN
        patient.withdrawal_date = date.today()
        patient.withdrawal_reason = reason
        
        logger.info(f"Withdrew patient {patient_id}: {reason}")
        return True
    
    def get_enrollment_summary(self) -> Dict[str, Any]:
        """Get enrollment summary statistics"""
        total_enrolled = len(self.enrolled_patients)
        arm_counts = {arm.value: 0 for arm in RandomizationArm}
        phase_counts = {phase.value: 0 for phase in StudyPhase}
        status_counts = {status.value: 0 for status in PatientStatus}
        
        for patient in self.enrolled_patients.values():
            arm_counts[patient.randomization_arm.value] += 1
            phase_counts[patient.current_phase.value] += 1
            status_counts[patient.status.value] += 1
        
        return {
            "total_screened": len(self.screened_patients),
            "total_enrolled": total_enrolled,
            "randomization_arms": arm_counts,
            "current_phases": phase_counts,
            "status_distribution": status_counts,
            "adverse_events_total": sum(len(p.adverse_events) for p in self.enrolled_patients.values())
        }
    
    def get_patient_details(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information for a specific patient"""
        if patient_id not in self.enrolled_patients:
            return None
        
        patient = self.enrolled_patients[patient_id]
        return patient.model_dump()


# Singleton instance
_patient_enrollment = None

def get_patient_enrollment() -> PatientEnrollment:
    """Get singleton patient enrollment instance"""
    global _patient_enrollment
    if _patient_enrollment is None:
        _patient_enrollment = PatientEnrollment()
    return _patient_enrollment
