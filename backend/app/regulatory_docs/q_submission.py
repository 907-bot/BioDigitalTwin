"""
FDA Q-Submission Manager
Pre-submission documentation for FDA regulatory review
"""
import json
from datetime import datetime, date
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QSubmission(BaseModel):
    """FDA Q-Submission document"""
    submission_id: str
    submission_type: str = "Pre-Submission"  # Pre-Sub, 513(g), Q-Sub
    device_name: str = "Bio-Digital Twin for Diabetes Management"
    submission_date: date = Field(default_factory=date.today)
    
    # Product information
    product_code: str = "DQZ"  # Diabetes Management Software
    regulation_number: str = "21 CFR 880.5525"
    device_classification: str = "Class II"
    predicate_devices: List[str] = Field(default_factory=lambda: [
        "Dexcom G6 CGM System",
        "Medtronic MiniMed 770G"
    ])
    
    # Intended use
    intended_use: str = "Software as a Medical Device (SaMD) for Type 1 Diabetes management"
    indications_for_use: str = "For adults with Type 1 Diabetes to provide insulin dosing recommendations based on CGM data"
    
    # Technology description
    technology_summary: str = "AI-powered digital twin using UKF estimation and MCMC parameter learning"
    
    # Submission content
    sections: List[str] = Field(default_factory=lambda: [
        "Cover Letter",
        "User Needs and Intended Use",
        "Device Description",
        "Proposed Indications for Use",
        "Non-Clinical Performance Testing",
        "Clinical Performance Testing",
        "Labeling Draft",
        "Risk Analysis",
        "Software Documentation",
        "Regulatory Strategy"
    ])
    
    # Status
    status: str = "Draft"
    fda_feedback: Optional[str] = None
    approval_date: Optional[date] = None


class QSubmissionManager:
    """
    FDA Q-Submission management
    Handles pre-submission documentation and FDA communication
    """
    
    def __init__(self):
        self.submission = QSubmission(submission_id="Q-SUB-2024-001")
        self.document_sections: Dict[str, Any] = {}
        self.communications: List[Dict[str, Any]] = []
    
    def generate_q_submission_package(self) -> Dict[str, Any]:
        """Generate complete Q-Submission package"""
        return {
            "submission_header": {
                "submission_id": self.submission.submission_id,
                "submission_type": self.submission.submission_type,
                "device_name": self.submission.device_name,
                "submission_date": self.submission.submission_date.isoformat()
            },
            "product_information": {
                "product_code": self.submission.product_code,
                "regulation_number": self.submission.regulation_number,
                "device_classification": self.submission.device_classification,
                "predicate_devices": self.submission.predicate_devices
            },
            "intended_use": {
                "intended_use": self.submission.intended_use,
                "indications_for_use": self.submission.indications_for_use
            },
            "technology_summary": self.submission.technology_summary,
            "submission_sections": self.submission.sections,
            "document_sections": self.document_sections,
            "status": self.submission.status
        }
    
    def add_document_section(self, section_name: str, content: Dict[str, Any]) -> bool:
        """Add document section to submission"""
        self.document_sections[section_name] = content
        return True
    
    def generate_cover_letter(self) -> str:
        """Generate FDA cover letter"""
        return f"""
FDA Center for Devices and Radiological Health
10903 New Hampshire Avenue
Silver Spring, MD 20993

RE: Pre-Submission for {self.submission.device_name}
Submission ID: {self.submission.submission_id}
Date: {self.submission.submission_date.isoformat()}

Dear FDA Reviewers,

We are submitting this Pre-Submission for the {self.submission.device_name}, a Software as a Medical Device (SaMD) 
intended for Type 1 Diabetes management.

DEVICE DESCRIPTION
{self.submission.device_name} is an AI-powered digital twin system that uses continuous glucose monitoring (CGM) data 
to provide personalized insulin dosing recommendations. The system employs Unscented Kalman Filtering (UKF) for 
real-time state estimation and Markov Chain Monte Carlo (MCMC) for parameter learning.

INTENDED USE
{self.submission.intended_use}

INDICATIONS FOR USE
{self.submission.indications_for_use}

REGULATORY STRATEGY
We believe this device falls under Class II medical device classification under 21 CFR 880.5525. 
We are seeking FDA feedback on our proposed regulatory pathway and testing requirements.

QUESTIONS FOR FDA
1. Is our proposed classification as Class II appropriate?
2. What non-clinical testing requirements should we plan for?
3. What clinical study design would be acceptable for market authorization?
4. Are there any specific software documentation requirements for AI/ML-based SaMD?
5. What predicate devices should we consider for substantial equivalence?

We look forward to your feedback and guidance.

Sincerely,
{self.submission.principal_investigator if hasattr(self.submission, 'principal_investigator') else 'Principal Investigator'}
"""
    
    def generate_device_description(self) -> Dict[str, Any]:
        """Generate detailed device description"""
        return {
            "device_name": self.submission.device_name,
            "device_type": "Software as a Medical Device (SaMD)",
            "platform": "Cloud-based with mobile application",
            "components": [
                "CGM data ingestion module",
                "UKF state estimation engine",
                "MCMC parameter learning module",
                "Insulin dosing recommendation algorithm",
                "Patient monitoring dashboard",
                "Clinical alert system"
            ],
            "algorithms": [
                {
                    "name": "Unscented Kalman Filter",
                    "purpose": "Real-time glucose state estimation",
                    "inputs": ["CGM glucose values", "insulin delivery data", "carbohydrate intake"],
                    "outputs": ["Estimated glucose state", "Prediction confidence"]
                },
                {
                    "name": "Markov Chain Monte Carlo",
                    "purpose": "Physiological parameter estimation",
                    "inputs": ["Historical glucose data", "insulin history", "meal data"],
                    "outputs": ["Insulin sensitivity", "Carb absorption rate", "Glucose effectiveness"]
                }
            ],
            "data_sources": [
                "Dexcom G6/G7 CGM",
                "Abbott FreeStyle Libre",
                "Medtronic CGM",
                "Samsung Health wearable data",
                "EHR FHIR observations"
            ],
            "security_features": [
                "HIPAA compliance",
                "End-to-end encryption",
                "Secure authentication",
                "Audit logging",
                "Data anonymization"
            ]
        }
    
    def record_fda_communication(
        self,
        date: date,
        communication_type: str,
        summary: str,
        fda_contact: Optional[str] = None
    ) -> Dict[str, Any]:
        """Record FDA communication"""
        communication = {
            "date": date.isoformat(),
            "type": communication_type,
            "summary": summary,
            "fda_contact": fda_contact
        }
        self.communications.append(communication)
        return communication
    
    def get_submission_status(self) -> Dict[str, Any]:
        """Get current submission status"""
        return {
            "submission_id": self.submission.submission_id,
            "status": self.submission.status,
            "submission_date": self.submission.submission_date.isoformat(),
            "sections_completed": len(self.document_sections),
            "total_sections": len(self.submission.sections),
            "communications_count": len(self.communications),
            "fda_feedback": self.submission.fda_feedback,
            "approval_date": self.submission.approval_date.isoformat() if self.submission.approval_date else None
        }


# Singleton instance
_q_submission_manager = None

def get_q_submission_manager() -> QSubmissionManager:
    """Get singleton Q-Submission manager instance"""
    global _q_submission_manager
    if _q_submission_manager is None:
        _q_submission_manager = QSubmissionManager()
    return _q_submission_manager
