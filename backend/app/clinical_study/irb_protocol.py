"""
IRB Protocol for Clinical Study BDT-001
Institutional Review Board protocol documentation
"""
import json
from datetime import datetime, date
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IRBProtocol(BaseModel):
    """IRB Protocol document"""
    protocol_id: str = "BDT-001-IRB-001"
    protocol_title: str = "Bio-Digital Twin for Type 1 Diabetes Management: A Randomized Crossover Trial"
    version: str = "1.0"
    date: date = Field(default_factory=date.today)
    
    # Study information
    study_id: str = "BDT-001"
    principal_investigator: str = "To be assigned"
    institution: str = "To be assigned"
    study_duration_months: int = 36
    
    # IRB classification
    irb_category: str = "Minimal Risk"
    fda_regulated: bool = True
    investigational_device: bool = True
    samd_classification: str = "Class II Medical Device"
    
    # Study population
    target_enrollment: int = 200
    age_range: str = "18-65 years"
    population: str = "Adults with Type 1 Diabetes"
    
    # Risk assessment
    risk_level: str = "Low"
    primary_risks: List[str] = Field(default_factory=lambda: [
        "Privacy breach of health data",
        "Device malfunction leading to incorrect insulin dosing recommendations",
        "Psychological impact of continuous monitoring",
        "Skin irritation from CGM/sensor wear"
    ])
    
    risk_mitigation: List[str] = Field(default_factory=lambda: [
        "End-to-end encryption of all health data",
        "FDA-cleared CGM devices only",
        "Clinical oversight of all insulin dosing recommendations",
        "Regular dermatological assessments",
        "Data Safety Monitoring Board (DSMB) oversight"
    ])
    
    # Benefits
    potential_benefits: List[str] = Field(default_factory=lambda: [
        "Improved glucose control",
        "Reduced hypoglycemia episodes",
        "Enhanced quality of life",
        "Personalized diabetes management"
    ])
    
    # Informed consent
    consent_elements: List[str] = Field(default_factory=lambda: [
        "Study purpose and procedures",
        "Risks and benefits",
        "Alternatives to participation",
        "Confidentiality protections",
        "Voluntary nature of participation",
        "Right to withdraw without penalty",
        "Contact information for questions"
    ])
    
    # Data protection
    data_protection: List[str] = Field(default_factory=lambda: [
        "HIPAA compliance",
        "De-identification of study data",
        "Secure cloud storage with encryption",
        "Limited access to identifiable information",
        "Data retention for 10 years post-study"
    ])
    
    # Safety monitoring
    dsmb_required: bool = True
    dsmb_meeting_frequency: str = "Quarterly"
    adverse_event_reporting: str = "Within 24 hours for serious events"
    stopping_rules: List[str] = Field(default_factory=lambda: [
        "Severe hypoglycemia rate > 5%",
        "HbA1c increase > 1.0%",
        "Serious adverse event related to intervention",
        "DSMB recommendation"
    ])


class IRBProtocolManager:
    """
    IRB Protocol management
    Generates and manages IRB protocol documentation
    """
    
    def __init__(self):
        self.protocol = IRBProtocol()
        self.amendments: List[Dict[str, Any]] = []
    
    def generate_protocol_document(self) -> Dict[str, Any]:
        """Generate complete IRB protocol document"""
        return {
            "protocol_header": {
                "protocol_id": self.protocol.protocol_id,
                "protocol_title": self.protocol.protocol_title,
                "version": self.protocol.version,
                "date": self.protocol.date.isoformat()
            },
            "study_information": {
                "study_id": self.protocol.study_id,
                "principal_investigator": self.protocol.principal_investigator,
                "institution": self.protocol.institution,
                "study_duration_months": self.protocol.study_duration_months
            },
            "irb_classification": {
                "category": self.protocol.irb_category,
                "fda_regulated": self.protocol.fda_regulated,
                "investigational_device": self.protocol.investigational_device,
                "samd_classification": self.protocol.samd_classification
            },
            "study_population": {
                "target_enrollment": self.protocol.target_enrollment,
                "age_range": self.protocol.age_range,
                "population": self.protocol.population
            },
            "risk_assessment": {
                "risk_level": self.protocol.risk_level,
                "primary_risks": self.protocol.primary_risks,
                "risk_mitigation": self.protocol.risk_mitigation
            },
            "potential_benefits": self.protocol.potential_benefits,
            "informed_consent": {
                "consent_elements": self.protocol.consent_elements
            },
            "data_protection": self.protocol.data_protection,
            "safety_monitoring": {
                "dsmb_required": self.protocol.dsmb_required,
                "dsmb_meeting_frequency": self.protocol.dsmb_meeting_frequency,
                "adverse_event_reporting": self.protocol.adverse_event_reporting,
                "stopping_rules": self.protocol.stopping_rules
            },
            "amendments": self.amendments
        }
    
    def add_amendment(
        self,
        amendment_number: str,
        description: str,
        reason: str,
        date: date = None
    ) -> Dict[str, Any]:
        """Add protocol amendment"""
        if date is None:
            date = date.today()
        
        amendment = {
            "amendment_number": amendment_number,
            "description": description,
            "reason": reason,
            "date": date.isoformat(),
            "approved": False
        }
        
        self.amendments.append(amendment)
        self.protocol.version = f"1.{len(self.amendments)}"
        
        return amendment
    
    def approve_amendment(self, amendment_number: str) -> bool:
        """Mark amendment as approved"""
        for amendment in self.amendments:
            if amendment["amendment_number"] == amendment_number:
                amendment["approved"] = True
                return True
        return False
    
    def generate_consent_form_template(self) -> str:
        """Generate informed consent form template"""
        return f"""
INFORMED CONSENT FORM
{self.protocol.protocol_title}
Protocol ID: {self.protocol.protocol_id}
Version: {self.protocol.version}
Date: {self.protocol.date.isoformat()}

INTRODUCTION
You are being asked to participate in a research study. This form provides information about the study. 
Please read it carefully and ask any questions you may have before deciding whether to participate.

PURPOSE
The purpose of this study is to evaluate a Bio-Digital Twin system for Type 1 Diabetes management. 
This system uses continuous glucose monitoring (CGM) data and artificial intelligence to provide 
personalized insulin dosing recommendations.

PROCEDURES
If you agree to participate, you will:
- Wear a CGM device for the duration of the study (approximately 36 weeks)
- Use a smartphone app to record your health data
- Be randomized to one of two intervention sequences
- Complete study visits every 4 weeks
- Provide blood samples for HbA1c testing

RISKS
The potential risks of participation include:
{chr(10).join(f"- {risk}" for risk in self.protocol.primary_risks)}

BENEFITS
The potential benefits of participation include:
{chr(10).join(f"- {benefit}" for benefit in self.protocol.potential_benefits)}

CONFIDENTIALITY
Your privacy will be protected as follows:
{chr(10).join(f"- {protection}" for protection in self.protocol.data_protection)}

VOLUNTARY PARTICIPATION
Your participation is entirely voluntary. You may withdraw at any time without penalty.

CONTACT
For questions about this study, contact: {self.protocol.principal_investigator}
For questions about your rights as a research participant, contact the IRB office.

SIGNATURE
I have read this consent form and agree to participate in this study.
Participant Signature: __________________________ Date: __________
Investigator Signature: __________________________ Date: __________
"""
    
    def get_protocol_status(self) -> Dict[str, Any]:
        """Get current protocol status"""
        return {
            "protocol_id": self.protocol.protocol_id,
            "version": self.protocol.version,
            "date": self.protocol.date.isoformat(),
            "amendments_count": len(self.amendments),
            "approved_amendments": sum(1 for a in self.amendments if a["approved"]),
            "pending_amendments": sum(1 for a in self.amendments if not a["approved"])
        }


# Singleton instance
_irb_protocol_manager = None

def get_irb_protocol_manager() -> IRBProtocolManager:
    """Get singleton IRB protocol manager instance"""
    global _irb_protocol_manager
    if _irb_protocol_manager is None:
        _irb_protocol_manager = IRBProtocolManager()
    return _irb_protocol_manager
