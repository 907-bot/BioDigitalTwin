"""
ISO 14971 Risk Management
Medical device risk management process documentation
"""
import json
from datetime import datetime, date as dt_date
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class SeverityLevel(str, Enum):
    """Risk severity levels"""
    NEGLIGIBLE = "Negligible"
    MINOR = "Minor"
    SERIOUS = "Serious"
    CRITICAL = "Critical"
    CATASTROPHIC = "Catastrophic"


class ProbabilityLevel(str, Enum):
    """Risk probability levels"""
    RARE = "Rare"
    UNLIKELY = "Unlikely"
    POSSIBLE = "Possible"
    LIKELY = "Likely"
    FREQUENT = "Frequent"


class RiskPriority(str, Enum):
    """Risk priority levels"""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    UNACCEPTABLE = "Unacceptable"


class Hazard(BaseModel):
    """Potential hazard"""
    hazard_id: str
    hazard_name: str
    description: str
    severity: SeverityLevel
    probability: ProbabilityLevel
    priority: RiskPriority
    causes: List[str]
    harms: List[str]
    controls: List[str]
    residual_risk: RiskPriority
    acceptability: str


class ISO14971RiskManagement(BaseModel):
    """ISO 14971 risk management documentation"""
    document_id: str = "ISO-14971-001"
    document_title: str = "Medical Device Risk Management"
    version: str = "1.0"
    date: dt_date = Field(default_factory=dt_date.today)
    
    # Risk management process
    risk_analysis_complete: bool = True
    risk_evaluation_complete: bool = True
    risk_control_complete: bool = True
    overall_residual_risk_acceptability: str = "Acceptable"
    
    # Risk management file
    hazards: List[Hazard] = Field(default_factory=list)
    
    # Risk management activities
    risk_management_activities: List[str] = Field(default_factory=lambda: [
        "Risk analysis",
        "Risk evaluation",
        "Risk control",
        "Production and post-production information",
        "Risk management review"
    ])


class ISO14971RiskManagementManager:
    """
    ISO 14971 risk management documentation manager
    Manages hazard identification, risk assessment, and control measures
    """
    
    def __init__(self):
        self.documentation = ISO14971RiskManagement()
        self.hazards: List[Hazard] = []
        self.risk_acceptance_criteria: Dict[str, Any] = {}
        self.post_production_monitoring: List[Dict[str, Any]] = []
    
    def generate_risk_management_file(self) -> Dict[str, Any]:
        """Generate complete risk management file"""
        return {
            "document_header": {
                "document_id": self.documentation.document_id,
                "document_title": self.documentation.document_title,
                "version": self.documentation.version,
                "date": self.documentation.date.isoformat()
            },
            "risk_management_process": {
                "risk_analysis_complete": self.documentation.risk_analysis_complete,
                "risk_evaluation_complete": self.documentation.risk_evaluation_complete,
                "risk_control_complete": self.documentation.risk_control_complete,
                "overall_residual_risk_acceptability": self.documentation.overall_residual_risk_acceptability
            },
            "risk_management_activities": self.documentation.risk_management_activities,
            "hazards": [h.model_dump() for h in self.hazards],
            "risk_acceptance_criteria": self.risk_acceptance_criteria,
            "post_production_monitoring": self.post_production_monitoring
        }
    
    def add_hazard(self, hazard: Hazard) -> bool:
        """Add hazard to risk management file"""
        self.hazards.append(hazard)
        return True
    
    def generate_standard_hazards(self) -> List[Hazard]:
        """Generate standard hazards for diabetes management SaMD"""
        hazards = [
            Hazard(
                hazard_id="HZ-001",
                hazard_name="Incorrect Insulin Dosing Recommendation",
                description="System provides incorrect insulin dosing recommendation leading to hypoglycemia or hyperglycemia",
                severity=SeverityLevel.CRITICAL,
                probability=ProbabilityLevel.UNLIKELY,
                priority=RiskPriority.HIGH,
                causes=[
                    "Algorithm error in UKF estimation",
                    "Incorrect CGM data input",
                    "Parameter estimation error in MCMC",
                    "Software bug in dosing calculation"
                ],
                harms=[
                    "Severe hypoglycemia",
                    "Diabetic ketoacidosis",
                    "Hyperosmolar hyperglycemic state",
                    "Death"
                ],
                controls=[
                    "Redundant validation of dosing recommendations",
                    "Clinical oversight of recommendations",
                    "Alert system for out-of-range recommendations",
                    "Regular algorithm validation",
                    "User confirmation required for dose changes"
                ],
                residual_risk=RiskPriority.MEDIUM,
                acceptability="Acceptable with controls"
            ),
            Hazard(
                hazard_id="HZ-002",
                hazard_name="Data Privacy Breach",
                description="Unauthorized access to patient health data",
                severity=SeverityLevel.SERIOUS,
                probability=ProbabilityLevel.RARE,
                priority=RiskPriority.MEDIUM,
                causes=[
                    "Insufficient encryption",
                    "Authentication bypass",
                    "SQL injection attack",
                    "Insider threat"
                ],
                harms=[
                    "Privacy violation",
                    "Identity theft",
                    "Discrimination",
                    "Legal liability"
                ],
                controls=[
                    "End-to-end encryption",
                    "Multi-factor authentication",
                    "Regular security audits",
                    "HIPAA compliance",
                    "Data anonymization for analysis"
                ],
                residual_risk=RiskPriority.LOW,
                acceptability="Acceptable"
            ),
            Hazard(
                hazard_id="HZ-003",
                hazard_name="CGM Data Failure",
                description="CGM device malfunction or data transmission failure",
                severity=SeverityLevel.SERIOUS,
                probability=ProbabilityLevel.POSSIBLE,
                priority=RiskPriority.MEDIUM,
                causes=[
                    "CGM sensor failure",
                    "Bluetooth connectivity loss",
                    "Device battery depletion",
                    "Data corruption"
                ],
                harms=[
                    "Missed hypoglycemia events",
                    "Inaccurate glucose estimation",
                    "Loss of monitoring capability"
                ],
                controls=[
                    "Multiple CGM device support",
                    "Data validation and quality checks",
                    "Alert for data loss",
                    "Manual glucose entry fallback",
                    "Redundant data storage"
                ],
                residual_risk=RiskPriority.LOW,
                acceptability="Acceptable"
            ),
            Hazard(
                hazard_id="HZ-004",
                hazard_name="Algorithm Convergence Failure",
                description="UKF or MCMC algorithm fails to converge",
                severity=SeverityLevel.MINOR,
                probability=ProbabilityLevel.UNLIKELY,
                priority=RiskPriority.LOW,
                causes=[
                    "Poor data quality",
                    "Insufficient data points",
                    "Numerical instability",
                    "Parameter initialization issues"
                ],
                harms=[
                    "Inaccurate state estimates",
                    "Reduced prediction accuracy",
                    "System unavailability"
                ],
                controls=[
                    "Convergence monitoring",
                    "Fallback to simpler models",
                    "Data quality requirements",
                    "Parameter initialization best practices",
                    "Error handling and logging"
                ],
                residual_risk=RiskPriority.LOW,
                acceptability="Acceptable"
            ),
            Hazard(
                hazard_id="HZ-005",
                hazard_name="User Interface Misinterpretation",
                description="User misinterprets displayed information or recommendations",
                severity=SeverityLevel.MINOR,
                probability=ProbabilityLevel.POSSIBLE,
                priority=RiskPriority.LOW,
                causes=[
                    "Unclear display design",
                    "Complex terminology",
                    "Information overload",
                    "Poor color contrast"
                ],
                harms=[
                    "Incorrect manual insulin dosing",
                    "Missed alerts",
                    "User frustration",
                    "Reduced treatment adherence"
                ],
                controls=[
                    "Usability testing",
                    "Clear terminology",
                    "Standardized color coding",
                    "User training materials",
                    "Help documentation"
                ],
                residual_risk=RiskPriority.LOW,
                acceptability="Acceptable"
            ),
            Hazard(
                hazard_id="HZ-006",
                hazard_name="System Unavailability",
                description="System downtime preventing access to recommendations",
                severity=SeverityLevel.MINOR,
                probability=ProbabilityLevel.RARE,
                priority=RiskPriority.LOW,
                causes=[
                    "Server failure",
                    "Network outage",
                    "Software crash",
                    "Maintenance downtime"
                ],
                harms=[
                    "Temporary loss of decision support",
                    "User inconvenience",
                    "Data recording interruption"
                ],
                controls=[
                    "High availability architecture",
                    "Redundant servers",
                    "Offline mode capability",
                    "Regular backups",
                    "SLA monitoring"
                ],
                residual_risk=RiskPriority.LOW,
                acceptability="Acceptable"
            )
        ]
        
        self.hazards = hazards
        return hazards
    
    def generate_risk_acceptance_criteria(self) -> Dict[str, Any]:
        """Generate risk acceptance criteria matrix"""
        criteria = {
            "risk_matrix": {
                "Catastrophic": {
                    "Frequent": "Unacceptable",
                    "Likely": "Unacceptable",
                    "Possible": "Unacceptable",
                    "Unlikely": "High",
                    "Rare": "Medium"
                },
                "Critical": {
                    "Frequent": "Unacceptable",
                    "Likely": "Unacceptable",
                    "Possible": "High",
                    "Unlikely": "Medium",
                    "Rare": "Low"
                },
                "Serious": {
                    "Frequent": "Unacceptable",
                    "Likely": "High",
                    "Possible": "Medium",
                    "Unlikely": "Low",
                    "Rare": "Low"
                },
                "Minor": {
                    "Frequent": "High",
                    "Likely": "Medium",
                    "Possible": "Low",
                    "Unlikely": "Low",
                    "Rare": "Low"
                },
                "Negligible": {
                    "Frequent": "Medium",
                    "Likely": "Low",
                    "Possible": "Low",
                    "Unlikely": "Low",
                    "Rare": "Low"
                }
            },
            "overall_acceptability": "Overall residual risk must be Low or Medium",
            "benefit_risk_ratio": "Benefits must outweigh risks for the intended use"
        }
        
        self.risk_acceptance_criteria = criteria
        return criteria
    
    def record_post_production_event(
        self,
        event_id: str,
        event_type: str,
        description: str,
        severity: str,
        date: dt_date = None,
        corrective_action: Optional[str] = None
    ) -> Dict[str, Any]:
        """Record post-production safety event"""
        if date is None:
            date = dt_date.today()
        
        event = {
            "event_id": event_id,
            "event_type": event_type,
            "description": description,
            "severity": severity,
            "date": date.isoformat(),
            "corrective_action": corrective_action,
            "status": "Open" if not corrective_action else "Closed"
        }
        
        self.post_production_monitoring.append(event)
        return event
    
    def generate_risk_summary(self) -> Dict[str, Any]:
        """Generate risk summary statistics"""
        priority_counts = {priority.value: 0 for priority in RiskPriority}
        severity_counts = {severity.value: 0 for severity in SeverityLevel}
        
        for hazard in self.hazards:
            priority_counts[hazard.priority.value] += 1
            severity_counts[hazard.severity.value] += 1
        
        return {
            "total_hazards": len(self.hazards),
            "priority_distribution": priority_counts,
            "severity_distribution": severity_counts,
            "unacceptable_risks": sum(1 for h in self.hazards if h.priority == RiskPriority.UNACCEPTABLE),
            "high_priority_risks": sum(1 for h in self.hazards if h.priority == RiskPriority.HIGH),
            "post_production_events": len(self.post_production_monitoring),
            "open_events": sum(1 for e in self.post_production_monitoring if e["status"] == "Open")
        }
    
    def get_risk_management_status(self) -> Dict[str, Any]:
        """Get current risk management status"""
        return {
            "document_id": self.documentation.document_id,
            "version": self.documentation.version,
            "date": self.documentation.date.isoformat(),
            "hazards_identified": len(self.hazards),
            "risk_analysis_complete": self.documentation.risk_analysis_complete,
            "risk_evaluation_complete": self.documentation.risk_evaluation_complete,
            "risk_control_complete": self.documentation.risk_control_complete,
            "overall_residual_risk_acceptability": self.documentation.overall_residual_risk_acceptability,
            "post_production_events": len(self.post_production_monitoring)
        }


# Singleton instance
_iso_14971_manager = None

def get_iso_14971_manager() -> ISO14971RiskManagementManager:
    """Get singleton ISO 14971 risk management manager instance"""
    global _iso_14971_manager
    if _iso_14971_manager is None:
        _iso_14971_manager = ISO14971RiskManagementManager()
    return _iso_14971_manager
