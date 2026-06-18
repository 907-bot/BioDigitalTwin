"""
IEC 62304 Documentation
Medical device software life cycle processes
Compliance documentation for SaMD regulatory requirements
"""
import json
from datetime import datetime, date
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SoftwareSafetyClass(str):
    """IEC 62304 software safety classification"""
    A = "A"  # No injury or damage to health is possible
    B = "B"  # Non-serious injury is possible
    C = "C"  # Death or serious injury is possible


class SoftwareProcess(BaseModel):
    """Software development process"""
    process_id: str
    process_name: str
    description: str
    inputs: List[str]
    outputs: List[str]
    verification_method: str
    validation_method: str


class IEC62304Documentation(BaseModel):
    """IEC 62304 compliance documentation"""
    document_id: str = "IEC-62304-001"
    document_title: str = "Medical Device Software Life Cycle Processes"
    version: str = "1.0"
    date: date = Field(default_factory=date.today)
    
    # Software classification
    safety_class: SoftwareSafetyClass = SoftwareSafetyClass.C  # Class C for insulin dosing
    classification_rationale: str = "Software provides insulin dosing recommendations; incorrect recommendations could cause death or serious injury"
    
    # Software lifecycle
    development_phases: List[str] = Field(default_factory=lambda: [
        "Software Planning",
        "Software Requirements Analysis",
        "Software Architecture Design",
        "Software Detailed Design",
        "Software Implementation and Verification",
        "Software Integration and Integration Testing",
        "Software System Testing",
        "Software Release"
    ])
    
    # Risk management integration
    risk_management_integration: bool = True
    risk_management_standard: str = "ISO 14971"
    
    # Configuration management
    configuration_management: bool = True
    version_control: str = "Git"
    change_control: bool = True
    
    # Verification and validation
    verification_activities: List[str] = Field(default_factory=lambda: [
        "Code reviews",
        "Unit testing",
        "Integration testing",
        "System testing",
        "Regression testing"
    ])
    
    validation_activities: List[str] = Field(default_factory=lambda: [
        "Clinical validation study",
        "User acceptance testing",
        "Performance testing",
        "Usability testing"
    ])


class IEC62304DocumentationManager:
    """
    IEC 62304 documentation management
    Generates and maintains compliance documentation
    """
    
    def __init__(self):
        self.documentation = IEC62304Documentation()
        self.processes: List[SoftwareProcess] = []
        self.test_results: Dict[str, Any] = {}
        self.traceability_matrix: Dict[str, List[str]] = {}
    
    def generate_compliance_package(self) -> Dict[str, Any]:
        """Generate complete IEC 62304 compliance package"""
        return {
            "document_header": {
                "document_id": self.documentation.document_id,
                "document_title": self.documentation.document_title,
                "version": self.documentation.version,
                "date": self.documentation.date.isoformat()
            },
            "software_classification": {
                "safety_class": self.documentation.safety_class,
                "classification_rationale": self.documentation.classification_rationale
            },
            "software_lifecycle": {
                "development_phases": self.documentation.development_phases
            },
            "risk_management": {
                "integrated": self.documentation.risk_management_integration,
                "standard": self.documentation.risk_management_standard
            },
            "configuration_management": {
                "enabled": self.documentation.configuration_management,
                "version_control": self.documentation.version_control,
                "change_control": self.documentation.change_control
            },
            "verification": {
                "activities": self.documentation.verification_activities
            },
            "validation": {
                "activities": self.documentation.validation_activities
            },
            "processes": [p.model_dump() for p in self.processes],
            "test_results": self.test_results,
            "traceability_matrix": self.traceability_matrix
        }
    
    def add_process(self, process: SoftwareProcess) -> bool:
        """Add software development process"""
        self.processes.append(process)
        return True
    
    def generate_standard_processes(self) -> List[SoftwareProcess]:
        """Generate standard IEC 62304 processes"""
        processes = [
            SoftwareProcess(
                process_id="SP-001",
                process_name="Software Planning",
                description="Plan software development activities and define software lifecycle",
                inputs=["Project requirements", "Regulatory requirements"],
                outputs=["Software plan", "Development schedule"],
                verification_method="Review by quality assurance",
                validation_method="Stakeholder approval"
            ),
            SoftwareProcess(
                process_id="SP-002",
                process_name="Software Requirements Analysis",
                description="Define software requirements including safety requirements",
                inputs=["User needs", "Intended use", "Risk analysis"],
                outputs=["Software requirements specification", "Safety requirements"],
                verification_method="Requirements review",
                validation_method="User needs traceability"
            ),
            SoftwareProcess(
                process_id="SP-003",
                process_name="Software Architecture Design",
                description="Design software architecture and identify software items",
                inputs=["Software requirements", "Safety classification"],
                outputs=["Software architecture description", "Software item list"],
                verification_method="Architecture review",
                validation_method="Requirements coverage analysis"
            ),
            SoftwareProcess(
                process_id="SP-004",
                process_name="Software Detailed Design",
                description="Design software units and interfaces",
                inputs=["Software architecture", "Software requirements"],
                outputs=["Detailed design specifications", "Interface specifications"],
                verification_method="Design review",
                validation_method="Traceability to requirements"
            ),
            SoftwareProcess(
                process_id="SP-005",
                process_name="Software Implementation",
                description="Implement software units according to design",
                inputs=["Detailed design specifications"],
                outputs=["Source code", "Unit test results"],
                verification_method="Code review, Unit testing",
                validation_method="Design compliance verification"
            ),
            SoftwareProcess(
                process_id="SP-006",
                process_name="Software Integration",
                description="Integrate software units and verify integration",
                inputs=["Software units", "Integration test specifications"],
                outputs=["Integrated software", "Integration test results"],
                verification_method="Integration testing",
                validation_method="System requirements verification"
            ),
            SoftwareProcess(
                process_id="SP-007",
                process_name="Software System Testing",
                description="Test complete software system against requirements",
                inputs=["Integrated software", "System test specifications"],
                outputs=["Test reports", "Defect reports"],
                verification_method="System testing",
                validation_method="User acceptance testing"
            ),
            SoftwareProcess(
                process_id="SP-008",
                process_name="Software Release",
                description="Prepare software for release including documentation",
                inputs=["Tested software", "Release criteria"],
                outputs=["Released software", "User documentation"],
                verification_method="Release review",
                validation_method="Installation testing"
            )
        ]
        
        self.processes = processes
        return processes
    
    def generate_traceability_matrix(self) -> Dict[str, List[str]]:
        """Generate requirements traceability matrix"""
        matrix = {
            "REQ-001: CGM Data Ingestion": [
                "SP-002: Software Requirements Analysis",
                "SP-003: Software Architecture Design",
                "SP-004: Software Detailed Design",
                "SP-005: Software Implementation",
                "SP-006: Software Integration"
            ],
            "REQ-002: UKF State Estimation": [
                "SP-002: Software Requirements Analysis",
                "SP-003: Software Architecture Design",
                "SP-004: Software Detailed Design",
                "SP-005: Software Implementation",
                "SP-006: Software Integration"
            ],
            "REQ-003: MCMC Parameter Learning": [
                "SP-002: Software Requirements Analysis",
                "SP-003: Software Architecture Design",
                "SP-004: Software Detailed Design",
                "SP-005: Software Implementation",
                "SP-006: Software Integration"
            ],
            "REQ-004: Insulin Dosing Recommendations": [
                "SP-002: Software Requirements Analysis",
                "SP-003: Software Architecture Design",
                "SP-004: Software Detailed Design",
                "SP-005: Software Implementation",
                "SP-006: Software Integration",
                "SP-007: Software System Testing"
            ],
            "REQ-005: Safety Alerts": [
                "SP-002: Software Requirements Analysis",
                "SP-003: Software Architecture Design",
                "SP-004: Software Detailed Design",
                "SP-005: Software Implementation",
                "SP-006: Software Integration",
                "SP-007: Software System Testing"
            ],
            "REQ-006: Data Security": [
                "SP-002: Software Requirements Analysis",
                "SP-003: Software Architecture Design",
                "SP-004: Software Detailed Design",
                "SP-005: Software Implementation",
                "SP-006: Software Integration"
            ]
        }
        
        self.traceability_matrix = matrix
        return matrix
    
    def record_test_result(
        self,
        test_id: str,
        test_name: str,
        result: str,
        date: date = None
    ) -> Dict[str, Any]:
        """Record test result"""
        if date is None:
            date = date.today()
        
        test_result = {
            "test_id": test_id,
            "test_name": test_name,
            "result": result,
            "date": date.isoformat()
        }
        
        self.test_results[test_id] = test_result
        return test_result
    
    def generate_software_unit_list(self) -> List[Dict[str, Any]]:
        """Generate software unit list per IEC 62304 requirements"""
        return [
            {
                "unit_id": "SU-001",
                "unit_name": "CGM Data Ingestion Module",
                "safety_class": "C",
                "description": "Ingests and validates CGM data from multiple manufacturers",
                "programming_language": "Python",
                "complexity": "Medium"
            },
            {
                "unit_id": "SU-002",
                "unit_name": "UKF State Estimation Engine",
                "safety_class": "C",
                "description": "Implements Unscented Kalman Filter for glucose state estimation",
                "programming_language": "Python",
                "complexity": "High"
            },
            {
                "unit_id": "SU-003",
                "unit_name": "MCMC Parameter Learning Module",
                "safety_class": "B",
                "description": "Performs Bayesian parameter estimation using MCMC",
                "programming_language": "Python",
                "complexity": "High"
            },
            {
                "unit_id": "SU-004",
                "unit_name": "Insulin Dosing Algorithm",
                "safety_class": "C",
                "description": "Calculates insulin dosing recommendations",
                "programming_language": "Python",
                "complexity": "High"
            },
            {
                "unit_id": "SU-005",
                "unit_name": "Safety Alert System",
                "safety_class": "C",
                "description": "Generates safety alerts for hypoglycemia and hyperglycemia",
                "programming_language": "Python",
                "complexity": "Medium"
            },
            {
                "unit_id": "SU-006",
                "unit_name": "Data Security Module",
                "safety_class": "B",
                "description": "Handles encryption, authentication, and data protection",
                "programming_language": "Python",
                "complexity": "High"
            },
            {
                "unit_id": "SU-007",
                "unit_name": "User Interface",
                "safety_class": "B",
                "description": "Web and mobile user interfaces",
                "programming_language": "TypeScript, React",
                "complexity": "Medium"
            },
            {
                "unit_id": "SU-008",
                "unit_name": "Data Persistence Layer",
                "safety_class": "B",
                "description": "Handles data storage in Redis and PostgreSQL",
                "programming_language": "Python, SQL",
                "complexity": "Medium"
            }
        ]
    
    def get_compliance_status(self) -> Dict[str, Any]:
        """Get current compliance status"""
        return {
            "document_id": self.documentation.document_id,
            "version": self.documentation.version,
            "date": self.documentation.date.isoformat(),
            "safety_class": self.documentation.safety_class,
            "processes_defined": len(self.processes),
            "test_results_recorded": len(self.test_results),
            "traceability_matrix_complete": len(self.traceability_matrix) > 0,
            "compliance_percentage": self._calculate_compliance_percentage()
        }
    
    def _calculate_compliance_percentage(self) -> float:
        """Calculate overall compliance percentage"""
        total_items = 8  # Total compliance items
        completed_items = 0
        
        if len(self.processes) >= 8:
            completed_items += 1
        if len(self.test_results) > 0:
            completed_items += 1
        if len(self.traceability_matrix) > 0:
            completed_items += 1
        if self.documentation.risk_management_integration:
            completed_items += 1
        if self.documentation.configuration_management:
            completed_items += 1
        if self.documentation.change_control:
            completed_items += 1
        if len(self.documentation.verification_activities) > 0:
            completed_items += 1
        if len(self.documentation.validation_activities) > 0:
            completed_items += 1
        
        return (completed_items / total_items) * 100


# Singleton instance
_iec_62304_manager = None

def get_iec_62304_manager() -> IEC62304DocumentationManager:
    """Get singleton IEC 62304 documentation manager instance"""
    global _iec_62304_manager
    if _iec_62304_manager is None:
        _iec_62304_manager = IEC62304DocumentationManager()
    return _iec_62304_manager
