"""
FHIR Observation Ingest Service
Handles FHIR R4 Observation resources from EHR systems
Supports HL7 FHIR standard for healthcare data exchange
"""
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FHIRObservation(BaseModel):
    """FHIR R4 Observation resource"""
    resource_type: str = "Observation"
    id: Optional[str] = None
    status: str = Field(..., description="final | preliminary | amended | corrected")
    category: List[Dict[str, Any]] = Field(default_factory=list)
    code: Dict[str, Any] = Field(..., description="LOINC or SNOMED CT code")
    subject: Dict[str, Any] = Field(..., description="Patient reference")
    effectiveDateTime: Optional[str] = None
    effectivePeriod: Optional[Dict[str, Any]] = None
    valueQuantity: Optional[Dict[str, Any]] = None
    valueCodeableConcept: Optional[Dict[str, Any]] = None
    valueString: Optional[str] = None
    valueBoolean: Optional[bool] = None
    valueInteger: Optional[int] = None
    valueRange: Optional[Dict[str, Any]] = None
    valueRatio: Optional[Dict[str, Any]] = None
    valueSampledData: Optional[Dict[str, Any]] = None
    valueTime: Optional[str] = None
    valueDateTime: Optional[str] = None
    valuePeriod: Optional[Dict[str, Any]] = None
    dataAbsentReason: Optional[Dict[str, Any]] = None
    interpretation: Optional[List[Dict[str, Any]]] = None
    note: Optional[List[Dict[str, Any]]] = None
    bodySite: Optional[Dict[str, Any]] = None
    method: Optional[Dict[str, Any]] = None
    specimen: Optional[Dict[str, Any]] = None
    device: Optional[Dict[str, Any]] = None
    referenceRange: Optional[List[Dict[str, Any]]] = None
    hasMember: Optional[List[Dict[str, Any]]] = None
    derivedFrom: Optional[List[Dict[str, Any]]] = None
    component: Optional[List[Dict[str, Any]]] = None
    
    @validator('status')
    def validate_status(cls, v):
        valid_statuses = ['final', 'preliminary', 'amended', 'corrected', 'entered-in-error', 'cancelled']
        if v not in valid_statuses:
            raise ValueError(f"Invalid status: {v}. Must be one of {valid_statuses}")
        return v


class FHIRIngestService:
    """
    FHIR Observation ingestion service
    Converts FHIR resources to internal observation format
    Supports LOINC and SNOMED CT coding systems
    """
    
    # LOINC codes for common observations
    LOINC_CODES = {
        "glucose": "15074-8",  # Glucose [Moles/volume] in Blood
        "glucose_mg": "2345-7",  # Glucose [Mass/volume] in Blood
        "heart_rate": "8867-4",  # Heart rate
        "blood_pressure_systolic": "8480-6",  # Systolic blood pressure
        "blood_pressure_diastolic": "8462-4",  # Diastolic blood pressure
        "oxygen_saturation": "2708-6",  # Oxygen saturation in Arterial blood
        "respiratory_rate": "9279-1",  # Respiratory rate
        "body_temperature": "8310-5",  # Body temperature
        "body_weight": "29463-7",  # Body weight
        "body_height": "8302-2",  # Body height
        "steps": "41950-7",  # Number of steps in 24 hour Measured
        "sleep_duration": "93832-2",  # Sleep duration
        "activity": "55423-8",  # Physical activity
    }
    
    # SNOMED CT codes
    SNOMED_CODES = {
        "glucose": "4165007",
        "heart_rate": "364075007",
        "blood_pressure": "75367002",
        "oxygen_saturation": "103228002",
    }
    
    def __init__(self):
        self.observation_mapping = {
            "15074-8": ("glucose", "mmol/L"),
            "2345-7": ("glucose", "mg/dL"),
            "8867-4": ("heart_rate", "bpm"),
            "8480-6": ("systolic_bp", "mmHg"),
            "8462-4": ("diastolic_bp", "mmHg"),
            "2708-6": ("spo2", "%"),
            "9279-1": ("respiratory_rate", "/min"),
            "8310-5": ("temperature", "C"),
            "29463-7": ("weight", "kg"),
            "8302-2": ("height", "cm"),
            "41950-7": ("steps", "count"),
            "93832-2": ("sleep_duration", "hours"),
        }
    
    def parse_fhir_observation(self, fhir_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parse FHIR Observation resource and convert to internal format
        Returns None if parsing fails
        """
        try:
            # Validate FHIR structure
            if fhir_data.get("resourceType") != "Observation":
                logger.error(f"Invalid resource type: {fhir_data.get('resourceType')}")
                return None
            
            # Extract patient ID
            subject = fhir_data.get("subject", {})
            patient_ref = subject.get("reference", "")
            patient_id = patient_ref.split("/")[-1] if "/" in patient_ref else patient_ref
            
            if not patient_id:
                logger.error("Missing patient ID in FHIR observation")
                return None
            
            # Extract observation code
            code = fhir_data.get("code", {})
            coding = code.get("coding", [])
            loinc_code = None
            observation_type = None
            unit = None
            
            for coding_item in coding:
                system = coding_item.get("system", "")
                code_value = coding_item.get("code", "")
                
                if "loinc.org" in system.lower():
                    loinc_code = code_value
                    if code_value in self.observation_mapping:
                        observation_type, unit = self.observation_mapping[code_value]
                    break
            
            if not observation_type:
                logger.warning(f"Unknown LOINC code: {loinc_code}")
                observation_type = "unknown"
                unit = "unknown"
            
            # Extract value
            value = None
            if "valueQuantity" in fhir_data:
                value_quantity = fhir_data["valueQuantity"]
                value = value_quantity.get("value")
                if unit == "unknown":
                    unit = value_quantity.get("code", "")
            elif "valueInteger" in fhir_data:
                value = fhir_data["valueInteger"]
            elif "valueString" in fhir_data:
                value = fhir_data["valueString"]
            elif "valueBoolean" in fhir_data:
                value = fhir_data["valueBoolean"]
            
            if value is None:
                logger.warning("No value found in FHIR observation")
                return None
            
            # Extract timestamp
            timestamp = None
            if "effectiveDateTime" in fhir_data:
                timestamp_str = fhir_data["effectiveDateTime"]
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            elif "effectivePeriod" in fhir_data:
                period = fhir_data["effectivePeriod"]
                if "start" in period:
                    timestamp = datetime.fromisoformat(period["start"].replace("Z", "+00:00"))
            
            if not timestamp:
                timestamp = datetime.now()
            
            # Extract device information
            device = fhir_data.get("device", {})
            device_id = device.get("reference", "").split("/")[-1] if device.get("reference") else None
            
            # Extract metadata
            metadata = {
                "fhir_id": fhir_data.get("id"),
                "loinc_code": loinc_code,
                "status": fhir_data.get("status"),
                "interpretation": fhir_data.get("interpretation"),
                "reference_range": fhir_data.get("referenceRange"),
                "note": fhir_data.get("note"),
                "method": fhir_data.get("method"),
            }
            
            return {
                "patient_id": patient_id,
                "observation_type": observation_type,
                "value": float(value) if isinstance(value, (int, float)) else str(value),
                "unit": unit,
                "timestamp": timestamp,
                "source": "fhir_ehr",
                "device_id": device_id,
                "quality_score": 1.0,  # FHIR data assumed high quality
                "metadata": metadata,
                "fhir_resource_type": "Observation",
                "status": fhir_data.get("status"),
                "code_system": "LOINC",
                "code": loinc_code,
            }
        except Exception as e:
            logger.error(f"Error parsing FHIR observation: {e}")
            return None
    
    def parse_fhir_bundle(self, fhir_bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse FHIR Bundle containing multiple Observation resources
        Returns list of parsed observations
        """
        observations = []
        
        if fhir_bundle.get("resourceType") != "Bundle":
            logger.error(f"Invalid bundle resource type: {fhir_bundle.get('resourceType')}")
            return observations
        
        entries = fhir_bundle.get("entry", [])
        
        for entry in entries:
            resource = entry.get("resource", {})
            parsed = self.parse_fhir_observation(resource)
            if parsed:
                observations.append(parsed)
        
        logger.info(f"Parsed {len(observations)} observations from FHIR bundle")
        return observations
    
    def validate_fhir_observation(self, fhir_data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate FHIR Observation resource
        Returns (is_valid, error_messages)
        """
        errors = []
        
        # Check required fields
        if "resourceType" not in fhir_data:
            errors.append("Missing resourceType")
        elif fhir_data["resourceType"] != "Observation":
            errors.append(f"Invalid resourceType: {fhir_data['resourceType']}")
        
        if "status" not in fhir_data:
            errors.append("Missing status")
        
        if "code" not in fhir_data:
            errors.append("Missing code")
        elif "coding" not in fhir_data["code"] or not fhir_data["code"]["coding"]:
            errors.append("Missing code.coding")
        
        if "subject" not in fhir_data:
            errors.append("Missing subject")
        elif "reference" not in fhir_data["subject"]:
            errors.append("Missing subject.reference")
        
        # Check for value
        value_fields = [
            "valueQuantity", "valueCodeableConcept", "valueString",
            "valueBoolean", "valueInteger", "valueRange", "valueRatio",
            "valueSampledData", "valueTime", "valueDateTime", "valuePeriod"
        ]
        has_value = any(field in fhir_data for field in value_fields)
        if not has_value and "dataAbsentReason" not in fhir_data:
            errors.append("Missing value or dataAbsentReason")
        
        # Check for effective time
        if "effectiveDateTime" not in fhir_data and "effectivePeriod" not in fhir_data:
            errors.append("Missing effectiveDateTime or effectivePeriod")
        
        return len(errors) == 0, errors
    
    def create_fhir_observation(
        self,
        patient_id: str,
        observation_type: str,
        value: float,
        unit: str,
        timestamp: datetime,
        device_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create FHIR Observation resource from internal format
        Useful for exporting data to EHR systems
        """
        # Find LOINC code
        loinc_code = None
        for code, (obs_type, obs_unit) in self.observation_mapping.items():
            if obs_type == observation_type:
                loinc_code = code
                break
        
        if not loinc_code:
            loinc_code = "unknown"
        
        fhir_observation = {
            "resourceType": "Observation",
            "status": "final",
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                    "code": "vital-signs",
                    "display": "Vital Signs"
                }]
            }],
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": loinc_code,
                    "display": observation_type.replace("_", " ").title()
                }],
                "text": observation_type.replace("_", " ").title()
            },
            "subject": {
                "reference": f"Patient/{patient_id}"
            },
            "effectiveDateTime": timestamp.isoformat(),
            "valueQuantity": {
                "value": value,
                "unit": unit,
                "system": "http://unitsofmeasure.org",
                "code": unit
            }
        }
        
        if device_id:
            fhir_observation["device"] = {
                "reference": f"Device/{device_id}"
            }
        
        return fhir_observation


# Singleton instance
_fhir_ingest_service = None

def get_fhir_ingest_service() -> FHIRIngestService:
    """Get singleton FHIR ingest service instance"""
    global _fhir_ingest_service
    if _fhir_ingest_service is None:
        _fhir_ingest_service = FHIRIngestService()
    return _fhir_ingest_service
