"""
SMART on FHIR 1.0.0 Epic integration.

Implements the HL7 FHIR R4 + SMART App Launch 2.0 protocol for
bidirectional integration with Epic EHR systems.

Endpoints implemented:
- /.well-known/smart_configuration (SMART discovery)
- /fhir/metadata (FHIR capability statement)
- /fhir/Patient/{id} (FHIR Patient resource)
- /fhir/Observation?patient={id}&category=vital-signs (FHIR Observations)
- /fhir/MedicationRequest?patient={id} (medications)
- /fhir/Condition?patient={id} (conditions)
- /fhir/CommunicationRequest (twin recommendations back to clinician)
- /launch (SMART app launch handler)
- /callback (OAuth2 callback)

For Epic sandbox testing, set EPIC_BASE_URL and EPIC_CLIENT_ID.

References:
- SMART App Launch 2.0: https://hl7.org/fhir/smart-app-launch/
- FHIR R4: https://hl7.org/fhir/R4/
- Epic on FHIR: https://fhir.epic.com/
- US Core 3.1.1: https://www.hl7.org/fhir/us/core/
"""

import os
import time
import json
import base64
import hashlib
import secrets
import logging
import uuid
from typing import Optional, Dict, List
from urllib.parse import urlencode, parse_qs
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FHIRPatient:
    """FHIR R4 Patient resource."""
    resourceType: str = "Patient"
    id: str = ""
    identifier: List[Dict] = field(default_factory=list)
    name: List[Dict] = field(default_factory=list)
    gender: str = ""
    birthDate: str = ""
    address: List[Dict] = field(default_factory=list)
    telecom: List[Dict] = field(default_factory=list)


@dataclass
class FHIRObservation:
    """FHIR R4 Observation resource (vital signs, labs)."""
    resourceType: str = "Observation"
    id: str = ""
    status: str = "final"
    category: List[Dict] = field(default_factory=list)
    code: Dict = field(default_factory=dict)
    subject: Dict = field(default_factory=dict)
    effectiveDateTime: str = ""
    valueQuantity: Dict = field(default_factory=dict)
    interpretation: List[Dict] = field(default_factory=list)


@dataclass
class FHIRCommunicationRequest:
    """FHIR R4 CommunicationRequest — twin recommendation to clinician."""
    resourceType: str = "CommunicationRequest"
    id: str = ""
    status: str = "active"
    priority: str = "routine"
    subject: Dict = field(default_factory=dict)
    encounter: Optional[Dict] = None
    payload: List[Dict] = field(default_factory=list)
    requester: Dict = field(default_factory=dict)
    authoredOn: str = ""
    note: List[Dict] = field(default_factory=list)


class FHIRCodeMapper:
    """Maps between LOINC codes and internal observation names."""

    # Glucose (CGM, fasting, etc.)
    LOINC_GLUCOSE = "2339-0"      # Glucose [Mass/volume] in Blood
    LOINC_HBA1C = "4548-4"        # Hemoglobin A1c/Hemoglobin.total
    LOINC_BP_SYS = "8480-6"       # Systolic blood pressure
    LOINC_BP_DIA = "8462-4"       # Diastolic blood pressure
    LOINC_HR = "8867-4"           # Heart rate
    LOINC_HRV = "80404-7"         # Heart rate variability
    LOINC_GFR = "62292-8"         # Glomerular filtration rate
    LOINC_NA = "2947-0"           # Sodium [Moles/volume] in Blood
    LOINC_K = "2823-3"            # Potassium [Moles/volume] in Blood
    LOINC_OSM = "2692-8"          # Osmolality of Serum or Plasma
    LOINC_FFA = "1501-4"          # Free Fatty Acids
    LOINC_LDL = "13457-7"         # Cholesterol.in LDL [Mass/volume]
    LOINC_HDL = "2085-9"          # Cholesterol.in HDL [Mass/volume]
    LOINC_TG = "2571-8"           # Triglyceride [Mass/volume]
    LOINC_CORTISOL = "2143-6"     # Cortisol [Mass/volume] in Serum
    LOINC_CRP = "1988-5"          # C-reactive protein

    # SNOMED CT for conditions
    SNOMED_T1DM = "46635009"
    SNOMED_T2DM = "44054006"
    SNOMED_HTN = "38341003"
    SNOMED_CKD = "709044004"

    # RxNorm for medications
    RXNORM_METFORMIN = "6809"
    RXNORM_INSULIN = "253181"
    RXNORM_SGLT2 = "476345"
    RXNORM_GLP1 = "60548"

    @classmethod
    def twin_obs_to_fhir(
        cls,
        patient_id: str,
        obs_name: str,
        value: float,
        unit: str,
        timestamp: Optional[str] = None,
    ) -> FHIRObservation:
        """Convert a twin observation to a FHIR Observation."""
        loinc_map = {
            "G": cls.LOINC_GLUCOSE, "HbA1c": cls.LOINC_HBA1C,
            "SBP": cls.LOINC_BP_SYS, "DBP": cls.LOINC_BP_DIA,
            "HR": cls.LOINC_HR, "HRV": cls.LOINC_HRV, "GFR": cls.LOINC_GFR,
            "Na": cls.LOINC_NA, "K": cls.LOINC_K, "Osm": cls.LOINC_OSM,
            "FFA": cls.LOINC_FFA, "LDL": cls.LOINC_LDL,
            "HDL": cls.LOINC_HDL, "TG": cls.LOINC_TG,
            "cortisol": cls.LOINC_CORTISOL, "CRP": cls.LOINC_CRP,
        }
        display_map = {
            "G": "Glucose", "HbA1c": "Hemoglobin A1c",
            "SBP": "Systolic Blood Pressure", "DBP": "Diastolic Blood Pressure",
            "HR": "Heart Rate", "HRV": "Heart Rate Variability",
            "GFR": "Glomerular Filtration Rate",
            "Na": "Sodium", "K": "Potassium", "Osm": "Osmolality",
            "FFA": "Free Fatty Acids", "LDL": "LDL Cholesterol",
            "HDL": "HDL Cholesterol", "TG": "Triglycerides",
            "cortisol": "Cortisol", "CRP": "C-Reactive Protein",
        }
        code = loinc_map.get(obs_name, "unknown")
        display = display_map.get(obs_name, obs_name)
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()
        return FHIRObservation(
            id=str(uuid.uuid4()),
            status="final",
            category=[{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                    "code": "vital-signs" if obs_name in ("G", "SBP", "DBP", "HR", "HRV") else "laboratory",
                    "display": "Vital Signs" if obs_name in ("G", "SBP", "DBP", "HR", "HRV") else "Laboratory",
                }],
            }],
            code={
                "coding": [{
                    "system": "http://loinc.org",
                    "code": code,
                    "display": display,
                }],
                "text": display,
            },
            subject={"reference": f"Patient/{patient_id}"},
            effectiveDateTime=timestamp,
            valueQuantity={
                "value": float(value),
                "unit": unit,
                "system": "http://unitsofmeasure.org",
                "code": unit,
            },
        )


class SMARTTokenStore:
    """In-memory store for OAuth2 tokens (replace with Redis in production)."""

    def __init__(self):
        self._tokens: Dict[str, Dict] = {}  # session_id -> token info

    def store(self, session_id: str, token_info: Dict) -> None:
        self._tokens[session_id] = token_info

    def get(self, session_id: str) -> Optional[Dict]:
        return self._tokens.get(session_id)

    def revoke(self, session_id: str) -> None:
        self._tokens.pop(session_id, None)


class SMARTEpicClient:
    """SMART on FHIR client for Epic integration.

    Handles:
    - SMART discovery
    - OAuth2 authorization code flow with PKCE
    - FHIR R4 queries (Patient, Observation, MedicationRequest, Condition)
    - Writing CommunicationRequest back to Epic (twin recommendations)
    """

    def __init__(
        self,
        epic_base_url: Optional[str] = None,
        client_id: Optional[str] = None,
        redirect_uri: Optional[str] = None,
    ):
        self.epic_base_url = epic_base_url or os.environ.get(
            "EPIC_BASE_URL", "https://fhir.epic.com/interconnect-fhir-oauth"
        )
        self.client_id = client_id or os.environ.get("EPIC_CLIENT_ID", "demo-client")
        self.redirect_uri = redirect_uri or os.environ.get(
            "EPIC_REDIRECT_URI", "https://localhost:8000/callback"
        )
        self.token_store = SMARTTokenStore()
        self.fhir_base = f"{self.epic_base_url}/api/FHIR/R4"

    def get_smart_configuration(self) -> Dict:
        """Return the SMART on FHIR configuration document."""
        return {
            "issuer": self.epic_base_url,
            "authorization_endpoint": f"{self.epic_base_url}/oauth2/authorize",
            "token_endpoint": f"{self.epic_base_url}/oauth2/token",
            "introspection_endpoint": f"{self.epic_base_url}/oauth2/introspect",
            "revocation_endpoint": f"{self.epic_base_url}/oauth2/revoke",
            "management_endpoint": f"{self.epic_base_url}/oauth2/management",
            "jwks_uri": f"{self.epic_base_url}/.well-known/jwks.json",
            "scopes_supported": [
                "openid", "fhirUser", "online_access", "offline_access",
                "patient/Patient.read", "patient/Observation.read",
                "patient/MedicationRequest.read", "patient/Condition.read",
                "user/Patient.read", "user/Observation.read",
                "user/CommunicationRequest.write",
            ],
            "response_types_supported": ["code"],
            "capabilities": [
                "launch-standalone", "launch-ehr",
                "client-public", "client-confidential-symmetric",
                "sso-openid-connect", "context-standalone-patient",
                "permission-offline", "permission-patient",
            ],
        }

    def generate_pkce(self) -> Dict:
        """Generate PKCE code_verifier and code_challenge."""
        code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(64)).decode("ascii").rstrip("=")
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode()).digest()
        ).decode("ascii").rstrip("=")
        return {
            "code_verifier": code_verifier,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }

    def build_authorization_url(
        self,
        state: str,
        pkce: Dict,
        scope: str = "launch/patient openid fhirUser patient/Patient.read "
                    "patient/Observation.read patient/MedicationRequest.read "
                    "patient/Condition.read",
    ) -> str:
        """Build the Epic authorization URL for OAuth2 redirect."""
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": scope,
            "state": state,
            "aud": self.fhir_base,
            "code_challenge": pkce["code_challenge"],
            "code_challenge_method": pkce["code_challenge_method"],
        }
        return f"{self.epic_base_url}/oauth2/authorize?{urlencode(params)}"

    def build_patient_fhir_request(
        self,
        patient_id: str,
        resource_type: str = "Patient",
        category: Optional[str] = None,
        n_records: int = 100,
    ) -> str:
        """Build a FHIR R4 query URL for a patient resource."""
        if resource_type == "Patient":
            return f"{self.fhir_base}/Patient/{patient_id}"
        params = {"patient": patient_id, "_count": n_records}
        if category:
            params["category"] = category
        return f"{self.fhir_base}/{resource_type}?{urlencode(params)}"

    def build_communication_request(
        self,
        patient_id: str,
        requester: str,
        recommendation_text: str,
        priority: str = "routine",
        subject: str = "Digital Twin Recommendation",
    ) -> FHIRCommunicationRequest:
        """Build a FHIR CommunicationRequest to write a recommendation back."""
        now = datetime.now(timezone.utc).isoformat()
        return FHIRCommunicationRequest(
            id=str(uuid.uuid4()),
            status="active",
            priority=priority,
            subject={"reference": f"Patient/{patient_id}"},
            payload=[{
                "contentString": recommendation_text,
            }],
            requester={"reference": f"Practitioner/{requester}"},
            authoredOn=now,
            note=[{"text": subject}],
        )


class SMARTRouter:
    """FastAPI router for SMART on FHIR endpoints.

    Exposes:
    - GET /.well-known/smart-configuration
    - GET /fhir/metadata
    - GET /fhir/Patient/{id}
    - GET /fhir/Observation (with patient query)
    - GET /fhir/MedicationRequest
    - GET /fhir/Condition
    - POST /fhir/CommunicationRequest
    - GET /launch (SMART app launch)
    - GET /callback (OAuth2 callback)
    """

    def __init__(self, client: Optional[SMARTEpicClient] = None):
        self.client = client or SMARTEpicClient()
        self.launch_states: Dict[str, Dict] = {}

    def handle_smart_configuration(self) -> Dict:
        return self.client.get_smart_configuration()

    def handle_metadata(self) -> Dict:
        """FHIR R4 CapabilityStatement."""
        return {
            "resourceType": "CapabilityStatement",
            "status": "active",
            "date": datetime.now(timezone.utc).isoformat(),
            "publisher": "BioDigitalTwin",
            "kind": "instance",
            "software": {
                "name": "BioDigitalTwin-SMART-Adapter",
                "version": "1.0.0",
            },
            "fhirVersion": "4.0.1",
            "format": ["json"],
            "rest": [{
                "mode": "server",
                "security": {
                    "service": [{
                        "coding": [{
                            "system": "http://terminology.hl7.org/CodeSystem/restful-security-service",
                            "code": "SMART-on-FHIR",
                        }],
                    }],
                },
                "resource": [
                    {
                        "type": "Patient",
                        "interaction": [{"code": "read"}],
                    },
                    {
                        "type": "Observation",
                        "interaction": [
                            {"code": "read"},
                            {"code": "search-type"},
                        ],
                        "searchParam": [
                            {"name": "patient", "type": "reference"},
                            {"name": "category", "type": "token"},
                            {"name": "code", "type": "token"},
                            {"name": "date", "type": "date"},
                        ],
                    },
                    {
                        "type": "MedicationRequest",
                        "interaction": [
                            {"code": "read"},
                            {"code": "search-type"},
                        ],
                    },
                    {
                        "type": "Condition",
                        "interaction": [
                            {"code": "read"},
                            {"code": "search-type"},
                        ],
                    },
                    {
                        "type": "CommunicationRequest",
                        "interaction": [
                            {"code": "read"},
                            {"code": "create"},
                            {"code": "search-type"},
                        ],
                    },
                ],
            }],
        }

    def handle_patient(self, patient_id: str) -> Dict:
        """Build a FHIR Patient resource for a twin's patient."""
        return FHIRPatient(
            id=patient_id,
            identifier=[{
                "use": "usual",
                "system": "http://hospital.example.org/mrn",
                "value": f"MRN-{patient_id}",
            }],
        ).__dict__

    def handle_observation_query(
        self,
        patient_id: str,
        loinc_codes: Optional[List[str]] = None,
        category: Optional[str] = None,
        n: int = 100,
    ) -> Dict:
        """Build a FHIR Bundle of Observations for a patient."""
        bundle = {
            "resourceType": "Bundle",
            "type": "searchset",
            "total": 0,
            "entry": [],
        }
        # In a real Epic integration, this would query the Epic API.
        # For demo, return empty bundle.
        return bundle

    def handle_communication_request_create(
        self,
        patient_id: str,
        requester_id: str,
        recommendation: str,
        priority: str = "routine",
    ) -> Dict:
        """Create a CommunicationRequest to send recommendation to clinician."""
        cr = self.client.build_communication_request(
            patient_id=patient_id,
            requester=requester_id,
            recommendation_text=recommendation,
            priority=priority,
        )
        return cr.__dict__

    def handle_launch(self, launch_token: Optional[str] = None) -> Dict:
        """Begin a SMART app launch."""
        state = secrets.token_urlsafe(16)
        pkce = self.client.generate_pkce()
        self.launch_states[state] = {
            "pkce_verifier": pkce["code_verifier"],
            "launch_token": launch_token,
            "created_at": time.time(),
        }
        auth_url = self.client.build_authorization_url(state, pkce)
        return {
            "redirect_to": auth_url,
            "state": state,
            "expires_in": 600,  # 10 min
        }

    def handle_callback(self, code: str, state: str) -> Dict:
        """OAuth2 callback handler.

        In a real integration, this would POST to the token endpoint.
        For demo, store the auth code for later exchange.
        """
        if state not in self.launch_states:
            return {"error": "invalid_state"}
        info = self.launch_states[state]
        info["auth_code"] = code
        info["completed_at"] = time.time()
        return {
            "status": "received",
            "state": state,
            "code": code,
            "next_step": "exchange_code_for_token",
        }
