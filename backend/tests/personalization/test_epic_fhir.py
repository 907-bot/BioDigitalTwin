"""
Tests for Epic SMART on FHIR integration.
"""

import time
import json
import pytest
import numpy as np

from app.personalization.epic_fhir import (
    SMARTEpicClient, SMARTRouter, FHIRCodeMapper,
    FHIRPatient, FHIRObservation, FHIRCommunicationRequest,
)


class TestFHIRCodeMapper:
    def test_glucose_mapping(self):
        obs = FHIRCodeMapper.twin_obs_to_fhir(
            patient_id="P001",
            obs_name="G",
            value=120.0,
            unit="mg/dL",
        )
        assert obs.resourceType == "Observation"
        assert obs.code["coding"][0]["code"] == "2339-0"
        assert obs.valueQuantity["value"] == 120.0
        assert obs.valueQuantity["unit"] == "mg/dL"

    def test_blood_pressure_mapping(self):
        obs = FHIRCodeMapper.twin_obs_to_fhir(
            patient_id="P001",
            obs_name="SBP",
            value=125.0,
            unit="mmHg",
        )
        assert obs.code["coding"][0]["code"] == "8480-6"

    def test_hba1c_mapping(self):
        obs = FHIRCodeMapper.twin_obs_to_fhir(
            patient_id="P001",
            obs_name="HbA1c",
            value=7.2,
            unit="%",
        )
        assert obs.code["coding"][0]["code"] == "4548-4"

    def test_unknown_obs_uses_unknown_code(self):
        obs = FHIRCodeMapper.twin_obs_to_fhir(
            patient_id="P001",
            obs_name="XYZ",
            value=1.0,
            unit="x",
        )
        assert obs.code["coding"][0]["code"] == "unknown"


class TestSMARTEpicClient:
    def test_smart_configuration(self):
        client = SMARTEpicClient()
        config = client.get_smart_configuration()
        assert "authorization_endpoint" in config
        assert "token_endpoint" in config
        assert "scopes_supported" in config
        assert "launch-standalone" in config["capabilities"]

    def test_pkce_generation(self):
        client = SMARTEpicClient()
        pkce = client.generate_pkce()
        assert "code_verifier" in pkce
        assert "code_challenge" in pkce
        assert pkce["code_challenge_method"] == "S256"
        assert len(pkce["code_verifier"]) >= 43
        assert len(pkce["code_challenge"]) >= 43

    def test_authorization_url_contains_pkce(self):
        client = SMARTEpicClient()
        pkce = client.generate_pkce()
        url = client.build_authorization_url(state="abc123", pkce=pkce)
        assert "code_challenge=" in url
        assert "code_challenge_method=S256" in url
        assert "state=abc123" in url
        assert "client_id=" in url

    def test_patient_fhir_request(self):
        client = SMARTEpicClient()
        url = client.build_patient_fhir_request("P123", resource_type="Patient")
        assert "Patient/P123" in url

    def test_observation_fhir_request(self):
        client = SMARTEpicClient()
        url = client.build_patient_fhir_request(
            "P123", resource_type="Observation", category="vital-signs"
        )
        assert "Observation" in url
        assert "patient=P123" in url
        assert "category=vital-signs" in url

    def test_communication_request_build(self):
        client = SMARTEpicClient()
        cr = client.build_communication_request(
            patient_id="P123",
            requester="dr_smith",
            recommendation_text="Reduce morning basal by 10%",
        )
        assert cr.resourceType == "CommunicationRequest"
        assert cr.status == "active"
        assert "Reduce morning basal" in cr.payload[0]["contentString"]


class TestSMARTRouter:
    def test_smart_configuration_endpoint(self):
        router = SMARTRouter()
        config = router.handle_smart_configuration()
        assert "authorization_endpoint" in config

    def test_metadata_endpoint(self):
        router = SMARTRouter()
        metadata = router.handle_metadata()
        assert metadata["resourceType"] == "CapabilityStatement"
        assert metadata["fhirVersion"] == "4.0.1"
        resource_types = [r["type"] for r in metadata["rest"][0]["resource"]]
        assert "Patient" in resource_types
        assert "Observation" in resource_types
        assert "CommunicationRequest" in resource_types

    def test_patient_endpoint(self):
        router = SMARTRouter()
        patient = router.handle_patient("P001")
        assert patient["resourceType"] == "Patient"
        assert patient["id"] == "P001"

    def test_communication_request_creation(self):
        router = SMARTRouter()
        cr = router.handle_communication_request_create(
            patient_id="P001",
            requester_id="dr_smith",
            recommendation="Increase insulin sensitivity",
        )
        assert cr["resourceType"] == "CommunicationRequest"
        assert cr["subject"]["reference"] == "Patient/P001"

    def test_launch_returns_redirect(self):
        router = SMARTRouter()
        result = router.handle_launch()
        assert "redirect_to" in result
        assert "state" in result
        assert "https://" in result["redirect_to"]

    def test_callback_with_invalid_state(self):
        router = SMARTRouter()
        result = router.handle_callback(code="xyz", state="invalid")
        assert "error" in result

    def test_callback_with_valid_state(self):
        router = SMARTRouter()
        launch = router.handle_launch()
        state = launch["state"]
        result = router.handle_callback(code="auth_code_123", state=state)
        assert result["status"] == "received"
        assert result["code"] == "auth_code_123"
