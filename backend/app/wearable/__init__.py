"""
Phase 17 — Wearable/EHR Integration
FHIR Observation + CGM webhook → /daily-checkin
Samsung Galaxy Watch 4 Classic integration
"""
from .fhir_ingest import FHIRIngestService, get_fhir_ingest_service
from .cgm_webhook import CGMWebhookService, get_cgm_webhook_service
from .samsung_health import SamsungHealthIntegration, get_samsung_health_integration
from .daily_checkin import DailyCheckinService, get_daily_checkin_service

__all__ = [
    "FHIRIngestService",
    "get_fhir_ingest_service",
    "CGMWebhookService",
    "get_cgm_webhook_service",
    "SamsungHealthIntegration",
    "get_samsung_health_integration",
    "DailyCheckinService",
    "get_daily_checkin_service",
]

