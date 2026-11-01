"""
Phase 17 Router
Twin Persistence, Wearable Integration, MIMIC-IV, Clinical Study, FDA Documentation
"""
from fastapi import APIRouter, HTTPException, Query, Path, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, date

# Import Phase 17 modules
from app.twin_persistence import get_ukf_store, get_observation_history, create_dual_estimation_engine
from app.wearable import get_fhir_ingest_service, get_cgm_webhook_service, get_samsung_health_integration, get_daily_checkin_service
from app.mimic import get_mimic_connector
from app.clinical_study import get_clinical_study, get_patient_enrollment, get_clinical_endpoints, get_irb_protocol_manager
from app.regulatory_docs import get_q_submission_manager, get_iec_62304_manager, get_iso_14971_manager

router = APIRouter(prefix="/phase17", tags=["Phase 17"])


# =============================================================================
# Twin Persistence Endpoints
# =============================================================================

@router.get("/ukf-state/{patient_id}")
async def get_ukf_state(patient_id: str = Path(..., description="Patient ID")):
    """Get current UKF state for a patient"""
    store = get_ukf_store()
    state = await store.get_state(patient_id)
    if state is None:
        raise HTTPException(status_code=404, detail="UKF state not found")
    return state.model_dump()


@router.post("/ukf-state")
async def store_ukf_state(state_data: Dict[str, Any]):
    """Store UKF state for a patient"""
    from app.twin_persistence.ukf_store import UKFState
    store = get_ukf_store()
    state = UKFState(**state_data)
    success = await store.store_state(state)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to store UKF state")
    return {"status": "success", "patient_id": state.patient_id}


@router.get("/observations/{patient_id}")
async def get_observations(
    patient_id: str = Path(..., description="Patient ID"),
    observation_type: Optional[str] = Query(None),
    limit: int = Query(1000)
):
    """Get observations for a patient"""
    history = get_observation_history()
    observations = await history.get_observations(patient_id, observation_type, limit=limit)
    return {
        "patient_id": patient_id,
        "observation_type": observation_type,
        "count": len(observations),
        "observations": [obs.model_dump() for obs in observations]
    }


@router.post("/observations")
async def add_observation(obs_data: Dict[str, Any]):
    """Add a single observation"""
    from app.twin_persistence.observation_history import Observation
    history = get_observation_history()
    obs = Observation(**obs_data)
    success = await history.add_observation(obs)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to add observation")
    return {"status": "success"}


class DualEstimationUpdateRequest(BaseModel):
    patient_id: str
    glucose_measurement: float
    dt: float = 5.0


class EstimateParametersRequest(BaseModel):
    patient_id: str
    glucose_data: list[float]
    insulin_data: list[float]
    carb_data: list[float]
    timestamps: list[float]


@router.post("/dual-estimation/update")
async def update_dual_estimation(
    req: DualEstimationUpdateRequest
):
    """Update dual estimation with new glucose measurement"""
    engine = create_dual_estimation_engine()
    result = engine.update_ukf(req.glucose_measurement, req.dt)
    return result


@router.post("/dual-estimation/estimate-parameters")
async def estimate_parameters(
    req: EstimateParametersRequest
):
    """Estimate physiological parameters using MCMC"""
    engine = create_dual_estimation_engine()
    result = await engine.estimate_parameters_mcmc(
        np.array(req.glucose_data),
        np.array(req.insulin_data),
        np.array(req.carb_data),
        np.array(req.timestamps)
    )
    return result


# =============================================================================
# Wearable/EHR Ingest Endpoints
# =============================================================================

@router.post("/fhir/ingest")
async def ingest_fhir_observation(fhir_data: Dict[str, Any]):
    """Ingest FHIR Observation resource"""
    service = get_fhir_ingest_service()
    parsed = service.parse_fhir_observation(fhir_data)
    if parsed is None:
        raise HTTPException(status_code=400, detail="Failed to parse FHIR observation")
    
    history = get_observation_history()
    from app.twin_persistence.observation_history import Observation
    obs = Observation(**parsed)
    success = await history.add_observation(obs)
    
    return {
        "status": "success" if success else "failed",
        "parsed_observation": parsed
    }


@router.post("/fhir/batch")
async def ingest_fhir_bundle(fhir_bundle: Dict[str, Any]):
    """Ingest FHIR Bundle with multiple observations"""
    service = get_fhir_ingest_service()
    parsed_observations = service.parse_fhir_bundle(fhir_bundle)
    
    history = get_observation_history()
    from app.twin_persistence.observation_history import Observation
    obs_objects = [Observation(**obs) for obs in parsed_observations]
    
    success_count = await history.add_observations_batch(obs_objects)
    
    return {
        "status": "success",
        "total_parsed": len(parsed_observations),
        "successfully_added": success_count
    }


@router.post("/cgm/webhook")
async def cgm_webhook(payload: Dict[str, Any], device_type: Optional[str] = None):
    """Process CGM webhook payload"""
    service = get_cgm_webhook_service()
    parsed = service.parse_webhook_payload(payload, device_type)
    
    if parsed is None:
        raise HTTPException(status_code=400, detail="Failed to parse CGM payload")
    
    # Store observations
    history = get_observation_history()
    from app.twin_persistence.observation_history import Observation
    from app.wearable.cgm_webhook import CGMReading
    
    success_count = 0
    for reading in parsed.readings:
        obs = Observation(
            patient_id=parsed.patient_id,
            observation_type="glucose",
            value=reading.glucose_value,
            unit=reading.units,
            timestamp=reading.timestamp,
            source=f"cgm_{parsed.device_type}",
            device_id=reading.device_id,
            quality_score=1.0 if reading.quality == "good" else 0.8,
            metadata={"trend": reading.trend, "trend_description": reading.trend_description}
        )
        if await history.add_observation(obs):
            success_count += 1
    
    return {
        "status": "success",
        "device_type": parsed.device_type,
        "readings_received": len(parsed.readings),
        "successfully_stored": success_count
    }


@router.post("/samsung-health/connect")
async def connect_samsung_watch(device_address: str):
    """Connect to Samsung Galaxy Watch via Bluetooth"""
    integration = get_samsung_health_integration()
    success = await integration.connect_bluetooth(device_address)
    return {
        "status": "connected" if success else "failed",
        "device_address": device_address
    }


@router.post("/samsung-health/disconnect")
async def disconnect_samsung_watch():
    """Disconnect from Samsung Galaxy Watch"""
    integration = get_samsung_health_integration()
    success = await integration.disconnect_bluetooth()
    return {"status": "disconnected" if success else "failed"}


@router.post("/samsung-health/ingest")
async def ingest_samsung_health_data(data: Dict[str, Any], patient_id: str):
    """Ingest Samsung Health export data"""
    integration = get_samsung_health_integration()
    health_data = integration.parse_samsung_health_export(data)
    
    history = get_observation_history()
    from app.twin_persistence.observation_history import Observation
    
    success_count = 0
    for health_point in health_data:
        obs_dict = integration.convert_to_observation_format(health_point, patient_id)
        obs = Observation(**obs_dict)
        if await history.add_observation(obs):
            success_count += 1
    
    return {
        "status": "success",
        "data_points_received": len(health_data),
        "successfully_stored": success_count
    }


@router.post("/samsung-health/sync-to-twin")
async def sync_samsung_health_to_twin(data: Dict[str, Any], patient_id: str):
    """Sync Samsung Health watch data directly into the initialized Personal Digital Twin."""
    integration = get_samsung_health_integration()
    health_data = integration.parse_samsung_health_export(data)
    
    from app.personalization.router import _engines
    eng = _engines.get(patient_id)
    if eng is None:
        raise HTTPException(status_code=404, detail=f"Digital Twin for patient '{patient_id}' not initialized. Call /personalization/v2/seed-real-twin first.")

    latest_values = {}
    for hp in health_data:
        latest_values[hp.data_type] = hp.value

    # Perform UKF state update with synced metrics
    if "heart_rate" in latest_values or "systolic_bp" in latest_values:
        obs = np.array([
            latest_values.get("glucose", 95.0),
            latest_values.get("systolic_bp", 120.0),
            latest_values.get("diastolic_bp", 80.0),
            latest_values.get("heart_rate", 70.0),
            latest_values.get("hrv", 45.0),
            latest_values.get("gfr", 100.0),
            140.0, 4.2, 290.0,
            latest_values.get("ffa", 0.5),
            latest_values.get("ldl", 100.0),
            latest_values.get("hdl", 50.0),
            latest_values.get("tg", 120.0),
            latest_values.get("cortisol", 350.0),
            latest_values.get("sleep_pressure", 0.3)
        ])
        eng.update(obs)

    twin_state = eng.get_twin_state()
    return {
        "status": "synced",
        "patient_id": patient_id,
        "synced_metrics_count": len(latest_values),
        "synced_metrics": latest_values,
        "updated_twin_summary": {
            "resting_hr": float(twin_state[7]),
            "systolic_bp": float(twin_state[5]),
            "muscle_mass_kg": float(twin_state[30]) if len(twin_state) > 30 else 30.0,
            "bmd_g_cm2": float(twin_state[33]) if len(twin_state) > 33 else 1.2,
        }
    }


@router.get("/daily-checkin/{patient_id}")
async def get_daily_checkin(
    patient_id: str = Path(..., description="Patient ID"),
    checkin_date: Optional[date] = None
):
    """Get daily checkin summary for a patient"""
    service = get_daily_checkin_service()
    summary = await service.generate_daily_checkin(patient_id, checkin_date)
    return summary.model_dump()


# =============================================================================
# MIMIC-IV Integration Endpoints
# =============================================================================

@router.get("/mimic/connect")
async def connect_mimic(database_url: Optional[str] = None):
    """Connect to MIMIC-IV database"""
    connector = get_mimic_connector()
    success = await connector.connect(database_url)
    return {"status": "connected" if success else "failed"}


@router.get("/mimic/summary")
async def get_mimic_summary():
    """Get MIMIC-IV database summary"""
    connector = get_mimic_connector()
    summary = await connector.get_mimic_summary_stats()
    return summary


@router.post("/mimic/validate")
async def run_mimic_validation(
    synthetic_data: Dict[str, Any],
    validation_id: Optional[str] = None
):
    """Run external validation against MIMIC-IV"""
    connector = get_mimic_connector()
    df = pd.DataFrame(synthetic_data)
    result = await connector.run_external_validation(df, validation_id)
    return result.model_dump()


# =============================================================================
# Clinical Study Endpoints
# =============================================================================

@router.post("/clinical-study/initialize")
async def initialize_clinical_study(start_date: Optional[date] = None):
    """Initialize clinical study BDT-001"""
    study = get_clinical_study()
    result = study.initialize_study(start_date)
    return result


@router.get("/clinical-study/summary")
async def get_study_summary():
    """Get clinical study summary"""
    study = get_clinical_study()
    return study.get_study_summary()


@router.get("/clinical-study/timeline")
async def get_study_timeline(patient_id: Optional[str] = None):
    """Get study timeline"""
    study = get_clinical_study()
    return study.get_study_timeline(patient_id)


@router.get("/clinical-study/inclusion-criteria")
async def get_inclusion_criteria():
    """Get study inclusion criteria"""
    study = get_clinical_study()
    return {"inclusion_criteria": study.get_inclusion_criteria()}


@router.get("/clinical-study/exclusion-criteria")
async def get_exclusion_criteria():
    """Get study exclusion criteria"""
    study = get_clinical_study()
    return {"exclusion_criteria": study.get_exclusion_criteria()}


@router.post("/clinical-study/screen")
async def screen_patient(screening_data: Dict[str, Any]):
    """Screen patient for eligibility"""
    from app.clinical_study.patient_enrollment import ScreeningData
    enrollment = get_patient_enrollment()
    screening = ScreeningData(**screening_data)
    result = enrollment.screen_patient(screening)
    return result


@router.post("/clinical-study/enroll")
async def enroll_patient(patient_id: str):
    """Enroll eligible patient"""
    enrollment = get_patient_enrollment()
    patient = enrollment.enroll_patient(patient_id)
    if patient is None:
        raise HTTPException(status_code=400, detail="Failed to enroll patient")
    return patient.model_dump()


@router.get("/clinical-study/enrollment-summary")
async def get_enrollment_summary():
    """Get enrollment summary"""
    enrollment = get_patient_enrollment()
    return enrollment.get_enrollment_summary()


class PatientPhaseUpdateRequest(BaseModel):
    patient_id: str
    new_phase: str


@router.post("/clinical-study/phase")
async def update_patient_phase(req: PatientPhaseUpdateRequest):
    """Update patient's current study phase"""
    from app.clinical_study.study_design import StudyPhase
    enrollment = get_patient_enrollment()
    success = enrollment.update_patient_phase(req.patient_id, StudyPhase(req.new_phase))
    return {"status": "success" if success else "failed"}


class AdverseEventRequest(BaseModel):
    patient_id: str
    event_type: str
    severity: str
    description: str
    related_to_intervention: bool = False


@router.post("/clinical-study/adverse-event")
async def record_adverse_event(req: AdverseEventRequest):
    """Record adverse event"""
    enrollment = get_patient_enrollment()
    success = enrollment.record_adverse_event(
        req.patient_id, req.event_type, req.severity, req.description, req.related_to_intervention
    )
    return {"status": "success" if success else "failed"}


class WithdrawPatientRequest(BaseModel):
    patient_id: str
    reason: str


@router.post("/clinical-study/withdraw")
async def withdraw_patient(req: WithdrawPatientRequest):
    """Withdraw patient from study"""
    enrollment = get_patient_enrollment()
    success = enrollment.withdraw_patient(req.patient_id, req.reason)
    return {"status": "success" if success else "failed"}


class CalculateTIRRequest(BaseModel):
    patient_id: str
    intervention: str
    glucose_values: list
    timestamps: list


@router.post("/clinical-study/calculate-tir")
async def calculate_tir(req: CalculateTIRRequest):
    """Calculate Time in Range (primary endpoint)"""
    endpoints = get_clinical_endpoints()
    timestamps_dt = [datetime.fromisoformat(ts) for ts in req.timestamps]
    result = endpoints.calculate_tir(req.glucose_values, timestamps_dt)
    result.patient_id = req.patient_id
    result.intervention = req.intervention
    return result.model_dump()


@router.get("/clinical-study/irb-protocol")
async def get_irb_protocol():
    """Get IRB protocol document"""
    manager = get_irb_protocol_manager()
    return manager.generate_protocol_document()


@router.get("/clinical-study/consent-form")
async def get_consent_form():
    """Get informed consent form template"""
    manager = get_irb_protocol_manager()
    return {"consent_form": manager.generate_consent_form_template()}


# =============================================================================
# FDA Regulatory Documentation Endpoints
# =============================================================================

@router.get("/regulatory/q-submission")
async def get_q_submission():
    """Get FDA Q-Submission package"""
    manager = get_q_submission_manager()
    return manager.generate_q_submission_package()


@router.get("/regulatory/q-submission/cover-letter")
async def get_cover_letter():
    """Get FDA cover letter"""
    manager = get_q_submission_manager()
    return {"cover_letter": manager.generate_cover_letter()}


@router.get("/regulatory/q-submission/device-description")
async def get_device_description():
    """Get device description"""
    manager = get_q_submission_manager()
    return manager.generate_device_description()


class FDACommunicationRequest(BaseModel):
    date: date
    communication_type: str
    summary: str
    fda_contact: Optional[str] = None


@router.post("/regulatory/q-submission/communication")
async def record_fda_communication(req: FDACommunicationRequest):
    """Record FDA communication"""
    manager = get_q_submission_manager()
    comm = manager.record_fda_communication(req.date, req.communication_type, req.summary, req.fda_contact)
    return comm


@router.get("/regulatory/iec-62304")
async def get_iec_62304_documentation():
    """Get IEC 62304 compliance documentation"""
    manager = get_iec_62304_manager()
    manager.generate_standard_processes()
    manager.generate_traceability_matrix()
    return manager.generate_compliance_package()


@router.get("/regulatory/iec-62304/software-units")
async def get_software_units():
    """Get software unit list"""
    manager = get_iec_62304_manager()
    return {"software_units": manager.generate_software_unit_list()}


@router.get("/regulatory/iso-14971")
async def get_iso_14971_documentation():
    """Get ISO 14971 risk management documentation"""
    manager = get_iso_14971_manager()
    manager.generate_standard_hazards()
    manager.generate_risk_acceptance_criteria()
    return manager.generate_risk_management_file()


@router.get("/regulatory/iso-14971/risk-summary")
async def get_risk_summary():
    """Get risk summary"""
    manager = get_iso_14971_manager()
    return manager.generate_risk_summary()


class PostProductionEventRequest(BaseModel):
    event_id: str
    event_type: str
    description: str
    severity: str
    date: Optional[date] = None
    corrective_action: Optional[str] = None


@router.post("/regulatory/iso-14971/post-production-event")
async def record_post_production_event(req: PostProductionEventRequest):
    """Record post-production safety event"""
    manager = get_iso_14971_manager()
    event = manager.record_post_production_event(
        req.event_id, req.event_type, req.description, req.severity, req.date, req.corrective_action
    )
    return event
