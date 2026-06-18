"""
CGM Webhook Service
Handles continuous glucose monitoring data from various CGM systems
Supports Dexcom, Abbott FreeStyle Libre, Medtronic, and generic CGM
"""
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator
import logging
import hashlib
import hmac
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CGMReading(BaseModel):
    """CGM glucose reading"""
    glucose_value: float = Field(..., description="Glucose value in mg/dL")
    timestamp: datetime = Field(..., description="Reading timestamp")
    trend: Optional[int] = Field(None, description="Trend arrow: 1=up, 2=up45, 3=flat, 4=down45, 5=down")
    trend_description: Optional[str] = Field(None, description="Trend description")
    device_id: Optional[str] = Field(None, description="CGM device identifier")
    sensor_age: Optional[int] = Field(None, description="Sensor age in hours")
    quality: Optional[str] = Field("good", description="Data quality: good, questionable, poor")
    units: str = Field("mg/dL", description="Units: mg/dL or mmol/L")


class CGMWebhookPayload(BaseModel):
    """CGM webhook payload structure"""
    patient_id: str
    device_type: str = Field(..., description="dexcom, freestyle, medtronic, generic")
    readings: List[CGMReading]
    webhook_timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CGMWebhookService:
    """
    CGM webhook service for real-time glucose data ingestion
    Supports multiple CGM manufacturers and provides data validation
    """
    
    def __init__(self):
        self.webhook_secret = os.getenv("CGM_WEBHOOK_SECRET", "")
        self.supported_devices = ["dexcom", "freestyle", "medtronic", "generic"]
        self.trend_map = {
            1: "Rising",
            2: "Rising Rapidly",
            3: "Flat",
            4: "Falling",
            5: "Falling Rapidly"
        }
    
    def validate_webhook_signature(
        self,
        payload: str,
        signature: str,
        secret: Optional[str] = None
    ) -> bool:
        """
        Validate webhook signature for security
        Uses HMAC-SHA256 for signature verification
        """
        if not self.webhook_secret and not secret:
            logger.warning("No webhook secret configured, skipping signature validation")
            return True
        
        secret_to_use = secret or self.webhook_secret
        expected_signature = hmac.new(
            secret_to_use.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(signature, expected_signature)
    
    def parse_dexcom_payload(self, payload: Dict[str, Any]) -> Optional[CGMWebhookPayload]:
        """
        Parse Dexcom G6/G7 webhook payload
        """
        try:
            patient_id = payload.get("patientId") or payload.get("subject", {}).get("reference", "").split("/")[-1]
            
            readings = []
            for record in payload.get("records", []):
                trend = record.get("trend")
                reading = CGMReading(
                    glucose_value=record.get("value") or record.get("glucoseValue"),
                    timestamp=datetime.fromisoformat(record.get("timestamp") or record.get("dateTime")),
                    trend=trend,
                    trend_description=self.trend_map.get(trend),
                    device_id=record.get("deviceId") or payload.get("deviceId"),
                    sensor_age=record.get("sensorAge"),
                    quality=record.get("quality", "good"),
                    units="mg/dL"
                )
                readings.append(reading)
            
            return CGMWebhookPayload(
                patient_id=patient_id,
                device_type="dexcom",
                readings=readings,
                metadata={"raw_payload": payload}
            )
        except Exception as e:
            logger.error(f"Error parsing Dexcom payload: {e}")
            return None
    
    def parse_freestyle_payload(self, payload: Dict[str, Any]) -> Optional[CGMWebhookPayload]:
        """
        Parse Abbott FreeStyle Libre webhook payload
        """
        try:
            patient_id = payload.get("patientId") or payload.get("userId")
            
            readings = []
            for record in payload.get("glucoseData", []):
                reading = CGMReading(
                    glucose_value=record.get("glucose") or record.get("value"),
                    timestamp=datetime.fromisoformat(record.get("timestamp")),
                    trend=record.get("trend"),
                    trend_description=record.get("trendText"),
                    device_id=record.get("sensorId") or payload.get("sensorId"),
                    sensor_age=record.get("sensorAge"),
                    quality=record.get("quality", "good"),
                    units="mg/dL"
                )
                readings.append(reading)
            
            return CGMWebhookPayload(
                patient_id=patient_id,
                device_type="freestyle",
                readings=readings,
                metadata={"raw_payload": payload}
            )
        except Exception as e:
            logger.error(f"Error parsing FreeStyle payload: {e}")
            return None
    
    def parse_medtronic_payload(self, payload: Dict[str, Any]) -> Optional[CGMWebhookPayload]:
        """
        Parse Medtronic CGM webhook payload
        """
        try:
            patient_id = payload.get("patientId")
            
            readings = []
            for record in payload.get("sgv", []):  # Sensor Glucose Values
                reading = CGMReading(
                    glucose_value=record.get("sgv") or record.get("value"),
                    timestamp=datetime.fromisoformat(record.get("timestamp") or record.get("dateString")),
                    trend=record.get("trend"),
                    trend_description=record.get("direction"),
                    device_id=record.get("device") or payload.get("device"),
                    quality=record.get("quality", "good"),
                    units="mg/dL"
                )
                readings.append(reading)
            
            return CGMWebhookPayload(
                patient_id=patient_id,
                device_type="medtronic",
                readings=readings,
                metadata={"raw_payload": payload}
            )
        except Exception as e:
            logger.error(f"Error parsing Medtronic payload: {e}")
            return None
    
    def parse_generic_payload(self, payload: Dict[str, Any]) -> Optional[CGMWebhookPayload]:
        """
        Parse generic CGM webhook payload
        """
        try:
            patient_id = payload.get("patient_id")
            
            readings = []
            glucose_data = payload.get("readings", payload.get("data", []))
            
            for record in glucose_data:
                reading = CGMReading(
                    glucose_value=record.get("glucose") or record.get("value"),
                    timestamp=datetime.fromisoformat(record.get("timestamp")),
                    trend=record.get("trend"),
                    trend_description=record.get("trend_description"),
                    device_id=record.get("device_id"),
                    sensor_age=record.get("sensor_age"),
                    quality=record.get("quality", "good"),
                    units=record.get("units", "mg/dL")
                )
                readings.append(reading)
            
            return CGMWebhookPayload(
                patient_id=patient_id,
                device_type="generic",
                readings=readings,
                metadata={"raw_payload": payload}
            )
        except Exception as e:
            logger.error(f"Error parsing generic payload: {e}")
            return None
    
    def parse_webhook_payload(
        self,
        payload: Dict[str, Any],
        device_type: Optional[str] = None
    ) -> Optional[CGMWebhookPayload]:
        """
        Parse webhook payload based on device type
        Auto-detects device type if not specified
        """
        if device_type:
            device_type = device_type.lower()
        else:
            # Auto-detect device type
            if "records" in payload and ("value" in payload.get("records", [{}])[0] or "glucoseValue" in payload.get("records", [{}])[0]):
                device_type = "dexcom"
            elif "glucoseData" in payload:
                device_type = "freestyle"
            elif "sgv" in payload:
                device_type = "medtronic"
            elif "readings" in payload or "data" in payload:
                device_type = "generic"
            else:
                logger.error("Unable to auto-detect device type")
                return None
        
        if device_type not in self.supported_devices:
            logger.error(f"Unsupported device type: {device_type}")
            return None
        
        parsers = {
            "dexcom": self.parse_dexcom_payload,
            "freestyle": self.parse_freestyle_payload,
            "medtronic": self.parse_medtronic_payload,
            "generic": self.parse_generic_payload
        }
        
        return parsers[device_type](payload)
    
    def validate_cgm_reading(self, reading: CGMReading) -> tuple[bool, List[str]]:
        """
        Validate CGM reading for data quality
        Returns (is_valid, error_messages)
        """
        errors = []
        
        # Check glucose value range (physiologically plausible)
        if reading.glucose_value < 20 or reading.glucose_value > 600:
            errors.append(f"Glucose value out of range: {reading.glucose_value}")
        
        # Check timestamp
        now = datetime.now()
        if reading.timestamp > now + timedelta(minutes=5):
            errors.append("Timestamp is in the future")
        if reading.timestamp < now - timedelta(days=30):
            errors.append("Timestamp is too old")
        
        # Check trend value
        if reading.trend is not None and reading.trend not in self.trend_map:
            errors.append(f"Invalid trend value: {reading.trend}")
        
        return len(errors) == 0, errors
    
    def convert_units(self, value: float, from_unit: str, to_unit: str) -> float:
        """
        Convert glucose units
        mg/dL to mmol/L: divide by 18.018
        mmol/L to mg/dL: multiply by 18.018
        """
        if from_unit == to_unit:
            return value
        
        if from_unit == "mg/dL" and to_unit == "mmol/L":
            return value / 18.018
        elif from_unit == "mmol/L" and to_unit == "mg/dL":
            return value * 18.018
        else:
            logger.warning(f"Unsupported unit conversion: {from_unit} to {to_unit}")
            return value
    
    def calculate_glucose_statistics(
        self,
        readings: List[CGMReading],
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Calculate glucose statistics from readings
        Includes mean, std, min, max, TIR (Time in Range), etc.
        """
        if not readings:
            return {}
        
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        recent_readings = [r for r in readings if r.timestamp > cutoff_time]
        
        if not recent_readings:
            return {}
        
        values = [r.glucose_value for r in recent_readings]
        
        stats = {
            "count": len(values),
            "mean": sum(values) / len(values),
            "std": (sum((x - sum(values)/len(values))**2 for x in values) / len(values))**0.5,
            "min": min(values),
            "max": max(values),
            "median": sorted(values)[len(values)//2],
        }
        
        # Time in Range (TIR) calculations
        # Target range: 70-180 mg/dL (3.9-10.0 mmol/L)
        in_range = sum(1 for v in values if 70 <= v <= 180)
        below_range = sum(1 for v in values if v < 70)
        above_range = sum(1 for v in values if v > 180)
        
        stats["time_in_range_percent"] = (in_range / len(values)) * 100
        stats["time_below_range_percent"] = (below_range / len(values)) * 100
        stats["time_above_range_percent"] = (above_range / len(values)) * 100
        
        # GV (Glycemic Variability) - CV
        if stats["mean"] > 0:
            stats["cv_percent"] = (stats["std"] / stats["mean"]) * 100
        
        return stats
    
    def detect_anomalies(
        self,
        readings: List[CGMReading],
        threshold_std: float = 3.0
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalous glucose readings
        Uses statistical outlier detection
        """
        if len(readings) < 10:
            return []
        
        values = [r.glucose_value for r in readings]
        mean = sum(values) / len(values)
        std = (sum((x - mean)**2 for x in values) / len(values))**0.5
        
        anomalies = []
        for reading in readings:
            z_score = abs(reading.glucose_value - mean) / std if std > 0 else 0
            if z_score > threshold_std:
                anomalies.append({
                    "reading": reading,
                    "z_score": z_score,
                    "reason": f"Z-score {z_score:.2f} exceeds threshold {threshold_std}"
                })
        
        return anomalies


# Singleton instance
_cgm_webhook_service = None

def get_cgm_webhook_service() -> CGMWebhookService:
    """Get singleton CGM webhook service instance"""
    global _cgm_webhook_service
    if _cgm_webhook_service is None:
        _cgm_webhook_service = CGMWebhookService()
    return _cgm_webhook_service
