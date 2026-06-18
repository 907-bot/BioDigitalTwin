"""
Samsung Health Integration
Samsung Galaxy Watch 4 Classic integration for wearable data
Supports health data sync via Samsung Health API and Bluetooth LE
"""
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
import logging
import asyncio
import os

try:
    import bleak
    BLEAK_AVAILABLE = True
except ImportError:
    BLEAK_AVAILABLE = False
    logging.warning("bleak not available, Bluetooth LE features disabled")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SamsungHealthData(BaseModel):
    """Samsung health data point"""
    data_type: str = Field(..., description="heart_rate, steps, sleep, stress, spo2, etc.")
    value: float
    unit: str
    timestamp: datetime
    device_id: str
    device_model: str = "Galaxy Watch 4 Classic"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SamsungHealthIntegration:
    """
    Samsung Health integration for Galaxy Watch 4 Classic
    Supports:
    - Bluetooth LE connection to watch
    - Health data synchronization
    - Real-time monitoring
    - Historical data retrieval
    """
    
    # Samsung Galaxy Watch 4 Classic Bluetooth service UUIDs
    SAMSUNG_SERVICE_UUID = "00001800-0000-1000-8000-00805f9b34fb"
    HEART_RATE_SERVICE_UUID = "0000180d-0000-1000-8000-00805f9b34fb"
    HEART_RATE_MEASUREMENT_UUID = "00002a37-0000-1000-8000-00805f9b34fb"
    
    def __init__(self):
        self.device_address = os.getenv("SAMSUNG_WATCH_ADDRESS", "")
        self.client = None
        self.connected = False
        self.data_buffer = []
        self.sync_enabled = False
        
    async def connect_bluetooth(self, device_address: Optional[str] = None) -> bool:
        """
        Connect to Samsung Galaxy Watch via Bluetooth LE
        Returns True if connection successful
        """
        if not BLEAK_AVAILABLE:
            logger.error("bleak not available, cannot connect via Bluetooth")
            return False
        
        address = device_address or self.device_address
        if not address:
            logger.error("No device address provided")
            return False
        
        try:
            self.client = bleak.BleakClient(address)
            await self.client.connect()
            self.connected = True
            logger.info(f"Connected to Samsung Watch at {address}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Samsung Watch: {e}")
            return False
    
    async def disconnect_bluetooth(self) -> bool:
        """Disconnect from Samsung Watch"""
        if self.client and self.connected:
            try:
                await self.client.disconnect()
                self.connected = False
                logger.info("Disconnected from Samsung Watch")
                return True
            except Exception as e:
                logger.error(f"Error disconnecting: {e}")
                return False
        return True
    
    async def start_heart_rate_monitoring(self, callback=None) -> bool:
        """
        Start real-time heart rate monitoring from Samsung Watch
        """
        if not self.connected or not self.client:
            logger.error("Not connected to Samsung Watch")
            return False
        
        try:
            # Subscribe to heart rate notifications
            def notification_handler(sender, data):
                # Parse heart rate data (standard Bluetooth Heart Rate format)
                if len(data) >= 2:
                    flags = data[0]
                    heart_rate = data[1]
                    if flags & 0x01:  # 16-bit heart rate value
                        heart_rate = (data[1] << 8) | data[2]
                    
                    health_data = SamsungHealthData(
                        data_type="heart_rate",
                        value=float(heart_rate),
                        unit="bpm",
                        timestamp=datetime.now(),
                        device_id=self.device_address,
                        device_model="Galaxy Watch 4 Classic"
                    )
                    self.data_buffer.append(health_data)
                    
                    if callback:
                        callback(health_data)
            
            await self.client.start_notify(
                self.HEART_RATE_MEASUREMENT_UUID,
                notification_handler
            )
            self.sync_enabled = True
            logger.info("Started heart rate monitoring")
            return True
        except Exception as e:
            logger.error(f"Error starting heart rate monitoring: {e}")
            return False
    
    async def stop_heart_rate_monitoring(self) -> bool:
        """Stop heart rate monitoring"""
        if self.client and self.sync_enabled:
            try:
                await self.client.stop_notify(self.HEART_RATE_MEASUREMENT_UUID)
                self.sync_enabled = False
                logger.info("Stopped heart rate monitoring")
                return True
            except Exception as e:
                logger.error(f"Error stopping monitoring: {e}")
                return False
        return False
    
    def parse_samsung_health_export(self, json_data: Dict[str, Any]) -> List[SamsungHealthData]:
        """
        Parse Samsung Health export data (JSON format)
        Samsung Health allows data export via the app
        """
        health_data = []
        
        try:
            # Parse different data types from Samsung Health export
            # Structure varies based on export format
            
            # Heart rate data
            if "heart_rate" in json_data:
                for record in json_data["heart_rate"]:
                    data = SamsungHealthData(
                        data_type="heart_rate",
                        value=float(record.get("heart_rate", record.get("value", 0))),
                        unit="bpm",
                        timestamp=datetime.fromisoformat(record.get("date_time", record.get("timestamp"))),
                        device_id=record.get("device_id", "samsung_watch"),
                        device_model="Galaxy Watch 4 Classic",
                        metadata={"source": "samsung_health_export"}
                    )
                    health_data.append(data)
            
            # Steps data
            if "steps" in json_data:
                for record in json_data["steps"]:
                    data = SamsungHealthData(
                        data_type="steps",
                        value=float(record.get("steps", record.get("value", 0))),
                        unit="count",
                        timestamp=datetime.fromisoformat(record.get("date_time", record.get("timestamp"))),
                        device_id=record.get("device_id", "samsung_watch"),
                        device_model="Galaxy Watch 4 Classic",
                        metadata={"source": "samsung_health_export"}
                    )
                    health_data.append(data)
            
            # Sleep data
            if "sleep" in json_data:
                for record in json_data["sleep"]:
                    data = SamsungHealthData(
                        data_type="sleep_duration",
                        value=float(record.get("duration", record.get("value", 0))),
                        unit="hours",
                        timestamp=datetime.fromisoformat(record.get("date_time", record.get("timestamp"))),
                        device_id=record.get("device_id", "samsung_watch"),
                        device_model="Galaxy Watch 4 Classic",
                        metadata={
                            "source": "samsung_health_export",
                            "sleep_stages": record.get("stages"),
                            "sleep_quality": record.get("quality")
                        }
                    )
                    health_data.append(data)
            
            # Stress data
            if "stress" in json_data:
                for record in json_data["stress"]:
                    data = SamsungHealthData(
                        data_type="stress_level",
                        value=float(record.get("stress", record.get("value", 0))),
                        unit="score",
                        timestamp=datetime.fromisoformat(record.get("date_time", record.get("timestamp"))),
                        device_id=record.get("device_id", "samsung_watch"),
                        device_model="Galaxy Watch 4 Classic",
                        metadata={"source": "samsung_health_export"}
                    )
                    health_data.append(data)
            
            # SpO2 data
            if "spo2" in json_data:
                for record in json_data["spo2"]:
                    data = SamsungHealthData(
                        data_type="spo2",
                        value=float(record.get("spo2", record.get("value", 0))),
                        unit="%",
                        timestamp=datetime.fromisoformat(record.get("date_time", record.get("timestamp"))),
                        device_id=record.get("device_id", "samsung_watch"),
                        device_model="Galaxy Watch 4 Classic",
                        metadata={"source": "samsung_health_export"}
                    )
                    health_data.append(data)
            
            # Blood pressure data
            if "blood_pressure" in json_data:
                for record in json_data["blood_pressure"]:
                    # Create separate entries for systolic and diastolic
                    systolic = SamsungHealthData(
                        data_type="systolic_bp",
                        value=float(record.get("systolic", record.get("systolic_bp", 0))),
                        unit="mmHg",
                        timestamp=datetime.fromisoformat(record.get("date_time", record.get("timestamp"))),
                        device_id=record.get("device_id", "samsung_watch"),
                        device_model="Galaxy Watch 4 Classic",
                        metadata={"source": "samsung_health_export"}
                    )
                    diastolic = SamsungHealthData(
                        data_type="diastolic_bp",
                        value=float(record.get("diastolic", record.get("diastolic_bp", 0))),
                        unit="mmHg",
                        timestamp=datetime.fromisoformat(record.get("date_time", record.get("timestamp"))),
                        device_id=record.get("device_id", "samsung_watch"),
                        device_model="Galaxy Watch 4 Classic",
                        metadata={"source": "samsung_health_export"}
                    )
                    health_data.extend([systolic, diastolic])
            
            logger.info(f"Parsed {len(health_data)} Samsung Health data points")
            return health_data
            
        except Exception as e:
            logger.error(f"Error parsing Samsung Health export: {e}")
            return []
    
    def get_buffered_data(self, clear_buffer: bool = True) -> List[SamsungHealthData]:
        """
        Get buffered health data from real-time monitoring
        """
        data = self.data_buffer.copy()
        if clear_buffer:
            self.data_buffer.clear()
        return data
    
    async def sync_historical_data(
        self,
        start_date: datetime,
        end_date: datetime,
        data_types: List[str] = None
    ) -> List[SamsungHealthData]:
        """
        Sync historical data from Samsung Health
        Note: This requires Samsung Health API access
        """
        # This would integrate with Samsung Health API
        # For now, return empty list as API integration requires OAuth setup
        logger.info("Historical data sync requires Samsung Health API integration")
        return []
    
    def get_supported_data_types(self) -> List[str]:
        """Return list of supported data types from Samsung Health"""
        return [
            "heart_rate",
            "steps",
            "sleep_duration",
            "stress_level",
            "spo2",
            "systolic_bp",
            "diastolic_bp",
            "oxygen_saturation",
            "respiratory_rate",
            "body_temperature",
            "calories_burned",
            "distance",
            "floors_climbed"
        ]
    
    def convert_to_observation_format(
        self,
        health_data: SamsungHealthData,
        patient_id: str
    ) -> Dict[str, Any]:
        """
        Convert Samsung Health data to internal observation format
        """
        type_mapping = {
            "heart_rate": "heart_rate",
            "steps": "steps",
            "sleep_duration": "sleep",
            "stress_level": "stress",
            "spo2": "spo2",
            "systolic_bp": "systolic_bp",
            "diastolic_bp": "diastolic_bp",
            "oxygen_saturation": "spo2",
            "respiratory_rate": "respiratory_rate",
            "body_temperature": "temperature",
        }
        
        unit_mapping = {
            "heart_rate": "bpm",
            "steps": "count",
            "sleep_duration": "hours",
            "stress_level": "score",
            "spo2": "%",
            "systolic_bp": "mmHg",
            "diastolic_bp": "mmHg",
            "oxygen_saturation": "%",
            "respiratory_rate": "/min",
            "body_temperature": "C",
        }
        
        obs_type = type_mapping.get(health_data.data_type, health_data.data_type)
        unit = unit_mapping.get(health_data.data_type, health_data.unit)
        
        return {
            "patient_id": patient_id,
            "observation_type": obs_type,
            "value": health_data.value,
            "unit": unit,
            "timestamp": health_data.timestamp,
            "source": "samsung_watch",
            "device_id": health_data.device_id,
            "quality_score": 1.0,  # Samsung Health data assumed high quality
            "metadata": health_data.metadata,
            "fhir_resource_type": "Observation",
            "status": "final",
            "code_system": "SNOMED",
            "code": None,  # Would map to SNOMED codes
        }


# Singleton instance
_samsung_health_integration = None

def get_samsung_health_integration() -> SamsungHealthIntegration:
    """Get singleton Samsung Health integration instance"""
    global _samsung_health_integration
    if _samsung_health_integration is None:
        _samsung_health_integration = SamsungHealthIntegration()
    return _samsung_health_integration
