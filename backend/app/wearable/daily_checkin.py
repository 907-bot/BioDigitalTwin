"""
Daily Checkin Service
Aggregates daily health data from multiple sources
Provides comprehensive daily health summary for patients
"""
import json
from datetime import datetime, timedelta, date
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
import logging
import asyncio

from .fhir_ingest import get_fhir_ingest_service
from .cgm_webhook import get_cgm_webhook_service
from .samsung_health import get_samsung_health_integration
from ..twin_persistence.observation_history import get_observation_history
from ..twin_persistence.ukf_store import get_ukf_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DailyCheckinSummary(BaseModel):
    """Daily health checkin summary"""
    patient_id: str
    checkin_date: date
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Glucose metrics
    glucose_avg: Optional[float] = None
    glucose_std: Optional[float] = None
    glucose_min: Optional[float] = None
    glucose_max: Optional[float] = None
    glucose_tir_percent: Optional[float] = None  # Time in Range
    glucose_below_range_percent: Optional[float] = None
    glucose_above_range_percent: Optional[float] = None
    
    # Heart metrics
    heart_rate_avg: Optional[float] = None
    heart_rate_std: Optional[float] = None
    heart_rate_min: Optional[float] = None
    heart_rate_max: Optional[float] = None
    hrv_avg: Optional[float] = None
    
    # Activity metrics
    steps_total: Optional[int] = None
    steps_goal_percent: Optional[float] = None
    calories_burned: Optional[float] = None
    active_minutes: Optional[int] = None
    
    # Sleep metrics
    sleep_duration_hours: Optional[float] = None
    sleep_quality_score: Optional[float] = None
    sleep_stages: Optional[Dict[str, float]] = None
    
    # Blood pressure
    systolic_bp_avg: Optional[float] = None
    diastolic_bp_avg: Optional[float] = None
    
    # Other vitals
    spo2_avg: Optional[float] = None
    stress_avg: Optional[float] = None
    temperature_avg: Optional[float] = None
    
    # Data sources
    data_sources: List[str] = Field(default_factory=list)
    observation_count: int = 0
    data_quality_score: float = 1.0
    
    # UKF state
    ukf_state_available: bool = False
    ukf_glucose_estimate: Optional[float] = None
    ukf_rmse: Optional[float] = None
    
    # Alerts and recommendations
    alerts: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    
    # Overall health score
    overall_health_score: Optional[float] = None


class DailyCheckinService:
    """
    Daily checkin service that aggregates data from multiple sources
    Provides comprehensive daily health summary for clinical review
    """
    
    def __init__(self):
        self.observation_history = get_observation_history()
        self.ukf_store = get_ukf_store()
        self.fhir_service = get_fhir_ingest_service()
        self.cgm_service = get_cgm_webhook_service()
        self.samsung_service = get_samsung_health_integration()
        
        # Health goals and thresholds
        self.glucose_target_range = (70, 180)  # mg/dL
        self.heart_rate_normal_range = (60, 100)  # bpm
        self.bp_normal_range = {"systolic": (90, 120), "diastolic": (60, 80)}
        self.spo2_normal_range = (95, 100)  # %
        self.steps_goal = 10000
        self.sleep_goal_hours = 7.5
    
    async def generate_daily_checkin(
        self,
        patient_id: str,
        checkin_date: Optional[date] = None
    ) -> DailyCheckinSummary:
        """
        Generate comprehensive daily checkin summary
        Aggregates data from all available sources
        """
        if checkin_date is None:
            checkin_date = date.today()
        
        start_time = datetime.combine(checkin_date, datetime.min.time())
        end_time = start_time + timedelta(days=1)
        
        summary = DailyCheckinSummary(
            patient_id=patient_id,
            checkin_date=checkin_date
        )
        
        # Get observations for the day
        observations = await self.observation_history.get_observations(
            patient_id=patient_id,
            start_time=start_time,
            end_time=end_time,
            limit=10000
        )
        
        summary.observation_count = len(observations)
        
        # Group observations by type
        obs_by_type = {}
        for obs in observations:
            obs_type = obs.observation_type
            if obs_type not in obs_by_type:
                obs_by_type[obs_type] = []
            obs_by_type[obs_type].append(obs)
        
        # Track data sources
        sources = set(obs.source for obs in observations)
        summary.data_sources = list(sources)
        
        # Calculate metrics for each type
        if "glucose" in obs_by_type:
            glucose_values = [obs.value for obs in obs_by_type["glucose"]]
            summary.glucose_avg = sum(glucose_values) / len(glucose_values)
            summary.glucose_std = (sum((x - summary.glucose_avg)**2 for x in glucose_values) / len(glucose_values))**0.5
            summary.glucose_min = min(glucose_values)
            summary.glucose_max = max(glucose_values)
            
            # Time in Range calculations
            in_range = sum(1 for v in glucose_values if self.glucose_target_range[0] <= v <= self.glucose_target_range[1])
            below_range = sum(1 for v in glucose_values if v < self.glucose_target_range[0])
            above_range = sum(1 for v in glucose_values if v > self.glucose_target_range[1])
            
            summary.glucose_tir_percent = (in_range / len(glucose_values)) * 100
            summary.glucose_below_range_percent = (below_range / len(glucose_values)) * 100
            summary.glucose_above_range_percent = (above_range / len(glucose_values)) * 100
        
        if "heart_rate" in obs_by_type:
            hr_values = [obs.value for obs in obs_by_type["heart_rate"]]
            summary.heart_rate_avg = sum(hr_values) / len(hr_values)
            summary.heart_rate_std = (sum((x - summary.heart_rate_avg)**2 for x in hr_values) / len(hr_values))**0.5
            summary.heart_rate_min = min(hr_values)
            summary.heart_rate_max = max(hr_values)
        
        if "hrv" in obs_by_type:
            hrv_values = [obs.value for obs in obs_by_type["hrv"]]
            summary.hrv_avg = sum(hrv_values) / len(hrv_values)
        
        if "steps" in obs_by_type:
            steps_values = [obs.value for obs in obs_by_type["steps"]]
            summary.steps_total = int(sum(steps_values))
            summary.steps_goal_percent = (summary.steps_total / self.steps_goal) * 100
        
        if "calories_burned" in obs_by_type:
            cal_values = [obs.value for obs in obs_by_type["calories_burned"]]
            summary.calories_burned = sum(cal_values)
        
        if "active_minutes" in obs_by_type:
            active_values = [obs.value for obs in obs_by_type["active_minutes"]]
            summary.active_minutes = int(sum(active_values))
        
        if "sleep" in obs_by_type:
            sleep_values = [obs.value for obs in obs_by_type["sleep"]]
            summary.sleep_duration_hours = sum(sleep_values)
        
        if "systolic_bp" in obs_by_type:
            sbp_values = [obs.value for obs in obs_by_type["systolic_bp"]]
            summary.systolic_bp_avg = sum(sbp_values) / len(sbp_values)
        
        if "diastolic_bp" in obs_by_type:
            dbp_values = [obs.value for obs in obs_by_type["diastolic_bp"]]
            summary.diastolic_bp_avg = sum(dbp_values) / len(dbp_values)
        
        if "spo2" in obs_by_type:
            spo2_values = [obs.value for obs in obs_by_type["spo2"]]
            summary.spo2_avg = sum(spo2_values) / len(spo2_values)
        
        if "stress" in obs_by_type:
            stress_values = [obs.value for obs in obs_by_type["stress"]]
            summary.stress_avg = sum(stress_values) / len(stress_values)
        
        if "temperature" in obs_by_type:
            temp_values = [obs.value for obs in obs_by_type["temperature"]]
            summary.temperature_avg = sum(temp_values) / len(temp_values)
        
        # Get UKF state
        ukf_state = await self.ukf_store.get_state(patient_id)
        if ukf_state:
            summary.ukf_state_available = True
            summary.ukf_glucose_estimate = ukf_state.glucose_mean
            summary.ukf_rmse = ukf_state.glucose_std
        
        # Calculate data quality score
        if observations:
            quality_scores = [obs.quality_score for obs in observations]
            summary.data_quality_score = sum(quality_scores) / len(quality_scores)
        
        # Generate alerts
        summary.alerts = self._generate_alerts(summary)
        
        # Generate recommendations
        summary.recommendations = self._generate_recommendations(summary)
        
        # Calculate overall health score
        summary.overall_health_score = self._calculate_health_score(summary)
        
        return summary
    
    def _generate_alerts(self, summary: DailyCheckinSummary) -> List[str]:
        """Generate health alerts based on metrics"""
        alerts = []
        
        # Glucose alerts
        if summary.glucose_avg:
            if summary.glucose_avg < 70:
                alerts.append(f"⚠️ Low average glucose: {summary.glucose_avg:.1f} mg/dL")
            elif summary.glucose_avg > 180:
                alerts.append(f"⚠️ High average glucose: {summary.glucose_avg:.1f} mg/dL")
        
        if summary.glucose_tir_percent is not None:
            if summary.glucose_tir_percent < 70:
                alerts.append(f"⚠️ Low Time in Range: {summary.glucose_tir_percent:.1f}%")
        
        # Heart rate alerts
        if summary.heart_rate_avg:
            if summary.heart_rate_avg < 60:
                alerts.append(f"⚠️ Low heart rate: {summary.heart_rate_avg:.1f} bpm")
            elif summary.heart_rate_avg > 100:
                alerts.append(f"⚠️ High heart rate: {summary.heart_rate_avg:.1f} bpm")
        
        # Blood pressure alerts
        if summary.systolic_bp_avg and summary.systolic_bp_avg > 140:
            alerts.append(f"⚠️ High systolic BP: {summary.systolic_bp_avg:.1f} mmHg")
        
        if summary.diastolic_bp_avg and summary.diastolic_bp_avg > 90:
            alerts.append(f"⚠️ High diastolic BP: {summary.diastolic_bp_avg:.1f} mmHg")
        
        # SpO2 alerts
        if summary.spo2_avg and summary.spo2_avg < 95:
            alerts.append(f"⚠️ Low SpO2: {summary.spo2_avg:.1f}%")
        
        # Sleep alerts
        if summary.sleep_duration_hours and summary.sleep_duration_hours < 6:
            alerts.append(f"⚠️ Insufficient sleep: {summary.sleep_duration_hours:.1f} hours")
        
        return alerts
    
    def _generate_recommendations(self, summary: DailyCheckinSummary) -> List[str]:
        """Generate health recommendations based on metrics"""
        recommendations = []
        
        # Glucose recommendations
        if summary.glucose_tir_percent is not None and summary.glucose_tir_percent < 70:
            recommendations.append("Consider reviewing meal timing and carbohydrate intake")
        
        if summary.glucose_avg and summary.glucose_avg > 180:
            recommendations.append("Monitor insulin timing and dosage")
        
        # Activity recommendations
        if summary.steps_total and summary.steps_total < self.steps_goal:
            recommendations.append(f"Try to increase daily steps to reach {self.steps_goal} goal")
        
        # Sleep recommendations
        if summary.sleep_duration_hours and summary.sleep_duration_hours < self.sleep_goal_hours:
            recommendations.append(f"Aim for {self.sleep_goal_hours} hours of sleep for optimal recovery")
        
        # Heart rate recommendations
        if summary.hrv_avg and summary.hrv_avg < 30:
            recommendations.append("Consider stress management techniques to improve HRV")
        
        # Blood pressure recommendations
        if summary.systolic_bp_avg and summary.systolic_bp_avg > 130:
            recommendations.append("Monitor sodium intake and consider relaxation techniques")
        
        return recommendations
    
    def _calculate_health_score(self, summary: DailyCheckinSummary) -> float:
        """
        Calculate overall health score (0-100)
        Based on weighted combination of metrics
        """
        score = 100.0
        deductions = 0.0
        
        # Glucose score (30% weight)
        if summary.glucose_tir_percent is not None:
            glucose_score = summary.glucose_tir_percent
            deductions += (100 - glucose_score) * 0.3
        
        # Heart rate score (15% weight)
        if summary.heart_rate_avg:
            if self.heart_rate_normal_range[0] <= summary.heart_rate_avg <= self.heart_rate_normal_range[1]:
                hr_score = 100
            else:
                deviation = min(
                    abs(summary.heart_rate_avg - self.heart_rate_normal_range[0]),
                    abs(summary.heart_rate_avg - self.heart_rate_normal_range[1])
                )
                hr_score = max(0, 100 - deviation * 2)
            deductions += (100 - hr_score) * 0.15
        
        # Activity score (15% weight)
        if summary.steps_goal_percent is not None:
            activity_score = min(100, summary.steps_goal_percent)
            deductions += (100 - activity_score) * 0.15
        
        # Sleep score (15% weight)
        if summary.sleep_duration_hours:
            sleep_score = min(100, (summary.sleep_duration_hours / self.sleep_goal_hours) * 100)
            deductions += (100 - sleep_score) * 0.15
        
        # Blood pressure score (15% weight)
        bp_score = 100
        if summary.systolic_bp_avg:
            if not (self.bp_normal_range["systolic"][0] <= summary.systolic_bp_avg <= self.bp_normal_range["systolic"][1]):
                bp_score -= 20
        if summary.diastolic_bp_avg:
            if not (self.bp_normal_range["diastolic"][0] <= summary.diastolic_bp_avg <= self.bp_normal_range["diastolic"][1]):
                bp_score -= 20
        deductions += (100 - bp_score) * 0.15
        
        # SpO2 score (10% weight)
        if summary.spo2_avg:
            if summary.spo2_avg >= 95:
                spo2_score = 100
            else:
                spo2_score = max(0, summary.spo2_avg)
            deductions += (100 - spo2_score) * 0.1
        
        final_score = max(0, score - deductions)
        return round(final_score, 1)
    
    async def get_checkin_history(
        self,
        patient_id: str,
        start_date: date,
        end_date: date
    ) -> List[DailyCheckinSummary]:
        """
        Get daily checkin history for a date range
        """
        summaries = []
        current_date = start_date
        
        while current_date <= end_date:
            try:
                summary = await self.generate_daily_checkin(patient_id, current_date)
                summaries.append(summary)
            except Exception as e:
                logger.error(f"Error generating checkin for {current_date}: {e}")
            
            current_date += timedelta(days=1)
        
        return summaries
    
    async def process_fhir_observation_batch(
        self,
        patient_id: str,
        fhir_bundle: Dict[str, Any]
    ) -> int:
        """
        Process a batch of FHIR observations and store them
        Returns number of successfully processed observations
        """
        parsed_observations = self.fhir_service.parse_fhir_bundle(fhir_bundle)
        
        from ..twin_persistence.observation_history import Observation
        observation_objects = []
        
        for obs_dict in parsed_observations:
            obs_dict["patient_id"] = patient_id  # Ensure patient_id is set
            obs = Observation(**obs_dict)
            observation_objects.append(obs)
        
        success_count = await self.observation_history.add_observations_batch(observation_objects)
        return success_count


# Singleton instance
_daily_checkin_service = None

def get_daily_checkin_service() -> DailyCheckinService:
    """Get singleton daily checkin service instance"""
    global _daily_checkin_service
    if _daily_checkin_service is None:
        _daily_checkin_service = DailyCheckinService()
    return _daily_checkin_service
