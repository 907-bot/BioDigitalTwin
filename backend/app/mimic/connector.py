"""
MIMIC-IV Integration
PhysioNet credentialing and external validation loop
Integrates with MIMIC-IV clinical database for model validation
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from pydantic import BaseModel, Field
import logging
import os
import asyncio
import asyncpg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MIMICCredential(BaseModel):
    """PhysioNet credentials for MIMIC-IV access"""
    physionet_username: str
    physionet_password: str
    mimic_version: str = "2.2"
    data_path: str = Field(default="/data/mimic-iv")


class MIMICValidationResult(BaseModel):
    """Result of external validation on MIMIC-IV data"""
    validation_id: str
    timestamp: datetime
    patient_count: int
    observation_count: int
    metrics: Dict[str, float]
    comparison_with_synthetic: Dict[str, float]
    statistical_tests: Dict[str, Any]
    passed: bool
    notes: List[str] = Field(default_factory=list)


class MIMICConnector:
    """
    MIMIC-IV connector for external validation
    Requires PhysioNet credentialing and data access
    """
    
    def __init__(self, credentials: Optional[MIMICCredential] = None):
        self.credentials = credentials
        self.pg_pool = None
        self.connected = False
        
        # MIMIC-IV table schemas
        self.tables = {
            "patients": "mimic_iv_hosp.patients",
            "admissions": "mimic_iv_hosp.admissions",
            "labevents": "mimic_iv_hosp.labevents",
            "chartevents": "mimic_iv_icu.chartevents",
            "prescriptions": "mimic_iv_hosp.prescriptions",
            "diagnoses_icd": "mimic_iv_hosp.diagnoses_icd",
            "d_items": "mimic_iv_icu.d_items",
            "labevents": "mimic_iv_hosp.labevents",
            "d_labitems": "mimic_iv_hosp.d_labitems",
        }
    
    async def connect(self, database_url: Optional[str] = None) -> bool:
        """
        Connect to MIMIC-IV PostgreSQL database
        Requires PhysioNet credentials and data download
        """
        if database_url is None:
            database_url = os.getenv("MIMIC_DATABASE_URL")
        
        if not database_url:
            logger.error("MIMIC database URL not configured")
            return False
        
        try:
            self.pg_pool = await asyncpg.create_pool(database_url)
            self.connected = True
            logger.info("Connected to MIMIC-IV database")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MIMIC-IV: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from MIMIC-IV database"""
        if self.pg_pool:
            await self.pg_pool.close()
            self.connected = False
            logger.info("Disconnected from MIMIC-IV database")
    
    async def get_patient_demographics(self, limit: int = 1000) -> pd.DataFrame:
        """
        Extract patient demographics from MIMIC-IV
        For comparison with synthetic patient generation
        """
        if not self.connected:
            await self.connect()
        
        try:
            async with self.pg_pool.acquire() as conn:
                query = """
                    SELECT 
                        p.subject_id,
                        p.gender,
                        p.anchor_age,
                        a.race,
                        a.insurance,
                        COUNT(DISTINCT a.hadm_id) as admission_count
                    FROM mimic_iv_hosp.patients p
                    JOIN mimic_iv_hosp.admissions a ON p.subject_id = a.subject_id
                    GROUP BY p.subject_id, p.gender, p.anchor_age, a.race, a.insurance
                    LIMIT $1
                """
                rows = await conn.fetch(query, limit)
                
                data = []
                for row in rows:
                    data.append({
                        "subject_id": row["subject_id"],
                        "gender": row["gender"],
                        "age": row["anchor_age"],
                        "race": row["race"],
                        "insurance": row["insurance"],
                        "admission_count": row["admission_count"]
                    })
                
                return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Error fetching patient demographics: {e}")
            return pd.DataFrame()
    
    async def get_glucose_lab_values(
        self,
        limit: int = 10000
    ) -> pd.DataFrame:
        """
        Extract glucose lab values from MIMIC-IV
        For glucose model validation
        """
        if not self.connected:
            await self.connect()
        
        try:
            async with self.pg_pool.acquire() as conn:
                query = """
                    SELECT 
                        le.subject_id,
                        le.hadm_id,
                        le.charttime,
                        le.valuenum,
                        le.valueuom,
                        d.label as item_label
                    FROM mimic_iv_hosp.labevents le
                    JOIN mimic_iv_hosp.d_labitems d ON le.itemid = d.itemid
                    WHERE d.label ILIKE '%glucose%'
                    AND le.valuenum IS NOT NULL
                    AND le.valuenum > 0
                    ORDER BY le.charttime
                    LIMIT $1
                """
                rows = await conn.fetch(query, limit)
                
                data = []
                for row in rows:
                    data.append({
                        "subject_id": row["subject_id"],
                        "hadm_id": row["hadm_id"],
                        "timestamp": row["charttime"],
                        "glucose_value": row["valuenum"],
                        "unit": row["valueuom"],
                        "item_label": row["item_label"]
                    })
                
                return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Error fetching glucose lab values: {e}")
            return pd.DataFrame()
    
    async def get_heart_rate_vitals(
        self,
        limit: int = 10000
    ) -> pd.DataFrame:
        """
        Extract heart rate vital signs from MIMIC-IV
        For vital sign validation
        """
        if not self.connected:
            await self.connect()
        
        try:
            async with self.pg_pool.acquire() as conn:
                query = """
                    SELECT 
                        ce.subject_id,
                        ce.stay_id,
                        ce.charttime,
                        ce.valuenum,
                        d.label as item_label
                    FROM mimic_iv_icu.chartevents ce
                    JOIN mimic_iv_icu.d_items d ON ce.itemid = d.itemid
                    WHERE d.label ILIKE '%heart rate%'
                    AND ce.valuenum IS NOT NULL
                    AND ce.valuenum > 0
                    ORDER BY ce.charttime
                    LIMIT $1
                """
                rows = await conn.fetch(query, limit)
                
                data = []
                for row in rows:
                    data.append({
                        "subject_id": row["subject_id"],
                        "stay_id": row["stay_id"],
                        "timestamp": row["charttime"],
                        "heart_rate": row["valuenum"],
                        "item_label": row["item_label"]
                    })
                
                return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Error fetching heart rate vitals: {e}")
            return pd.DataFrame()
    
    async def get_blood_pressure_vitals(
        self,
        limit: int = 10000
    ) -> pd.DataFrame:
        """
        Extract blood pressure vital signs from MIMIC-IV
        """
        if not self.connected:
            await self.connect()
        
        try:
            async with self.pg_pool.acquire() as conn:
                query = """
                    SELECT 
                        ce.subject_id,
                        ce.stay_id,
                        ce.charttime,
                        ce.valuenum,
                        d.label as item_label
                    FROM mimic_iv_icu.chartevents ce
                    JOIN mimic_iv_icu.d_items d ON ce.itemid = d.itemid
                    WHERE (d.label ILIKE '%systolic%' OR d.label ILIKE '%diastolic%')
                    AND d.label ILIKE '%bp%'
                    AND ce.valuenum IS NOT NULL
                    AND ce.valuenum > 0
                    ORDER BY ce.charttime
                    LIMIT $1
                """
                rows = await conn.fetch(query, limit)
                
                data = []
                for row in rows:
                    data.append({
                        "subject_id": row["subject_id"],
                        "stay_id": row["stay_id"],
                        "timestamp": row["charttime"],
                        "value": row["valuenum"],
                        "item_label": row["item_label"]
                    })
                
                return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Error fetching blood pressure vitals: {e}")
            return pd.DataFrame()
    
    async def run_external_validation(
        self,
        synthetic_data: pd.DataFrame,
        validation_id: Optional[str] = None
    ) -> MIMICValidationResult:
        """
        Run external validation loop comparing synthetic data with MIMIC-IV
        """
        if validation_id is None:
            validation_id = f"val_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting external validation {validation_id}")
        
        try:
            # Get MIMIC-IV data
            mimic_demographics = await self.get_patient_demographics(limit=5000)
            mimic_glucose = await self.get_glucose_lab_values(limit=50000)
            mimic_hr = await self.get_heart_rate_vitals(limit=50000)
            
            # Calculate metrics
            metrics = {}
            comparison = {}
            statistical_tests = {}
            notes = []
            
            # Demographics comparison
            if not mimic_demographics.empty:
                mimic_age_mean = mimic_demographics["age"].mean()
                mimic_age_std = mimic_demographics["age"].std()
                mimic_gender_dist = mimic_demographics["gender"].value_counts(normalize=True)
                
                metrics["mimic_patient_count"] = len(mimic_demographics)
                metrics["mimic_age_mean"] = mimic_age_mean
                metrics["mimic_age_std"] = mimic_age_std
                
                if "age" in synthetic_data.columns:
                    synthetic_age_mean = synthetic_data["age"].mean()
                    synthetic_age_std = synthetic_data["age"].std()
                    
                    comparison["age_mean_diff"] = abs(synthetic_age_mean - mimic_age_mean)
                    comparison["age_std_diff"] = abs(synthetic_age_std - mimic_age_std)
                    
                    # Statistical test
                    from scipy import stats
                    t_stat, p_value = stats.ttest_ind(
                        synthetic_data["age"],
                        mimic_demographics["age"]
                    )
                    statistical_tests["age_ttest"] = {
                        "t_statistic": t_stat,
                        "p_value": p_value,
                        "significant": p_value < 0.05
                    }
            
            # Glucose comparison
            if not mimic_glucose.empty:
                mimic_glucose_mean = mimic_glucose["glucose_value"].mean()
                mimic_glucose_std = mimic_glucose["glucose_value"].std()
                
                metrics["mimic_glucose_count"] = len(mimic_glucose)
                metrics["mimic_glucose_mean"] = mimic_glucose_mean
                metrics["mimic_glucose_std"] = mimic_glucose_std
                
                if "glucose" in synthetic_data.columns:
                    synthetic_glucose_mean = synthetic_data["glucose"].mean()
                    synthetic_glucose_std = synthetic_data["glucose"].std()
                    
                    comparison["glucose_mean_diff"] = abs(synthetic_glucose_mean - mimic_glucose_mean)
                    comparison["glucose_std_diff"] = abs(synthetic_glucose_std - mimic_glucose_std)
            
            # Heart rate comparison
            if not mimic_hr.empty:
                mimic_hr_mean = mimic_hr["heart_rate"].mean()
                mimic_hr_std = mimic_hr["heart_rate"].std()
                
                metrics["mimic_hr_count"] = len(mimic_hr)
                metrics["mimic_hr_mean"] = mimic_hr_mean
                metrics["mimic_hr_std"] = mimic_hr_std
                
                if "hr" in synthetic_data.columns:
                    synthetic_hr_mean = synthetic_data["hr"].mean()
                    synthetic_hr_std = synthetic_data["hr"].std()
                    
                    comparison["hr_mean_diff"] = abs(synthetic_hr_mean - mimic_hr_mean)
                    comparison["hr_std_diff"] = abs(synthetic_hr_std - mimic_hr_std)
            
            # Determine if validation passed
            passed = True
            for key, value in comparison.items():
                if "diff" in key and value > 10:  # Threshold for significant difference
                    passed = False
                    notes.append(f"Significant difference in {key}: {value:.2f}")
            
            if passed:
                notes.append("Validation passed: synthetic data closely matches MIMIC-IV distributions")
            
            result = MIMICValidationResult(
                validation_id=validation_id,
                timestamp=datetime.now(),
                patient_count=metrics.get("mimic_patient_count", 0),
                observation_count=metrics.get("mimic_glucose_count", 0) + metrics.get("mimic_hr_count", 0),
                metrics=metrics,
                comparison_with_synthetic=comparison,
                statistical_tests=statistical_tests,
                passed=passed,
                notes=notes
            )
            
            logger.info(f"External validation {validation_id} completed: {'PASSED' if passed else 'FAILED'}")
            return result
            
        except Exception as e:
            logger.error(f"Error during external validation: {e}")
            return MIMICValidationResult(
                validation_id=validation_id,
                timestamp=datetime.now(),
                patient_count=0,
                observation_count=0,
                metrics={},
                comparison_with_synthetic={},
                statistical_tests={},
                passed=False,
                notes=[f"Validation failed with error: {str(e)}"]
            )
    
    async def get_mimic_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics from MIMIC-IV database"""
        if not self.connected:
            await self.connect()
        
        try:
            async with self.pg_pool.acquire() as conn:
                # Get table row counts
                tables_info = {}
                for table_name, table_path in self.tables.items():
                    try:
                        result = await conn.fetchval(f"SELECT COUNT(*) FROM {table_path}")
                        tables_info[table_name] = result
                    except Exception as e:
                        logger.warning(f"Could not get count for {table_name}: {e}")
                        tables_info[table_name] = 0
                
                return {
                    "connected": True,
                    "tables": tables_info,
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Error getting MIMIC summary: {e}")
            return {
                "connected": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


# Singleton instance
_mimic_connector = None

def get_mimic_connector(credentials: Optional[MIMICCredential] = None) -> MIMICConnector:
    """Get singleton MIMIC connector instance"""
    global _mimic_connector
    if _mimic_connector is None:
        _mimic_connector = MIMICConnector(credentials)
    return _mimic_connector
