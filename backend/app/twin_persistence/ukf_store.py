"""
UKF State Store using Redis for fast access and Postgres for persistence
Implements state-of-the-art twin state management with dual storage
"""
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import redis
import asyncio
import asyncpg
from pydantic import BaseModel, Field
import os


class UKFState(BaseModel):
    """UKF filter state representation"""
    patient_id: str
    timestamp: datetime
    state_vector: list[float]  # x_k: state estimate
    covariance_matrix: list[list[float]]  # P_k: covariance estimate
    process_noise: list[list[float]]  # Q: process noise covariance
    measurement_noise: list[list[float]]  # R: measurement noise covariance
    
    # Glucose-specific parameters
    glucose_mean: float
    glucose_std: float
    insulin_sensitivity: float
    carb_ratio: float
    
    # Metadata
    last_observation_time: Optional[datetime] = None
    observation_count: int = 0
    convergence_score: float = 0.0


class UKFStateStore:
    """
    Dual-storage UKF state manager:
    - Redis: Hot storage for real-time access (sub-ms latency)
    - Postgres: Cold storage for persistence and historical analysis
    """
    
    def __init__(self):
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=0,
            decode_responses=True
        )
        self.pg_pool = None
        self._redis_ttl = 86400  # 24 hours TTL for Redis
        
    async def init_postgres(self):
        """Initialize Postgres connection pool"""
        if self.pg_pool is None:
            self.pg_pool = await asyncpg.create_pool(
                os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/biodigital")
            )
            await self._create_tables()
    
    async def _create_tables(self):
        """Create UKF state tables in Postgres"""
        async with self.pg_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ukf_states (
                    id SERIAL PRIMARY KEY,
                    patient_id VARCHAR(20) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    state_vector JSONB NOT NULL,
                    covariance_matrix JSONB NOT NULL,
                    process_noise JSONB NOT NULL,
                    measurement_noise JSONB NOT NULL,
                    glucose_mean FLOAT NOT NULL,
                    glucose_std FLOAT NOT NULL,
                    insulin_sensitivity FLOAT NOT NULL,
                    carb_ratio FLOAT NOT NULL,
                    last_observation_time TIMESTAMP,
                    observation_count INTEGER DEFAULT 0,
                    convergence_score FLOAT DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(patient_id, timestamp)
                );
                
                CREATE INDEX IF NOT EXISTS idx_ukf_patient_id ON ukf_states(patient_id);
                CREATE INDEX IF NOT EXISTS idx_ukf_timestamp ON ukf_states(timestamp);
                CREATE INDEX IF NOT EXISTS idx_ukf_patient_timestamp ON ukf_states(patient_id, timestamp);
            """)
    
    async def store_state(self, state: UKFState) -> bool:
        """
        Store UKF state in both Redis and Postgres
        Returns True if successful
        """
        try:
            # Store in Redis (hot storage)
            redis_key = f"ukf:state:{state.patient_id}"
            state_dict = state.model_dump()
            state_dict["timestamp"] = state_dict["timestamp"].isoformat()
            if state_dict["last_observation_time"]:
                state_dict["last_observation_time"] = state_dict["last_observation_time"].isoformat()
            
            self.redis_client.setex(
                redis_key,
                self._redis_ttl,
                json.dumps(state_dict)
            )
            
            # Store in Postgres (cold storage)
            await self.init_postgres()
            async with self.pg_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO ukf_states (
                        patient_id, timestamp, state_vector, covariance_matrix,
                        process_noise, measurement_noise, glucose_mean, glucose_std,
                        insulin_sensitivity, carb_ratio, last_observation_time,
                        observation_count, convergence_score
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    ON CONFLICT (patient_id, timestamp) 
                    DO UPDATE SET
                        state_vector = EXCLUDED.state_vector,
                        covariance_matrix = EXCLUDED.covariance_matrix,
                        process_noise = EXCLUDED.process_noise,
                        measurement_noise = EXCLUDED.measurement_noise,
                        glucose_mean = EXCLUDED.glucose_mean,
                        glucose_std = EXCLUDED.glucose_std,
                        insulin_sensitivity = EXCLUDED.insulin_sensitivity,
                        carb_ratio = EXCLUDED.carb_ratio,
                        last_observation_time = EXCLUDED.last_observation_time,
                        observation_count = EXCLUDED.observation_count,
                        convergence_score = EXCLUDED.convergence_score
                """,
                    state.patient_id,
                    state.timestamp,
                    json.dumps(state.state_vector),
                    json.dumps(state.covariance_matrix),
                    json.dumps(state.process_noise),
                    json.dumps(state.measurement_noise),
                    state.glucose_mean,
                    state.glucose_std,
                    state.insulin_sensitivity,
                    state.carb_ratio,
                    state.last_observation_time,
                    state.observation_count,
                    state.convergence_score
                )
            
            return True
        except Exception as e:
            print(f"Error storing UKF state: {e}")
            return False
    
    async def get_state(self, patient_id: str) -> Optional[UKFState]:
        """
        Retrieve UKF state from Redis (hot) or Postgres (cold)
        Returns None if not found
        """
        try:
            # Try Redis first (hot storage)
            redis_key = f"ukf:state:{patient_id}"
            cached = self.redis_client.get(redis_key)
            
            if cached:
                state_dict = json.loads(cached)
                state_dict["timestamp"] = datetime.fromisoformat(state_dict["timestamp"])
                if state_dict.get("last_observation_time"):
                    state_dict["last_observation_time"] = datetime.fromisoformat(state_dict["last_observation_time"])
                return UKFState(**state_dict)
            
            # Fallback to Postgres (cold storage)
            await self.init_postgres()
            async with self.pg_pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM ukf_states WHERE patient_id = $1 ORDER BY timestamp DESC LIMIT 1",
                    patient_id
                )
                
                if row:
                    state = UKFState(
                        patient_id=row["patient_id"],
                        timestamp=row["timestamp"],
                        state_vector=json.loads(row["state_vector"]),
                        covariance_matrix=json.loads(row["covariance_matrix"]),
                        process_noise=json.loads(row["process_noise"]),
                        measurement_noise=json.loads(row["measurement_noise"]),
                        glucose_mean=row["glucose_mean"],
                        glucose_std=row["glucose_std"],
                        insulin_sensitivity=row["insulin_sensitivity"],
                        carb_ratio=row["carb_ratio"],
                        last_observation_time=row["last_observation_time"],
                        observation_count=row["observation_count"],
                        convergence_score=row["convergence_score"]
                    )
                    
                    # Cache back to Redis
                    state_dict = state.model_dump()
                    state_dict["timestamp"] = state_dict["timestamp"].isoformat()
                    if state_dict["last_observation_time"]:
                        state_dict["last_observation_time"] = state_dict["last_observation_time"].isoformat()
                    self.redis_client.setex(
                        redis_key,
                        self._redis_ttl,
                        json.dumps(state_dict)
                    )
                    
                    return state
            
            return None
        except Exception as e:
            print(f"Error retrieving UKF state: {e}")
            return None
    
    async def get_state_history(
        self, 
        patient_id: str, 
        start_time: datetime,
        end_time: datetime
    ) -> list[UKFState]:
        """
        Retrieve historical UKF states from Postgres
        Useful for analysis and re-training
        """
        try:
            await self.init_postgres()
            async with self.pg_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT * FROM ukf_states 
                    WHERE patient_id = $1 
                    AND timestamp BETWEEN $2 AND $3
                    ORDER BY timestamp ASC
                    """,
                    patient_id, start_time, end_time
                )
                
                states = []
                for row in rows:
                    state = UKFState(
                        patient_id=row["patient_id"],
                        timestamp=row["timestamp"],
                        state_vector=json.loads(row["state_vector"]),
                        covariance_matrix=json.loads(row["covariance_matrix"]),
                        process_noise=json.loads(row["process_noise"]),
                        measurement_noise=json.loads(row["measurement_noise"]),
                        glucose_mean=row["glucose_mean"],
                        glucose_std=row["glucose_std"],
                        insulin_sensitivity=row["insulin_sensitivity"],
                        carb_ratio=row["carb_ratio"],
                        last_observation_time=row["last_observation_time"],
                        observation_count=row["observation_count"],
                        convergence_score=row["convergence_score"]
                    )
                    states.append(state)
                
                return states
        except Exception as e:
            print(f"Error retrieving UKF state history: {e}")
            return []
    
    async def delete_state(self, patient_id: str) -> bool:
        """Delete UKF state from both Redis and Postgres"""
        try:
            # Delete from Redis
            redis_key = f"ukf:state:{patient_id}"
            self.redis_client.delete(redis_key)
            
            # Delete from Postgres
            await self.init_postgres()
            async with self.pg_pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM ukf_states WHERE patient_id = $1",
                    patient_id
                )
            
            return True
        except Exception as e:
            print(f"Error deleting UKF state: {e}")
            return False
    
    async def get_patient_list(self) -> list[str]:
        """Get list of all patients with UKF states"""
        try:
            await self.init_postgres()
            async with self.pg_pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT DISTINCT patient_id FROM ukf_states ORDER BY patient_id"
                )
                return [row["patient_id"] for row in rows]
        except Exception as e:
            print(f"Error getting patient list: {e}")
            return []


# Singleton instance
_ukf_store = None

def get_ukf_store() -> UKFStateStore:
    """Get singleton UKF state store instance"""
    global _ukf_store
    if _ukf_store is None:
        _ukf_store = UKFStateStore()
    return _ukf_store
