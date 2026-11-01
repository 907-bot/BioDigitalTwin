"""
Observation History Manager
Stores and retrieves patient observations (glucose, HR, activity, etc.)
Supports FHIR Observation format and CGM data
"""
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
try:
    import redis
except ImportError:
    redis = None

try:
    import asyncpg
except ImportError:
    asyncpg = None
import os


class Observation(BaseModel):
    """FHIR-compliant observation representation"""
    patient_id: str
    observation_type: str  # glucose, heart_rate, steps, sleep, etc.
    value: float
    unit: str
    timestamp: datetime
    source: str  # cgm, wearable, manual, ehr
    device_id: Optional[str] = None
    quality_score: float = 1.0  # 0-1 data quality metric
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # FHIR-specific fields
    fhir_resource_type: str = "Observation"
    status: str = "final"
    code_system: Optional[str] = None
    code: Optional[str] = None


class ObservationHistory:
    """
    Observation history manager with dual storage:
    - Redis: Recent observations (last 7 days) for fast access
    - Postgres: Full history for analysis and compliance
    """
    
    def __init__(self):
        if redis is not None:
            try:
                self.redis_client = redis.Redis(
                    host=os.getenv("REDIS_HOST", "localhost"),
                    port=int(os.getenv("REDIS_PORT", 6379)),
                    db=1,  # Use different DB for observations
                    decode_responses=True
                )
            except Exception:
                self.redis_client = None
        else:
            self.redis_client = None
        self._in_memory_obs = {}
        self.pg_pool = None
        self._redis_ttl = 604800  # 7 days TTL for Redis
        
    async def init_postgres(self):
        """Initialize Postgres connection pool"""
        if asyncpg is None:
            return
        if self.pg_pool is None:
            try:
                self.pg_pool = await asyncpg.create_pool(
                    os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/biodigital")
                )
                await self._create_tables()
            except Exception:
                self.pg_pool = None
    
    async def _create_tables(self):
        """Create observation tables in Postgres"""
        async with self.pg_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS observations (
                    id SERIAL PRIMARY KEY,
                    patient_id VARCHAR(20) NOT NULL,
                    observation_type VARCHAR(50) NOT NULL,
                    value FLOAT NOT NULL,
                    unit VARCHAR(20) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    source VARCHAR(20) NOT NULL,
                    device_id VARCHAR(100),
                    quality_score FLOAT DEFAULT 1.0,
                    metadata JSONB DEFAULT '{}',
                    fhir_resource_type VARCHAR(50) DEFAULT 'Observation',
                    status VARCHAR(20) DEFAULT 'final',
                    code_system VARCHAR(100),
                    code VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_obs_patient_id ON observations(patient_id);
                CREATE INDEX IF NOT EXISTS idx_obs_timestamp ON observations(timestamp);
                CREATE INDEX IF NOT EXISTS idx_obs_type ON observations(observation_type);
                CREATE INDEX IF NOT EXISTS idx_obs_patient_timestamp ON observations(patient_id, timestamp);
                CREATE INDEX IF NOT EXISTS idx_obs_patient_type_timestamp ON observations(patient_id, observation_type, timestamp);
            """)
    
    async def add_observation(self, observation: Observation) -> bool:
        """
        Add observation to both Redis and Postgres
        Returns True if successful
        """
        try:
            # Store in Redis (hot storage)
            redis_key = f"obs:{observation.patient_id}:{observation.observation_type}"
            obs_dict = observation.model_dump()
            obs_dict["timestamp"] = obs_dict["timestamp"].isoformat()
            
            # Use Redis sorted set for time-series data
            score = observation.timestamp.timestamp()
            if self.redis_client is not None:
                try:
                    self.redis_client.zadd(redis_key, {json.dumps(obs_dict): score})
                    self.redis_client.expire(redis_key, self._redis_ttl)
                except Exception:
                    self._in_memory_obs.setdefault(redis_key, []).append((score, obs_dict))
            else:
                self._in_memory_obs.setdefault(redis_key, []).append((score, obs_dict))
            
            # Store in Postgres (cold storage)
            await self.init_postgres()
            if self.pg_pool is not None:
                async with self.pg_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO observations (
                            patient_id, observation_type, value, unit, timestamp,
                            source, device_id, quality_score, metadata,
                            fhir_resource_type, status, code_system, code
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    """,
                        observation.patient_id,
                        observation.observation_type,
                        observation.value,
                        observation.unit,
                        observation.timestamp,
                        observation.source,
                        observation.device_id,
                        observation.quality_score,
                        json.dumps(observation.metadata),
                        observation.fhir_resource_type,
                        observation.status,
                        observation.code_system,
                        observation.code
                    )
            
            return True
        except Exception as e:
            print(f"Error adding observation: {e}")
            return False
    
    async def add_observations_batch(self, observations: List[Observation]) -> int:
        """
        Batch add observations for efficiency
        Returns number of successfully added observations
        """
        success_count = 0
        for obs in observations:
            if await self.add_observation(obs):
                success_count += 1
        return success_count
    
    async def get_observations(
        self,
        patient_id: str,
        observation_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Observation]:
        """
        Retrieve observations from Redis (recent) or Postgres (full history)
        """
        try:
            # Default to last 7 days if not specified
            if start_time is None:
                start_time = datetime.now() - timedelta(days=7)
            if end_time is None:
                end_time = datetime.now()
            
            # Try Redis first for recent data
            if observation_type and (end_time - start_time) <= timedelta(days=7):
                redis_key = f"obs:{patient_id}:{observation_type}"
                min_score = start_time.timestamp()
                max_score = end_time.timestamp()
                
                results = []
                if self.redis_client is not None:
                    try:
                        results = self.redis_client.zrangebyscore(
                            redis_key, min_score, max_score, start=0, num=limit
                        )
                    except Exception:
                        results = []
                else:
                    items = self._in_memory_obs.get(redis_key, [])
                    results = [json.dumps(d) for s, d in items if min_score <= s <= max_score][:limit]
                
                if results:
                    observations = []
                    for result in results:
                        obs_dict = json.loads(result)
                        obs_dict["timestamp"] = datetime.fromisoformat(obs_dict["timestamp"])
                        observations.append(Observation(**obs_dict))
                    return observations
            
            # Fallback to Postgres
            await self.init_postgres()
            if self.pg_pool is not None:
                async with self.pg_pool.acquire() as conn:
                    query = """
                        SELECT * FROM observations 
                        WHERE patient_id = $1 
                        AND timestamp BETWEEN $2 AND $3
                    """
                    params = [patient_id, start_time, end_time]
                    
                    if observation_type:
                        query += " AND observation_type = $4"
                        params.append(observation_type)
                    
                    query += " ORDER BY timestamp DESC LIMIT $5"
                    params.append(limit)
                    
                    rows = await conn.fetch(query, *params)
                    
                    observations = []
                    for row in rows:
                        obs = Observation(
                            patient_id=row["patient_id"],
                            observation_type=row["observation_type"],
                            value=row["value"],
                            unit=row["unit"],
                            timestamp=row["timestamp"],
                            source=row["source"],
                            device_id=row["device_id"],
                            quality_score=row["quality_score"],
                            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                            fhir_resource_type=row["fhir_resource_type"],
                            status=row["status"],
                            code_system=row["code_system"],
                            code=row["code"]
                        )
                        observations.append(obs)
                    
                    return observations
        except Exception as e:
            print(f"Error retrieving observations: {e}")
            return []
    
    async def get_latest_observation(
        self,
        patient_id: str,
        observation_type: str
    ) -> Optional[Observation]:
        """Get the most recent observation of a specific type"""
        observations = await self.get_observations(
            patient_id, observation_type, limit=1
        )
        return observations[0] if observations else None
    
    async def get_observations_as_dataframe(
        self,
        patient_id: str,
        observation_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Retrieve observations as pandas DataFrame for analysis
        """
        observations = await self.get_observations(
            patient_id, observation_type, start_time, end_time, limit=10000
        )
        
        if not observations:
            return pd.DataFrame()
        
        data = []
        for obs in observations:
            data.append({
                "patient_id": obs.patient_id,
                "observation_type": obs.observation_type,
                "value": obs.value,
                "unit": obs.unit,
                "timestamp": obs.timestamp,
                "source": obs.source,
                "device_id": obs.device_id,
                "quality_score": obs.quality_score
            })
        
        return pd.DataFrame(data)
    
    async def get_observation_types(self, patient_id: str) -> List[str]:
        """Get list of observation types for a patient"""
        try:
            await self.init_postgres()
            async with self.pg_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT DISTINCT observation_type 
                    FROM observations 
                    WHERE patient_id = $1 
                    ORDER BY observation_type
                    """,
                    patient_id
                )
                return [row["observation_type"] for row in rows]
        except Exception as e:
            print(f"Error getting observation types: {e}")
            return []
    
    async def get_daily_summary(
        self,
        patient_id: str,
        date: datetime
    ) -> Dict[str, Any]:
        """
        Get daily summary of observations for a patient
        Useful for daily check-in endpoints
        """
        start_time = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = start_time + timedelta(days=1)
        
        observations = await self.get_observations(
            patient_id, start_time=start_time, end_time=end_time, limit=10000
        )
        
        summary = {
            "patient_id": patient_id,
            "date": date.date(),
            "total_observations": len(observations),
            "by_type": {}
        }
        
        for obs in observations:
            obs_type = obs.observation_type
            if obs_type not in summary["by_type"]:
                summary["by_type"][obs_type] = {
                    "count": 0,
                    "values": [],
                    "sources": set()
                }
            summary["by_type"][obs_type]["count"] += 1
            summary["by_type"][obs_type]["values"].append(obs.value)
            summary["by_type"][obs_type]["sources"].add(obs.source)
        
        # Calculate statistics per type
        for obs_type, data in summary["by_type"].items():
            values = data["values"]
            if values:
                data["mean"] = sum(values) / len(values)
                data["min"] = min(values)
                data["max"] = max(values)
                data["std"] = (sum((x - data["mean"]) ** 2 for x in values) / len(values)) ** 0.5
            data["sources"] = list(data["sources"])
        
        return summary
    
    async def delete_old_observations(self, days_to_keep: int = 365) -> int:
        """
        Delete observations older than specified days
        Returns number of deleted records
        """
        try:
            await self.init_postgres()
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            async with self.pg_pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM observations WHERE timestamp < $1",
                    cutoff_date
                )
                # Parse "DELETE n" response
                return int(result.split()[-1]) if result else 0
        except Exception as e:
            print(f"Error deleting old observations: {e}")
            return 0


# Singleton instance
_observation_history = None

def get_observation_history() -> ObservationHistory:
    """Get singleton observation history instance"""
    global _observation_history
    if _observation_history is None:
        _observation_history = ObservationHistory()
    return _observation_history
