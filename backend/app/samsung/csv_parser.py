"""
Samsung Health CSV Parser — Phase 18
Parses the real Samsung Health data export (44 CSV files).

Samsung CSV files have a non-standard 2-line header:
  Line 1: com.samsung.shealth.<type>,<ver>,<schema_ver>   ← metadata, skip
  Line 2: col1,col2,...colN                                ← actual column names
  Line 3+: data rows

All timestamps include timezone offset (UTC+0530 → converted to UTC).
"""
from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timezone helper
# ---------------------------------------------------------------------------
_TZ_OFFSET_CACHE: Dict[str, timezone] = {}

def _parse_tz(tz_str: Optional[str]) -> timezone:
    """Parse 'UTC+0530' style offset strings into timezone objects."""
    if not tz_str or tz_str.strip() == "":
        return timezone.utc
    tz_str = tz_str.strip()
    if tz_str in _TZ_OFFSET_CACHE:
        return _TZ_OFFSET_CACHE[tz_str]
    try:
        sign = 1 if "+" in tz_str else -1
        part = tz_str.replace("UTC", "").replace("+", "").replace("-", "")
        hours = int(part[:2]) if len(part) >= 2 else 0
        minutes = int(part[2:4]) if len(part) >= 4 else 0
        offset = timedelta(hours=hours, minutes=minutes) * sign
        tz = timezone(offset)
        _TZ_OFFSET_CACHE[tz_str] = tz
        return tz
    except Exception:
        return timezone.utc


def _to_utc(ts_str: Optional[str], tz_str: Optional[str] = None) -> Optional[datetime]:
    """Parse a Samsung timestamp string and return UTC datetime."""
    if not ts_str or str(ts_str).strip() in ("", "nan", "None"):
        return None
    ts_str = str(ts_str).strip()
    # Samsung format: "2025-01-10 17:20:29.296"
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(ts_str, fmt)
            tz = _parse_tz(tz_str)
            dt = dt.replace(tzinfo=tz)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue
    return None


def _safe_float(val: Any) -> Optional[float]:
    try:
        f = float(val)
        return f if not np.isnan(f) else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Read a Samsung CSV (handles BOM + 2-line header)
# ---------------------------------------------------------------------------
def _read_samsung_csv(filepath: str) -> Optional[pd.DataFrame]:
    """
    Read a Samsung Health CSV file, handling the 2-line header format.
    Returns DataFrame with clean column names, or None on error.
    """
    try:
        with open(filepath, encoding="utf-8-sig") as f:
            lines = f.readlines()

        if len(lines) < 3:
            return None

        # Line 0: metadata (skip)
        # Line 1: real column headers
        # Line 2+: data
        header_line = lines[1].strip()
        data_lines = lines[2:]

        # Re-parse via csv module for proper quoting
        reader = csv.reader([header_line])
        columns = next(reader)
        # Strip BOM from first column if present
        if columns and columns[0].startswith("\ufeff"):
            columns[0] = columns[0][1:]

        rows = []
        for line in data_lines:
            line = line.strip()
            if not line:
                continue
            r = csv.reader([line])
            row = next(r)
            # Pad short rows
            while len(row) < len(columns):
                row.append("")
            rows.append(row[:len(columns)])

        if not rows:
            return None

        df = pd.DataFrame(rows, columns=columns)
        return df
    except Exception as e:
        logger.warning(f"Failed to read {filepath}: {e}")
        return None


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class HeartRateRecord:
    timestamp: datetime
    hr_bpm: float
    device_id: str = ""

@dataclass
class StressRecord:
    timestamp: datetime
    score: float          # 0–100
    device_id: str = ""

@dataclass
class SleepStageRecord:
    start: datetime
    end: datetime
    stage_code: int       # 40001=awake 40002=light 40003=deep 40004=REM
    sleep_session_id: str = ""

@dataclass
class PedometerDaySummary:
    day: datetime
    step_count: int
    distance_m: float
    calorie: float
    active_time_ms: int
    run_step_count: int = 0
    walk_step_count: int = 0

@dataclass
class WeightRecord:
    timestamp: datetime
    weight_kg: float
    body_fat_pct: float = 0.0
    skeletal_muscle_mass_kg: float = 0.0
    fat_free_mass_kg: float = 0.0
    bmi_est: float = 0.0
    basal_metabolic_rate: float = 0.0

@dataclass
class ExerciseSession:
    start: datetime
    end: datetime
    exercise_type: int
    duration_ms: int
    distance_m: float = 0.0
    calorie: float = 0.0
    mean_heart_rate: float = 0.0
    max_heart_rate: float = 0.0
    min_heart_rate: float = 0.0
    mean_speed_ms: float = 0.0
    max_speed_ms: float = 0.0
    vo2_max: float = 0.0

@dataclass
class SpO2Record:
    timestamp: datetime
    spo2_pct: float

@dataclass
class HRVRecord:
    timestamp: datetime
    hrv_ms: float

@dataclass
class FloorsSummary:
    day: datetime
    floors: float = 0.0


@dataclass
class SamsungPatientRecord:
    """Unified real-world patient record parsed from Samsung Health CSVs."""
    patient_id: str = "P_REAL_001"
    data_dir: str = ""

    heart_rate: List[HeartRateRecord] = field(default_factory=list)
    stress: List[StressRecord] = field(default_factory=list)
    sleep_stages: List[SleepStageRecord] = field(default_factory=list)
    pedometer: List[PedometerDaySummary] = field(default_factory=list)
    weight: List[WeightRecord] = field(default_factory=list)
    exercise: List[ExerciseSession] = field(default_factory=list)
    spo2: List[SpO2Record] = field(default_factory=list)
    hrv: List[HRVRecord] = field(default_factory=list)
    floors: List[FloorsSummary] = field(default_factory=list)

    # Quick aggregates (populated by aggregate())
    aggregates: Dict[str, Any] = field(default_factory=dict)

    def aggregate(self) -> Dict[str, Any]:
        """Compute summary statistics across all data types."""
        agg: Dict[str, Any] = {}

        # --- Heart Rate ---
        if self.heart_rate:
            hrs = [r.hr_bpm for r in self.heart_rate]
            agg["hr_mean"] = float(np.mean(hrs))
            agg["hr_std"] = float(np.std(hrs))
            agg["hr_min"] = float(np.min(hrs))
            agg["hr_max"] = float(np.max(hrs))
            agg["hr_resting"] = float(np.percentile(hrs, 10))  # 10th pct ≈ resting
            agg["hr_count"] = len(hrs)
            # Date range
            ts = [r.timestamp for r in self.heart_rate]
            agg["hr_start"] = min(ts).isoformat()
            agg["hr_end"] = max(ts).isoformat()
        else:
            agg.update(dict(hr_mean=0, hr_std=0, hr_min=0, hr_max=0,
                            hr_resting=0, hr_count=0))

        # --- Stress ---
        if self.stress:
            scores = [r.score for r in self.stress if r.score >= 0]
            agg["stress_mean"] = float(np.mean(scores)) if scores else 0.0
            agg["stress_max"] = float(np.max(scores)) if scores else 0.0
            agg["stress_min"] = float(np.min(scores)) if scores else 0.0
            agg["stress_std"] = float(np.std(scores)) if scores else 0.0
            agg["stress_high_pct"] = float(np.mean([s > 60 for s in scores]) * 100) if scores else 0.0
            agg["stress_count"] = len(scores)
        else:
            agg.update(dict(stress_mean=0, stress_max=0, stress_min=0,
                            stress_std=0, stress_high_pct=0, stress_count=0))

        # --- Sleep ---
        if self.sleep_stages:
            total_ms = sum((r.end - r.start).total_seconds() * 1000 for r in self.sleep_stages)
            stage_ms = {40001: 0, 40002: 0, 40003: 0, 40004: 0}
            for r in self.sleep_stages:
                dur = (r.end - r.start).total_seconds() * 1000
                if r.stage_code in stage_ms:
                    stage_ms[r.stage_code] += dur
            agg["sleep_total_hours"] = total_ms / 3_600_000
            agg["sleep_awake_pct"] = (stage_ms[40001] / max(total_ms, 1)) * 100
            agg["sleep_light_pct"] = (stage_ms[40002] / max(total_ms, 1)) * 100
            agg["sleep_deep_pct"] = (stage_ms[40003] / max(total_ms, 1)) * 100
            agg["sleep_rem_pct"] = (stage_ms[40004] / max(total_ms, 1)) * 100
        else:
            agg.update(dict(sleep_total_hours=0, sleep_awake_pct=0,
                            sleep_light_pct=0, sleep_deep_pct=0, sleep_rem_pct=0))

        # --- Pedometer ---
        if self.pedometer:
            steps_per_day = [r.step_count for r in self.pedometer]
            agg["steps_daily_mean"] = float(np.mean(steps_per_day))
            agg["steps_daily_max"] = float(np.max(steps_per_day))
            agg["steps_daily_std"] = float(np.std(steps_per_day))
            agg["distance_daily_mean_km"] = float(np.mean([r.distance_m / 1000 for r in self.pedometer]))
            agg["calories_daily_mean"] = float(np.mean([r.calorie for r in self.pedometer]))
            agg["active_time_daily_mean_min"] = float(np.mean([r.active_time_ms / 60_000 for r in self.pedometer]))
        else:
            agg.update(dict(steps_daily_mean=0, steps_daily_max=0,
                            steps_daily_std=0, distance_daily_mean_km=0,
                            calories_daily_mean=0, active_time_daily_mean_min=0))

        # --- Weight ---
        if self.weight:
            agg["weight_kg_latest"] = self.weight[-1].weight_kg
            agg["weight_kg_mean"] = float(np.mean([r.weight_kg for r in self.weight]))
            agg["body_fat_pct_latest"] = self.weight[-1].body_fat_pct
            agg["skeletal_muscle_kg_latest"] = self.weight[-1].skeletal_muscle_mass_kg
            agg["bmi_est"] = self.weight[-1].bmi_est if self.weight[-1].bmi_est > 0 else (
                self.weight[-1].weight_kg / (1.75 ** 2)  # fallback: assume 175cm
            )
            agg["bmr"] = self.weight[-1].basal_metabolic_rate
        else:
            agg.update(dict(weight_kg_latest=0, weight_kg_mean=0,
                            body_fat_pct_latest=0, skeletal_muscle_kg_latest=0,
                            bmi_est=0, bmr=0))

        # --- Exercise ---
        if self.exercise:
            agg["exercise_session_count"] = len(self.exercise)
            agg["exercise_mean_hr"] = float(np.mean([e.mean_heart_rate for e in self.exercise if e.mean_heart_rate > 0]))
            agg["exercise_max_hr"] = float(np.max([e.max_heart_rate for e in self.exercise if e.max_heart_rate > 0]))
            agg["exercise_mean_duration_min"] = float(np.mean([e.duration_ms / 60_000 for e in self.exercise]))
            agg["vo2_max"] = float(np.max([e.vo2_max for e in self.exercise if e.vo2_max > 0]) if any(e.vo2_max > 0 for e in self.exercise) else 0)
            agg["exercise_total_calories"] = float(sum(e.calorie for e in self.exercise))
        else:
            agg.update(dict(exercise_session_count=0, exercise_mean_hr=0,
                            exercise_max_hr=0, exercise_mean_duration_min=0,
                            vo2_max=0, exercise_total_calories=0))

        # --- SpO2 ---
        if self.spo2:
            spo2_vals = [r.spo2_pct for r in self.spo2]
            agg["spo2_mean"] = float(np.mean(spo2_vals))
            agg["spo2_min"] = float(np.min(spo2_vals))
        else:
            agg.update(dict(spo2_mean=0, spo2_min=0))

        # --- HRV ---
        if self.hrv:
            hrv_vals = [r.hrv_ms for r in self.hrv]
            agg["hrv_mean"] = float(np.mean(hrv_vals))
        else:
            agg["hrv_mean"] = 0.0

        # --- Floors ---
        if self.floors:
            agg["floors_daily_mean"] = float(np.mean([r.floors for r in self.floors]))
        else:
            agg["floors_daily_mean"] = 0.0

        self.aggregates = agg
        return agg


# ---------------------------------------------------------------------------
# Main Parser
# ---------------------------------------------------------------------------
class SamsungCSVParser:
    """
    Parses all Samsung Health CSV files from a directory into a SamsungPatientRecord.
    Handles the 2-line Samsung header format, BOM, and timezone normalization.
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def _find_file(self, pattern: str) -> Optional[Path]:
        """Find first CSV file matching pattern (case-insensitive)."""
        for f in self.data_dir.glob("*.csv"):
            if pattern.lower() in f.name.lower():
                return f
        return None

    # --- Parsers for each data type ---

    def parse_heart_rate(self) -> List[HeartRateRecord]:
        path = self._find_file("tracker.heart_rate")
        if not path:
            return []
        df = _read_samsung_csv(str(path))
        if df is None:
            return []
        records = []
        # Columns: com.samsung.health.heart_rate.start_time, heart_rate, time_offset, deviceuuid
        ts_col = next((c for c in df.columns if "start_time" in c), None)
        hr_col = next((c for c in df.columns if c.endswith(".heart_rate") or c == "com.samsung.health.heart_rate.heart_rate"), None)
        tz_col = next((c for c in df.columns if "time_offset" in c), None)
        dev_col = next((c for c in df.columns if "deviceuuid" in c), None)
        if ts_col is None or hr_col is None:
            logger.warning("heart_rate: missing required columns")
            return []
        for _, row in df.iterrows():
            ts = _to_utc(row.get(ts_col), row.get(tz_col) if tz_col else None)
            hr = _safe_float(row.get(hr_col))
            if ts and hr and 30 <= hr <= 220:  # physiological bounds
                records.append(HeartRateRecord(
                    timestamp=ts, hr_bpm=hr,
                    device_id=str(row.get(dev_col, "")) if dev_col else ""
                ))
        logger.info(f"Parsed {len(records)} heart rate records")
        return sorted(records, key=lambda r: r.timestamp)

    def parse_stress(self) -> List[StressRecord]:
        path = self._find_file("shealth.stress.2")
        if not path:
            path = self._find_file("shealth.stress.")
        if not path:
            return []
        df = _read_samsung_csv(str(path))
        if df is None:
            return []
        records = []
        ts_col = next((c for c in df.columns if "start_time" in c), None)
        sc_col = next((c for c in df.columns if c == "score"), None)
        tz_col = next((c for c in df.columns if "time_offset" in c), None)
        dev_col = next((c for c in df.columns if "deviceuuid" in c), None)
        if ts_col is None or sc_col is None:
            logger.warning("stress: missing required columns")
            return []
        for _, row in df.iterrows():
            ts = _to_utc(row.get(ts_col), row.get(tz_col) if tz_col else None)
            score = _safe_float(row.get(sc_col))
            if ts and score is not None and 0 <= score <= 100:
                records.append(StressRecord(
                    timestamp=ts, score=score,
                    device_id=str(row.get(dev_col, "")) if dev_col else ""
                ))
        logger.info(f"Parsed {len(records)} stress records")
        return sorted(records, key=lambda r: r.timestamp)

    def parse_sleep_stages(self) -> List[SleepStageRecord]:
        path = self._find_file("health.sleep_stage")
        if not path:
            return []
        df = _read_samsung_csv(str(path))
        if df is None:
            return []
        records = []
        ts_col = next((c for c in df.columns if "start_time" in c), None)
        end_col = next((c for c in df.columns if "end_time" in c), None)
        stage_col = next((c for c in df.columns if c == "stage"), None)
        sid_col = next((c for c in df.columns if "sleep_id" in c), None)
        tz_col = next((c for c in df.columns if "time_offset" in c), None)
        if ts_col is None or stage_col is None:
            logger.warning("sleep_stage: missing required columns")
            return []
        for _, row in df.iterrows():
            ts = _to_utc(row.get(ts_col), row.get(tz_col) if tz_col else None)
            end = _to_utc(row.get(end_col) if end_col else None, row.get(tz_col) if tz_col else None)
            stage = _safe_float(row.get(stage_col))
            if ts and end and stage:
                records.append(SleepStageRecord(
                    start=ts, end=end,
                    stage_code=int(stage),
                    sleep_session_id=str(row.get(sid_col, "")) if sid_col else ""
                ))
        logger.info(f"Parsed {len(records)} sleep stage records")
        return sorted(records, key=lambda r: r.start)

    def parse_pedometer(self) -> List[PedometerDaySummary]:
        path = self._find_file("pedometer_day_summary")
        if not path:
            return []
        df = _read_samsung_csv(str(path))
        if df is None:
            return []
        records = []
        # Deduplicate by day (multiple device sources)
        day_best: Dict[str, PedometerDaySummary] = {}
        day_col = next((c for c in df.columns if "day_time" in c), None)
        steps_col = next((c for c in df.columns if c == "step_count"), None)
        dist_col = next((c for c in df.columns if c == "distance"), None)
        cal_col = next((c for c in df.columns if c == "calorie"), None)
        at_col = next((c for c in df.columns if c == "active_time"), None)
        run_col = next((c for c in df.columns if c == "run_step_count"), None)
        walk_col = next((c for c in df.columns if c == "walk_step_count"), None)
        if day_col is None or steps_col is None:
            return []
        for _, row in df.iterrows():
            day = _to_utc(row.get(day_col))
            steps = _safe_float(row.get(steps_col))
            if not day or not steps:
                continue
            day_key = day.strftime("%Y-%m-%d")
            rec = PedometerDaySummary(
                day=day, step_count=int(steps),
                distance_m=_safe_float(row.get(dist_col)) or 0.0,
                calorie=_safe_float(row.get(cal_col)) or 0.0,
                active_time_ms=int(_safe_float(row.get(at_col)) or 0),
                run_step_count=int(_safe_float(row.get(run_col)) or 0),
                walk_step_count=int(_safe_float(row.get(walk_col)) or 0),
            )
            # Keep highest step count per day (usually most complete device)
            if day_key not in day_best or rec.step_count > day_best[day_key].step_count:
                day_best[day_key] = rec
        records = sorted(day_best.values(), key=lambda r: r.day)
        logger.info(f"Parsed {len(records)} pedometer day summaries")
        return records

    def parse_weight(self) -> List[WeightRecord]:
        path = self._find_file("health.weight")
        if not path:
            return []
        df = _read_samsung_csv(str(path))
        if df is None:
            return []
        records = []
        ts_col = next((c for c in df.columns if "start_time" in c), None)
        w_col = next((c for c in df.columns if c == "weight"), None)
        bf_col = next((c for c in df.columns if "body_fat" in c and "mass" not in c), None)
        sm_col = next((c for c in df.columns if "skeletal_muscle_mass" in c), None)
        ffm_col = next((c for c in df.columns if "fat_free_mass" in c), None)
        bmr_col = next((c for c in df.columns if "basal_metabolic_rate" in c), None)
        h_col = next((c for c in df.columns if c == "height"), None)
        tz_col = next((c for c in df.columns if "time_offset" in c), None)
        if ts_col is None or w_col is None:
            return []
        for _, row in df.iterrows():
            ts = _to_utc(row.get(ts_col), row.get(tz_col) if tz_col else None)
            w = _safe_float(row.get(w_col))
            if not ts or not w:
                continue
            height_m = (_safe_float(row.get(h_col)) or 0) / 100.0  # cm → m
            bmi_est = w / (height_m ** 2) if height_m > 0 else 0.0
            records.append(WeightRecord(
                timestamp=ts, weight_kg=w,
                body_fat_pct=_safe_float(row.get(bf_col)) or 0.0,
                skeletal_muscle_mass_kg=_safe_float(row.get(sm_col)) or 0.0,
                fat_free_mass_kg=_safe_float(row.get(ffm_col)) or 0.0,
                bmi_est=round(bmi_est, 2),
                basal_metabolic_rate=_safe_float(row.get(bmr_col)) or 0.0,
            ))
        logger.info(f"Parsed {len(records)} weight records")
        return sorted(records, key=lambda r: r.timestamp)

    def parse_exercise(self) -> List[ExerciseSession]:
        path = self._find_file("shealth.exercise.2")
        if not path:
            path = self._find_file("shealth.exercise.")
        if not path:
            return []
        df = _read_samsung_csv(str(path))
        if df is None:
            return []
        records = []
        st_col = next((c for c in df.columns if "start_time" in c), None)
        en_col = next((c for c in df.columns if "end_time" in c), None)
        dur_col = next((c for c in df.columns if "duration" in c), None)
        type_col = next((c for c in df.columns if "exercise_type" in c), None)
        dist_col = next((c for c in df.columns if "distance" in c and "decline" not in c and "incline" not in c), None)
        cal_col = next((c for c in df.columns if "calorie" in c and "caloricburn" not in c), None)
        mhr_col = next((c for c in df.columns if "mean_heart_rate" in c), None)
        xhr_col = next((c for c in df.columns if "max_heart_rate" in c), None)
        nhr_col = next((c for c in df.columns if "min_heart_rate" in c), None)
        mspd_col = next((c for c in df.columns if "mean_speed" in c), None)
        xspd_col = next((c for c in df.columns if "max_speed" in c), None)
        vo2_col = next((c for c in df.columns if "vo2_max" in c), None)
        tz_col = next((c for c in df.columns if "time_offset" in c), None)
        if st_col is None:
            return []
        for _, row in df.iterrows():
            ts = _to_utc(row.get(st_col), row.get(tz_col) if tz_col else None)
            end = _to_utc(row.get(en_col) if en_col else None, row.get(tz_col) if tz_col else None)
            dur = _safe_float(row.get(dur_col)) if dur_col else 0
            if not ts:
                continue
            if end is None and dur:
                end = ts + timedelta(milliseconds=dur)
            if end is None:
                end = ts
            records.append(ExerciseSession(
                start=ts, end=end,
                exercise_type=int(_safe_float(row.get(type_col)) or 0) if type_col else 0,
                duration_ms=int(dur or 0),
                distance_m=_safe_float(row.get(dist_col)) or 0.0 if dist_col else 0.0,
                calorie=_safe_float(row.get(cal_col)) or 0.0 if cal_col else 0.0,
                mean_heart_rate=_safe_float(row.get(mhr_col)) or 0.0 if mhr_col else 0.0,
                max_heart_rate=_safe_float(row.get(xhr_col)) or 0.0 if xhr_col else 0.0,
                min_heart_rate=_safe_float(row.get(nhr_col)) or 0.0 if nhr_col else 0.0,
                mean_speed_ms=_safe_float(row.get(mspd_col)) or 0.0 if mspd_col else 0.0,
                max_speed_ms=_safe_float(row.get(xspd_col)) or 0.0 if xspd_col else 0.0,
                vo2_max=_safe_float(row.get(vo2_col)) or 0.0 if vo2_col else 0.0,
            ))
        logger.info(f"Parsed {len(records)} exercise sessions")
        return sorted(records, key=lambda r: r.start)

    def parse_spo2(self) -> List[SpO2Record]:
        path = self._find_file("oxygen_saturation")
        if not path:
            return []
        df = _read_samsung_csv(str(path))
        if df is None:
            return []
        records = []
        ts_col = next((c for c in df.columns if "start_time" in c), None)
        val_col = next((c for c in df.columns if "spo2" in c or "saturation" in c), None)
        tz_col = next((c for c in df.columns if "time_offset" in c), None)
        if ts_col is None or val_col is None:
            return []
        for _, row in df.iterrows():
            ts = _to_utc(row.get(ts_col), row.get(tz_col) if tz_col else None)
            val = _safe_float(row.get(val_col))
            if ts and val and 50 <= val <= 100:
                records.append(SpO2Record(timestamp=ts, spo2_pct=val))
        logger.info(f"Parsed {len(records)} SpO2 records")
        return records

    def parse_hrv(self) -> List[HRVRecord]:
        path = self._find_file("health.hrv")
        if not path:
            return []
        df = _read_samsung_csv(str(path))
        if df is None:
            return []
        records = []
        ts_col = next((c for c in df.columns if "start_time" in c), None)
        val_col = next((c for c in df.columns if "hrv" in c.lower() and "time" not in c), None)
        tz_col = next((c for c in df.columns if "time_offset" in c), None)
        if ts_col is None or val_col is None:
            return []
        for _, row in df.iterrows():
            ts = _to_utc(row.get(ts_col), row.get(tz_col) if tz_col else None)
            val = _safe_float(row.get(val_col))
            if ts and val and val > 0:
                records.append(HRVRecord(timestamp=ts, hrv_ms=val))
        logger.info(f"Parsed {len(records)} HRV records")
        return records

    def parse_floors(self) -> List[FloorsSummary]:
        path = self._find_file("floors_day_summary")
        if not path:
            return []
        df = _read_samsung_csv(str(path))
        if df is None:
            return []
        records = []
        day_col = next((c for c in df.columns if "day_time" in c or "start_time" in c), None)
        fl_col = next((c for c in df.columns if "floor_count" in c or "floors" in c), None)
        if day_col is None or fl_col is None:
            return []
        for _, row in df.iterrows():
            ts = _to_utc(row.get(day_col))
            val = _safe_float(row.get(fl_col))
            if ts and val is not None:
                records.append(FloorsSummary(day=ts, floors=val))
        return sorted(records, key=lambda r: r.day)

    # --- Main entry point ---

    def parse_all(self) -> SamsungPatientRecord:
        """Parse all Samsung Health CSV files and return a unified patient record."""
        logger.info(f"Parsing Samsung Health data from {self.data_dir}")
        record = SamsungPatientRecord(data_dir=str(self.data_dir))
        record.heart_rate = self.parse_heart_rate()
        record.stress = self.parse_stress()
        record.sleep_stages = self.parse_sleep_stages()
        record.pedometer = self.parse_pedometer()
        record.weight = self.parse_weight()
        record.exercise = self.parse_exercise()
        record.spo2 = self.parse_spo2()
        record.hrv = self.parse_hrv()
        record.floors = self.parse_floors()
        record.aggregate()

        total = (len(record.heart_rate) + len(record.stress) +
                 len(record.sleep_stages) + len(record.pedometer) +
                 len(record.weight) + len(record.exercise) +
                 len(record.spo2) + len(record.hrv))
        logger.info(f"Samsung parse complete: {total} total data points across all types")
        return record


# ---------------------------------------------------------------------------
# Build synthetic_patients.csv-compatible row from real record
# ---------------------------------------------------------------------------
def samsung_record_to_patient_row(record: SamsungPatientRecord) -> Dict[str, Any]:
    """
    Convert a SamsungPatientRecord into a row compatible with synthetic_patients.csv.
    Maps real biometrics to the expected schema used by downstream phases.
    """
    agg = record.aggregates if record.aggregates else record.aggregate()
    return {
        "patient_id": record.patient_id,
        "age": 25,  # real patient — unknown, defaults to 25; override if known
        "gender": "Male",
        "bmi": round(agg.get("bmi_est", 0) or 26.5, 2),
        "hr": round(agg.get("hr_resting", 0) or agg.get("hr_mean", 72), 0),
        "hrv": round(agg.get("hrv_mean", 0) or 45, 0),
        "spo2": round(agg.get("spo2_mean", 0) or 97.0, 1),
        "glucose": 100,  # not in Samsung data; use normal default
        "systolic_bp": 120,  # not in Samsung data
        "diastolic_bp": 78,  # not in Samsung data
        "steps_daily": round(agg.get("steps_daily_mean", 0), 0),
        "stress_score": round(agg.get("stress_mean", 0), 1),
        "sleep_hours": round(agg.get("sleep_total_hours", 0) / max(
            len(set(s.start.date() for s in record.sleep_stages)), 1
        ), 2),
        "body_fat_pct": round(agg.get("body_fat_pct_latest", 0), 2),
        "vo2_max": round(agg.get("vo2_max", 0), 1),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
