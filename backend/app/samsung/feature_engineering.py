"""
Feature Engineering — Phase 18 Samsung Health
Converts a SamsungPatientRecord into a rich, fixed-length feature vector
suitable for ML model training and prediction.
"""
from __future__ import annotations

import logging
from datetime import timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from .csv_parser import SamsungPatientRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature schema (order matters — must stay stable across train/predict)
# ---------------------------------------------------------------------------
FEATURE_NAMES: List[str] = [
    # Heart Rate features (0-8)
    "hr_mean",
    "hr_std",
    "hr_min",
    "hr_max",
    "hr_resting",           # 10th percentile ≈ resting HR
    "hr_range",             # max - min
    "hr_high_pct",          # % readings > 100 bpm
    "hr_tachycardia_pct",   # % readings > 110 bpm
    "hr_bradycardia_pct",   # % readings < 60 bpm

    # HRV features (9-11)
    "hrv_mean",
    "hrv_rmssd_proxy",      # std of successive HR differences (proxy for RMSSD)
    "hrv_cardiac_fitness",  # VO2max-to-resting-HR ratio

    # Stress features (12-17)
    "stress_mean",
    "stress_max",
    "stress_std",
    "stress_high_pct",      # % time score > 60
    "stress_morning_mean",  # 06:00-12:00 UTC
    "stress_evening_mean",  # 18:00-00:00 UTC

    # Sleep features (18-24)
    "sleep_total_hours",
    "sleep_awake_pct",
    "sleep_light_pct",
    "sleep_deep_pct",
    "sleep_rem_pct",
    "sleep_efficiency",     # (total - awake) / total
    "sleep_consistency",    # std of nightly sleep duration (lower = more consistent)

    # Activity features (25-31)
    "steps_daily_mean",
    "steps_daily_std",
    "steps_goal_achievement",  # days > 10000 steps / total days
    "active_time_daily_min",
    "floors_daily_mean",
    "distance_daily_km",
    "calories_daily_mean",

    # Exercise features (32-38)
    "exercise_session_count",
    "exercise_mean_duration_min",
    "exercise_mean_hr",
    "exercise_max_hr",
    "exercise_hr_reserve_pct",  # (max_ex_hr - resting_hr) / (220 - age_est - resting_hr)
    "exercise_intensity_score",  # composite
    "vo2_max",

    # Body composition (39-42)
    "weight_kg",
    "bmi",
    "body_fat_pct",
    "skeletal_muscle_kg",

    # SpO2 (43-44)
    "spo2_mean",
    "spo2_min",

    # Metabolic (45-46)
    "bmr",
    "caloric_balance_proxy",  # calories_burned - bmr
]

NUM_FEATURES = len(FEATURE_NAMES)
assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)), "Duplicate feature names!"


# ---------------------------------------------------------------------------
# Compute successive HR differences (proxy for RMSSD)
# ---------------------------------------------------------------------------
def _hr_rmssd_proxy(hr_records) -> float:
    """RMSSD proxy: sqrt(mean(squared successive differences of sequential HR readings))."""
    if len(hr_records) < 2:
        return 0.0
    bpms = [r.hr_bpm for r in hr_records]
    diffs = np.diff(bpms)
    return float(np.sqrt(np.mean(diffs ** 2)))


def _sleep_consistency(sleep_stages) -> float:
    """Std of nightly sleep durations (in hours). Lower = more consistent."""
    from collections import defaultdict
    nightly: Dict[str, float] = defaultdict(float)
    for s in sleep_stages:
        night_key = s.start.strftime("%Y-%m-%d")
        dur_h = (s.end - s.start).total_seconds() / 3600
        nightly[night_key] += dur_h
    if len(nightly) < 2:
        return 0.0
    return float(np.std(list(nightly.values())))


def _stress_by_hour(stress_records, hour_min: int, hour_max: int) -> float:
    """Mean stress score for readings in given UTC hour range."""
    filtered = [r.score for r in stress_records
                if hour_min <= r.timestamp.hour < hour_max]
    return float(np.mean(filtered)) if filtered else 0.0


def _hr_by_condition(hr_records, cond) -> float:
    """Fraction of HR readings satisfying a condition."""
    if not hr_records:
        return 0.0
    return float(np.mean([cond(r.hr_bpm) for r in hr_records]) * 100)


# ---------------------------------------------------------------------------
# Main feature engineering function
# ---------------------------------------------------------------------------
def engineer_features(record: SamsungPatientRecord, age_est: int = 30) -> np.ndarray:
    """
    Convert a SamsungPatientRecord into a fixed-length feature vector.

    Args:
        record: Parsed Samsung patient record
        age_est: Estimated age (used for HR reserve calculation)

    Returns:
        numpy array of shape (NUM_FEATURES,) in the order of FEATURE_NAMES
    """
    agg = record.aggregates if record.aggregates else record.aggregate()
    hr = record.heart_rate
    stress = record.stress
    sleep = record.sleep_stages
    ped = record.pedometer
    ex = record.exercise
    floors = record.floors

    # --- Heart Rate ---
    hr_bpms = [r.hr_bpm for r in hr] if hr else [72]
    hr_mean = agg.get("hr_mean", 72)
    hr_std = agg.get("hr_std", 8)
    hr_min = agg.get("hr_min", 50)
    hr_max = agg.get("hr_max", 120)
    hr_resting = agg.get("hr_resting", 65)
    hr_range = hr_max - hr_min
    hr_high_pct = _hr_by_condition(hr, lambda x: x > 100)
    hr_tachy_pct = _hr_by_condition(hr, lambda x: x > 110)
    hr_brady_pct = _hr_by_condition(hr, lambda x: x < 60)

    # --- HRV ---
    hrv_mean = agg.get("hrv_mean", 45)
    rmssd_proxy = _hr_rmssd_proxy(hr)
    vo2_max = agg.get("vo2_max", 0)
    # Cardiac fitness: VO2max/resting_hr ratio (higher = better)
    cardiac_fitness = vo2_max / max(hr_resting, 1) if vo2_max > 0 else hrv_mean / max(hr_resting, 1)

    # --- Stress ---
    stress_mean = agg.get("stress_mean", 0)
    stress_max = agg.get("stress_max", 0)
    stress_std = agg.get("stress_std", 0)
    stress_high_pct = agg.get("stress_high_pct", 0)
    stress_morning = _stress_by_hour(stress, 1, 7)   # 06:00-12:00 IST = 01:00-07:00 UTC
    stress_evening = _stress_by_hour(stress, 13, 19) # 18:00-00:00 IST = 13:00-19:00 UTC

    # --- Sleep ---
    sleep_total_hours = agg.get("sleep_total_hours", 0)
    # Per-night average: divide by number of unique nights
    unique_nights = len(set(s.start.date() for s in sleep)) if sleep else 1
    sleep_total_hours_per_night = sleep_total_hours / max(unique_nights, 1)

    awake_pct = agg.get("sleep_awake_pct", 0)
    light_pct = agg.get("sleep_light_pct", 0)
    deep_pct = agg.get("sleep_deep_pct", 0)
    rem_pct = agg.get("sleep_rem_pct", 0)
    sleep_eff = max(0.0, 100 - awake_pct)  # efficiency = non-awake %
    sleep_consistency = _sleep_consistency(sleep)

    # --- Activity ---
    steps_mean = agg.get("steps_daily_mean", 0)
    steps_std = agg.get("steps_daily_std", 0)
    goal_days = sum(1 for p in ped if p.step_count >= 10000)
    total_days = max(len(ped), 1)
    steps_goal_pct = (goal_days / total_days) * 100
    active_time_min = agg.get("active_time_daily_mean_min", 0)
    floors_mean = agg.get("floors_daily_mean", 0)
    dist_km = agg.get("distance_daily_mean_km", 0)
    cal_mean = agg.get("calories_daily_mean", 0)

    # --- Exercise ---
    ex_count = agg.get("exercise_session_count", 0)
    ex_dur_min = agg.get("exercise_mean_duration_min", 0)
    ex_mean_hr = agg.get("exercise_mean_hr", 0)
    ex_max_hr = agg.get("exercise_max_hr", 0)
    # HR reserve %: (max_exercise_hr - resting_hr) / (max_predicted_hr - resting_hr)
    max_predicted_hr = 220 - age_est
    hr_reserve_pct = (
        (ex_max_hr - hr_resting) / max(max_predicted_hr - hr_resting, 1) * 100
        if ex_max_hr > 0 else 0
    )
    # Exercise intensity score: composite of duration, frequency, HR reserve
    ex_intensity = min(100, (ex_count * 0.3 + ex_dur_min * 0.01 + hr_reserve_pct * 0.5))

    # --- Body composition ---
    weight_kg = agg.get("weight_kg_latest", 70)
    bmi = agg.get("bmi_est", 24)
    body_fat_pct = agg.get("body_fat_pct_latest", 0)
    skeletal_muscle_kg = agg.get("skeletal_muscle_kg_latest", 0)

    # --- SpO2 ---
    spo2_mean = agg.get("spo2_mean", 97)
    spo2_min = agg.get("spo2_min", 95)

    # --- Metabolic ---
    bmr = agg.get("bmr", 1600)
    ex_total_cal = agg.get("exercise_total_calories", 0)
    # Caloric balance proxy: total exercise calories / days vs BMR
    days_tracked = max(total_days, 1)
    cal_per_day_exercise = ex_total_cal / days_tracked
    caloric_balance = cal_per_day_exercise - bmr

    # --- Assemble vector ---
    vector = np.array([
        # HR (0-8)
        hr_mean, hr_std, hr_min, hr_max, hr_resting,
        hr_range, hr_high_pct, hr_tachy_pct, hr_brady_pct,
        # HRV (9-11)
        hrv_mean, rmssd_proxy, cardiac_fitness,
        # Stress (12-17)
        stress_mean, stress_max, stress_std, stress_high_pct,
        stress_morning, stress_evening,
        # Sleep (18-24)
        sleep_total_hours_per_night, awake_pct, light_pct,
        deep_pct, rem_pct, sleep_eff, sleep_consistency,
        # Activity (25-31)
        steps_mean, steps_std, steps_goal_pct,
        active_time_min, floors_mean, dist_km, cal_mean,
        # Exercise (32-38)
        ex_count, ex_dur_min, ex_mean_hr, ex_max_hr,
        hr_reserve_pct, ex_intensity, vo2_max,
        # Body comp (39-42)
        weight_kg, bmi, body_fat_pct, skeletal_muscle_kg,
        # SpO2 (43-44)
        spo2_mean, spo2_min,
        # Metabolic (45-46)
        bmr, caloric_balance,
    ], dtype=np.float32)

    assert len(vector) == NUM_FEATURES, f"Feature vector length mismatch: {len(vector)} vs {NUM_FEATURES}"
    # Replace NaN/Inf with 0
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    return vector


def feature_vector_to_dict(vec: np.ndarray) -> Dict[str, float]:
    """Convert feature vector to named dict."""
    return {name: float(val) for name, val in zip(FEATURE_NAMES, vec)}


# ---------------------------------------------------------------------------
# Augmented cohort generation (for supervised model bootstrapping)
# ---------------------------------------------------------------------------
def generate_augmented_cohort(
    base_record: SamsungPatientRecord,
    n_synthetic: int = 600,
    noise_pct: float = 0.15,
    rng_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a synthetic cohort from one real patient for supervised model training.

    Strategy:
      1. Extract the real feature vector from the Samsung record
      2. Generate n_synthetic perturbations using Gaussian noise (±noise_pct)
      3. Derive clinical labels from feature thresholds (AHA / NIOSH / sleep guidelines)

    Returns:
      X: (n_synthetic+1, NUM_FEATURES) feature matrix (first row = real patient)
      y_cardio: (n_synthetic+1,) binary cardiovascular risk label
      y_sleep: (n_synthetic+1,) sleep quality index 0–100
      y_stress_class: (n_synthetic+1,) stress class 0=low, 1=med, 2=high
    """
    rng = np.random.default_rng(rng_seed)

    # Real patient features
    real_features = engineer_features(base_record)

    # Generate perturbations
    noise = rng.normal(0, noise_pct, (n_synthetic, NUM_FEATURES))
    # Only perturb relative to real values (multiplicative for non-zero features)
    base = np.tile(real_features, (n_synthetic, 1))
    perturbed = base * (1.0 + noise)
    # Clip to physiological bounds
    perturbed[:, 0] = np.clip(perturbed[:, 0], 40, 200)   # hr_mean
    perturbed[:, 39] = np.clip(perturbed[:, 39], 40, 200) # weight_kg
    perturbed[:, 40] = np.clip(perturbed[:, 40], 14, 50)  # bmi
    perturbed[:, 41] = np.clip(perturbed[:, 41], 3, 60)   # body_fat_pct
    perturbed[:, 43] = np.clip(perturbed[:, 43], 85, 100) # spo2_mean
    perturbed[:, 44] = np.clip(perturbed[:, 44], 80, 100) # spo2_min

    # Combine real + synthetic
    X = np.vstack([real_features.reshape(1, -1), perturbed]).astype(np.float32)

    # --- Derive labels ---
    def _cardio_risk(x: np.ndarray) -> int:
        """
        Elevated cardiovascular risk if:
        - Resting HR > 80 bpm (AHA risk factor)
        - OR stress_high_pct > 40% (chronic stress → cortisol → CVD)
        - OR low HRV (< 30ms proxy) indicating autonomic dysfunction
        - OR BMI > 30 (obesity)
        - OR SpO2 < 95% mean (hypoxemia risk)
        """
        resting = x[FEATURE_NAMES.index("hr_resting")]
        stress_hi = x[FEATURE_NAMES.index("stress_high_pct")]
        hrv = x[FEATURE_NAMES.index("hrv_mean")]
        rmssd = x[FEATURE_NAMES.index("hrv_rmssd_proxy")]
        bmi = x[FEATURE_NAMES.index("bmi")]
        spo2 = x[FEATURE_NAMES.index("spo2_mean")]
        risk_score = 0
        if resting > 80:     risk_score += 2
        if resting > 90:     risk_score += 1
        if stress_hi > 40:   risk_score += 2
        if hrv < 30 and hrv > 0:  risk_score += 1
        if rmssd > 15:       risk_score -= 1  # good HRV variability = protective
        if bmi > 30:         risk_score += 2
        if bmi > 35:         risk_score += 1
        if spo2 < 95 and spo2 > 0: risk_score += 2
        return int(risk_score >= 3)

    def _sleep_quality(x: np.ndarray) -> float:
        """
        Sleep quality index 0–100.
        Optimal: 7-9h/night, >20% deep, >20% REM, <15% awake.
        """
        hours = x[FEATURE_NAMES.index("sleep_total_hours")]
        awake = x[FEATURE_NAMES.index("sleep_awake_pct")]
        deep = x[FEATURE_NAMES.index("sleep_deep_pct")]
        rem = x[FEATURE_NAMES.index("sleep_rem_pct")]
        consistency = x[FEATURE_NAMES.index("sleep_consistency")]
        score = 50.0
        # Duration score
        if 7 <= hours <= 9:   score += 20
        elif 6 <= hours < 7:  score += 10
        elif 9 < hours <= 10: score += 10
        elif hours < 5:       score -= 20
        elif hours > 10:      score -= 10
        # Quality score
        score += min(20, deep * 0.5)   # more deep sleep = better (up to 20pt)
        score += min(15, rem * 0.4)    # more REM = better (up to 15pt)
        score -= min(25, awake * 0.5)  # more awake time = worse
        score -= min(10, consistency * 3)  # inconsistency = worse
        return float(np.clip(score, 0, 100))

    def _stress_class(x: np.ndarray) -> int:
        """0=low, 1=medium, 2=high"""
        s_mean = x[FEATURE_NAMES.index("stress_mean")]
        s_max = x[FEATURE_NAMES.index("stress_max")]
        s_high_pct = x[FEATURE_NAMES.index("stress_high_pct")]
        if s_mean < 35 and s_high_pct < 15:  return 0
        if s_mean >= 60 or s_high_pct >= 40: return 2
        return 1

    y_cardio = np.array([_cardio_risk(x) for x in X], dtype=np.int32)
    y_sleep = np.array([_sleep_quality(x) for x in X], dtype=np.float32)
    y_stress = np.array([_stress_class(x) for x in X], dtype=np.int32)

    logger.info(
        f"Augmented cohort: {len(X)} samples | "
        f"cardio_risk={y_cardio.mean():.2%} | "
        f"sleep_quality_mean={y_sleep.mean():.1f} | "
        f"stress: low={( y_stress==0).mean():.1%} med={(y_stress==1).mean():.1%} high={(y_stress==2).mean():.1%}"
    )
    return X, y_cardio, y_sleep, y_stress
