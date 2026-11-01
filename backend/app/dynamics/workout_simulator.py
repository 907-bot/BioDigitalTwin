"""
Workout & Progressive Overload Simulator Module.

Simulates acute physiological responses and longitudinal adaptation (hypertrophy,
strength gains, bone mineral density remodeling, resting HR reduction, recovery)
over 1 to 90 days of structured physical exercise.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class WorkoutSessionRequest(BaseModel):
    patient_id: str = Field(..., example="P_USER_001")
    workout_type: str = Field("hypertrophy", pattern="^(hypertrophy|strength|endurance|plyometric)$")
    intensity_pct: float = Field(75.0, ge=30.0, le=100.0, description="% 1RM or % HRmax")
    duration_mins: float = Field(60.0, ge=15.0, le=180.0)
    sets: int = Field(4, ge=1, le=20)
    reps: int = Field(10, ge=1, le=50)
    frequency_per_week: int = Field(4, ge=1, le=7)
    protein_intake_g_day: float = Field(140.0, ge=40.0, le=300.0)
    horizon_days: int = Field(30, ge=1, le=90)


def simulate_workout_trajectory(
    initial_muscle_mass: float = 34.0,
    initial_strength: float = 50.0,
    initial_bmd: float = 1.20,
    initial_resting_hr: float = 70.0,
    initial_fat_mass: float = 14.0,
    workout_type: str = "hypertrophy",
    intensity_pct: float = 75.0,
    duration_mins: float = 60.0,
    sets: int = 4,
    reps: int = 10,
    frequency_per_week: int = 4,
    protein_intake_g_day: float = 140.0,
    horizon_days: int = 30,
) -> Dict[str, Any]:
    """
    Simulates acute workout bouts and multi-week progressive overload adaptations.
    """
    days = list(range(1, horizon_days + 1))
    
    # Workout schedule (days of the week)
    workout_days_set = set()
    interval = max(1, round(7 / frequency_per_week))
    for d in range(1, horizon_days + 1):
        if (d - 1) % interval == 0:
            workout_days_set.add(d)

    # Base parameters per workout type
    type_multipliers = {
        "hypertrophy": {"mtor": 1.2, "strain": 0.8, "cardio": 0.4, "fat_burn": 0.7},
        "strength":    {"mtor": 0.9, "strain": 1.5, "cardio": 0.3, "fat_burn": 0.5},
        "endurance":   {"mtor": 0.3, "strain": 0.5, "cardio": 1.5, "fat_burn": 1.4},
        "plyometric":  {"mtor": 0.7, "strain": 1.8, "cardio": 0.8, "fat_burn": 0.9},
    }
    mult = type_multipliers.get(workout_type, type_multipliers["hypertrophy"])

    volume_load = (sets * reps * (intensity_pct / 100.0)) / 10.0
    mtor_stimulus = mult["mtor"] * volume_load * (protein_intake_g_day / 120.0)
    strain_stimulus = mult["strain"] * volume_load
    cardio_stimulus = mult["cardio"] * (intensity_pct / 100.0) * (duration_mins / 60.0)

    trajectory = []
    
    current_muscle = initial_muscle_mass
    current_strength = initial_strength
    current_bmd = initial_bmd
    current_hr = initial_resting_hr
    current_fat = initial_fat_mass
    mtor_signal = 0.1
    recovery_score = 90.0

    for d in days:
        is_workout_day = d in workout_days_set
        
        if is_workout_day:
            mtor_signal = min(5.0, mtor_signal + mtor_stimulus * 0.5)
            acute_il6_spike = 3.5 * (intensity_pct / 100.0)
            acute_cortisol_spike = 120.0 * (duration_mins / 60.0)
            recovery_score = max(40.0, recovery_score - 25.0)
        else:
            mtor_signal *= 0.65  # decay
            acute_il6_spike = 0.5
            acute_cortisol_spike = 0.0
            recovery_score = min(100.0, recovery_score + 20.0)

        # Net Muscle Protein Synthesis (MPS) gain over time
        net_mps = 0.02 * mtor_signal * (protein_intake_g_day / 140.0) - 0.005
        current_muscle += net_mps * (1 / 7.0)  # weekly scale
        current_muscle = max(10.0, current_muscle)

        # Strength capacity progressive overload
        strength_gain = 0.15 * (intensity_pct / 100.0) * (sets / 4.0) if is_workout_day else 0.02
        current_strength += strength_gain
        
        # Bone mineral density remodeling (Wolff's law)
        bmd_gain = 0.0003 * strain_stimulus if is_workout_day else 0.00005
        current_bmd += bmd_gain

        # Resting HR reduction from cardio adaptation
        hr_reduction = 0.08 * cardio_stimulus if is_workout_day else 0.01
        current_hr = max(45.0, current_hr - hr_reduction)

        # Fat mass loss from caloric expenditure
        fat_loss = 0.04 * mult["fat_burn"] if is_workout_day else 0.01
        current_fat = max(3.0, current_fat - fat_loss)

        trajectory.append({
            "day": d,
            "is_workout_day": is_workout_day,
            "muscle_mass_kg": round(float(current_muscle), 3),
            "strength_index": round(float(current_strength), 2),
            "hypertrophy_signal": round(float(mtor_signal), 3),
            "bmd_g_cm2": round(float(current_bmd), 4),
            "resting_hr": round(float(current_hr), 1),
            "fat_mass_kg": round(float(current_fat), 2),
            "recovery_score": round(float(recovery_score), 1),
            "acute_il6_spike": round(float(acute_il6_spike), 2),
        })

    total_muscle_gain = round(current_muscle - initial_muscle_mass, 2)
    total_strength_gain = round(current_strength - initial_strength, 1)
    total_bmd_gain_pct = round(((current_bmd - initial_bmd) / initial_bmd) * 100.0, 2)
    total_hr_drop = round(initial_resting_hr - current_hr, 1)

    return {
        "workout_type": workout_type,
        "horizon_days": horizon_days,
        "frequency_per_week": frequency_per_week,
        "total_workouts_completed": len(workout_days_set),
        "summary": {
            "initial_muscle_mass_kg": initial_muscle_mass,
            "final_muscle_mass_kg": round(current_muscle, 2),
            "muscle_mass_delta_kg": total_muscle_gain,
            "initial_strength_index": initial_strength,
            "final_strength_index": round(current_strength, 1),
            "strength_delta": total_strength_gain,
            "bmd_gain_pct": total_bmd_gain_pct,
            "resting_hr_reduction_bpm": total_hr_drop,
        },
        "trajectory": trajectory,
    }
