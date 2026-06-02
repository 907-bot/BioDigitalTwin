"""
Phase 1 — Synthetic patient generator.

Produces a cohort of patients with correlated biomarkers using a
hand-defined physiological dependency structure. We deliberately avoid
a full GAN/CTGAN training step for the MVP — the dependency graph
below is medically motivated and produces realistic correlations out
of the box, which is enough for Phase 1/2.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class BiomarkerSpec:
    name: str
    mean: float
    std: float
    lo: float
    hi: float
    unit: str
    healthy_lo: float
    healthy_hi: float


BIOMARKER_SPECS: dict[str, BiomarkerSpec] = {
    "hr":            BiomarkerSpec("hr",            72,  8,  50, 110, "bpm",   60,  100),
    "hrv":           BiomarkerSpec("hrv",           45, 18,  10, 120, "ms",    20,  70),
    "spo2":          BiomarkerSpec("spo2",         96.5, 1.8, 88, 100, "%",     95,  100),
    "glucose":       BiomarkerSpec("glucose",     105,  25,  70, 250, "mg/dL", 70,  110),
    "systolic_bp":   BiomarkerSpec("systolic_bp", 125,  15,  90, 180, "mmHg",  90,  130),
    "diastolic_bp":  BiomarkerSpec("diastolic_bp", 78,  10,  60, 110, "mmHg",  60,  85),
    "bmi":           BiomarkerSpec("bmi",          26.5, 5.5, 15,  45, "kg/m2",18.5, 25),
}


def _draw_age_bmi(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    age = np.clip(rng.normal(45, 15, n).astype(int), 18, 85)
    bmi = np.clip(rng.normal(26.5, 5.5, n), 15, 45)
    return age, bmi


def _draw_biomarkers(
    n: int,
    age: np.ndarray,
    bmi: np.ndarray,
    gender: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Draw biomarkers with BMI/age/gender-driven correlations."""
    bmi_z = (bmi - 26.5) / 5.5
    age_z = (age - 45) / 15
    male = (gender == "Male").astype(float)

    rows: dict[str, np.ndarray] = {}

    # HR: higher with stress/low HRV; slightly higher in females
    rows["hr"] = np.clip(
        rng.normal(72, 8, n) - 1.2 * (rng.normal(45, 18, n) - 45) / 18 + 2 * (1 - male),
        BIOMARKER_SPECS["hr"].lo,
        BIOMARKER_SPECS["hr"].hi,
    ).astype(int)

    # HRV: decreases with age and BMI, increases with fitness (proxied inversely by HR)
    rows["hrv"] = np.clip(
        rng.normal(45, 18, n) - 8 * age_z - 6 * bmi_z,
        BIOMARKER_SPECS["hrv"].lo,
        BIOMARKER_SPECS["hrv"].hi,
    ).astype(int)

    # SpO2: mostly normal, slight dip in older/heavier patients
    rows["spo2"] = np.clip(
        rng.normal(96.5, 1.8, n) - 0.4 * bmi_z - 0.2 * age_z,
        BIOMARKER_SPECS["spo2"].lo,
        BIOMARKER_SPECS["spo2"].hi,
    )

    # Glucose: BMI is the dominant driver (insulin resistance)
    rows["glucose"] = np.clip(
        rng.normal(105, 25, n) + 18 * bmi_z + 6 * age_z,
        BIOMARKER_SPECS["glucose"].lo,
        BIOMARKER_SPECS["glucose"].hi,
    ).astype(int)

    # Blood pressure: age + BMI driven, slightly higher in males
    rows["systolic_bp"] = np.clip(
        rng.normal(125, 15, n) + 12 * age_z + 8 * bmi_z + 3 * male,
        BIOMARKER_SPECS["systolic_bp"].lo,
        BIOMARKER_SPECS["systolic_bp"].hi,
    ).astype(int)
    rows["diastolic_bp"] = np.clip(
        rng.normal(78, 10, n) + 6 * age_z + 4 * bmi_z + 2 * male,
        BIOMARKER_SPECS["diastolic_bp"].lo,
        BIOMARKER_SPECS["diastolic_bp"].hi,
    ).astype(int)

    rows["bmi"] = bmi
    return pd.DataFrame(rows)


def generate_patients(n: int = 500, seed: Optional[int] = 42) -> pd.DataFrame:
    """Return a DataFrame of `n` synthetic patients with correlated biomarkers."""
    rng = np.random.default_rng(seed)
    age, bmi = _draw_age_bmi(n, rng)
    gender = rng.choice(["Male", "Female"], n, p=[0.52, 0.48])
    biomarkers = _draw_biomarkers(n, age, bmi, gender, rng)

    df = pd.DataFrame({
        "patient_id": [f"P{str(i).zfill(6)}" for i in range(1, n + 1)],
        "age": age,
        "gender": gender,
    })
    df = pd.concat([df, biomarkers], axis=1)
    df["created_at"] = datetime.utcnow().isoformat()
    return df


def to_records(df: pd.DataFrame) -> list[dict]:
    return df.to_dict(orient="records")
