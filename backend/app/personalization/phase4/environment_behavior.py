"""
Phase 4: Environmental and Behavioral Models.

Models external factors that modulate the twin:
  - Air quality (AQI, PM2.5, PM10, NO2) → inflammatory & CV effects
  - Ambient temperature → thermoregulation & CV load
  - Treatment adherence stochastic process
  - Lifestyle stability (diet, exercise, sleep regularity)

Each factor produces coupling signals that the Multi-Scale Engine
injects into the appropriate layer (molecular, cellular, organ).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum


# ── Environment State ─────────────────────────────────────────

@dataclass
class EnvironmentState:
    """
    Current environmental conditions.
    All values are normalized to [0, 1] for model consumption.
    """
    aqi: float = 0.2           # Air Quality Index (0=pristine, 1=very unhealthy)
    pm25: float = 0.15         # PM2.5 concentration (normalized)
    pm10: float = 0.15         # PM10 concentration (normalized)
    no2: float = 0.1           # NO2 concentration (normalized)
    temperature: float = 0.5   # Ambient temp (0=very cold, 0.5=optimal, 1=very hot)
    humidity: float = 0.5      # Relative humidity (0-1)
    noise_level: float = 0.2   # Noise pollution (0=silent, 1=extreme)
    season: int = 0            # 0=spring, 1=summer, 2=autumn, 3=winter

    def to_dict(self) -> Dict[str, float]:
        return {
            "aqi": self.aqi, "pm25": self.pm25, "pm10": self.pm10,
            "no2": self.no2, "temperature": self.temperature,
            "humidity": self.humidity, "noise_level": self.noise_level,
            "season": float(self.season),
        }


@dataclass
class BehavioralState:
    """
    Current patient behavioral state.
    """
    diet_quality: float = 0.5       # 0=very poor, 1=optimal
    exercise_minutes: float = 30.0  # daily exercise minutes
    exercise_adherence: float = 0.7  # 0-1
    sleep_hours: float = 7.0        # hours per night
    sleep_regularity: float = 0.8   # 0=chaotic, 1=perfectly regular
    medication_adherence: float = 0.85  # 0-1
    stress_level: float = 0.3       # 0=calm, 1=extreme stress
    smoking: float = 0.0            # 0=non-smoker, 1=heavy smoker
    alcohol: float = 0.1            # 0=abstinent, 1=heavy use
    physical_activity: float = 0.5  # overall activity level (0-1)

    def to_dict(self) -> Dict[str, float]:
        return {
            "diet_quality": self.diet_quality,
            "exercise_minutes": self.exercise_minutes,
            "exercise_adherence": self.exercise_adherence,
            "sleep_hours": self.sleep_hours,
            "sleep_regularity": self.sleep_regularity,
            "medication_adherence": self.medication_adherence,
            "stress_level": self.stress_level,
            "smoking": self.smoking,
            "alcohol": self.alcohol,
            "physical_activity": self.physical_activity,
        }


# ── Environmental Model ───────────────────────────────────────

class EnvironmentalModel:
    """
    Models how environmental factors affect physiological signals.

    Provides coupling signals to molecular, cellular, and organ layers.
    """

    def __init__(self):
        pass

    def compute_coupling_signals(
        self, env: EnvironmentState, behaviors: BehavioralState,
    ) -> Dict[str, float]:
        """
        Compute environmental coupling signals for each twin layer.

        Returns:
            oxidative_stress_mod: multiplier for molecular oxidative stress
            inflammatory_mod: additive inflammation signal from pollution
            cv_load_mod: cardiovascular load from temperature + AQI
            thermoregulation_signal: hot/cold stress signal
            pulmonary_stress: lung irritation from pollution
        """
        # Air quality → oxidative stress + inflammation
        aqi_effect = 0.5 * env.aqi + 0.3 * env.pm25 + 0.2 * env.no2
        oxidative_stress_mod = 1.0 + 2.0 * aqi_effect

        # Pollution → inflammatory signal (CRP-like)
        inflam_from_pollution = 0.3 * env.aqi + 0.2 * env.no2

        # Temperature → CV load
        temp_deviation = abs(env.temperature - 0.5) * 2.0  # 0 at optimal, 1 at extremes
        cv_load = 1.0 + 0.3 * temp_deviation + 0.2 * aqi_effect

        # Extreme temperatures → thermoregulatory stress
        if env.temperature < 0.2:  # Very cold
            thermo_stress = 1.0 - env.temperature / 0.2
        elif env.temperature > 0.8:  # Very hot
            thermo_stress = (env.temperature - 0.8) / 0.2
        else:
            thermo_stress = 0.0

        # PM2.5/PM10 → pulmonary / vascular stress
        pulmonary = 0.5 * env.pm25 + 0.3 * env.pm10

        # Synergy: pollution + heat is worse
        synergy = aqi_effect * thermo_stress * 2.0

        return {
            "oxidative_stress_mod": float(oxidative_stress_mod),
            "inflammatory_mod": float(inflam_from_pollution + synergy),
            "cv_load_mod": float(cv_load),
            "thermoregulation_signal": float(thermo_stress),
            "pulmonary_stress": float(pulmonary),
        }


# ── Behavioral Model ──────────────────────────────────────────

class BehavioralModel:
    """
    Models behavioral dynamics and their physiological coupling.

    Includes stochastic adherence processes.
    """

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng()

    def compute_coupling_signals(
        self, behaviors: BehavioralState,
    ) -> Dict[str, float]:
        """
        Compute behavioral coupling signals for each twin layer.

        Returns:
            insulin_sensitivity_bhv: multiplier from exercise (1.0-1.4)
            inflammation_bhv: additive from stress + smoking + alcohol
            metabolic_health_bhv: multiplier from diet quality
            cardiovascular_bhv: multiplier from exercise + smoking
            sleep_quality_signal: sleep quality score (0-1)
            adherence_composite: overall adherence (0-1)
        """
        # Exercise → insulin sensitivity (0-40% improvement)
        ex_benefit = 0.01 * behaviors.exercise_minutes * behaviors.exercise_adherence
        insulin_sens_mod = 1.0 + 0.4 * min(ex_benefit / 30.0, 1.0)

        # Diet → metabolic health
        met_health = 0.6 + 0.4 * behaviors.diet_quality

        # Stress + smoking + alcohol → inflammation stress
        inflam = (
            0.3 * behaviors.stress_level +
            0.4 * behaviors.smoking +
            0.2 * behaviors.alcohol
        )

        # Exercise + smoking → CV health
        cv_benefit = 0.3 * min(ex_benefit / 30.0, 1.0)
        cv_risk = 0.3 * behaviors.smoking + 0.2 * behaviors.alcohol
        cv_mod = 1.0 + cv_benefit - cv_risk

        # Sleep quality
        sleep_quality = (
            0.5 * (1.0 - abs(behaviors.sleep_hours - 7.5) / 4.0) +
            0.5 * behaviors.sleep_regularity
        )
        sleep_quality = max(0.0, min(1.0, sleep_quality))

        # Composite adherence
        adherence = (
            0.4 * behaviors.medication_adherence +
            0.3 * behaviors.exercise_adherence +
            0.3 * behaviors.sleep_regularity
        )

        return {
            "insulin_sensitivity_bhv": float(insulin_sens_mod),
            "inflammation_bhv": float(inflam),
            "metabolic_health_bhv": float(met_health),
            "cardiovascular_bhv": float(cv_mod),
            "sleep_quality_signal": float(sleep_quality),
            "adherence_composite": float(adherence),
        }


# ── Advanced Adherence Model ──────────────────────────────────

class AdherenceModel:
    """
    Stochastic adherence model with state-dependent transitions.

    Uses a two-state Markov chain (adherent / non-adherent) with
    transition probabilities influenced by:
      - Habit strength (τ_days)
      - Side effects
      - Cognitive burden (number of medications)
      - Social support
      - Time since last visit
    """

    def __init__(self, habit_strength: float = 0.3,
                 side_effect_tolerance: float = 0.8,
                 cognitive_load: float = 0.2,
                 social_support: float = 0.5,
                 base_adherence: float = 0.85,
                 rng: Optional[np.random.Generator] = None):
        self.habit_strength = habit_strength
        self.side_effect_tolerance = side_effect_tolerance
        self.cognitive_load = cognitive_load
        self.social_support = social_support
        self.base_adherence = base_adherence
        self.rng = rng or np.random.default_rng()
        self._state = 1.0  # 1 = adherent, 0 = non-adherent

    def step(self, dt: float = 1.0) -> float:
        """
        Advance adherence state by dt days.
        Returns current adherence (0 or 1).
        """
        # Transition probabilities
        logit_stay_adherent = (
            1.5 +
            2.0 * self.habit_strength +
            1.0 * self.social_support -
            1.0 * (1.0 - self.side_effect_tolerance) -
            1.5 * self.cognitive_load
        )
        p_stay_adherent = 1.0 / (1.0 + np.exp(-logit_stay_adherent))

        logit_stay_non = (
            -1.0 +
            2.0 * (1.0 - self.side_effect_tolerance) +
            1.5 * self.cognitive_load -
            1.0 * self.social_support -
            0.5 * self.habit_strength
        )
        p_stay_non_adherent = 1.0 / (1.0 + np.exp(-logit_stay_non))

        # dt-scaled transition rates
        if self._state > 0.5:
            p_transition = (1.0 - p_stay_adherent) * min(dt / 7.0, 1.0)
        else:
            p_transition = (1.0 - p_stay_non_adherent) * min(dt / 14.0, 1.0)

        if self.rng.random() < p_transition:
            self._state = 1.0 - self._state

        return self._state

    def get_adherence_rate(self, window_days: int = 30) -> float:
        """Compute effective adherence rate."""
        return self.base_adherence * (0.5 + 0.5 * self._state)

    def reset(self) -> None:
        self._state = 1.0


# ── Lifestyle Stability ───────────────────────────────────────

class LifestyleModel:
    """
    Models day-to-day variability in lifestyle factors.
    Generates realistic behavioral trajectories.
    """

    def __init__(self, base_state: Optional[BehavioralState] = None,
                 rng: Optional[np.random.Generator] = None):
        self.base = base_state or BehavioralState()
        self.rng = rng or np.random.default_rng()
        self._current = BehavioralState(
            diet_quality=self.base.diet_quality,
            exercise_minutes=self.base.exercise_minutes,
            exercise_adherence=self.base.exercise_adherence,
            sleep_hours=self.base.sleep_hours,
            sleep_regularity=self.base.sleep_regularity,
            medication_adherence=self.base.medication_adherence,
            stress_level=self.base.stress_level,
            smoking=self.base.smoking,
            alcohol=self.base.alcohol,
            physical_activity=self.base.physical_activity,
        )

    def step(self, dt: float = 1.0) -> BehavioralState:
        """
        Advance lifestyle by dt days with stochastic variation.
        Returns the new behavioral state.
        """
        # Mean-reverting random walk
        daily_noise_scale = 0.05 * np.sqrt(dt)

        self._current.diet_quality += (
            0.1 * (self.base.diet_quality - self._current.diet_quality) * dt +
            self.rng.normal(0, daily_noise_scale)
        )
        self._current.exercise_minutes += (
            0.05 * (self.base.exercise_minutes - self._current.exercise_minutes) * dt +
            self.rng.normal(0, 3.0 * np.sqrt(dt))
        )
        self._current.exercise_adherence += (
            0.1 * (self.base.exercise_adherence - self._current.exercise_adherence) * dt +
            self.rng.normal(0, daily_noise_scale)
        )
        self._current.sleep_hours += (
            0.1 * (self.base.sleep_hours - self._current.sleep_hours) * dt +
            self.rng.normal(0, 0.3 * np.sqrt(dt))
        )
        self._current.sleep_regularity += (
            0.05 * (self.base.sleep_regularity - self._current.sleep_regularity) * dt +
            self.rng.normal(0, daily_noise_scale)
        )
        self._current.medication_adherence += (
            0.1 * (self.base.medication_adherence - self._current.medication_adherence) * dt +
            self.rng.normal(0, daily_noise_scale)
        )
        self._current.stress_level += (
            0.05 * (self.base.stress_level - self._current.stress_level) * dt +
            self.rng.normal(0, daily_noise_scale)
        )
        self._current.smoking += (
            0.05 * (self.base.smoking - self._current.smoking) * dt +
            self.rng.normal(0, daily_noise_scale * 0.5)
        )
        self._current.alcohol += (
            0.05 * (self.base.alcohol - self._current.alcohol) * dt +
            self.rng.normal(0, daily_noise_scale)
        )
        self._current.physical_activity += (
            0.1 * (self.base.physical_activity - self._current.physical_activity) * dt +
            self.rng.normal(0, daily_noise_scale)
        )

        # Clamp to valid ranges
        for attr in ["diet_quality", "exercise_adherence", "sleep_regularity",
                      "medication_adherence", "stress_level", "smoking", "alcohol",
                      "physical_activity"]:
            setattr(self._current, attr,
                    np.clip(getattr(self._current, attr), 0.0, 1.0))
        self._current.exercise_minutes = max(0.0, self._current.exercise_minutes)
        self._current.sleep_hours = np.clip(self._current.sleep_hours, 3.0, 12.0)

        return self._current

    def reset(self) -> None:
        self._current = BehavioralState(
            diet_quality=self.base.diet_quality,
            exercise_minutes=self.base.exercise_minutes,
            exercise_adherence=self.base.exercise_adherence,
            sleep_hours=self.base.sleep_hours,
            sleep_regularity=self.base.sleep_regularity,
            medication_adherence=self.base.medication_adherence,
            stress_level=self.base.stress_level,
            smoking=self.base.smoking,
            alcohol=self.base.alcohol,
            physical_activity=self.base.physical_activity,
        )
