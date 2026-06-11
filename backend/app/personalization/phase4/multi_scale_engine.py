"""
Phase 4: Multi-Scale Twin Engine.

Hierarchical multi-rate estimator coordinating:
  - Molecular layer (75D, dt=1min, τ~hours)
  - Cellular layer (25D, dt=15min, τ~days)
  - Organ layer (30D physio + 25D params, dt=1min, from Phase 3 UKF)
  - Whole-body layer (daily behavior/environment, dt=1440min)

Each layer runs at its native dt with coupling signals flowing
up and down the hierarchy via interpolation.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from enum import Enum

from app.personalization.dynamics import (
    full_dynamics, full_observation,
    compute_metabolic_dynamics, compute_cardio_dynamics,
    compute_renal_dynamics, compute_inflammation_dynamics,
    compute_circadian_dynamics, compute_adipose_dynamics,
    compute_immune_dynamics,
)
from app.personalization.core import UnscentedKalmanFilter
from app.personalization.state import PHYSIO_DIM, PARAM_DIM, OBS_DIM
from app.personalization.phase4.molecular import (
    MolecularState, compute_molecular_dynamics, molecular_to_cellular_signals,
)
from app.personalization.phase4.cellular import (
    CellularState, CellPopulationDynamics, cellular_to_organ_signals,
    CELL_TYPES,
)
from app.personalization.phase4.environment_behavior import (
    EnvironmentState, BehavioralState,
    EnvironmentalModel, BehavioralModel, AdherenceModel, LifestyleModel,
)


# ── Layer Identifiers ─────────────────────────────────────────

class TwinLayer(Enum):
    MOLECULAR = "molecular"
    CELLULAR = "cellular"
    ORGAN = "organ"
    WHOLE_BODY = "whole_body"


# ── Multi-Rate Clock ──────────────────────────────────────────

@dataclass
class LayerClock:
    """
    Tracks simulation time for a single layer.
    Each layer runs at its native step size.
    """
    dt: float                    # native time step (minutes)
    last_step: float = 0.0       # last simulation time (minutes)
    steps_per_day: int = 0

    def __post_init__(self):
        self.steps_per_day = max(1, int(1440.0 / self.dt))

    def is_ready(self, current_time: float) -> bool:
        """Check if the layer should step at current_time."""
        return current_time - self.last_step >= self.dt - 1e-9

    def step(self, current_time: float) -> None:
        self.last_step = current_time

    def reset(self) -> None:
        self.last_step = 0.0


# ── Multi-Scale State ─────────────────────────────────────────

@dataclass
class MultiScaleState:
    """
    Complete multi-scale twin state at a point in time.

    This composes all layers into a single unified state.
    """
    molecular: MolecularState
    cellular: CellularState
    organ_physio: np.ndarray       # (PHYSIO_DIM,) — from Phase 3
    organ_params: np.ndarray       # (PARAM_DIM,) — from Phase 3
    environment: EnvironmentState
    behaviors: BehavioralState
    timestamp: float = 0.0         # simulation time (minutes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "molecular": {
                "genes": self.molecular.gene_expression.tolist(),
                "proteins": self.molecular.protein_activity.tolist(),
                "metabolites": self.molecular.metabolite_levels.tolist(),
                "pathways": list(self.molecular.pathway_activity.values()),
            },
            "cellular": {
                "population": self.cellular.population.tolist(),
                "stress": self.cellular.stress.tolist(),
                "health": self.cellular.health.tolist(),
            },
            "organ": {
                "physio": self.organ_physio.tolist(),
                "params": self.organ_params.tolist(),
            },
            "environment": self.environment.to_dict(),
            "behaviors": self.behaviors.to_dict(),
        }


# ── Coupling Signal Manager ───────────────────────────────────

class CouplingSignalManager:
    """
    Manages coupling signals between layers.

    Upward signals (molecular → cellular → organ):
      - Molecular → Cellular: insulin, inflammatory, metabolic stress, growth signals
      - Cellular → Organ: insulin sensitivity mod, inflammation mod, beta cell fn, etc.

    Downward signals (organ → cellular → molecular):
      - Organ → Cellular: glucose toxicity, BP overload, etc.
      - Cellular → Molecular: cellular health modulates gene expression rates
    """

    def __init__(self):
        # Upward signals from molecular → cellular
        self.mol_to_cell: Dict[str, float] = {
            "insulin_signal": 0.3,
            "inflammatory_signal": 0.1,
            "metabolic_stress": 0.1,
            "growth_signals": 0.5,
        }
        # Upward signals from cellular → organ
        self.cell_to_organ: Dict[str, float] = {
            "insulin_sensitivity_mod": 1.0,
            "inflammation_mod": 0.0,
            "beta_cell_function": 1.0,
            "vascular_health_mod": 1.0,
            "metabolic_rate_mod": 1.0,
        }
        # Downward signals from organ → cellular
        self.organ_to_cell: Dict[str, float] = {
            "glucose_toxicity": 0.0,
            "bp_overload": 0.0,
        }
        # Downward signals from cellular → molecular
        self.cell_to_mol: Dict[str, float] = {
            "cellular_health_mod": 1.0,
            "cellular_stress_mod": 1.0,
        }
        # Environmental coupling signals
        self.env_to_all: Dict[str, float] = {
            "oxidative_stress_mod": 1.0,
            "inflammatory_mod": 0.0,
            "cv_load_mod": 1.0,
            "thermoregulation_signal": 0.0,
            "pulmonary_stress": 0.0,
        }
        # Behavioral coupling signals
        self.bhv_to_all: Dict[str, float] = {
            "insulin_sensitivity_bhv": 1.0,
            "inflammation_bhv": 0.0,
            "metabolic_health_bhv": 1.0,
            "cardiovascular_bhv": 1.0,
            "sleep_quality_signal": 1.0,
            "adherence_composite": 1.0,
        }

    def update_upward(self, mol_to_cell: Dict[str, float],
                      cell_to_organ: Dict[str, float]) -> None:
        self.mol_to_cell.update(mol_to_cell)
        self.cell_to_organ.update(cell_to_organ)

    def update_downward(self, organ_to_cell: Dict[str, float],
                        cell_to_mol: Dict[str, float]) -> None:
        self.organ_to_cell.update(organ_to_cell)
        self.cell_to_mol.update(cell_to_mol)

    def update_env(self, signals: Dict[str, float]) -> None:
        self.env_to_all.update(signals)

    def update_behavior(self, signals: Dict[str, float]) -> None:
        self.bhv_to_all.update(signals)


# ── Multi-Scale Engine ────────────────────────────────────────

class MultiScaleTwinEngine:
    """
    Coordinates all four twin layers with multi-rate estimation.

    Standard usage:
        engine = MultiScaleTwinEngine()
        engine.initialize(patient_id, molecular_state, cellular_state, ...)
        for step in range(n_steps):
            engine.step(dt_days=1.0, observation=obs)
    """

    def __init__(
        self,
        ukf: Optional[UnscentedKalmanFilter] = None,
        cell_dynamics: Optional[CellPopulationDynamics] = None,
        env_model: Optional[EnvironmentalModel] = None,
        bhv_model: Optional[BehavioralModel] = None,
        lifestyle: Optional[LifestyleModel] = None,
        adherence: Optional[AdherenceModel] = None,
    ):
        # Layer dynamics
        self.ukf = ukf
        self.cell_dynamics = cell_dynamics or CellPopulationDynamics()

        # Environmental / behavioral models
        self.env_model = env_model or EnvironmentalModel()
        self.bhv_model = bhv_model or BehavioralModel()
        self.lifestyle = lifestyle or LifestyleModel()
        self.adherence = adherence or AdherenceModel()

        # Coupling signals
        self.coupling = CouplingSignalManager()

        # Multi-rate clocks (all times in minutes)
        self.clocks = {
            TwinLayer.MOLECULAR: LayerClock(dt=1.0),
            TwinLayer.CELLULAR: LayerClock(dt=15.0),
            TwinLayer.ORGAN: LayerClock(dt=1.0),
            TwinLayer.WHOLE_BODY: LayerClock(dt=1440.0),
        }

        # State
        self.state: Optional[MultiScaleState] = None
        self._patient_id: Optional[str] = None
        self._current_time: float = 0.0
        self._step_count: int = 0
        self._history: List[MultiScaleState] = []

    # ── Initialization ──

    def initialize(
        self,
        patient_id: str,
        molecular: MolecularState,
        cellular: CellularState,
        organ_physio: np.ndarray,
        organ_params: np.ndarray,
        environment: Optional[EnvironmentState] = None,
        behaviors: Optional[BehavioralState] = None,
    ) -> None:
        self._patient_id = patient_id
        self.state = MultiScaleState(
            molecular=molecular,
            cellular=cellular,
            organ_physio=organ_physio.copy(),
            organ_params=organ_params.copy(),
            environment=environment or EnvironmentState(),
            behaviors=behaviors or BehavioralState(),
            timestamp=0.0,
        )
        self._current_time = 0.0
        self._step_count = 0
        self._history = [self.state]
        for clock in self.clocks.values():
            clock.reset()

    # ── Step ──

    def step(self, dt_days: float = 1.0,
             observation: Optional[np.ndarray] = None,
             rl_action: Optional[np.ndarray] = None) -> MultiScaleState:
        """
        Advance the multi-scale twin by dt_days.

        Each layer advances at its native dt, with coupling signals
        interpolated between layer steps.
        """
        if self.state is None:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        dt_minutes = dt_days * 1440.0
        end_time = self._current_time + dt_minutes

        while self._current_time < end_time - 1e-9:
            step_minutes = min(dt_minutes, 1.0)  # base micro-step = 1 min
            self._micro_step(step_minutes, observation)
            self._current_time += step_minutes

        self.state.timestamp = self._current_time
        self._step_count += 1
        return self.state

    def _micro_step(self, dt: float, observation: Optional[np.ndarray] = None) -> None:
        """Single 1-minute micro-step across all layers."""
        s = self.state

        # ── 1. Whole-Body Layer (daily) ──
        if self.clocks[TwinLayer.WHOLE_BODY].is_ready(self._current_time):
            wb_dt = self.clocks[TwinLayer.WHOLE_BODY].dt
            # Step lifestyle with stochastic variation
            s.behaviors = self.lifestyle.step(dt=wb_dt / 1440.0)

            # Compute environmental coupling signals
            env_signals = self.env_model.compute_coupling_signals(
                s.environment, s.behaviors,
            )
            self.coupling.update_env(env_signals)

            # Compute behavioral coupling signals
            bhv_signals = self.bhv_model.compute_coupling_signals(s.behaviors)
            self.coupling.update_behavior(bhv_signals)

            self.clocks[TwinLayer.WHOLE_BODY].step(self._current_time)

        # ── 2. Molecular Layer (every 1 min) ──
        if self.clocks[TwinLayer.MOLECULAR].is_ready(self._current_time):
            # Compute cellular stress signal (0-1) from coupling
            cellular_stress = (
                self.coupling.cell_to_mol.get("cellular_stress_mod", 1.0) - 1.0 +
                self.coupling.env_to_all.get("oxidative_stress_mod", 1.0) * 0.1
            )
            cellular_stress = float(np.clip(cellular_stress, 0.0, 1.0))
            s.molecular = compute_molecular_dynamics(
                s.molecular, cellular_stress=cellular_stress, dt=1.0,
            )

            # Molecular → Cellular coupling signals
            mol_cell = molecular_to_cellular_signals(s.molecular)
            self.coupling.update_upward(mol_to_cell=mol_cell, cell_to_organ={})

            self.clocks[TwinLayer.MOLECULAR].step(self._current_time)

        # ── 3. Organ Layer (every 1 min) ──
        if self.clocks[TwinLayer.ORGAN].is_ready(self._current_time):
            if self.ukf is not None:
                # Build organ inputs with cellular + environmental + behavioral modulation
                organ_inputs = self._build_organ_inputs()

                # Predict (propagate UKF dynamics)
                self.ukf.predict(u=organ_inputs)

                # UKF update if observation available
                if observation is not None:
                    self.ukf.update(observation)

                # Extract organ state
                physio, params = self.ukf.get_physio_state(), self.ukf.get_parameters()[0]
                s.organ_physio = physio.copy()
                s.organ_params = params.copy()

            self.clocks[TwinLayer.ORGAN].step(self._current_time)

        # ── 4. Cellular Layer (every 15 min) ──
        if self.clocks[TwinLayer.CELLULAR].is_ready(self._current_time):
            # Build cellular coupling signals from molecular + organ + env
            organ_to_cell = self._extract_organ_downstream()

            # Cellular dynamics
            s.cellular = self.cell_dynamics.compute_dynamics(
                s.cellular,
                mol_signals=self.coupling.mol_to_cell,
                organ_signals=organ_to_cell,
                dt=15.0,
            )

            # Cellular → Organ coupling
            cell_org = cellular_to_organ_signals(s.cellular)
            self.coupling.update_upward(mol_to_cell={}, cell_to_organ=cell_org)

            # Cellular → Molecular feedback
            self.coupling.cell_to_mol["cellular_health_mod"] = float(np.mean(s.cellular.health))
            self.coupling.cell_to_mol["cellular_stress_mod"] = float(np.mean(s.cellular.stress) + 1.0)

            self.clocks[TwinLayer.CELLULAR].step(self._current_time)

    def _build_organ_inputs(self) -> Dict[str, float]:
        """Build organ-layer inputs modulated by cellular and environmental signals."""
        s = self.state
        inputs = {
            # Meal / exercise defaults (overridden by RL actions later)
            "meal": 0.0,
            "exercise": 0.0,
            "medication": 0.0,
            # Light / sleep (from circadian)
            "light": 1.0 if 6.0 <= (self._current_time / 60.0) % 24.0 < 22.0 else 0.0,
            "sleep": 1.0 if (self._current_time / 60.0) % 24.0 >= 22.0 or
                           (self._current_time / 60.0) % 24.0 < 6.0 else 0.0,
        }

        # Apply cellular modulation to organ parameters
        cell_org = self.coupling.cell_to_organ
        inputs["SI_mod"] = cell_org.get("insulin_sensitivity_mod", 1.0)
        inputs["inflam_mod"] = cell_org.get("inflammation_mod", 0.0)
        inputs["beta_fn_mod"] = cell_org.get("beta_cell_function", 1.0)
        inputs["vasc_mod"] = cell_org.get("vascular_health_mod", 1.0)

        # Apply environmental modulation
        inputs["cv_load"] = self.coupling.env_to_all.get("cv_load_mod", 1.0)
        inputs["env_inflam"] = self.coupling.env_to_all.get("inflammatory_mod", 0.0)
        inputs["thermo_stress"] = self.coupling.env_to_all.get("thermoregulation_signal", 0.0)

        # Apply behavioral modulation
        inputs["bhv_ins_sens"] = self.coupling.bhv_to_all.get("insulin_sensitivity_bhv", 1.0)
        inputs["bhv_inflam"] = self.coupling.bhv_to_all.get("inflammation_bhv", 0.0)
        inputs["bhv_met_health"] = self.coupling.bhv_to_all.get("metabolic_health_bhv", 1.0)
        inputs["bhv_cv"] = self.coupling.bhv_to_all.get("cardiovascular_bhv", 1.0)
        inputs["sleep_quality"] = self.coupling.bhv_to_all.get("sleep_quality_signal", 1.0)

        return inputs

    def _extract_organ_downstream(self) -> Dict[str, float]:
        """Extract organ-level signals that flow down to cellular layer."""
        s = self.state
        G = float(s.organ_physio[0])          # glucose
        SBP = float(s.organ_physio[5])         # systolic BP
        IR = float(s.organ_physio[4])          # insulin resistance
        TNFa = float(s.organ_physio[26])       # TNF-alpha

        # Glucose toxicity: scaled glucose above 180 mg/dL
        gluc_tox = max(0.0, (G - 180.0) / 300.0)
        # BP overload: scaled SBP above 140 mmHg
        bp_overload = max(0.0, (SBP - 140.0) / 80.0)
        # Insulin resistance stress
        ir_stress = float(min(1.0, max(0.0, (IR - 3.0) / 15.0)))
        # Inflammatory cytokine signal
        cyto_stress = float(min(1.0, TNFa / 10.0))

        return {
            "glucose_toxicity": gluc_tox,
            "bp_overload": bp_overload,
            "ir_stress": ir_stress,
            "cytokine_stress": cyto_stress,
        }

    # ── Query ──

    def get_current_state(self) -> MultiScaleState:
        if self.state is None:
            raise RuntimeError("Engine not initialized")
        return self.state

    def get_molecular_state(self) -> MolecularState:
        return self.state.molecular

    def get_cellular_state(self) -> CellularState:
        return self.state.cellular

    def get_organ_state(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.state.organ_physio, self.state.organ_params

    def get_history(self) -> List[MultiScaleState]:
        return self._history

    def get_current_time(self) -> float:
        return self._current_time

    # ── Reset ──

    def reset(self) -> None:
        self.state = None
        self._patient_id = None
        self._current_time = 0.0
        self._step_count = 0
        self._history = []
        self.coupling = CouplingSignalManager()
        for clock in self.clocks.values():
            clock.reset()


# ── Convenience: Build a Default Engine ───────────────────────

def create_default_multi_scale_engine(
    ukf_kwargs: Optional[Dict] = None,
) -> MultiScaleTwinEngine:
    """Create a MultiScaleTwinEngine with default models."""
    state_dim = PHYSIO_DIM + PARAM_DIM
    process_noise = np.eye(state_dim) * 0.01
    obs_noise = np.eye(OBS_DIM) * 0.1

    def _param_prior_fn() -> np.ndarray:
        from app.personalization.priors import PRIORS
        return np.array([p.sample() for p in PRIORS], dtype=np.float64)

    ukf = UnscentedKalmanFilter(
        state_dim=state_dim,
        process_noise=process_noise,
        obs_noise=obs_noise,
        dynamics_fn=full_dynamics,
        obs_fn=full_observation,
        param_prior_fn=_param_prior_fn,
        **(ukf_kwargs or {}),
    )
    return MultiScaleTwinEngine(
        ukf=ukf,
        cell_dynamics=CellPopulationDynamics(),
        env_model=EnvironmentalModel(),
        bhv_model=BehavioralModel(),
        lifestyle=LifestyleModel(),
        adherence=AdherenceModel(),
    )
