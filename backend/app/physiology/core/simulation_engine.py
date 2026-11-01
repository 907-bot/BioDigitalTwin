"""
BioDigitalTwin — Coupled Multi-Organ Simulation Engine.

Coordinates the coupled whole-body differential system:
dx/dt = f_mechanistic(x, theta, u, t) + g_ml(x, u)

Workflow per step:
1. Snapshot current organ states and shared circulating environment
2. Evaluate internal organ derivatives and inter-organ exchange fluxes
3. Verify flux conservation and balance shared transport pools
4. Numerically integrate coupled state via RK4 or adaptive Runge-Kutta solver
5. Enforce physiological clamping and conservation constraints
6. Emit structured clinical snapshot and diagnostics
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
import logging

from .state_registry import StateRegistry, get_global_registry
from .shared_environment import SharedEnvironment
from .organ_interface import OrganModel
from .flux_registry import FluxRegistry, Flux
from .conservation import ConservationValidator

logger = logging.getLogger(__name__)


@dataclass
class SimulationEvent:
    """Discrete clinical or behavioral event (meal, exercise, drug dose)."""
    time_minutes: float
    event_type: str                     # "meal", "exercise_start", "exercise_end", "drug_dose"
    payload: Dict[str, float]           # e.g. {"carbs_g": 75.0}, {"intensity": 0.6}, {"dose_mg": 10.0}


@dataclass
class WholeBodySnapshot:
    """Complete diagnostic snapshot of the human twin at time t."""
    time_minutes: float
    circulating_glucose_mg_dL: float
    circulating_insulin_uU_mL: float
    circulating_glucagon_pg_mL: float
    heart_rate_bpm: float
    cardiac_output_L_min: float
    mean_arterial_pressure_mmHg: float
    systolic_bp_mmHg: float
    diastolic_bp_mmHg: float
    arterial_paO2_mmHg: float
    arterial_paCO2_mmHg: float
    arterial_saO2_pct: float
    arterial_pH: float
    extracellular_fluid_L: float
    organ_states: Dict[str, Dict[str, float]]
    active_fluxes: List[Dict[str, Any]]
    is_physiologically_valid: bool
    violations: List[str]


class CoupledSimulationEngine:
    """
    Main driver executing the coupled multi-organ physiological world model.
    """

    def __init__(
        self,
        patient_weight_kg: float = 70.0,
        patient_height_cm: float = 175.0,
        registry: Optional[StateRegistry] = None,
    ):
        self.patient_weight_kg = patient_weight_kg
        self.patient_height_cm = patient_height_cm
        self.registry = registry or get_global_registry()

        self.environment = SharedEnvironment(patient_weight_kg, patient_height_cm)
        self.organs: Dict[str, OrganModel] = {}
        self.flux_registry = FluxRegistry()
        self.conservation_validator = ConservationValidator()

        self.current_time_minutes: float = 0.0
        self.events: List[SimulationEvent] = []
        self._history: List[WholeBodySnapshot] = []

    def register_organ(self, organ: OrganModel) -> None:
        """Adds an organ model into the coupled network."""
        if organ.organ_id in self.organs:
            raise ValueError(f"Organ '{organ.organ_id}' already registered in simulation engine")
        self.organs[organ.organ_id] = organ

    def schedule_event(self, event: SimulationEvent) -> None:
        self.events.append(event)
        self.events.sort(key=lambda e: e.time_minutes)

    def initialize(self, patient_profile: Optional[Dict[str, Any]] = None) -> None:
        """Initializes shared environment and all registered organs."""
        profile = patient_profile or {
            "weight_kg": self.patient_weight_kg,
            "height_cm": self.patient_height_cm,
            "age": 35.0,
            "sex": "male",
        }
        for organ in self.organs.values():
            organ.initialize(profile, self.environment)
        self.current_time_minutes = 0.0
        self._history.clear()

    def _process_scheduled_events(self, t_current: float, dt: float) -> Dict[str, float]:
        """Dispatches active events within window [t_current, t_current + dt)."""
        active_inputs: Dict[str, float] = {}
        for event in self.events:
            if t_current <= event.time_minutes < t_current + dt:
                if event.event_type == "meal":
                    carbs = event.payload.get("carbs_g", 0.0)
                    active_inputs["meal_carbs_g"] = active_inputs.get("meal_carbs_g", 0.0) + carbs
                    if "gastrointestinal" in self.organs:
                        gi_organ = self.organs["gastrointestinal"]
                        if hasattr(gi_organ, "ingest_meal"):
                            gi_organ.ingest_meal(carbs)
                elif event.event_type == "exercise":
                    active_inputs["exercise"] = event.payload.get("intensity", 0.0)
                elif event.event_type == "drug_dose":
                    drug_name = event.payload.get("drug", "generic")
                    dose_mg = event.payload.get("dose_mg", 0.0)
                    active_inputs[f"dose_{drug_name}_mg"] = dose_mg
        return active_inputs

    def step(
        self,
        dt_minutes: float = 1.0,
        inputs: Optional[Dict[str, float]] = None,
        solver: str = "rk4",
    ) -> WholeBodySnapshot:
        """
        Advances the entire coupled multi-organ system by dt_minutes.
        """
        if inputs is None:
            inputs = {}

        # 1. Process scheduled events and merge with runtime inputs
        event_inputs = self._process_scheduled_events(self.current_time_minutes, dt_minutes)
        combined_inputs = {**inputs, **event_inputs}
        self.environment.exercise_intensity = float(combined_inputs.get("exercise", 0.0))

        # 2. Clear flux registry for this integration interval
        self.flux_registry.clear()

        # 3. Compute derivatives & fluxes for all organs
        all_derivs: Dict[str, Dict[str, float]] = {}
        for organ_id, organ in self.organs.items():
            state_copy = organ.get_state()
            derivs = organ.compute_derivatives(self.current_time_minutes, state_copy, self.environment)
            fluxes = organ.exchange_fluxes(self.current_time_minutes, state_copy, self.environment)
            all_derivs[organ_id] = derivs
            self.flux_registry.register_fluxes(fluxes)

        # Peripheral glucose uptake (Brain CNS GLUT1 + Muscle/Adipose GLUT4 + Exercise translocation)
        g_curr = self.environment.chemistry.glucose_mg_dL
        i_curr = self.environment.chemistry.insulin_uU_mL
        ex_curr = self.environment.exercise_intensity

        # Basal CNS glucose consumption: ~82 mg/min (insulin-independent)
        u_cns = 82.0 * (g_curr / 95.0)
        # Insulin-dependent muscle & adipose disposal:
        # Basal ~65 mg/min; robustly accelerates with postprandial insulinemia (GLUT4)
        if i_curr >= 8.0:
            insulin_multiplier = 1.0 + 0.55 * ((i_curr - 8.0) / 8.0)
        else:
            insulin_multiplier = max(0.2, i_curr / 8.0)
        u_periph = 65.0 * insulin_multiplier * (g_curr / 95.0)
        # Exercise-induced GLUT4 translocation independent of insulin
        u_ex = 300.0 * ex_curr * (g_curr / 95.0)
        total_uptake = max(0.0, u_cns + u_periph + u_ex)

        self.flux_registry.register_fluxes([
            Flux(
                source="blood",
                destination="periphery",
                quantity="glucose",
                value=float(total_uptake),
                unit="mg/min",
                mechanism="Peripheral Glucose Disposal (GLUT1 Brain + GLUT4 Muscle/Adipose)",
            )
        ])

        # 4. Integrate each organ's internal state
        for organ_id, organ in self.organs.items():
            curr_state = organ.get_state()
            derivs = all_derivs[organ_id]

            if solver == "rk4":
                # Compute intermediate RK4 sub-stages
                # Half-step k2
                half_state = {k: curr_state[k] + 0.5 * dt_minutes * derivs.get(k, 0.0) for k in curr_state}
                k2 = organ.compute_derivatives(self.current_time_minutes + 0.5 * dt_minutes, half_state, self.environment)
                # Half-step k3
                half_state2 = {k: curr_state[k] + 0.5 * dt_minutes * k2.get(k, 0.0) for k in curr_state}
                k3 = organ.compute_derivatives(self.current_time_minutes + 0.5 * dt_minutes, half_state2, self.environment)
                # Full-step k4
                full_state = {k: curr_state[k] + dt_minutes * k3.get(k, 0.0) for k in curr_state}
                k4 = organ.compute_derivatives(self.current_time_minutes + dt_minutes, full_state, self.environment)

                new_state = {}
                for k in curr_state:
                    rk_deriv = (derivs.get(k, 0.0) + 2.0 * k2.get(k, 0.0) + 2.0 * k3.get(k, 0.0) + k4.get(k, 0.0)) / 6.0
                    new_state[k] = curr_state[k] + dt_minutes * rk_deriv
            else:
                # Forward Euler
                new_state = {k: curr_state[k] + dt_minutes * derivs.get(k, 0.0) for k in curr_state}

            organ.set_state(new_state)

        # 5. Apply inter-organ fluxes to circulating shared environment
        # A. Glucose balance: Influxes (gut absorption + HGP) - Effluxes (PGU uptake + Renal glycosuria)
        net_glucose = self.flux_registry.get_net_flux_to_compartment("blood", "glucose")
        self.environment.apply_solute_flux("glucose", net_glucose, dt_minutes)

        # B. Insulin balance: Secretion from beta cells into blood
        net_insulin = self.flux_registry.get_net_flux_to_compartment("blood", "insulin")
        self.environment.apply_solute_flux("insulin", net_insulin, dt_minutes)

        # C. Glucagon balance: Secretion from alpha cells into blood
        net_glucagon = self.flux_registry.get_net_flux_to_compartment("blood", "glucagon")
        self.environment.apply_solute_flux("glucagon", net_glucagon, dt_minutes)

        # D. Incretin (GLP-1) balance
        net_glp1 = self.flux_registry.get_net_flux_to_compartment("blood", "glp1")
        self.environment.apply_solute_flux("glp1", net_glp1, dt_minutes)

        # E. Electrolytes & fluid
        net_na = self.flux_registry.get_net_flux_to_compartment("blood", "sodium")
        if abs(net_na) > 0.001:
            self.environment.apply_solute_flux("sodium", net_na, dt_minutes)

        # F. Sync organ states to environment hemodynamics & respiration
        if "cardiovascular" in self.organs:
            c_st = self.organs["cardiovascular"].get_state()
            self.environment.update_hemodynamics(
                hr=c_st["cardio.heart_rate_bpm"],
                sv=c_st["cardio.stroke_volume_mL"],
                tpr=c_st["cardio.tpr_mmHg_min_L"],
            )
        if "respiratory" in self.organs:
            r_st = self.organs["respiratory"].get_state()
            self.environment.update_acid_base_and_gas_equilibrium(
                pao2=r_st.get("resp.alveolar_pao2_mmHg", 102.0) - 7.0,
                paco2=r_st.get("resp.alveolar_paco2_mmHg", 40.0),
            )

        # G. Update regional blood flow based on exercise & ANS
        ex = combined_inputs.get("exercise", 0.0)
        symp = self.environment.hormones.plasma_renin_activity_ng_mL_h  # ANS tone proxy
        self.environment.compute_regional_perfusion(exercise_intensity=ex, sympathetic_tone=symp)

        # 6. Audit mass conservation and physical bounds
        ok_hemo, hemo_issues = self.conservation_validator.verify_hemodynamic_conservation(self.environment)
        ok_bounds, bound_issues = self.conservation_validator.verify_physical_bounds(
            self.environment, {k: v.get_state() for k, v in self.organs.items()}
        )
        violations = hemo_issues + bound_issues

        # Advance master clock
        self.current_time_minutes += dt_minutes

        # 7. Construct Snapshot
        active_flux_list = [
            {
                "source": f.source,
                "destination": f.destination,
                "quantity": f.quantity,
                "value": round(f.value, 3),
                "unit": f.unit,
                "mechanism": f.mechanism,
            }
            for f in self.flux_registry.all_fluxes
        ]

        snapshot = WholeBodySnapshot(
            time_minutes=self.current_time_minutes,
            circulating_glucose_mg_dL=round(self.environment.chemistry.glucose_mg_dL, 2),
            circulating_insulin_uU_mL=round(self.environment.chemistry.insulin_uU_mL, 2),
            circulating_glucagon_pg_mL=round(self.environment.chemistry.glucagon_pg_mL, 2),
            heart_rate_bpm=round(self.environment.circulation.heart_rate_bpm, 1),
            cardiac_output_L_min=round(self.environment.circulation.cardiac_output_L_min, 2),
            mean_arterial_pressure_mmHg=round(self.environment.circulation.mean_arterial_pressure_mmHg, 1),
            systolic_bp_mmHg=round(self.environment.circulation.systolic_bp_mmHg, 1),
            diastolic_bp_mmHg=round(self.environment.circulation.diastolic_bp_mmHg, 1),
            arterial_paO2_mmHg=round(self.environment.chemistry.paO2_mmHg, 1),
            arterial_paCO2_mmHg=round(self.environment.chemistry.paCO2_mmHg, 1),
            arterial_saO2_pct=round(self.environment.chemistry.saO2_pct, 1),
            arterial_pH=round(self.environment.chemistry.pH, 3),
            extracellular_fluid_L=round(self.environment.fluids.extracellular_fluid_L, 2),
            organ_states={k: {sk: round(sv, 4) for sk, sv in v.get_state().items()} for k, v in self.organs.items()},
            active_fluxes=active_flux_list,
            is_physiologically_valid=(len(violations) == 0),
            violations=violations,
        )

        self._history.append(snapshot)
        return snapshot

    def run_simulation(
        self,
        duration_minutes: Optional[float] = None,
        dt_minutes: float = 1.0,
        inputs: Optional[Dict[str, float]] = None,
        solver: str = "rk4",
        total_minutes: Optional[float] = None,
    ) -> List[WholeBodySnapshot]:
        """Runs simulation for the given duration."""
        dur = duration_minutes if duration_minutes is not None else (total_minutes or 60.0)
        steps = int(dur / dt_minutes)
        trajectory = []
        for _ in range(steps):
            snap = self.step(dt_minutes=dt_minutes, inputs=inputs, solver=solver)
            trajectory.append(snap)
        return trajectory

    # Method alias for convenience
    simulate = run_simulation
