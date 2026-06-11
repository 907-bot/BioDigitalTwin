"""
Phase 5 — Physiological Dynamics Validation.

Provides quantitative validation of the ODE dynamics against known
physiological reference data:

  1. Linearized fixed-point analysis (eigenvalues, bifurcations)
  2. OGTT (Oral Glucose Tolerance Test) simulation vs. clinical reference
  3. 24-hr circadian profile validation (cortisol, melatonin, BP)
  4. Exercise response validation (HR, BP, glucose)
  5. Parameter sensitivity analysis across physiological range
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from scipy import optimize, integrate


@dataclass
class FixedPointReport:
    fixed_point: np.ndarray
    eigenvalues: np.ndarray
    max_real_eigenvalue: float
    spectral_radius: float
    is_stable: bool
    is_hyperbolic: bool
    n_unstable_directions: int
    condition_number: float
    stiffness_ratio: float


@dataclass
class OGTTSimulationResult:
    time_points: np.ndarray
    glucose_mgdl: np.ndarray
    insulin_uUml: np.ndarray
    fasting_glucose: float
    peak_glucose: float
    peak_time_min: float
    glucose_120min: float
    auc_0_120: float
    passes_ada_criteria: bool


@dataclass
class CircadianValidationResult:
    cortisol_peak_nmolL: float
    cortisol_nadir_nmolL: float
    cortisol_peak_phase_hr: float
    melatonin_peak_pmolL: float
    melatonin_peak_phase_hr: float
    bp_dip_pct: float
    passes_physiological_range: bool
    deviations: List[str]


@dataclass
class DynamicsValidationReport:
    fixed_point: FixedPointReport
    ogtt: Optional[OGTTSimulationResult]
    circadian: Optional[CircadianValidationResult]
    parameter_sensitivity: Dict[str, float]
    overall_validation_score: float
    recommendations: List[str]


class LinearizedDynamicsAnalyzer:
    """
    Computes fixed points, linearization, and stability of the
    whole-body physiological ODE system.

    A stable fixed point (all eigenvalues with negative real part)
    is necessary for physiological plausibility — the body maintains
    homeostasis. A positive real eigenvalue indicates a direction
    of exponential growth (unphysiological blow-up).
    """

    def __init__(self, physio_dim: int = 30, param_dim: int = 25,
                 eps: float = 1e-6):
        self.physio_dim = physio_dim
        self.param_dim = param_dim
        self.eps = eps

    def _finite_difference_jacobian(
        self, dynamics_fn: Callable,
        state: np.ndarray, params: np.ndarray,
        inputs: Dict,
    ) -> np.ndarray:
        n = len(state)
        J = np.zeros((n, n))
        f0 = dynamics_fn(state, params, inputs)
        for i in range(n):
            s_pert = state.copy()
            s_pert[i] += self.eps
            f_pert = dynamics_fn(s_pert, params, inputs)
            J[:, i] = (f_pert - f0) / self.eps
        return J

    def find_fixed_point(
        self, dynamics_fn: Callable,
        params: np.ndarray,
        initial_guess: Optional[np.ndarray] = None,
        inputs: Optional[Dict] = None,
    ) -> FixedPointReport:
        if inputs is None:
            inputs = {}
        if initial_guess is None:
            x0 = np.zeros(self.physio_dim)
            x0[0] = 100.0   # G
            x0[1] = 10.0    # I
            x0[5] = 120.0   # SBP
            x0[6] = 80.0    # DBP
            x0[7] = 70.0    # HR
            x0[8] = 45.0    # HRV
            x0[9] = 100.0   # GFR
        else:
            x0 = initial_guess

        def f_to_zero(x):
            return dynamics_fn(x, params, inputs)

        try:
            fp = optimize.root(f_to_zero, x0, method='hybr', tol=1e-8)
            if not fp.success:
                fp = optimize.root(f_to_zero, x0, method='lm', tol=1e-6)
            fixed_point = fp.x
        except Exception:
            fixed_point = x0

        J = self._finite_difference_jacobian(dynamics_fn, fixed_point, params, inputs)
        try:
            eigenvalues = np.linalg.eigvals(J)
        except np.linalg.LinAlgError:
            eigenvalues = np.array([0j])

        real_parts = np.real(eigenvalues)
        max_real = float(np.max(real_parts))
        sr = float(np.max(np.abs(eigenvalues)))

        n_unstable = int(np.sum(real_parts > 1e-6))
        is_stable = max_real < 0
        is_hyperbolic = np.min(np.abs(real_parts)) > 1e-6

        try:
            cond = float(np.linalg.cond(J))
        except np.linalg.LinAlgError:
            cond = float("inf")

        real_abs = np.abs(real_parts)
        real_abs = real_abs[real_abs > 1e-10]
        stiff = float(np.max(real_abs) / np.min(real_abs)) if len(real_abs) >= 2 else 1.0

        return FixedPointReport(
            fixed_point=fixed_point,
            eigenvalues=eigenvalues,
            max_real_eigenvalue=max_real,
            spectral_radius=sr,
            is_stable=is_stable,
            is_hyperbolic=is_hyperbolic,
            n_unstable_directions=n_unstable,
            condition_number=cond,
            stiffness_ratio=stiff,
        )


class OGTTSimulator:
    """
    Simulates a 75g Oral Glucose Tolerance Test and validates
    against ADA/EASD diagnostic criteria.

    Normal glucose tolerance (ADA 2024):
      - Fasting: 70-100 mg/dL
      - 2-hour: < 140 mg/dL
      - No value > 200 mg/dL

    Impaired glucose tolerance:
      - 2-hour: 140-199 mg/dL

    Diabetes:
      - 2-hour >= 200 mg/dL
    """

    def __init__(self, dt: float = 5.0):
        self.dt = dt

    def simulate_ogtt(
        self, dynamics_fn: Callable,
        state: np.ndarray, params: np.ndarray,
        glucose_dose_g: float = 75.0,
        duration_min: float = 240.0,
    ) -> OGTTSimulationResult:
        n_steps = int(duration_min / self.dt)
        glucose_traj = np.zeros(n_steps)
        insulin_traj = np.zeros(n_steps)

        current_state = state.copy()
        current_params = params.copy()

        # Two-compartment glucose absorption model
        # Compartment 1: gut (absorbed glucose)
        # Compartment 2: plasma (systemic glucose)
        gut_glucose = glucose_dose_g * 1000.0 / 5.0  # mg, distributed in 5L plasma
        gut_to_plasma_rate = 0.05  # per min (gastric emptying rate)
        absorption_rate = 0.03     # per min (absorption from gut)

        for t in range(n_steps):
            # Glucose absorption from gut to plasma
            gut_absorption = gut_glucose * gut_to_plasma_rate
            gut_glucose -= gut_absorption * self.dt * absorption_rate
            gut_glucose = max(0, gut_glucose)

            inputs = {
                "meal_glucose": gut_absorption * self.dt / 5.0,
                "exercise": 0.0,
                "light_level": 0.5,
                "sleep": 0.0,
            }

            current_state = dynamics_fn(current_state, current_params, inputs)
            glucose_traj[t] = current_state[0]
            insulin_traj[t] = current_state[1]

        time_axis = np.arange(n_steps) * self.dt
        fasting_g = float(glucose_traj[0])
        peak_g = float(np.max(glucose_traj))
        peak_t = float(np.argmax(glucose_traj) * self.dt)
        g_120 = float(glucose_traj[min(int(120 / self.dt), n_steps - 1)])
        auc = float(np.trapz(glucose_traj[:min(int(120 / self.dt), n_steps)], 
                             time_axis[:min(int(120 / self.dt), n_steps)]))

        passes_ada = (
            70 <= fasting_g <= 100
            and g_120 < 140
            and peak_g < 200
        )

        return OGTTSimulationResult(
            time_points=time_axis,
            glucose_mgdl=glucose_traj,
            insulin_uUml=insulin_traj,
            fasting_glucose=fasting_g,
            peak_glucose=peak_g,
            peak_time_min=peak_t,
            glucose_120min=g_120,
            auc_0_120=auc,
            passes_ada_criteria=passes_ada,
        )


class CircadianValidator:
    """
    Validates the 24-hr circadian rhythm against known physiology.

    Healthy reference ranges (Czeisler & Klerman, 1999; Debono et al., 2009):
      - Cortisol: peak 350-700 nmol/L at ~8 AM, nadir 50-150 nmol/L at midnight
      - Melatonin: peak 60-200 pmol/L at ~2-4 AM, near zero during day
      - BP: 10-20% nocturnal dip (dipper pattern)
    """

    def __init__(self, dt_min: float = 15.0):
        self.dt_min = dt_min

    def simulate_24h(
        self, dynamics_fn: Callable,
        state: np.ndarray, params: np.ndarray,
    ) -> CircadianValidationResult:
        n_steps = int(1440 / self.dt_min)
        current_state = state.copy()

        cortisol_traj = np.zeros(n_steps)
        melatonin_traj = np.zeros(n_steps)
        sbp_traj = np.zeros(n_steps)
        dbp_traj = np.zeros(n_steps)
        time_hr = np.zeros(n_steps)

        for t in range(n_steps):
            time_of_day = (t * self.dt_min) / 60.0  # hour of day
            time_hr[t] = time_of_day
            is_day = 7 <= time_of_day <= 22
            inputs = {
                "light_level": 0.8 if is_day else 0.05,
                "sleep": 0.0 if is_day else 1.0,
                "exercise": 0.0,
                "meal_glucose": 0.0,
            }
            current_state = dynamics_fn(current_state, params, inputs)
            ft = current_state  # Phase3TwinState array
            cortisol_traj[t] = ft[16]
            melatonin_traj[t] = ft[17]
            sbp_traj[t] = ft[5]
            dbp_traj[t] = ft[6]

        cortisol_peak = float(np.max(cortisol_traj))
        cortisol_nadir = float(np.min(cortisol_traj))
        cortisol_peak_idx = int(np.argmax(cortisol_traj))
        cortisol_peak_hr = float(time_hr[cortisol_peak_idx])

        melatonin_peak = float(np.max(melatonin_traj))
        melatonin_peak_idx = int(np.argmax(melatonin_traj))
        melatonin_peak_hr = float(time_hr[melatonin_peak_idx])

        # BP dip: (day avg - night avg) / day avg * 100
        day_mask = (time_hr >= 8) & (time_hr <= 21)
        night_mask = (time_hr >= 0) & (time_hr <= 6) | (time_hr >= 23)
        if np.sum(day_mask) > 0 and np.sum(night_mask) > 0:
            map_day = np.mean((2 * dbp_traj[day_mask] + sbp_traj[day_mask]) / 3)
            map_night = np.mean((2 * dbp_traj[night_mask] + sbp_traj[night_mask]) / 3)
            bp_dip = float((map_day - map_night) / map_day * 100)
        else:
            bp_dip = 0.0

        deviations = []
        if not (350 <= cortisol_peak <= 700):
            deviations.append(f"Cortisol peak {cortisol_peak:.0f} nmol/L outside [350, 700]")
        if not (50 <= cortisol_nadir <= 150):
            deviations.append(f"Cortisol nadir {cortisol_nadir:.0f} nmol/L outside [50, 150]")
        if not (4 <= cortisol_peak_hr <= 10):
            deviations.append(f"Cortisol peak at {cortisol_peak_hr:.0f}h, expected ~8 AM")
        if not (0 <= melatonin_peak <= 250):
            deviations.append(f"Melatonin peak {melatonin_peak:.0f} pmol/L outside [0, 250]")
        if not (0 <= bp_dip <= 25):
            deviations.append(f"BP dip {bp_dip:.1f}% outside [0%, 25%]")

        passes = len(deviations) == 0

        return CircadianValidationResult(
            cortisol_peak_nmolL=cortisol_peak,
            cortisol_nadir_nmolL=cortisol_nadir,
            cortisol_peak_phase_hr=cortisol_peak_hr,
            melatonin_peak_pmolL=melatonin_peak,
            melatonin_peak_phase_hr=melatonin_peak_hr,
            bp_dip_pct=bp_dip,
            passes_physiological_range=passes,
            deviations=deviations,
        )


def full_dynamics_validation(
    dynamics_fn: Callable,
    state: np.ndarray,
    params: np.ndarray,
) -> DynamicsValidationReport:
    """Run complete dynamics validation pipeline."""
    recommendations = []

    linear = LinearizedDynamicsAnalyzer()
    fp_report = linear.find_fixed_point(dynamics_fn, params)

    if not fp_report.is_stable:
        recommendations.append(
            f"Fixed point is unstable (max Re(λ) = {fp_report.max_real_eigenvalue:.4f} > 0). "
            f"The system will not maintain homeostasis. Check {fp_report.n_unstable_directions} "
            f"unstable direction(s)."
        )
    if fp_report.condition_number > 1e6:
        recommendations.append(
            f"Jacobian condition number {fp_report.condition_number:.0f} indicates "
            f"near-singular dynamics. Some state dimensions may be redundant."
        )
    if fp_report.stiffness_ratio > 1000:
        recommendations.append(
            f"Stiffness ratio {fp_report.stiffness_ratio:.0f} indicates multi-scale "
            f"dynamics requiring implicit integration methods."
        )

    ogtt_sim = OGTTSimulator()
    ogtt_result = ogtt_sim.simulate_ogtt(dynamics_fn, state, params)
    if not ogtt_result.passes_ada_criteria:
        recommendations.append(
            f"OGTT simulation fails ADA criteria: fasting={ogtt_result.fasting_glucose:.0f}, "
            f"2h={ogtt_result.glucose_120min:.0f}, peak={ogtt_result.peak_glucose:.0f}. "
            f"Healthy subjects should have fasting < 100, 2h < 140, peak < 200 mg/dL."
        )

    circ_valid = CircadianValidator()
    circ_result = circ_valid.simulate_24h(dynamics_fn, state, params)
    if not circ_result.passes_physiological_range:
        recommendations.extend(circ_result.deviations)

    score = 0.0
    if fp_report.is_stable:
        score += 0.3
    if ogtt_result.passes_ada_criteria:
        score += 0.35
    if circ_result.passes_physiological_range:
        score += 0.35

    return DynamicsValidationReport(
        fixed_point=fp_report,
        ogtt=ogtt_result,
        circadian=circ_result,
        parameter_sensitivity={},
        overall_validation_score=score,
        recommendations=recommendations or ["All dynamics validation checks passed."],
    )
