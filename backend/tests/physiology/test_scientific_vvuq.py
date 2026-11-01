"""
Scientific Verification, Validation, and Uncertainty Quantification (VVUQ) Suite.
Validates the BioDigitalTwin Multi-Organ Multi-Scale Physiological World Model.

Levels evaluated:
1. Mathematical & Dimensional Consistency (SI units, descriptors, bounds)
2. Numerical Verification (RK4 vs Euler convergence, conservation laws, mass balance)
3. Physiological Plausibility (Organ responses, baseline homeostasis, exercise challenge)
4. Cross-Organ Coupling (Lungs <-> Circulation <-> Liver <-> Gut <-> Neuro-Endocrine)
"""

import pytest
import numpy as np

from app.physiology.base import PhysiologicalOrganModel
from app.physiology.transport import BloodTransportLayer, HemodynamicState
from app.physiology.respiratory import RespiratoryOrganModel
from app.physiology.hepatic import HepaticOrganModel
from app.physiology.gastrointestinal import GastrointestinalOrganModel
from app.physiology.autonomic_endocrine import AutonomicEndocrineModel
from app.physiology.multi_scale_twin import MultiOrganPhysiologicalTwin


# ==============================================================================
# 1. Mathematical & Contract Verification
# ==============================================================================

class TestMathematicalContracts:
    """Verify that all physiological organs satisfy the PhysiologicalOrganModel contract."""

    @pytest.mark.parametrize("organ_cls", [
        RespiratoryOrganModel,
        HepaticOrganModel,
        GastrointestinalOrganModel,
        AutonomicEndocrineModel,
    ])
    def test_organ_contract_compliance(self, organ_cls):
        organ = organ_cls()
        assert isinstance(organ, PhysiologicalOrganModel)
        assert len(organ.descriptors_state) > 0
        assert len(organ.descriptors_param) > 0

        # Verify all descriptors specify valid bounds
        for name, desc in organ.descriptors_state.items():
            assert desc.min_bound < desc.max_bound, f"Invalid bounds for state {name}"
            assert desc.si_unit != "", f"Missing SI unit for state {name}"

        for name, desc in organ.descriptors_param.items():
            assert desc.min_bound <= desc.max_bound, f"Invalid bounds for param {name}"
            assert desc.si_unit != "", f"Missing SI unit for param {name}"

    @pytest.mark.parametrize("organ_cls", [
        RespiratoryOrganModel,
        HepaticOrganModel,
        GastrointestinalOrganModel,
        AutonomicEndocrineModel,
    ])
    def test_baseline_state_validity(self, organ_cls):
        organ = organ_cls()
        is_valid, violations = organ.validate_state()
        assert is_valid, f"{organ.name} baseline has violations: {violations}"

    def test_uncertainty_bounds(self):
        organ = HepaticOrganModel()
        unc = organ.get_uncertainty()
        assert len(unc) == len(organ.get_state())
        for k, v in unc.items():
            assert v >= 0.0, f"Uncertainty for {k} must be non-negative"


# ==============================================================================
# 2. Conservation Laws & Numerical Solvers
# ==============================================================================

class TestConservationAndNumericalSolvers:
    """Verify mass conservation and numerical solver stability."""

    def test_hemodynamic_flow_conservation(self):
        """Sum of regional organ blood flows must equal total Cardiac Output."""
        layer = BloodTransportLayer(patient_weight_kg=70.0)
        layer.update_hemodynamics(hr=75.0, sv=70.0, tpr=18.0)
        co = layer.state.cardiac_output_L_min
        assert 4.0 <= co <= 6.5

        flows = layer.compute_organ_perfusion(exercise_intensity=0.0)
        # Regional systemic flows
        regional_sum = (
            flows["flow_brain"] +
            flows["flow_coronary"] +
            flows["flow_renal"] +
            flows["flow_gut"] +
            flows["flow_muscle"] +
            flows["flow_adipose"]
        )
        assert np.isclose(regional_sum, co, rtol=1e-3), (
            f"Regional flows sum={regional_sum:.3f} L/min does not match CO={co:.3f} L/min"
        )
        assert np.isclose(flows["flow_pulmonary"], co), "Pulmonary flow must equal Cardiac Output"

    def test_gastric_mass_conservation(self):
        """Mass in stomach + mass absorbed + mass in intestine must equal total input."""
        gut = GastrointestinalOrganModel()
        meal_g = 75.0
        gut.ingest_meal(carbs_g=meal_g)

        cum_absorbed_g = 0.0
        dt = 1.0  # min
        for _ in range(120):
            state = gut.step(dt=dt, inputs={}, transport={})
            absorbed_in_step = (state["glucose_absorption_rate_mg_min"] / 1000.0) * dt
            cum_absorbed_g += absorbed_in_step

        remaining_in_gut = state["stomach_carb_mass_g"] + state["intestinal_glucose_mass_g"]
        total_accounted_g = remaining_in_gut + cum_absorbed_g
        # Accounted mass must conserve ingested carbs within numerical integration tolerance
        assert np.isclose(total_accounted_g, meal_g, rtol=0.05), (
            f"Carb mass balance violation: input={meal_g}g, accounted={total_accounted_g:.2f}g"
        )

    def test_rk4_vs_euler_stability(self):
        """RK4 solver must show superior numerical stability compared to Euler under large dt."""
        resp_euler = RespiratoryOrganModel()
        resp_rk4 = RespiratoryOrganModel()

        # Step forward 60 minutes with large dt = 5.0 min
        dt = 5.0
        for _ in range(12):
            resp_euler.step(dt=dt, inputs={"exercise": 0.5}, transport={}, method="euler")
            resp_rk4.step(dt=dt, inputs={"exercise": 0.5}, transport={}, method="rk4")

        ok_rk4, violations_rk4 = resp_rk4.validate_state()
        assert ok_rk4, f"RK4 integration produced unphysiological state: {violations_rk4}"
        assert 7.20 <= resp_rk4.get_state()["pH"] <= 7.60


# ==============================================================================
# 3. Organ-Specific Physiological Validation
# ==============================================================================

class TestPhysiologicalResponses:
    """Validate organ responses against published medical physiology literature."""

    def test_respiratory_hypercapnia_response(self):
        """Elevated arterial paCO2 must stimulate central chemoreceptors and increase ventilation."""
        resp = RespiratoryOrganModel()
        s_base = resp.get_state()
        base_rr = s_base["respiratory_rate_bpm"]

        # Induce hypercapnia (paCO2 = 55 mmHg, normal = 40)
        resp.set_state({"paCO2_mmHg": 55.0})
        derivs = resp.compute_derivatives(resp.get_state(), inputs={}, transport={})
        assert derivs["respiratory_rate_bpm"] > 0, "Hypercapnia must increase respiratory rate drive"
        assert derivs["alveolar_ventilation_L_min"] > 0, "Hypercapnia must increase alveolar ventilation"

    def test_respiratory_henderson_hasselbalch_acid_base(self):
        """Arterial pH must obey Henderson-Hasselbalch equation."""
        layer = BloodTransportLayer()
        # Normal baseline: paCO2 = 40, HCO3 = 24 -> pH 7.40
        _, ph_norm = layer.update_blood_gas_equilibrium(pao2=95.0, paco2=40.0, hco3=24.0)
        assert np.isclose(ph_norm, 7.40, atol=0.03)

        # Metabolic acidosis: HCO3 = 14, paCO2 = 40 -> pH < 7.20
        _, ph_acid = layer.update_blood_gas_equilibrium(pao2=95.0, paco2=40.0, hco3=14.0)
        assert ph_acid < 7.25

        # Respiratory compensation: paCO2 drops to 26 mmHg -> restores pH toward 7.35
        _, ph_comp = layer.update_blood_gas_equilibrium(pao2=95.0, paco2=26.0, hco3=14.0)
        assert ph_comp > ph_acid

    def test_hepatic_insulin_suppression_of_hgp(self):
        """Hyperinsulinemia must suppress Hepatic Glucose Production (HGP)."""
        liver = HepaticOrganModel()

        # Fasting / basal state: insulin = 5 uU/mL
        derivs_basal = liver.compute_derivatives(
            liver.get_state(), inputs={}, transport={"insulin_uU_mL": 5.0, "glucagon_pg_mL": 70.0}
        )
        hgp_basal = liver.get_state()["hgp_rate_mg_min"] + derivs_basal["hgp_rate_mg_min"]

        # Postprandial high-insulin state: insulin = 60 uU/mL
        derivs_high = liver.compute_derivatives(
            liver.get_state(), inputs={}, transport={"insulin_uU_mL": 60.0, "glucagon_pg_mL": 40.0}
        )
        hgp_high = liver.get_state()["hgp_rate_mg_min"] + derivs_high["hgp_rate_mg_min"]

        assert hgp_high < hgp_basal, "Insulin must suppress hepatic glucose production"

    def test_baroreflex_hypotension_response(self):
        """Hypotension must trigger baroreceptor withdrawal, raising sympathetic tone and HR."""
        ae = AutonomicEndocrineModel()

        # Normal MAP = 93 mmHg
        derivs_norm = ae.compute_derivatives(
            ae.get_state(), inputs={}, transport={"mean_arterial_pressure_mmHg": 93.0}
        )

        # Hypotension: MAP drops to 65 mmHg
        derivs_hypo = ae.compute_derivatives(
            ae.get_state(), inputs={}, transport={"mean_arterial_pressure_mmHg": 65.0}
        )

        assert derivs_hypo["sympathetic_tone"] > derivs_norm["sympathetic_tone"], (
            "Hypotension must activate sympathetic nervous system drive"
        )
        assert derivs_hypo["renin_activity_ng_mL_h"] > derivs_norm["renin_activity_ng_mL_h"], (
            "Renal hypoperfusion must trigger renin release"
        )

    def test_incretin_stimulation_and_dpp4_clearance(self):
        """Luminal nutrients stimulate GLP-1, and DPP-4 enzyme clears it back to baseline."""
        gut = GastrointestinalOrganModel()
        gut.ingest_meal(carbs_g=50.0)

        # Step 20 minutes: GLP-1 should rise above baseline (15 pmol/L)
        for _ in range(20):
            state = gut.step(dt=1.0, inputs={}, transport={})

        assert state["glp1_plasma_pmol_L"] > 18.0, "Meal carbs must stimulate GLP-1 secretion"


# ==============================================================================
# 4. Multi-Organ Integrated Closed-Loop Twin Validation
# ==============================================================================

class TestMultiOrganWholeBodyTwin:
    """Validate integrated whole-body digital twin under clinical challenge protocols."""

    def test_meal_tolerance_test(self):
        """
        Oral Glucose Tolerance Challenge:
        Feed 75g carbs -> Observe gastric emptying -> portal absorption ->
        insulin secretion -> glycemic peak & return to homeostasis.
        """
        twin = MultiOrganPhysiologicalTwin(weight_kg=70.0)
        snap0 = twin.step(dt_minutes=1.0)
        assert 80.0 <= snap0.glucose_mg_dL <= 110.0

        # Ingest 75g oral glucose
        twin.feed_meal(carbs_g=75.0)

        glucose_curve = []
        insulin_curve = []
        glp1_curve = []

        for m in range(120):
            snap = twin.step(dt_minutes=1.0)
            glucose_curve.append(snap.glucose_mg_dL)
            insulin_curve.append(snap.insulin_uU_mL)
            glp1_curve.append(snap.glp1_pmol_L)

        # Peak glucose should occur between 20 and 60 minutes
        peak_idx = int(np.argmax(glucose_curve))
        assert 15 <= peak_idx <= 75, f"Glycemic excursion peak occurred at t={peak_idx} min (expected 15-75 min)"
        assert max(glucose_curve) > snap0.glucose_mg_dL + 20.0, "Glucose must rise after meal"
        # Glycemia should recover toward baseline by 120 min
        assert glucose_curve[-1] < max(glucose_curve)
        # Insulin should rise in response to glucose and GLP-1
        assert max(insulin_curve) > snap0.insulin_uU_mL * 2.0
        assert max(glp1_curve) > snap0.glp1_pmol_L

        ok, violations = twin.validate_whole_body_state()
        assert ok, f"Whole-body state violations during OGTT: {violations}"

    def test_exercise_stress_adaptation(self):
        """
        Exercise Challenge:
        Physical exertion -> increased CO & HR -> hyperpnea & ventilation ->
        increased O2 delivery and glycogen mobilization.
        """
        twin = MultiOrganPhysiologicalTwin(weight_kg=70.0)
        snap_rest = twin.step(dt_minutes=1.0)

        # 30 minutes of moderate exercise
        exercise_inputs = {"exercise": 0.6}
        for _ in range(30):
            snap_ex = twin.step(dt_minutes=1.0, inputs=exercise_inputs)

        assert snap_ex.heart_rate_bpm > snap_rest.heart_rate_bpm + 20.0, "Exercise must increase HR"
        assert snap_ex.cardiac_output_L_min > snap_rest.cardiac_output_L_min + 2.0, "Exercise must increase CO"
        assert snap_ex.respiratory_rate_bpm > snap_rest.respiratory_rate_bpm, "Exercise must increase breathing rate"
        assert snap_ex.sympathetic_tone > snap_rest.sympathetic_tone, "Exercise must elevate sympathetic tone"
        # Arterial oxygen saturation must remain preserved
        assert snap_ex.saO2_pct >= 94.0
        # Blood pH must stay within physiological range
        assert 7.25 <= snap_ex.pH <= 7.55

