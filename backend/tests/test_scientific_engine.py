"""
backend/tests/test_scientific_engine.py
Comprehensive Scientific & Mathematical Verification Test Suite.

Verifies:
1. Scientific Model Registry (contracts, equations, units, parameters)
2. Receptor Kinetics & Mass Balance Conservation
3. Full Black-Leff Operational Agonism & Schild Competitive Antagonism
4. SGLT2 Carrier-Mediated Cotransporter & Natriuresis (Non-GPCR)
5. Respiratory Gas Exchange, Henderson-Hasselbalch & Verified Bohr Effect Direction
6. RAAS Closed-Loop Hemodynamics & Pressure Natriuresis
7. End-to-End MultiScaleTwinBridge Integrated Simulation
"""
import pytest
import numpy as np

# 1. Scientific Registry
from app.registry.scientific_registry import (
    MODEL_REGISTRY,
    get_registered_model,
    list_registered_models,
)

# 2. Receptors & Transporters
from app.pkpd.receptors import (
    RECEPTOR_LIBRARY,
    ReceptorKineticsEngine,
    SGLT2TransporterEngine,
    SGLT2TransporterParameters,
)

# 3. Respiratory & Acid-Base
from app.dynamics.respiratory_acid_base import (
    RespiratoryAcidBaseEngine,
    RespiratoryAcidBaseState,
)

# 4. RAAS
from app.dynamics.raas_model import RAASClosedLoopEngine, RAASState

# 5. Integrated Multi-Scale Bridge
from app.personalization.integrated_bridge import MultiScaleTwinBridge
from app.pkpd.compartments import PKParams


class TestScientificModelRegistry:
    def test_registry_populated(self):
        models = list_registered_models()
        assert len(models) >= 4
        assert any(m["model_id"] == "black_leff_operational_agonism" for m in models)
        assert any(m["model_id"] == "sglt2_carrier_kinetics" for m in models)
        assert any(m["model_id"] == "respiratory_acid_base" for m in models)
        assert any(m["model_id"] == "raas_hemodynamics" for m in models)

    def test_model_contracts_have_units_and_equations(self):
        for m_id, contract in MODEL_REGISTRY.items():
            assert len(contract.governing_equations) > 0
            for var in contract.state_variables:
                assert var.unit != "", f"Missing unit for {var.symbol} in {m_id}"
                assert var.physio_min < var.physio_max


class TestReceptorKineticsAndOperationalModel:
    def test_mass_balance_conservation(self):
        """Verify that total receptor pool is conserved during transient ligand exposure."""
        # 1. Closed system trafficking conservation test
        params_closed = RECEPTOR_LIBRARY["INSR"]
        # Set synthesis and degradation to 0 to test pure mass conservation
        params_closed.k_synth = 0.0
        params_closed.k_deg = 0.0
        engine_closed = ReceptorKineticsEngine(params_closed)
        initial_total = engine_closed.total_receptors

        for _ in range(60):
            engine_closed.step(agonist_conc_nM=10.0, dt_min=1.0)

        # In a closed trafficking system, mass conservation is exact (< 0.01% numerical error)
        assert abs(engine_closed.total_receptors - initial_total) / initial_total < 0.001
        assert engine_closed.LR_complex > 0.0
        assert engine_closed.R_internalized > 0.0

    def test_black_leff_operational_curve(self):
        """Verify that tissue response follows Black-Leff operational concentration-response."""
        params = RECEPTOR_LIBRARY["ADRB1"]
        engine = ReceptorKineticsEngine(params)

        # Test at low, medium, and saturating agonist concentrations
        res_zero = engine.step(agonist_conc_nM=0.0, dt_min=0.1)["tissue_response"]
        res_low = engine.step(agonist_conc_nM=0.1, dt_min=0.1)["tissue_response"]
        res_med = engine.step(agonist_conc_nM=5.0, dt_min=0.1)["tissue_response"]
        res_high = engine.step(agonist_conc_nM=100.0, dt_min=0.1)["tissue_response"]

        assert res_zero == 0.0
        assert 0.0 < res_low < res_med < res_high <= 1.0
        # High efficiency (tau = 15) reaches near-maximal effect well before Kd saturation
        assert res_high > 0.90

    def test_competitive_antagonist_schild_shift(self):
        """Verify that a competitive antagonist right-shifts the concentration-response curve."""
        params = RECEPTOR_LIBRARY["ADRB1"]
        engine_control = ReceptorKineticsEngine(params)
        engine_blocked = ReceptorKineticsEngine(params)

        agonist = 2.0  # nM
        antagonist = 200.0  # nM (with Kb = 100 nM -> 3x rightward shift)

        resp_control = engine_control.step(agonist_conc_nM=agonist, dt_min=1.0)["tissue_response"]
        resp_blocked = engine_blocked.step(
            agonist_conc_nM=agonist,
            dt_min=1.0,
            antagonist_conc_nM=antagonist,
            antagonist_Kb_nM=100.0
        )["tissue_response"]

        assert resp_blocked < resp_control


class TestSGLT2TransporterKinetics:
    def test_sglt2_carrier_kinetics(self):
        """Verify carrier saturation (TmG) and physiological glucosuria threshold."""
        sglt2 = SGLT2TransporterEngine()

        # At normal fasting glucose (90 mg/dL), GFR 100 mL/min -> filtered 90 mg/min < TmG (375 mg/min)
        normal = sglt2.compute_renal_flux(plasma_glucose_mg_dL=90.0, gfr_mL_min=100.0)
        assert normal["urinary_glucose_mg_min"] == 0.0
        assert normal["effective_renal_threshold_mg_dL"] > 180.0

        # At severe hyperglycemia (350 mg/dL), filtered 350 mg/min approaches TmG
        hyper = sglt2.compute_renal_flux(plasma_glucose_mg_dL=350.0, gfr_mL_min=100.0)
        assert hyper["urinary_glucose_mg_min"] > 0.0

    def test_sglt2_pharmacological_inhibition_and_natriuresis(self):
        """Verify that Empagliflozin induces glucosuria and stoichiometric natriuresis."""
        sglt2 = SGLT2TransporterEngine()
        baseline = sglt2.compute_renal_flux(plasma_glucose_mg_dL=160.0, gfr_mL_min=100.0, inhibitor_conc_nM=0.0)
        inhibited = sglt2.compute_renal_flux(
            plasma_glucose_mg_dL=160.0,
            gfr_mL_min=100.0,
            inhibitor_conc_nM=20.0,  # >> Ki (1.3 nM)
            inhibitor_type="empagliflozin"
        )

        assert inhibited["urinary_glucose_mg_min"] > baseline["urinary_glucose_mg_min"]
        assert inhibited["sglt2_inhibition_pct"] > 80.0
        assert inhibited["induced_natriuresis_mEq_min"] > 0.0


class TestRespiratoryAcidBaseAndBohrEffect:
    def test_henderson_hasselbalch_normal_abg(self):
        """Verify normal human arterial blood gas parameters."""
        engine = RespiratoryAcidBaseEngine()
        pH = engine.compute_pH(PaCO2_mmHg=40.0, HCO3_mEq_L=24.0)
        assert 7.38 <= pH <= 7.42

    def test_bohr_effect_direction_acidosis_shifts_right(self):
        """
        CRITICAL TEST: Verify Bohr effect direction.
        Acidosis (pH < 7.40) MUST shift the oxygen dissociation curve to the RIGHT,
        increasing P50 so that hemoglobin unloads oxygen more readily.
        """
        engine = RespiratoryAcidBaseEngine()
        P50_normal = engine.compute_P50(pH=7.40, PaCO2_mmHg=40.0)
        P50_acidotic = engine.compute_P50(pH=7.20, PaCO2_mmHg=40.0)
        P50_alkalotic = engine.compute_P50(pH=7.60, PaCO2_mmHg=40.0)

        assert P50_acidotic > P50_normal, "Bohr effect inverted: Acidosis must increase P50 (shift curve right)!"
        assert P50_alkalotic < P50_normal, "Alkalosis must decrease P50 (shift curve left)!"

    def test_respiratory_compensation(self):
        """Verify hypercapnia triggers chemoreceptor hyperventilation."""
        engine = RespiratoryAcidBaseEngine()
        # Elevate PaCO2 to 60 mmHg
        engine.state.PaCO2_mmHg = 60.0
        new_state = engine.step(dt_min=1.0)
        assert new_state.V_A_L_min > 4.2  # Hyperventilation triggered


class TestRAASClosedLoop:
    def test_hypotension_stimulates_renin_and_ang_ii(self):
        """Verify negative feedback: MAP drop stimulates renin and Ang II."""
        raas = RAASClosedLoopEngine()
        # Normal MAP 93.3 mmHg
        norm = raas.step(MAP_current_mmHg=93.3, dt_min=1.0)
        # Hypotension 70 mmHg
        hypo = raas.step(MAP_current_mmHg=70.0, dt_min=10.0)

        assert hypo.plasma_renin_activity > norm.plasma_renin_activity
        assert hypo.angiotensin_II > norm.angiotensin_II

    def test_ace_inhibitor_blocks_ang_ii(self):
        """Verify Lisinopril inhibits ACE, lowering Ang II and SVR."""
        raas = RAASClosedLoopEngine()
        raas.state.angiotensin_I = 40.0
        state = raas.step(MAP_current_mmHg=85.0, dt_min=5.0, ace_inhibition_pct=95.0)
        assert state.angiotensin_II < 20.0
        assert state.SVR_multiplier <= 1.0


class TestMultiScaleTwinBridge:
    def test_integrated_simulation_step(self):
        """Verify that the multi-scale bridge runs without numerical breakdown."""
        bridge = MultiScaleTwinBridge()
        # Administer oral Empagliflozin 25 mg
        bridge.administer_drug(
            "empagliflozin",
            dose_mg=25.0,
            pk_spec=PKParams(ka=1.2, CL=10.0, Vc=40.0, Vp=30.0, Q=5.0, F=0.85),
        )

        # Run 30 simulation steps (30 minutes)
        for _ in range(30):
            state = bridge.step(dt_min=1.0)

        assert state.drug_plasma_conc_mg_L["empagliflozin"] > 0.0
        assert "SGLT2_inhibition_pct" in state.receptor_signals
        assert state.receptor_signals["SGLT2_inhibition_pct"] > 0.0
        assert 7.30 <= state.respiratory.pH <= 7.50
        assert 50.0 <= state.physio_state[0] <= 300.0  # Glucose in plausible range
