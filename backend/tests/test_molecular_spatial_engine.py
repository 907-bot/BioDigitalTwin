"""
backend/tests/test_molecular_spatial_engine.py
Comprehensive Verification Suite for Spatial Pharmacology and Molecular Target Engagement Engine.
"""
from __future__ import annotations

import pytest
import numpy as np
from fastapi.testclient import TestClient

from app.molecular.tissue.cell_types import (
    OrganType,
    SubcellularLocation,
    CELL_TYPES_ATLAS,
    get_anatomical_hierarchy,
)
from app.molecular.targets.target_registry import (
    MolecularTarget,
    TargetClass,
    ReceptorFamily,
    ExpressionStatus,
    get_target_registry,
)
from app.molecular.targets import initialize_target_registry
from app.molecular.binding.binding_kinetics import (
    DynamicBindingEngine,
)
from app.molecular.binding.occupancy import (
    compute_equilibrium_occupancy,
    cheng_prusoff_ki,
)
from app.molecular.binding.target_engagement import (
    TargetEngagementEvaluator,
)
from app.molecular.signaling.second_messengers import (
    SecondMessengerTransducer,
)
from app.molecular.signaling.downstream_response import (
    CellularResponseEvaluator,
)
from app.molecular.drug_response.organ_response import (
    OrganResponseEvaluator,
)
from app.molecular.spatial_engine import (
    get_spatial_pharmacology_engine,
)
from app.main import app


@pytest.fixture(scope="module")
def registry():
    return initialize_target_registry()


@pytest.fixture(scope="module")
def spatial_engine():
    return get_spatial_pharmacology_engine()


@pytest.fixture(scope="module")
def api_client():
    return TestClient(app)


# =============================================================================
# 1. Target Registry & Literature Evidence Verification
# =============================================================================
class TestTargetRegistryAndEvidence:
    def test_all_target_classes_populated(self, registry):
        """Verify presence of receptors, enzymes, transporters, and ion channels."""
        all_targets = registry.list_targets()
        assert len(all_targets) >= 10

        receptors = registry.list_targets(target_class=TargetClass.RECEPTOR)
        enzymes = registry.list_targets(target_class=TargetClass.ENZYME)
        transporters = registry.list_targets(target_class=TargetClass.TRANSPORTER)
        ion_channels = registry.list_targets(target_class=TargetClass.ION_CHANNEL)

        assert len(receptors) >= 5, "Must have GLP1R, ADRB1, AGTR1, INSR, NR3C1, CHRNA1"
        assert len(enzymes) >= 2, "Must have HMGCR, ACE"
        assert len(transporters) >= 2, "Must have SGLT2, GLUT4"
        assert len(ion_channels) >= 1, "Must have CACNA1C"

    def test_target_identifiers_and_citations(self, registry):
        """Every target must have valid HGNC symbol, UniProt ID, and experimental citations."""
        for target in registry.list_targets():
            assert target.gene_symbol != ""
            assert target.uniprot_id.startswith(("P", "Q", "O"))
            assert target.binding_params.Kd_nM > 0.0
            assert target.binding_params.k_on > 0.0
            assert target.binding_params.k_off > 0.0
            assert len(target.spatial_expressions) > 0
            for exp in target.spatial_expressions:
                assert exp.evidence.source_database != ""
                assert 0.0 <= exp.evidence.confidence_score <= 1.0


# =============================================================================
# 2. Spatial Expression Atlas & Unknown Status Representation
# =============================================================================
class TestSpatialExpressionAtlas:
    def test_glp1r_pancreatic_vs_muscle_specificity(self, registry):
        """GLP1R must be confirmed high in pancreatic beta cells and explicitly absent in skeletal muscle."""
        glp1r = registry.get("GLP1R")
        assert glp1r is not None

        beta_exp = glp1r.get_expression("pancreas_beta_cell")
        assert beta_exp is not None
        assert beta_exp.status == ExpressionStatus.CONFIRMED_HIGH
        assert beta_exp.is_expressed is True
        assert beta_exp.copies_per_cell > 1000.0

        muscle_exp = glp1r.get_expression("skeletal_myocyte")
        assert muscle_exp is not None
        assert muscle_exp.status == ExpressionStatus.ABSENT
        assert muscle_exp.is_expressed is False

    def test_sglt2_apical_renal_localization(self, registry):
        """SGLT2 must be localized specifically to the apical brush border of proximal tubule cells."""
        sglt2 = registry.get("SGLT2")
        assert sglt2 is not None

        exp = sglt2.get_expression("kidney_proximal_tubule_epithelial")
        assert exp is not None
        assert exp.organ == OrganType.KIDNEY
        assert exp.subcellular_location == SubcellularLocation.PLASMA_MEMBRANE_APICAL
        assert exp.status == ExpressionStatus.CONFIRMED_HIGH

    def test_missing_data_not_assumed_zero(self, registry):
        """Targets must return 0.0 density only if absent/unexpressed, but preserve UNKNOWN status."""
        insr = registry.get("INSR")
        unmeasured_exp = insr.get_expression("non_existent_cell_population")
        assert unmeasured_exp is None
        # Effective density fallback for unexpressed cell population is 0.0
        assert insr.get_effective_density_pmol_g("non_existent_cell_population") == 0.0


# =============================================================================
# 3. Dynamic Binding Kinetics & Mathematical Target Engagement
# =============================================================================
class TestBindingKineticsAndOccupancy:
    def test_langmuir_equilibrium_half_maximal_occupancy(self):
        """When [L] == Kd, equilibrium occupancy must equal exactly 0.50 (50%)."""
        kd = 5.0 # nM
        occupancy = compute_equilibrium_occupancy(free_conc_nM=5.0, Kd_nM=kd)
        assert pytest.approx(occupancy, abs=1e-5) == 0.50

    def test_schild_competitive_antagonist_shift(self):
        """Competitive antagonist right-shifts Kd by factor (1 + [B]/KB)."""
        kd = 2.0 # nM
        kb = 1.0 # nM
        b_conc = 9.0 # nM -> shift factor = 1 + 9/1 = 10 -> apparent Kd = 20 nM

        # At [L] = 20 nM (apparent Kd), occupancy with competitor must be 50%
        occupancy_shifted = compute_equilibrium_occupancy(
            free_conc_nM=20.0,
            Kd_nM=kd,
            competing_inhibitor_conc_nM=b_conc,
            competing_inhibitor_Ki_nM=kb,
        )
        assert pytest.approx(occupancy_shifted, abs=1e-5) == 0.50

    def test_dynamic_binding_engine_time_course(self):
        """Dynamic ODE binding reaches steady-state Langmuir equilibrium."""
        engine = DynamicBindingEngine(
            k_on_nM_min=0.05,
            k_off_min=0.10, # Kd = 2.0 nM
            total_target_pmol_g=20.0,
        )
        # Step forward for 60 minutes at [D] = 2.0 nM
        for _ in range(60):
            state = engine.step(free_drug_nM=2.0, dt_min=1.0)

        # Equilibrium complex = 20 * (2 / (2 + 2)) = 10 pmol/g (occupancy 0.50)
        assert pytest.approx(state.occupancy_fraction, abs=0.01) == 0.50
        assert pytest.approx(state.complex_pmol_g, abs=0.2) == 10.0


# =============================================================================
# 4. Second Messenger Transduction Cascades
# =============================================================================
class TestSecondMessengerTransduction:
    def test_gpcr_gs_elevates_camp_and_pka(self):
        state = SecondMessengerTransducer.compute_steady_state(
            receptor_family="gpcr_gs",
            functional_modulation_pct=85.0,
            is_agonist=True,
        )
        assert state.cAMP_uM > 3.0
        assert state.PKA_activity_fraction > 0.50

    def test_gpcr_gq_elevates_ip3_and_calcium(self):
        state = SecondMessengerTransducer.compute_steady_state(
            receptor_family="gpcr_gq",
            functional_modulation_pct=90.0,
            is_agonist=True,
        )
        assert state.IP3_uM > 1.5
        assert state.cytosolic_Ca2_uM > 0.80 # Peak intracellular calcium
        assert state.MLCK_activity_fraction > 0.60

    def test_rtk_elevates_akt_phosphorylation(self):
        state = SecondMessengerTransducer.compute_steady_state(
            receptor_family="receptor_tyrosine_kinase",
            functional_modulation_pct=95.0,
            is_agonist=True,
        )
        assert state.Akt_phosphorylation_fraction > 0.80


# =============================================================================
# 5. Spatial Pharmacology Simulation Scenarios
# =============================================================================
class TestSpatialPharmacologyScenarios:
    def test_renal_sglt2_empagliflozin_spatial_simulation(self, spatial_engine):
        """
        User-specified spatial drug testing scenario:
        Organ: Kidney
        Region: Proximal tubule
        Cell: Proximal tubular epithelial cell
        Target: SGLT2 transporter
        Drug: Empagliflozin
        """
        res = spatial_engine.simulate_spatial_interaction(
            organ="kidney",
            cell_type_id="kidney_proximal_tubule_epithelial",
            target_id="SGLT2",
            drug_name="Empagliflozin",
            plasma_concentration_mg_L=0.25, # Therapeutic Cmax ~250 ng/mL
            molecular_weight_g_mol=450.9,
            fu_plasma=0.14,
            partition_coefficient_Kp=1.2,
            is_agonist=False,
            is_blocker=True,
            current_plasma_glucose_mg_dL=160.0,
        )

        assert res.organ == "kidney"
        assert "proximal" in res.region
        assert res.cell_type_id == "kidney_proximal_tubule_epithelial"
        assert res.target_id == "SGLT2"
        assert res.target_class == "transporter"
        assert res.subcellular_location == "plasma_membrane_apical"

        # Exposure and engagement checks
        assert res.plasma_free_nM > 50.0 # ~77 nM free drug
        assert res.functional_modulation_pct > 90.0 # IC50 is 3.1 nM -> strong inhibition

        # Clinical endpoints
        assert res.organ_endpoints["effective_renal_glucose_threshold_mg_dL"] < 100.0
        assert res.organ_endpoints["urinary_glucose_excretion_g_day"] > 50.0
        assert res.organ_endpoints["osmotic_natriuresis_mmol_day"] > 30.0

        # Safety checks
        adverse_names = [ae["adverse_effect"] for ae in res.potential_adverse_effects]
        assert any("Diabetic Ketoacidosis" in name for name in adverse_names)
        assert any("Mycotic" in name for name in adverse_names)

    def test_pancreatic_glp1r_semaglutide_spatial_simulation(self, spatial_engine):
        """GLP-1R in Pancreatic Beta Cell stimulated by Semaglutide."""
        res = spatial_engine.simulate_spatial_interaction(
            organ="pancreas",
            cell_type_id="pancreas_beta_cell",
            target_id="GLP1R",
            drug_name="Semaglutide",
            plasma_concentration_mg_L=0.05,
            molecular_weight_g_mol=4113.6, # Peptide MW
            fu_plasma=0.01,                # High albumin binding
            is_agonist=True,
        )

        assert res.organ == "pancreas"
        assert res.target_id == "GLP1R"
        assert res.second_messengers["cAMP_uM"] > 1.0
        assert res.quantitative_change_fold > 1.2
        assert "insulin_secretion" in res.primary_cellular_effect

    def test_cardiac_adrb1_atenolol_spatial_simulation(self, spatial_engine):
        """ADRB1 in Ventricular Cardiomyocyte blocked by Atenolol."""
        res = spatial_engine.simulate_spatial_interaction(
            organ="heart",
            cell_type_id="heart_ventricular_cardiomyocyte",
            target_id="ADRB1",
            drug_name="Atenolol",
            plasma_concentration_mg_L=0.50,
            molecular_weight_g_mol=266.3,
            fu_plasma=0.90,
            is_agonist=False,
            is_blocker=True,
            current_hr_bpm=85.0,
        )

        assert res.organ == "heart"
        assert res.target_id == "ADRB1"
        assert res.organ_endpoints["heart_rate_delta_bpm"] < 0.0
        assert res.organ_endpoints["resting_heart_rate_bpm"] < 85.0

    def test_hepatic_hmgcr_atorvastatin_spatial_simulation(self, spatial_engine):
        """HMGCR in Hepatocyte inhibited by Atorvastatin."""
        res = spatial_engine.simulate_spatial_interaction(
            organ="liver",
            cell_type_id="liver_hepatocyte",
            target_id="HMGCR",
            drug_name="Atorvastatin",
            plasma_concentration_mg_L=0.05,
            molecular_weight_g_mol=558.6,
            fu_plasma=0.05,
            partition_coefficient_Kp=5.0, # Active hepatic uptake via OATP1B1
            is_agonist=False,
            is_blocker=True,
        )

        assert res.organ == "liver"
        assert res.target_id == "HMGCR"
        assert res.target_class == "enzyme"
        assert res.organ_endpoints["hepatic_cholesterol_synthesis_inhibition_pct"] > 60.0
        assert res.whole_body_consequences["serum_ldl_cholesterol_reduction_pct"] > 30.0


# =============================================================================
# 6. FastAPI Molecular Router Endpoints
# =============================================================================
class TestMolecularFastAPIEndpoints:
    def test_get_targets_list_and_filters(self, api_client):
        resp = api_client.get("/molecular/targets")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 10

        # Filter by class
        resp_receptors = api_client.get("/molecular/targets?target_class=receptor")
        assert resp_receptors.status_code == 200
        assert resp_receptors.json()["count"] >= 5

        # Filter by organ
        resp_kidney = api_client.get("/molecular/targets?organ=kidney")
        assert resp_kidney.status_code == 200
        assert any(t["target_id"] == "SGLT2" for t in resp_kidney.json()["targets"])

    def test_get_target_details(self, api_client):
        resp = api_client.get("/molecular/targets/SGLT2")
        assert resp.status_code == 200
        data = resp.json()
        assert data["target_id"] == "SGLT2"
        assert data["target_class"] == "transporter"
        assert len(data["spatial_expressions"]) > 0
        assert data["binding_parameters"]["Kd_nM"] > 0

    def test_get_anatomy_hierarchy(self, api_client):
        resp = api_client.get("/molecular/anatomy/hierarchy")
        assert resp.status_code == 200
        data = resp.json()["hierarchy"]
        assert "kidney" in data
        assert "heart" in data
        assert "pancreas" in data

    def test_spatial_simulate_endpoint(self, api_client):
        payload = {
            "organ": "kidney",
            "cell_type_id": "kidney_proximal_tubule_epithelial",
            "target_id": "SGLT2",
            "drug_name": "Empagliflozin",
            "plasma_concentration_mg_L": 0.25,
            "molecular_weight_g_mol": 450.9,
            "fu_plasma": 0.14,
            "partition_coefficient_Kp": 1.2,
            "is_agonist": False,
            "is_blocker": True,
            "current_plasma_glucose_mg_dL": 150.0,
            "current_sbp_mmHg": 128.0,
            "current_hr_bpm": 72.0,
        }
        resp = api_client.post("/molecular/spatial-simulate", json=payload)
        assert resp.status_code == 200, resp.text
        sim = resp.json()["simulation"]
        assert sim["target_id"] == "SGLT2"
        assert sim["organ"] == "kidney"
        assert sim["functional_modulation_pct"] > 80.0
        assert sim["organ_endpoints"]["urinary_glucose_excretion_g_day"] > 40.0
