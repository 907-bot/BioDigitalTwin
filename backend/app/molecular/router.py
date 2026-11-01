"""
backend/app/molecular/router.py
FastAPI Endpoints for Spatial Pharmacology and Molecular Target Testing.
"""
from __future__ import annotations

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.molecular.tissue.cell_types import OrganType, get_anatomical_hierarchy
from app.molecular.targets.target_registry import (
    TargetClass,
    get_target_registry,
)
from app.molecular.spatial_engine import (
    get_spatial_pharmacology_engine,
    SpatialSimulationResult,
)

router = APIRouter(prefix="/molecular", tags=["Spatial Pharmacology"])


class SpatialSimulationRequest(BaseModel):
    organ: str = Field(..., description="Target organ (e.g. kidney, heart, pancreas, liver, vasculature)")
    cell_type_id: str = Field(..., description="Specific cell population ID (e.g. kidney_proximal_tubule_epithelial, heart_ventricular_cardiomyocyte)")
    target_id: str = Field(..., description="Molecular target symbol (e.g. SGLT2, GLP1R, ADRB1, AGTR1, INSR, HMGCR, CACNA1C)")
    drug_name: str = Field(..., description="Investigational or approved therapeutic agent (e.g. Empagliflozin, Semaglutide, Atenolol)")
    plasma_concentration_mg_L: float = Field(0.25, ge=0.0, le=50.0, description="Systemic total plasma concentration [mg/L]")
    molecular_weight_g_mol: float = Field(450.0, ge=10.0, le=200000.0, description="Molecular weight of drug [g/mol]")
    fu_plasma: float = Field(0.15, ge=0.001, le=1.0, description="Unbound fraction in plasma")
    partition_coefficient_Kp: float = Field(1.0, ge=0.01, le=50.0, description="Tissue-to-plasma partition coefficient")
    is_agonist: bool = Field(True, description="True if agonist; False if antagonist or competitive inhibitor")
    is_blocker: bool = Field(False, description="True for pore/channel blockers")
    current_plasma_glucose_mg_dL: float = Field(120.0, ge=30.0, le=600.0)
    current_sbp_mmHg: float = Field(125.0, ge=50.0, le=250.0)
    current_hr_bpm: float = Field(72.0, ge=30.0, le=220.0)


@router.get("/targets")
def list_targets(
    target_class: Optional[str] = Query(None, description="Filter by target class (receptor, enzyme, transporter, ion_channel)"),
    organ: Optional[str] = Query(None, description="Filter by expressing organ (e.g. kidney, heart, pancreas)"),
):
    """
    List registered molecular targets with filtering capability.
    """
    registry = get_target_registry()
    tc = TargetClass(target_class.lower()) if target_class else None
    org = OrganType(organ.lower()) if organ else None
    targets = registry.list_targets(target_class=tc, organ=org)

    return {
        "count": len(targets),
        "targets": [
            {
                "target_id": t.target_id,
                "name": t.name,
                "gene_symbol": t.gene_symbol,
                "uniprot_id": t.uniprot_id,
                "target_class": t.target_class.value,
                "receptor_family": t.receptor_family.value if t.receptor_family else None,
                "known_drugs": t.known_drugs,
                "expressed_organs": list(set(exp.organ.value for exp in t.spatial_expressions if exp.is_expressed)),
            }
            for t in targets
        ],
    }


@router.get("/targets/{target_id}")
def get_target_details(target_id: str):
    """
    Get comprehensive spatial expression, binding parameters, and functional metadata for a target.
    """
    registry = get_target_registry()
    target = registry.get(target_id)
    if not target:
        raise HTTPException(status_code=404, detail=f"Target '{target_id}' not found.")

    return {
        "target_id": target.target_id,
        "name": target.name,
        "gene_symbol": target.gene_symbol,
        "uniprot_id": target.uniprot_id,
        "target_class": target.target_class.value,
        "receptor_family": target.receptor_family.value if target.receptor_family else None,
        "description": target.description,
        "known_drugs": target.known_drugs,
        "binding_parameters": {
            "endogenous_ligand": target.binding_params.endogenous_ligand,
            "Kd_nM": target.binding_params.Kd_nM,
            "k_on": target.binding_params.k_on,
            "k_off": target.binding_params.k_off,
            "residence_time_min": target.binding_params.residence_time_min,
            "hill_coefficient": target.binding_params.hill_coefficient,
        },
        "functional_parameters": {
            "mechanism_type": target.functional_params.mechanism_type,
            "E_max": target.functional_params.E_max,
            "tau": target.functional_params.tau,
            "IC50_nM": target.functional_params.IC50_nM,
            "EC50_nM": target.functional_params.EC50_nM,
            "Km_uM": target.functional_params.Km_uM,
        },
        "spatial_expressions": [
            {
                "organ": exp.organ.value,
                "region": exp.region,
                "cell_type_id": exp.cell_type_id,
                "subcellular_location": exp.subcellular_location.value,
                "status": exp.status.value,
                "density_pmol_mg_protein": exp.density_pmol_mg_protein,
                "copies_per_cell": exp.copies_per_cell,
                "relative_expression_tpm": exp.relative_expression_tpm,
                "evidence_database": exp.evidence.source_database,
                "confidence_score": exp.evidence.confidence_score,
                "notes": exp.notes,
            }
            for exp in target.spatial_expressions
        ],
    }


@router.get("/anatomy/hierarchy")
def get_anatomy():
    """
    Returns the anatomical hierarchy tree (Organ -> Region -> Cell Types) for UI selection.
    """
    return {
        "hierarchy": get_anatomical_hierarchy(),
    }


@router.post("/spatial-simulate")
def run_spatial_simulation(req: SpatialSimulationRequest):
    """
    Executes a spatially resolved drug-target interaction simulation.
    """
    engine = get_spatial_pharmacology_engine()
    try:
        result = engine.simulate_spatial_interaction(
            organ=req.organ,
            cell_type_id=req.cell_type_id,
            target_id=req.target_id,
            drug_name=req.drug_name,
            plasma_concentration_mg_L=req.plasma_concentration_mg_L,
            molecular_weight_g_mol=req.molecular_weight_g_mol,
            fu_plasma=req.fu_plasma,
            partition_coefficient_Kp=req.partition_coefficient_Kp,
            is_agonist=req.is_agonist,
            is_blocker=req.is_blocker,
            current_plasma_glucose_mg_dL=req.current_plasma_glucose_mg_dL,
            current_sbp_mmHg=req.current_sbp_mmHg,
            current_hr_bpm=req.current_hr_bpm,
        )
        return {
            "status": "success",
            "simulation": result.__dict__,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}")

