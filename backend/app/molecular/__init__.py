"""
backend/app/molecular/__init__.py
Spatial Pharmacology and Molecular Target Engagement Engine for BioDigitalTwin.

Modules:
- targets: Curated registry of human molecular targets (Receptors, Enzymes, Transporters, Ion Channels)
- binding: Kinetic and thermodynamic target engagement mathematics
- tissue: Anatomical compartments, cell types, local exposure, and spatial expression atlas
- signaling: Second messengers, pathway topology, and cellular responses
- drug_response: Tissue integration and organ-level / systemic clinical consequences
- spatial_engine: High-level orchestrator for spatial drug testing
"""
from app.molecular.targets import (
    MolecularTarget,
    TargetClass,
    ReceptorFamily,
    SpatialTargetExpression,
    ExpressionStatus,
    MolecularTargetRegistry,
    get_target_registry,
    initialize_target_registry,
)
from app.molecular.tissue import (
    OrganType,
    SubcellularLocation,
    CellTypeSpec,
    CELL_TYPES_ATLAS,
    get_anatomical_hierarchy,
    LocalExposureCalculator,
    LocalExposureProfile,
    SpatialExpressionResolver,
)
from app.molecular.binding import (
    DynamicBindingEngine,
    compute_equilibrium_occupancy,
    TargetEngagementEvaluator,
    TargetEngagementResult,
)
from app.molecular.signaling import (
    SecondMessengerState,
    SecondMessengerTransducer,
    CellularResponse,
    CellularResponseEvaluator,
)
from app.molecular.drug_response import (
    TissueResponse,
    TissueResponseIntegrator,
    OrganResponseReport,
    OrganResponseEvaluator,
)
from app.molecular.spatial_engine import (
    SpatialSimulationResult,
    SpatialPharmacologyEngine,
    get_spatial_pharmacology_engine,
)

__all__ = [
    "MolecularTarget",
    "TargetClass",
    "ReceptorFamily",
    "SpatialTargetExpression",
    "ExpressionStatus",
    "MolecularTargetRegistry",
    "get_target_registry",
    "initialize_target_registry",
    "OrganType",
    "SubcellularLocation",
    "CellTypeSpec",
    "CELL_TYPES_ATLAS",
    "get_anatomical_hierarchy",
    "LocalExposureCalculator",
    "LocalExposureProfile",
    "SpatialExpressionResolver",
    "DynamicBindingEngine",
    "compute_equilibrium_occupancy",
    "TargetEngagementEvaluator",
    "TargetEngagementResult",
    "SecondMessengerState",
    "SecondMessengerTransducer",
    "CellularResponse",
    "CellularResponseEvaluator",
    "TissueResponse",
    "TissueResponseIntegrator",
    "OrganResponseReport",
    "OrganResponseEvaluator",
    "SpatialSimulationResult",
    "SpatialPharmacologyEngine",
    "get_spatial_pharmacology_engine",
]
