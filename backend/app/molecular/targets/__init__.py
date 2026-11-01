"""
backend/app/molecular/targets/__init__.py
"""
from app.molecular.targets.target_registry import (
    MolecularTarget,
    TargetClass,
    ReceptorFamily,
    SpatialTargetExpression,
    ExpressionStatus,
    EvidenceCitation,
    BindingParameters,
    FunctionalParameters,
    MolecularTargetRegistry,
    get_target_registry,
)
from app.molecular.targets.receptor_models import register_all_receptors
from app.molecular.targets.enzyme_models import register_all_enzymes
from app.molecular.targets.transporter_models import register_all_transporters
from app.molecular.targets.ion_channel_models import register_all_ion_channels


def initialize_target_registry() -> MolecularTargetRegistry:
    """Initialize and populate the global registry with all curated molecular targets."""
    register_all_receptors()
    register_all_enzymes()
    register_all_transporters()
    register_all_ion_channels()
    return get_target_registry()


# Auto-populate on import
initialize_target_registry()

__all__ = [
    "MolecularTarget",
    "TargetClass",
    "ReceptorFamily",
    "SpatialTargetExpression",
    "ExpressionStatus",
    "EvidenceCitation",
    "BindingParameters",
    "FunctionalParameters",
    "MolecularTargetRegistry",
    "get_target_registry",
    "initialize_target_registry",
]
