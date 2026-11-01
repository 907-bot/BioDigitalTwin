"""
backend/app/molecular/tissue/target_expression.py
Spatial Target Expression Querier & Tissue Density Resolver.
"""
from __future__ import annotations

from typing import Dict, Any, Optional, List
from app.molecular.tissue.cell_types import OrganType, CellTypeSpec, CELL_TYPES_ATLAS


class SpatialExpressionResolver:
    """
    Resolves target expression status and quantitative densities across anatomical compartments.
    """
    @staticmethod
    def get_expression_summary(target_id: str) -> Dict[str, Any]:
        from app.molecular.targets.target_registry import get_target_registry
        registry = get_target_registry()
        target = registry.get(target_id)
        if not target:
            return {"error": f"Target '{target_id}' not found in registry."}

        expressions = []
        for exp in target.spatial_expressions:
            cell_spec = CELL_TYPES_ATLAS.get(exp.cell_type_id)
            cell_name = cell_spec.name if cell_spec else exp.cell_type_id
            expressions.append({
                "organ": exp.organ.value,
                "region": exp.region,
                "cell_type_id": exp.cell_type_id,
                "cell_name": cell_name,
                "subcellular_location": exp.subcellular_location.value,
                "status": exp.status.value,
                "density_pmol_mg_protein": exp.density_pmol_mg_protein,
                "copies_per_cell": exp.copies_per_cell,
                "relative_expression_tpm": exp.relative_expression_tpm,
                "confidence_score": exp.evidence.confidence_score,
                "source": exp.evidence.source_database,
                "notes": exp.notes,
            })

        return {
            "target_id": target.target_id,
            "name": target.name,
            "gene_symbol": target.gene_symbol,
            "target_class": target.target_class.value,
            "expressions": expressions,
        }
