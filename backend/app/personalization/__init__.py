"""
Phase 3: Whole-Body Cellular Digital Twin.
30-dim physiology, 25-dim parameters, UKF estimation,
circadian clock, adipose-lipid metabolism, immune-inflammatory signaling,
hierarchical Bayesian priors, virtual cohorts, counterfactual simulation,
uncertainty quantification, explainability, and RL-based recommendation.
"""

from .core import PersonalizationEngine, create_personalization_engine
from .priors import PRIORS, PriorDistribution, get_subgroup_priors
from .state import Phase3TwinState, PHYSIO_DIM, PARAM_DIM
from .router import router as personalization_router
from .phase5.router import router as phase5_router

__all__ = [
    "PersonalizationEngine",
    "create_personalization_engine",
    "PRIORS",
    "PriorDistribution",
    "get_subgroup_priors",
    "Phase3TwinState",
    "PHYSIO_DIM",
    "PARAM_DIM",
    "personalization_router",
    "phase5_router",
]
