"""
Phase 4: Adaptive Multi-Scale Human Digital Twin.

Layers:
  - Molecular Twin (PyG + RDKit): pathway graphs, drug-target models
  - Cellular Twin (Neural ODE): cell population dynamics
  - Graph Intelligence (Neo4j + PyG): drug-disease-pathway knowledge graph
  - Environmental + Behavioral: external factors and adherence
  - Multi-Scale Engine: hierarchical multi-rate estimation
  - Scientific Discovery: doWhy causal + LangChain hypotheses
  - Biomarkers 2.0: resilience, frailty, adaptability
  - Counterfactual V3: multi-objective optimization
  - Virtual Population V2: 1M+ GPU-accelerated
"""

from app.personalization.phase4.molecular import (
    MolecularState, build_pathway_graph, PathwayGNN, DrugTargetPredictor,
    compute_molecular_dynamics, molecular_to_cellular_signals,
)
from app.personalization.phase4.cellular import (
    CellularState, CellPopulationDynamics, cellular_to_organ_signals,
    CELL_TYPES, CELLULAR_DIM,
)
from app.personalization.phase4.graph_intelligence import (
    BioKnowledgeGraph, GraphEncoder, PatientSimilarityGraph, suggest_drugs_for_patient,
)
from app.personalization.phase4.environment_behavior import (
    EnvironmentState, BehavioralState,
    EnvironmentalModel, BehavioralModel, AdherenceModel, LifestyleModel,
)
from app.personalization.phase4.multi_scale_engine import (
    MultiScaleState, MultiScaleTwinEngine, CouplingSignalManager,
    TwinLayer, create_default_multi_scale_engine,
)
from app.personalization.phase4.scientific_discovery import (
    CausalDiscoveryEngine, HypothesisAgent, CausalGraph, CausalEdge,
    ScientificHypothesis, generate_twin_trial_data, CausalMethod,
)
from app.personalization.phase4.biomarkers20 import (
    compute_metabolic_age, compute_resilience_score, compute_frailty_index,
    compute_adaptability_score, compute_circadian_robustness,
    compute_inflammaging_score, compute_all_biomarkers_20, Biomarkers20,
)
from app.personalization.phase4.counterfactual_v3 import (
    CounterfactualEngineV3, InterventionDesign, MultiObjectiveOutcome, OutcomeMetric,
)
from app.personalization.phase4.virtual_population_v2 import (
    VirtualPopulationGeneratorV2, VirtualPatientV2,
)

__all__ = [
    "MolecularState", "build_pathway_graph", "PathwayGNN", "DrugTargetPredictor",
    "compute_molecular_dynamics", "molecular_to_cellular_signals",
    "CellularState", "CellPopulationDynamics", "cellular_to_organ_signals",
    "CELL_TYPES", "CELLULAR_DIM",
    "BioKnowledgeGraph", "GraphEncoder", "PatientSimilarityGraph", "suggest_drugs_for_patient",
    "EnvironmentState", "BehavioralState",
    "EnvironmentalModel", "BehavioralModel", "AdherenceModel", "LifestyleModel",
    "MultiScaleState", "MultiScaleTwinEngine", "CouplingSignalManager",
    "TwinLayer", "create_default_multi_scale_engine",
    "CausalDiscoveryEngine", "HypothesisAgent", "CausalGraph", "CausalEdge",
    "ScientificHypothesis", "generate_twin_trial_data", "CausalMethod",
    "compute_metabolic_age", "compute_resilience_score", "compute_frailty_index",
    "compute_adaptability_score", "compute_circadian_robustness",
    "compute_inflammaging_score", "compute_all_biomarkers_20", "Biomarkers20",
    "CounterfactualEngineV3", "InterventionDesign", "MultiObjectiveOutcome", "OutcomeMetric",
    "VirtualPopulationGeneratorV2", "VirtualPatientV2",
]
