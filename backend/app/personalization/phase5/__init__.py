"""
Phase 5: Autonomous Biological Intelligence Platform (ABIP).

Transforms the multi-scale digital twin into an autonomous scientific
platform capable of continuous learning, mechanism discovery,
in-silico experimentation, and hypothesis generation.

Pillars:
  1. Self-Updating Biological Knowledge Graph
  2. Mechanism Discovery Engine (causal/SCM/Bayesian)
  3. Scientific Hypothesis Generator
  4. Autonomous Virtual Clinical Trials (10M+)
  5. Multi-Agent Scientific Reasoning
  6. Adaptive Twin Evolution (online learning)
  7. Cellular & Tissue Simulation
  8. Foundation Model for Physiology
  9. Real-World Learning Network (federated + differential privacy)
 10. Validation Framework V2
"""

from app.personalization.phase5.knowledge_graph_engine import (
    KnowledgeGraphEngine, BiologicalNode, BiologicalEdge, EdgeType,
    LiteratureMiner, ClinicalTrialsMiner,
    NODE_TYPES as KG_NODE_TYPES,
)
from app.personalization.phase5.mechanism_discovery import (
    MechanismDiscoveryEngine, CausalMechanism, MechanismGraph,
    MechanismType, StructuralCausalModel,
    PCCausalDiscovery, BayesianNetworkDiscovery, TemporalCausalDiscovery,
    discover_causal_mechanisms,
)
from app.personalization.phase5.hypothesis_generator import (
    HypothesisGenerator, ScientificHypothesis,
    HypothesisEvidence, ValidationPlan,
    generate_hypotheses_from_cohort,
)
from app.personalization.phase5.clinical_trial_simulator import (
    ClinicalTrialSimulator, TrialDesign, TrialArm,
    TrialOutcome, TrialResult, TrialEndpoint, TrialPhase,
    simulate_comparative_trial,
)
from app.personalization.phase5.multi_agent_system import (
    MultiAgentSystem, PhysiologyAgent, PharmacologyAgent,
    GenomicsAgent, ClinicalAgent, ResearchAgent,
    AgentDebate, ConsensusResult,
    create_default_agent_system,
)
from app.personalization.phase5.adaptive_twin import (
    AdaptiveTwinEngine, OnlineBayesianUpdater,
    TwinEvolutionTracker, Observation, PredictionRecord, EvolutionEvent,
    create_adaptive_twin,
)
from app.personalization.phase5.tissue_simulation import (
    TissueSimulator, LiverTissue, KidneyTissue,
    CardiacTissue, AdiposeTissue,
    TissueState, simulate_tissue_response,
    TISSUE_TYPES,
)
from app.personalization.phase5.foundation_model import (
    PhysiologyFoundationModel, PhysiologyEncoder,
    PhysiologyDecoder, PhysiologyConfig,
    create_default_foundation_model,
)
from app.personalization.phase5.federated_learning import (
    FederatedLearningEngine, DifferentialPrivacyMechanism,
    PopulationKnowledgeBase,
    FederatedTwinClient,
    create_federated_network,
)
from app.personalization.phase5.validation_framework import (
    ValidationFrameworkV2, ValidationLevel,
    ValidationResult, ValidationCriterion,
    SyntheticTruthValidator, PublishedStudyValidator,
    ExternalCohortValidator, ProspectiveValidator,
    ClinicalTrialValidator,
    run_validation_pipeline,
)
from app.personalization.phase5.clinical_dataset import (
    ClinicalDataGenerator, ClinicalPatientRecord,
    RetrospectiveValidator, RetrospectiveValidationResult,
    generate_nhanes_style_dataset, run_retrospective_validation,
)
from app.personalization.phase5.calibration import (
    CalibrationAssessor, CalibrationReport, CalibrationPipeline,
    PlattCalibrator, BetaCalibrator, ConformalPredictor,
)
from app.personalization.phase5.causal_sensitivity import (
    CausalGraphSensitivity, SensitivityReport,
    EdgePerturbationResult, BootstrapResult,
    run_causal_sensitivity,
)
from app.personalization.phase5.population_broader import (
    POPULATION_MODULES, ETHNICITY_ADJUSTMENTS,
    get_population_adjustment, adjust_priors_for_population,
    PopulationModule,
)
from app.personalization.phase5.foundation_train import (
    FoundationModelTrainer, TrainingRun,
    PhysiologicalPretrainingDataset,
    train_foundation_model, load_foundation_model,
)
from app.personalization.phase5.overparameterization import (
    OverparameterizationAnalyzer, OverparameterizationReport,
    analyze_overparameterization,
)
from app.personalization.phase5.uncertainty_decomposition import (
    UncertaintyDecomposer, UncertaintyDecomposition,
    DecompositionReport, run_uncertainty_decomposition,
)
from app.personalization.phase5.stability_analysis import (
    StabilityAnalyzer, StabilityReport,
    analyze_multi_scale_stability,
)
from app.personalization.phase5.clinical_protocol import (
    ClinicalStudyProtocol, PowerAnalyzer,
    generate_twin_validation_protocol,
    compute_power_analysis, generate_study_report,
)
from app.personalization.phase5.counterfactual_sensitivity import (
    CounterfactualSensitivityAnalyzer,
    CounterfactualSensitivityReport,
    AdherenceSensitivityPoint, DoseResponsePoint,
    run_counterfactual_sensitivity,
)

__all__ = [
    "KnowledgeGraphEngine", "BiologicalNode", "BiologicalEdge", "EdgeType",
    "LiteratureMiner", "ClinicalTrialsMiner", "KG_NODE_TYPES",
    "MechanismDiscoveryEngine", "CausalMechanism", "MechanismGraph",
    "MechanismType", "StructuralCausalModel",
    "PCCausalDiscovery", "BayesianNetworkDiscovery", "TemporalCausalDiscovery",
    "discover_causal_mechanisms",
    "HypothesisGenerator", "ScientificHypothesis",
    "HypothesisEvidence", "ValidationPlan",
    "generate_hypotheses_from_cohort",
    "ClinicalTrialSimulator", "TrialDesign", "TrialArm",
    "TrialOutcome", "TrialResult", "TrialEndpoint", "TrialPhase",
    "simulate_comparative_trial",
    "MultiAgentSystem", "PhysiologyAgent", "PharmacologyAgent",
    "GenomicsAgent", "ClinicalAgent", "ResearchAgent",
    "AgentDebate", "ConsensusResult", "create_default_agent_system",
    "AdaptiveTwinEngine", "OnlineBayesianUpdater",
    "TwinEvolutionTracker", "Observation", "PredictionRecord", "EvolutionEvent",
    "create_adaptive_twin",
    "TissueSimulator", "LiverTissue", "KidneyTissue",
    "CardiacTissue", "AdiposeTissue",
    "TissueState", "simulate_tissue_response", "TISSUE_TYPES",
    "PhysiologyFoundationModel", "PhysiologyEncoder",
    "PhysiologyDecoder", "PhysiologyConfig",
    "create_default_foundation_model",
    "FederatedLearningEngine", "DifferentialPrivacyMechanism",
    "PopulationKnowledgeBase", "FederatedTwinClient",
    "create_federated_network",
    "ValidationFrameworkV2", "ValidationLevel",
    "ValidationResult", "ValidationCriterion",
    "SyntheticTruthValidator", "PublishedStudyValidator",
    "ExternalCohortValidator", "ProspectiveValidator",
    "ClinicalTrialValidator",
    "run_validation_pipeline",
    # Clinical dataset & retrospective validation
    "ClinicalDataGenerator", "ClinicalPatientRecord",
    "RetrospectiveValidator", "RetrospectiveValidationResult",
    "generate_nhanes_style_dataset", "run_retrospective_validation",
    # Calibration
    "CalibrationAssessor", "CalibrationReport", "CalibrationPipeline",
    "PlattCalibrator", "BetaCalibrator", "ConformalPredictor",
    # Causal sensitivity
    "CausalGraphSensitivity", "SensitivityReport",
    "EdgePerturbationResult", "BootstrapResult",
    "run_causal_sensitivity",
    # Broader population
    "POPULATION_MODULES", "ETHNICITY_ADJUSTMENTS",
    "get_population_adjustment", "adjust_priors_for_population",
    "PopulationModule",
    # Foundation model training
    "FoundationModelTrainer", "TrainingRun",
    "PhysiologicalPretrainingDataset",
    "train_foundation_model", "load_foundation_model",
    # Overparameterization
    "OverparameterizationAnalyzer", "OverparameterizationReport",
    "analyze_overparameterization",
    # Uncertainty decomposition
    "UncertaintyDecomposer", "UncertaintyDecomposition",
    "DecompositionReport", "run_uncertainty_decomposition",
    # Stability analysis
    "StabilityAnalyzer", "StabilityReport",
    "analyze_multi_scale_stability",
    # Clinical protocol
    "ClinicalStudyProtocol", "PowerAnalyzer",
    "generate_twin_validation_protocol",
    "compute_power_analysis", "generate_study_report",
    # Counterfactual sensitivity
    "CounterfactualSensitivityAnalyzer",
    "CounterfactualSensitivityReport",
    "AdherenceSensitivityPoint", "DoseResponsePoint",
    "run_counterfactual_sensitivity",
]
