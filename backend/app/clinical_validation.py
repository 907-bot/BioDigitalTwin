"""
Clinical Validation Documentation — Bio-Digital Twin Platform

This document provides transparency on the validation status of the Bio-Digital Twin
clinical decision support system and the evidence base for treatment recommendations.

IMPORTANT: This system is intended to support clinical decision-making, NOT replace it.
All recommendations must be reviewed by qualified healthcare professionals.

DISCLAIMER: The digital twin generates predictive estimates based on computational models.
These estimates have known limitations and should not be used as the sole basis for
clinical decisions without professional judgment.
"""

# =============================================================================
# Model Validation Status
# =============================================================================

VALIDATION_STATUS = {
    "ukf_state_estimation": {
        "status": "Validated",
        "description": "Unscented Kalman Filter for physiological state estimation",
        "validation_approach": [
            "Synthetic data testing with known ground truth",
            "Covariance stability bounds enforced",
            "Innovation-based outlier gating",
            "Cross-validation on historical patient cohorts"
        ],
        "known_limitations": [
            "Assumes Gaussian noise distribution",
            "Nonlinear dynamics may cause residual errors",
            "Parameters assumed slowly time-varying"
        ],
        "references": [
            "Simon, D. (2006). Optimal State Estimation. Wiley.",
            "Julier, S.J. & Uhlmann, J.K. (1997). Unscented Filtering and Nonlinear Estimation. IEEE.",
        ]
    },
    
    "gnn_patient_embeddings": {
        "status": "Research Use Only",
        "description": "Graph Neural Network for patient similarity embedding",
        "validation_approach": [
            "Synthetic patient data with controlled phenotype clusters",
            "Edge weight correlation with clinical similarity metrics",
            "k-NN validation against known patient subgroups",
            "External validation pending clinical data partnership"
        ],
        "clinical_validation": "PENDING",
        "required_for_clinical_use": [
            "External validation on real patient cohorts",
            "IRB-approved retrospective study",
            "Prospective clinical trial"
        ],
        "references": [
            "Zhou, J. et al. (2020). Graph Neural Networks: A Review. IEEE Trans. Neural Netw.",
            "Wu, Z. et al. (2021). A Comprehensive Survey on Graph Neural Networks. IEEE TPAMI.",
        ]
    },
    
    "drug_drug_interactions": {
        "status": "Validated (Curated)",
        "description": "Drug-Drug Interaction database",
        "data_sources": [
            "FDA Drug Interaction Database",
            "CPIC (Clinical Pharmacogenetics Implementation Consortium)",
            "Lexicomp Drug Interactions",
            "Micromedex",
            "FDA Adverse Event Reporting System (FAERS)"
        ],
        "validation_approach": [
            "Expert curation of clinically significant interactions",
            "Mechanism-based categorization (CYP450, P-gp, QT prolongation)",
            "Severity classification by clinical consensus"
        ],
        "references": [
            "FDA. Drug Interaction Studies - Study Design, Data Analysis. 2012.",
            "Caudle, K.E. et al. (2014). Standard practices for pharmacogenetic testing. CPT.",
        ]
    },
    
    "pharmacogenomics": {
        "status": "Validated (CPIC-aligned)",
        "description": "Pharmacogenomic variant interpretation",
        "guidelines": "CPIC (Clinical Pharmacogenetics Implementation Consortium)",
        "covered_genes": ["CYP2C19", "CYP2C9", "CYP3A5", "DPYP", "TPMT", "NUDT15", "SLCO1B1"],
        "validation_approach": [
            "Gene-drug pairs mapped to CPIC level A evidence",
            "Phenotype prediction from diplotype",
            "Dosing recommendations per CPIC guidelines"
        ],
        "references": [
            "Caudle, K.E. et al. (2020). Standardizing CYP2C19 Genotype Results. Clin Pharmacol Ther.",
            "Clinical Pharmacogenetics Implementation Consortium. www.cpicpgx.org",
        ]
    },
    
    "pkpd_modeling": {
        "status": "Literature-Validated",
        "description": "Pharmacokinetic/Pharmacodynamic compartment models",
        "validation_approach": [
            "Parameters derived from published population PK studies",
            "Allometric scaling per regulatory guidance (FDA, EMA)",
            "Cockcroft-Gault for renal function adjustment",
            "Between-subject variability (BSV) on CL, Vc, ka"
        ],
        "limitations": [
            "Healthy volunteer parameters may not generalize to all patient populations",
            "Drug-drug interactions not fully captured",
            "Special populations (hepatic impairment, obesity) require additional validation"
        ],
        "references": [
            "Rowland, M. & Tozer, T.N. (2010). Clinical Pharmacokinetics. Lippincott Williams & Wilkins.",
            "FDA. Physiologically Based Pharmacokinetic Analyses. 2018.",
        ]
    },
    
    "causal_inference_do_calculus": {
        "status": "Validated (Theoretical)",
        "description": "Pearlian do-calculus for counterfactual reasoning",
        "validation_approach": [
            "Graph surgery correctly implements do-operator",
            "Refutation tests (placebo, random common cause)",
            "ODE-based causal model with known structure"
        ],
        "clinical_applicability": "Research/decision support only",
        "limitations": [
            "Causal assumptions based on domain knowledge, not necessarily RCT-confirmed",
            "Unobserved confounders may violate causal assumptions",
            "Results are counterfactuals, not observed outcomes"
        ],
        "references": [
            "Pearl, J. (2009). Causality: Models, Reasoning, and Inference. Cambridge.",
            "Hernán, M.A. & Robins, J.M. (2020). Causal Inference. CRC Press.",
        ]
    },
}


# =============================================================================
# Clinical Evidence Levels
# =============================================================================

EVIDENCE_LEVELS = {
    "A": "High-quality randomized controlled trial(s) or meta-analysis",
    "B": "Limited randomized trials, well-designed observational studies",
    "C": "Case series, case-control studies, expert opinion",
    "D": "Theoretical reasoning, preclinical data, mechanism-based inference"
}


# =============================================================================
# Treatment Recommendation Evidence Base
# =============================================================================

TREATMENT_EVIDENCE = {
    "insulin_dosing": {
        "evidence_level": "A",
        "sources": [
            "DCCT/EDIC. Long-term effects of intensive therapy on complications. NEJM 2005.",
            "ADA Standards of Care in Diabetes. Diabetes Care 2024.",
        ]
    },
    "metformin_lifestyle": {
        "evidence_level": "A",
        "sources": [
            "UKPDS 34. Effect of intensive blood-glucose control with metformin. Lancet 1998.",
            "Diabetes Prevention Program Research Group. Reduction in incidence. NEJM 2002.",
        ]
    },
    "sglt2i_cardiovascular": {
        "evidence_level": "A",
        "sources": [
            "EMPA-REG OUTCOME. Empagliflozin and cardiovascular outcomes. NEJM 2015.",
            "CANVAS Program. Canagliflozin and cardiovascular outcomes. NEJM 2017.",
        ]
    },
    "glp1_ra_weight_loss": {
        "evidence_level": "A",
        "sources": [
            "STEP trials. Semaglutide and body weight. NEJM 2021.",
            "SUSTAIN trials. Semaglutide efficacy. Lancet 2018.",
        ]
    },
}


# =============================================================================
# Model Uncertainty Communication
# =============================================================================

UNCERTAINTY_GUIDANCE = """
The Bio-Digital Twin communicates uncertainty through:

1. Prediction Intervals: 95% credible intervals around all continuous predictions.
   Narrow intervals indicate high confidence; wide intervals indicate uncertainty.

2. Drift Detection: Multi-level alerting when model predictions deviate from 
   observations, indicating model may need recalibration.

3. Safety Abstention: When uncertainty is too high (drift level 3, physiological
   implausibility, or covariance explosion), the system abstains from making
   predictions and recommends clinical consultation.

4. Evidence Levels: Treatment recommendations are tagged with evidence levels
   (A-D) to communicate the strength of supporting clinical evidence.

CLINICAL USE GUIDANCE:
- Level A evidence recommendations: Strong support, use as primary guidance
- Level B evidence recommendations: Moderate support, consider in context
- Level C evidence recommendations: Suggestive, use cautiously
- Level D evidence recommendations: Theoretical, use only with extreme caution
"""


# =============================================================================
# Validation Update Log
# =============================================================================

VALIDATION_LOG = """
2024-06-12: Initial validation documentation created
- UKF state estimation: Validated on synthetic data with known ground truth
- GNN embeddings: Research use only, external validation pending
- DDI database: Curated from FDA/CPIC/Lexicomp/Micromedex
- PGx: CPIC-aligned gene-drug pairs
- PK/PD: Literature-derived parameters with allometric scaling
- do-calculus: Theoretical validation complete, clinical validation pending
"""