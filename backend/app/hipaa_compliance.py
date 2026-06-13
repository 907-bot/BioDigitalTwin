"""
HIPAA Compliance Documentation — Bio-Digital Twin Platform

This document describes the technical safeguards implemented in the Bio-Digital Twin
platform for compliance with the HIPAA Security Rule (45 CFR Part 164) and
SOC 2 Trust Services Criteria.

IMPORTANT: This document describes TECHNICAL safeguards only. Organizational and
administrative safeguards (policies, training, BAA agreements) must be implemented
separately and are outside the scope of this code.

DISCLAIMER: Implementation of technical safeguards does not guarantee HIPAA
compliance. Organizations must conduct their own risk assessments and implement
appropriate administrative and physical safeguards.

Reference: HHS HIPAA Security Rule - 45 CFR § 164.308, § 164.310, § 164.312
"""

# =============================================================================
# Technical Safeguards Implemented
# =============================================================================

TECHNICAL_SAFEGUARDS = {
    "access_control": {
        "requirement": "45 CFR § 164.312(a)(1)",
        "implemented": [
            "Unique user identification (patient_id pseudonymization via HMAC-SHA256)",
            "Automatic logoff (session timeout configurable via TOKEN_EXPIRY_MINUTES)",
            "Encryption and authentication (TLS 1.2+ required, API key auth for sensitive endpoints)",
            "Role-based access control (ADMIN, CLINICIAN, RESEARCHER, PATIENT, SERVICE, AUDITOR)"
        ],
        "implementation_file": "app/personalization/hipaa.py"
    },
    
    "audit_controls": {
        "requirement": "45 CFR § 164.312(b)",
        "implemented": [
            "Immutable audit log with HMAC-SHA256 signatures",
            "Log entries include: timestamp, user_id, action, resource, result, ip_address",
            "Tamper-evident log storage",
            "Query endpoint: GET /admin/audit-log"
        ],
        "implementation_file": "app/personalization/audit.py"
    },
    
    "integrity": {
        "requirement": "45 CFR § 164.312(c)(1)",
        "implemented": [
            "AES-256-GCM encryption at rest for PHI fields",
            "Envelope encryption with DEK/KEK hierarchy",
            "HMAC-SHA256 message integrity verification",
            "Immutable audit trail"
        ],
        "implementation_file": "app/personalization/hipaa.py"
    },
    
    "transmission_security": {
        "requirement": "45 CFR § 164.312(e)(1)",
        "implemented": [
            "TLS 1.2+ enforced via HSTS header (max-age=31536000)",
            "Certificate validation required",
            "HTTPS-only deployment via CORS_ORIGINS validation"
        ],
        "implementation_file": "app/main.py"
    },
    
    "encryption_at_rest": {
        "requirement": "45 CFR § 164.312(a)(2)(iv)",
        "implemented": [
            "AES-256-GCM for PHI field encryption",
            "Secure key derivation via PBKDF2",
            "Envelope encryption pattern for scalability"
        ],
        "implementation_file": "app/personalization/hipaa.py"
    },
}


# =============================================================================
# PHI Data Classification
# =============================================================================

PHI_FIELDS = {
    "identified": [
        "patient_id",  # Direct identifier
        "mrn",         # Medical Record Number
        "ssn",         # Social Security Number
        "name",
        "date_of_birth",
        "address"
    ],
    "quasi_identified": [
        "age",
        "gender",
        "zip_code",
        "admission_date",
        "treatment_date"
    ],
    "de_identified": [
        "glucose_values",      # Time series (de-identified if no linkage)
        "heart_rate",          # Physiological measurements
        "blood_pressure",
        "lab_results",
        "treatment_records"    # Without patient linkage
    ]
}


# =============================================================================
# De-identification Methods
# =============================================================================

DEIDENTIFICATION_METHODS = {
    "pseudonymization": {
        "method": "HMAC-SHA256 hash with deployment-specific salt",
        "purpose": "Enable data linkage within deployment without exposing identity",
        "implementation": "hash_identifier() in app/personalization/hipaa.py",
        "requirements": [
            "BIO_HASH_SALT must be set in production",
            "Salt must be stored securely (HSM or key vault recommended)",
            "Salt rotation requires re-hashing of all identifiers"
        ],
        "hipaa_note": "Pseudonymization is NOT equivalent to de-identification under HIPAA Safe Harbor."
    },
    
    "aggregation": {
        "method": "Minimum cell size of 10 patients",
        "purpose": "Prevent re-identification through small cell suppression",
        "implementation": "Applied in cohort analysis endpoints"
    },
    
    "k_anonymity": {
        "method": "Quasi-identifier generalization",
        "purpose": "Ensure each record is identical to at least k-1 others",
        "implementation": "Applied in data export endpoints"
    }
}


# =============================================================================
# Informed Consent Handling
# =============================================================================

INFORMED_CONSENT = """
The Bio-Digital Twin platform handles informed consent as follows:

1. CONSENT VERIFICATION (Technical Safeguard)
   - Consent_status field tracked per patient record
   - Required consent categories: data_processing, research_use, data_sharing
   - Consent expiration tracking with renewal reminders

2. DATA PROCESSING RESTRICTIONS
   - If consent withdrawn: anonymize existing data, stop new processing
   - Granular consent: patient can withdraw from research but allow treatment use
   - Consent audit trail maintained indefinitely

3. REQUIRED CONSENT CATEGORIES
   a) Treatment Use: Required for any clinical decision support
   b) Quality Improvement: Anonymized analytics on treatment outcomes
   c) Research Use: Requires separate IRB approval per study
   d) Data Sharing: Requires explicit patient authorization form

4. IMPLEMENTATION
   - Consent status checked before any PHI processing
   - Withdrawal triggers automatic data handling workflow
   - Patient-facing portal for consent management (future enhancement)

IMPORTANT: Technical consent tracking does NOT replace legal informed consent
documentation. Paper/digital consent forms must be collected and stored
separately per institutional requirements.
"""


# =============================================================================
# SOC 2 Trust Services Criteria Alignment
# =============================================================================

SOC2_ALIGNMENT = {
    "security": {
        "criteria": "Common Criteria (CC) aligned",
        "controls": [
            "Logical access controls (API key, role-based)",
            "System boundary protection (TLS, firewall rules)",
            "Vulnerability management (dependency scanning)",
            "Incident response procedures (documented)"
        ]
    },
    
    "availability": {
        "criteria": "A1 - Availability commitments",
        "controls": [
            "Health check endpoint: GET /health",
            "Graceful degradation when non-critical services fail",
            "Drift detection alerts before prediction failure"
        ]
    },
    
    "processing_integrity": {
        "criteria": "PI1 - Processing integrity",
        "controls": [
            "Input validation on all endpoints",
            "Output plausibility checks",
            "Audit trail for all data modifications"
        ]
    },
    
    "confidentiality": {
        "criteria": "C1 - Confidentiality commitments",
        "controls": [
            "PHI encryption at rest (AES-256-GCM)",
            "Role-based data access (minimum necessary)",
            "De-identification before research use"
        ]
    },
    
    "privacy": {
        "criteria": "P1 - Privacy notice, P2 - Choice, P3 - Collection",
        "controls": [
            "Consent tracking per patient",
            "Data minimization (only collect necessary fields)",
            "Retention and deletion policies (configurable)"
        ]
    }
}


# =============================================================================
# Risk Assessment Summary
# =============================================================================

RISK_ASSESSMENT = """
RESIDUAL RISKS (mitigations in place but not eliminated):

1. Model Uncertainty Risk
   - Risk: Incorrect predictions leading to suboptimal treatment
   - Mitigation: Safety abstention, prediction intervals, drift detection
   - Residual: Clinical judgment always required

2. Data Breach Risk
   - Risk: Unauthorized access to PHI
   - Mitigation: Encryption, access controls, audit logging
   - Residual: Defense-in-depth, but no system is 100% secure

3. Model Bias Risk
   - Risk: Predictions systematically inaccurate for subpopulations
   - Mitigation: Stratified validation, fairness metrics
   - Residual: Continuous monitoring required

4. Operational Risk
   - Risk: System downtime affecting care delivery
   - Mitigation: Graceful degradation, health monitoring
   - Residual: Backup procedures required

REQUIRED ORGANIZATIONAL SAFEGUARDS (not implemented in code):
- Business Associate Agreements (BAAs) with all service providers
- Workforce HIPAA training
- Incident response procedures
- Regular security risk assessments
- Policy documentation for access management
- Physical security controls
"""


# =============================================================================
# Compliance Update Log
# =============================================================================

COMPLIANCE_LOG = """
2024-06-12: Initial HIPAA technical safeguards documentation created
- Access controls: API key auth, RBAC
- Audit logging: HMAC-signed immutable logs
- Encryption: AES-256-GCM with envelope encryption
- Transmission: TLS 1.2+ enforcement via HSTS
- Consent: Status tracking implemented
- De-identification: HMAC-SHA256 pseudonymization

NOTE: This documentation describes TECHNICAL safeguards only.
Organizational and administrative safeguards must be implemented separately.
Consult with HIPAA compliance officer and legal counsel before deployment.
"""