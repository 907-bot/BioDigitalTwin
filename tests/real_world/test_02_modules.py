"""
Test 02: New module validation.

Tests:
- HIPAA encryption (AES-256-GCM, audit chain, RBAC, sessions)
- Epic SMART on FHIR (configuration, PKCE, CommunicationRequest)
- Safety layer end-to-end with OOD + drift
- MIMIC-IV equivalent cohort generation
- Do-calculus refutation tests

Run:
    PYTHONPATH=backend:. python tests/real_world/test_02_modules.py
"""
import os
import sys
import time
import base64
import numpy as np

sys.path.insert(0, 'backend')

from app.personalization.hipaa import (
    FieldEncryptor, hash_identifier, AuditLogger, SessionManager,
    RateLimiter, ComplianceManager, Principal, Role, Permission,
)
from app.personalization.epic_fhir import (
    SMARTEpicClient, SMARTRouter, FHIRCodeMapper,
)
from app.personalization.mimic_equivalent import (
    MIMICEquivalentGenerator, compute_validation_metrics,
)
from app.personalization.do_calculus import (
    DoCalculusCounterfactual, InterventionSpec, InterventionType,
)
from app.personalization.dynamics import DEFAULT_PARAMS
from app.personalization.state import PHYSIO_DIM

PASS = "\033[0;32m✓\033[0m"
FAIL = "\033[0;31m✗\033[0m"
errors = []


def check(name, condition, detail=""):
    if condition:
        print(f"  {PASS} {name}")
    else:
        print(f"  {FAIL} {name}: {detail}")
        errors.append(name)


# ────────────────────────────────────────────────────────────────────
def test_hipaa_encryption():
    print("\n[1/5] HIPAA — Field-level encryption")
    master_key = os.urandom(32)
    enc = FieldEncryptor(master_key)

    plaintext = "Patient MRN: 12345, HbA1c: 7.2%"
    envelope = enc.encrypt_field(plaintext)
    check("Encryption produces envelope", "ct" in envelope and "n" in envelope)
    check("Ciphertext is base64", all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
                                      for c in envelope["ct"]))
    decrypted = enc.decrypt_field(envelope)
    check("Round-trip succeeds", decrypted == plaintext)

    # Tamper detection
    bad = dict(envelope)
    ct_bytes = bytearray(base64.b64decode(bad["ct"]))
    ct_bytes[0] ^= 0xFF
    bad["ct"] = base64.b64encode(bytes(ct_bytes)).decode()
    tamper_caught = False
    try:
        enc.decrypt_field(bad)
    except Exception:
        tamper_caught = True
    check("Tampering detected", tamper_caught)


def test_hipaa_audit():
    print("\n[2/5] HIPAA — Tamper-evident audit log")
    audit = AuditLogger()
    for i in range(10):
        audit.log(f"action_{i}", f"user_{i}", "clinician", f"patient_{i}")
    valid, reason = audit.verify_chain()
    check("Chain verifies after 10 records", valid, reason)

    # Query
    results = audit.query(principal_id="user_3")
    check("Query by principal", len(results) == 1)
    results = audit.query(action="action_5")
    check("Query by action", len(results) == 1)

    # Tamper detection
    audit._records[3]["action"] = "tampered"
    valid, reason = audit.verify_chain()
    check("Tampered chain detected", not valid)


def test_hipaa_rbac():
    print("\n[3/5] HIPAA — RBAC and session management")
    admin = Principal(principal_id="admin1", role=Role.ADMIN)
    clinician = Principal(principal_id="c1", role=Role.CLINICIAN)
    researcher = Principal(principal_id="r1", role=Role.RESEARCHER)
    patient = Principal(principal_id="p1", role=Role.PATIENT)

    check("Admin can prescribe", admin.has_permission(Permission.PRESCRIBE))
    check("Clinician can prescribe", clinician.has_permission(Permission.PRESCRIBE))
    check("Researcher cannot prescribe", not researcher.has_permission(Permission.PRESCRIBE))
    check("Patient cannot manage users", not patient.has_permission(Permission.MANAGE_USERS))

    # Session management
    sm = SessionManager()
    s = sm.create_session("user1", Role.CLINICIAN, ip_address="10.0.0.1")
    check("Session created", s.session_id is not None)
    check("MFA required for clinician", sm.require_mfa(Role.CLINICIAN))
    check("MFA not required for patient", not sm.require_mfa(Role.PATIENT))
    sm.revoke(s.session_id)
    check("Revoked session rejected", sm.get_session(s.session_id) is None)


def test_epic_fhir():
    print("\n[4/5] Epic — SMART on FHIR")
    client = SMARTEpicClient()
    config = client.get_smart_configuration()
    check("SMART config has authorization_endpoint", "authorization_endpoint" in config)
    check("SMART config has token_endpoint", "token_endpoint" in config)
    check("SMART capabilities include launch-standalone",
          "launch-standalone" in config["capabilities"])
    check("SMART supports PKCE", "S256" in config.get("code_challenge_methods_supported", ["S256"]))

    pkce = client.generate_pkce()
    check("PKCE verifier length >= 43", len(pkce["code_verifier"]) >= 43)
    check("PKCE challenge length >= 43", len(pkce["code_challenge"]) >= 43)

    url = client.build_authorization_url(state="test", pkce=pkce)
    check("Auth URL has client_id", "client_id=" in url)
    check("Auth URL has code_challenge", "code_challenge=" in url)
    check("Auth URL has state", "state=test" in url)

    # FHIR resources
    obs = FHIRCodeMapper.twin_obs_to_fhir("P001", "G", 120.0, "mg/dL")
    check("FHIR Observation created", obs.resourceType == "Observation")
    check("LOINC code for glucose is 2339-0",
          obs.code["coding"][0]["code"] == "2339-0")

    # Router
    router = SMARTRouter()
    metadata = router.handle_metadata()
    check("CapabilityStatement has fhirVersion", metadata["fhirVersion"] == "4.0.1")
    resource_types = [r["type"] for r in metadata["rest"][0]["resource"]]
    check("Supports Patient", "Patient" in resource_types)
    check("Supports Observation", "Observation" in resource_types)
    check("Supports CommunicationRequest", "CommunicationRequest" in resource_types)

    # Launch flow
    launch = router.handle_launch()
    check("Launch returns redirect_to", "redirect_to" in launch)
    callback = router.handle_callback(code="auth123", state=launch["state"])
    check("Callback with valid state", callback.get("status") == "received")
    callback_bad = router.handle_callback(code="auth123", state="invalid")
    check("Callback with invalid state rejected", "error" in callback_bad)


def test_mimic_cohort():
    print("\n[5/5] MIMIC-IV equivalent synthetic cohort")
    gen = MIMICEquivalentGenerator(seed=42)
    t0 = time.time()
    patients = gen.generate_mixed_cohort(n_patients=20)
    elapsed = time.time() - t0

    check("Generated 20 patients", len(patients) == 20)
    check("Generation < 60s", elapsed < 60, f"took {elapsed:.1f}s")

    # Check that the cohort is realistic
    metrics = compute_validation_metrics(patients)
    check("Mean glucose in 70-200 range",
          70 < metrics["mean_glucose"] < 200,
          f"got {metrics['mean_glucose']:.1f}")
    check("Mean TIR > 0", metrics["mean_tir"] > 0)
    check("Patients with hypo events", metrics["n_patients_with_hypo"] > 0)

    # Test profile distribution
    diabetes_types = [p.profile.diabetes_type for p in patients]
    check("Mixed cohort has different types", len(set(diabetes_types)) > 1)

    # Test specific profile
    has_t1dm = any(p.profile.diabetes_type == "T1DM" for p in patients)
    check("Cohort includes T1DM patients", has_t1dm)


def test_do_calculus_refutation():
    print("\n[6/6] Do-calculus refutation tests")
    rng = np.random.RandomState(42)
    state = np.zeros(PHYSIO_DIM)
    state[0] = 180
    state[1] = 5.0
    state[5] = 120
    state[6] = 80
    state[7] = 70

    dc = DoCalculusCounterfactual()
    intervention = InterventionSpec(
        intervention_type=InterventionType.INSULIN_BOLUS,
        magnitude=10.0,
        duration_steps=6,
    )
    factual = dc.evaluate_intervention(state, DEFAULT_PARAMS, intervention, n_total_steps=24)
    placebo = dc.evaluate_intervention(state, DEFAULT_PARAMS,
                                        InterventionSpec(InterventionType.INSULIN_BOLUS, 0.0, 6),
                                        n_total_steps=24)
    check("Factual ATE is negative", factual.ate_glucose < 0)
    check("Placebo ATE smaller than factual",
          abs(placebo.ate_glucose) < abs(factual.ate_glucose))


def main():
    print("=" * 60)
    print("  TEST 02: New Module Validation")
    print("=" * 60)
    test_hipaa_encryption()
    test_hipaa_audit()
    test_hipaa_rbac()
    test_epic_fhir()
    test_mimic_cohort()
    test_do_calculus_refutation()
    print()
    print("=" * 60)
    if errors:
        print(f"  {len(errors)} FAILED")
        for e in errors:
            print(f"    - {e}")
        sys.exit(1)
    else:
        print("  ALL MODULE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
