"""
Tests for HIPAA/SOC 2 compliance layer.
"""

import os
import time
import pytest
import numpy as np

from app.personalization.hipaa import (
    FieldEncryptor, hash_identifier, AuditLogger, SessionManager,
    RateLimiter, ComplianceManager, ROLE_PERMISSIONS, Role, Permission,
    Principal,
)


@pytest.fixture
def master_key():
    return os.urandom(32)


class TestFieldEncryptor:
    def test_encrypt_decrypt_roundtrip(self, master_key):
        enc = FieldEncryptor(master_key)
        plaintext = "Patient SSN: 123-45-6789"
        envelope = enc.encrypt_field(plaintext)
        assert "ct" in envelope
        assert "n" in envelope
        assert "s" in envelope
        decrypted = enc.decrypt_field(envelope)
        assert decrypted == plaintext

    def test_different_nonces(self, master_key):
        enc = FieldEncryptor(master_key)
        e1 = enc.encrypt_field("test")
        e2 = enc.encrypt_field("test")
        assert e1["ct"] != e2["ct"]
        assert e1["n"] != e2["n"]

    def test_tamper_detection(self, master_key):
        enc = FieldEncryptor(master_key)
        envelope = enc.encrypt_field("original")
        # Tamper with ciphertext
        import base64
        ct = bytearray(base64.b64decode(envelope["ct"]))
        ct[0] ^= 1
        envelope["ct"] = base64.b64encode(bytes(ct)).decode()
        with pytest.raises(Exception):
            enc.decrypt_field(envelope)


class TestHashIdentifier:
    def test_same_input_same_hash(self):
        salt = "test-salt"
        h1 = hash_identifier("MRN-001", salt=salt)
        h2 = hash_identifier("MRN-001", salt=salt)
        assert h1 == h2

    def test_different_input_different_hash(self):
        salt = "test-salt"
        h1 = hash_identifier("MRN-001", salt=salt)
        h2 = hash_identifier("MRN-002", salt=salt)
        assert h1 != h2

    def test_different_salt_different_hash(self):
        h1 = hash_identifier("MRN-001", salt="salt1")
        h2 = hash_identifier("MRN-001", salt="salt2")
        assert h1 != h2

    def test_hash_length_16(self):
        h = hash_identifier("anything")
        assert len(h) == 16


class TestAuditLogger:
    def test_log_creates_record(self):
        audit = AuditLogger()
        rid = audit.log("test_action", "user1", "clinician", "patient/123")
        assert rid is not None
        assert len(audit._records) == 1

    def test_chain_validity(self):
        audit = AuditLogger()
        audit.log("action1", "user1", "admin", "resource1")
        audit.log("action2", "user2", "clinician", "resource2")
        valid, _ = audit.verify_chain()
        assert valid

    def test_tamper_detection(self):
        audit = AuditLogger()
        audit.log("a1", "u1", "admin", "r1")
        audit.log("a2", "u2", "admin", "r2")
        # Tamper with first record
        audit._records[0]["action"] = "tampered"
        valid, reason = audit.verify_chain()
        assert not valid
        assert reason is not None

    def test_query_by_principal(self):
        audit = AuditLogger()
        audit.log("a1", "alice", "clinician", "r1")
        audit.log("a2", "bob", "admin", "r2")
        results = audit.query(principal_id="alice")
        assert len(results) == 1
        assert results[0]["principal_id"] == "alice"

    def test_query_by_action(self):
        audit = AuditLogger()
        audit.log("read", "u1", "admin", "r1")
        audit.log("write", "u1", "admin", "r2")
        results = audit.query(action="read")
        assert len(results) == 1


class TestSessionManager:
    def test_create_session(self):
        sm = SessionManager()
        s = sm.create_session("user1", Role.CLINICIAN)
        assert s.session_id
        assert s.principal_id == "user1"

    def test_get_valid_session(self):
        sm = SessionManager()
        s = sm.create_session("user1", Role.CLINICIAN)
        retrieved = sm.get_session(s.session_id)
        assert retrieved is not None
        assert retrieved.principal_id == "user1"

    def test_revoked_session_invalid(self):
        sm = SessionManager()
        s = sm.create_session("user1", Role.CLINICIAN)
        sm.revoke(s.session_id)
        assert sm.get_session(s.session_id) is None

    def test_mfa_required_for_clinician(self):
        sm = SessionManager()
        assert sm.require_mfa(Role.CLINICIAN) is True
        assert sm.require_mfa(Role.PATIENT) is False


class TestRateLimiter:
    def test_allows_under_limit(self):
        rl = RateLimiter(max_requests=3, window_seconds=10)
        for _ in range(3):
            assert rl.is_allowed("key1")

    def test_blocks_over_limit(self):
        rl = RateLimiter(max_requests=2, window_seconds=10)
        assert rl.is_allowed("key1")
        assert rl.is_allowed("key1")
        assert not rl.is_allowed("key1")

    def test_separate_keys(self):
        rl = RateLimiter(max_requests=1, window_seconds=10)
        assert rl.is_allowed("key1")
        assert rl.is_allowed("key2")


class TestRBAC:
    def test_admin_has_all_permissions(self):
        admin = Principal(principal_id="admin1", role=Role.ADMIN)
        assert admin.has_permission(Permission.PRESCRIBE)
        assert admin.has_permission(Permission.MANAGE_USERS)

    def test_clinician_has_clinical_permissions(self):
        c = Principal(principal_id="c1", role=Role.CLINICIAN)
        assert c.has_permission(Permission.READ_TWIN)
        assert c.has_permission(Permission.PRESCRIBE)
        assert not c.has_permission(Permission.MANAGE_USERS)

    def test_researcher_limited(self):
        r = Principal(principal_id="r1", role=Role.RESEARCHER)
        assert r.has_permission(Permission.READ_TWIN)
        assert not r.has_permission(Permission.PRESCRIBE)

    def test_patient_minimal(self):
        p = Principal(principal_id="p1", role=Role.PATIENT)
        assert p.has_permission(Permission.READ_TWIN)
        assert not p.has_permission(Permission.READ_PHI) is False  # patient CAN read their own PHI


class TestComplianceManager:
    def test_compliance_report(self):
        cm = ComplianceManager()
        cm.audit.log("test", "u1", "admin", "r1")
        report = cm.generate_compliance_report()
        assert report.audit_chain_valid
        assert report.audit_records_count == 1
        assert report.overall_status() in ("COMPLIANT", "PARTIAL")
        report_dict = report.to_dict()
        assert "overall_status" in report_dict

    def test_export_audit(self):
        cm = ComplianceManager()
        cm.audit.log("a1", "u1", "admin", "r1")
        cm.audit.log("a2", "u2", "admin", "r2")
        exported = cm.export_audit_for_compliance_audit()
        assert len(exported) == 2
