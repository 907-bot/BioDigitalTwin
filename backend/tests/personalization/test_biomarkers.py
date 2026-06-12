"""Tests for digital biomarkers and drift detection."""
import numpy as np
from app.personalization.biomarkers import compute_recovery_score, compute_stress_score
from app.personalization.drift import DriftDetector


class TestBiomarkers:

    def test_recovery_score_baseline(self):
        score = compute_recovery_score([90.0, 91.0, 89.0], ir_state=1.0)
        assert 0 <= score <= 100

    def test_recovery_score_high_variability(self):
        low_var = compute_recovery_score([90.0] * 6, ir_state=1.0)
        high_var = compute_recovery_score([90.0, 180.0, 90.0, 180.0, 90.0, 180.0], ir_state=1.0)
        assert low_var > high_var

    def test_recovery_score_high_ir(self):
        low_ir = compute_recovery_score([90.0] * 6, ir_state=1.0)
        high_ir = compute_recovery_score([90.0] * 6, ir_state=50.0)
        assert low_ir > high_ir

    def test_stress_score_baseline(self):
        score = compute_stress_score([90.0, 91.0, 89.0], ir_state=1.0)
        assert 0 <= score <= 100

    def test_stress_score_high_variability(self):
        low_var = compute_stress_score([90.0] * 6, ir_state=1.0)
        high_var = compute_stress_score([90.0, 180.0, 90.0, 180.0, 90.0, 180.0], ir_state=1.0)
        assert low_var < high_var

    def test_stress_score_high_ir(self):
        low_ir = compute_stress_score([90.0] * 6, ir_state=1.0)
        high_ir = compute_stress_score([90.0] * 6, ir_state=50.0)
        assert low_ir < high_ir

    def test_short_window(self):
        score = compute_recovery_score([90.0], ir_state=1.0)
        assert score == 50.0
        score2 = compute_stress_score([90.0], ir_state=1.0)
        assert score2 == 50.0


class TestDriftDetector:

    def test_no_drift(self):
        dd = DriftDetector()
        for _ in range(10):
            dd.check(100.0, 100.0, 5.0)
        assert dd.level == 0
        assert dd.can_run_counterfactuals

    def test_level_1_warning(self):
        dd = DriftDetector()
        for _ in range(3):
            dd.check(200.0, 100.0, 5.0)
        assert dd.level >= 1

    def test_level_2_recalibrate(self):
        dd = DriftDetector()
        for _ in range(5):
            dd.check(200.0, 100.0, 5.0)
        assert dd.level >= 2

    def test_level_3_invalid(self):
        dd = DriftDetector()
        for _ in range(10):
            dd.check(200.0, 100.0, 5.0)
        assert dd.level == 3
        assert not dd.can_run_counterfactuals

    def test_reset(self):
        dd = DriftDetector()
        for _ in range(10):
            dd.check(200.0, 100.0, 5.0)
        assert dd.level == 3
        dd.reset()
        assert dd.level == 0
        assert dd.can_run_counterfactuals

    def test_label(self):
        dd = DriftDetector()
        assert dd.label == "normal"
        for _ in range(3):
            dd.check(200.0, 100.0, 5.0)
        assert dd.label == "warning"
        for _ in range(2):
            dd.check(200.0, 100.0, 5.0)
        assert dd.label == "recalibrate"
        for _ in range(5):
            dd.check(200.0, 100.0, 5.0)
        assert dd.label == "invalid"

    def test_recovery_after_reset(self):
        dd = DriftDetector()
        for _ in range(3):
            dd.check(200.0, 100.0, 5.0)
        assert dd.level == 1
        for _ in range(3):
            dd.check(100.0, 100.0, 5.0)
        assert dd.level == 0

    def test_status_dict(self):
        dd = DriftDetector()
        for _ in range(3):
            dd.check(200.0, 100.0, 5.0)
        status = dd.status()
        assert "level" in status
        assert "label" in status
        assert "subsystems" in status
        assert dd.can_run_counterfactuals
