"""
Pytest wrapper for the Digital Twin Benchmark Framework.

Run with:
    pytest backend/tests/personalization/test_benchmark_framework.py -v

This is a smoke test that runs the full benchmark suite at reduced speed.
"""

import pytest

from app.personalization.benchmark_framework import (
    run_all_benchmarks,
    benchmark_personalization,
    benchmark_parameter_recovery,
    benchmark_state_estimation,
    benchmark_counterfactual_validity,
    benchmark_calibration,
    benchmark_robustness,
    benchmark_physiological_realism,
    benchmark_generalization,
    benchmark_drift_detection,
    benchmark_clinical_usefulness,
)


def test_personalization():
    """Smoke test: personalization benchmark runs and returns a score."""
    result = benchmark_personalization(n_patients=1)
    assert 0.0 <= result.score <= 1.0
    assert len(result.sub_results) > 0


def test_parameter_recovery():
    """Smoke test: parameter recovery benchmark runs."""
    result = benchmark_parameter_recovery(n_patients=1)
    assert 0.0 <= result.score <= 1.0


def test_state_estimation():
    """Smoke test: state estimation benchmark runs."""
    result = benchmark_state_estimation(n_patients=1)
    assert 0.0 <= result.score <= 1.0


def test_counterfactual_validity():
    """Smoke test: counterfactual validity benchmark runs."""
    result = benchmark_counterfactual_validity(n_patients=1)
    assert 0.0 <= result.score <= 1.0


def test_calibration():
    """Smoke test: calibration benchmark runs."""
    result = benchmark_calibration(n_patients=1)
    assert 0.0 <= result.score <= 1.0


def test_robustness():
    """Smoke test: robustness benchmark runs."""
    result = benchmark_robustness(n_patients=1)
    assert 0.0 <= result.score <= 1.0


def test_physiological_realism():
    """Smoke test: physiological realism benchmark runs."""
    result = benchmark_physiological_realism(n_patients=1)
    assert 0.0 <= result.score <= 1.0


def test_generalization():
    """Smoke test: generalization benchmark runs."""
    result = benchmark_generalization(n_patients=1)
    assert 0.0 <= result.score <= 1.0


def test_drift_detection():
    """Smoke test: drift detection benchmark runs."""
    result = benchmark_drift_detection(n_patients=1)
    assert 0.0 <= result.score <= 1.0


def test_clinical_usefulness():
    """Smoke test: clinical usefulness benchmark runs."""
    result = benchmark_clinical_usefulness(n_patients=1)
    assert 0.0 <= result.score <= 1.0


@pytest.mark.slow
def test_full_benchmark_suite():
    """Run the full benchmark suite (slow, ~3 min)."""
    overall, dims = run_all_benchmarks(n_patients=1, verbose=False)
    assert 0.0 <= overall <= 1.0
    assert len(dims) == 10


if __name__ == "__main__":
    test_full_benchmark_suite()
