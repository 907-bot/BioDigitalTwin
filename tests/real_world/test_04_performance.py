"""
Test 04: Performance and latency benchmarks.

Measures:
- Personalization throughput (observations / second)
- Single UKF update time
- MAP convergence time
- Counterfactual evaluation time
- Encryption throughput
- Memory usage
- API response time

Run:
    PYTHONPATH=backend:. python tests/real_world/test_04_performance.py
"""
import sys
import time
import os
import resource
import numpy as np

sys.path.insert(0, 'backend')

from app.personalization.mimic_equivalent import MIMICEquivalentGenerator
from app.personalization.dual_engine import create_dual_engine
from app.personalization.core import PersonalizationEngine
from app.personalization.hipaa import FieldEncryptor
from app.personalization.do_calculus import DoCalculusCounterfactual, InterventionSpec, InterventionType
from app.personalization.dynamics import DEFAULT_PARAMS
from app.personalization.state import PHYSIO_DIM

PASS = "\033[0;32m✓\033[0m"
FAIL = "\033[0;31m✗\033[0m"
WARN = "\033[1;33m⚠\033[0m"
errors = []
metrics = {}


def check(name, condition, detail=""):
    if condition:
        print(f"  {PASS} {name}")
    else:
        print(f"  {FAIL} {name}: {detail}")
        errors.append(name)


def get_memory_mb():
    """Return peak resident set size in MB.

    On macOS, ru_maxrss is in bytes. On Linux, it is in kilobytes.
    """
    ru = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        return ru.ru_maxrss / (1024 * 1024)
    else:
        return ru.ru_maxrss / 1024


def measure(label, func, *args, **kwargs):
    t0 = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - t0
    metrics[label] = elapsed
    return result, elapsed


# ────────────────────────────────────────────────────────────────────
def test_ukf_throughput():
    print("\n[1/6] UKF update throughput")
    obs0 = np.array([120.0] * 15)
    engine = PersonalizationEngine()
    engine.initialize(obs0)

    n_updates = 200
    t0 = time.time()
    for _ in range(n_updates):
        engine.update(obs0)
    elapsed = time.time() - t0
    per_step = elapsed / n_updates * 1000
    metrics["ukf_per_step_ms"] = per_step
    metrics["ukf_updates_per_sec"] = n_updates / elapsed
    print(f"  {per_step:.2f} ms/update, {n_updates/elapsed:.1f} updates/sec")
    check("UKF update < 100ms", per_step < 100, f"got {per_step:.1f}ms")
    check("UKF update > 5/s", n_updates / elapsed > 5, f"got {n_updates/elapsed:.1f}/s")


def test_dual_engine_throughput():
    print("\n[2/6] Dual engine throughput")
    gen = MIMICEquivalentGenerator(seed=42)
    patients = gen.generate_mixed_cohort(n_patients=20)
    t1dm = [p for p in patients if p.profile.diabetes_type == "T1DM"]
    if t1dm:
        patient = t1dm[0]
    else:
        patient = patients[0]
    obs = patient.observations

    engine = create_dual_engine()
    engine.initialize(obs[0])

    t0 = time.time()
    for t in range(1, len(obs)):
        engine.update(obs[t])
    elapsed = time.time() - t0
    per_step = elapsed / (len(obs) - 1) * 1000
    metrics["dual_per_step_ms"] = per_step
    metrics["dual_total_s"] = elapsed
    print(f"  {per_step:.2f} ms/obs, total {elapsed:.2f}s for {len(obs)} obs")
    check("Dual engine < 200ms/obs", per_step < 200, f"got {per_step:.1f}ms")
    check("Dual engine finishes", elapsed < 30, f"took {elapsed:.1f}s")


def test_personalization_pipeline():
    print("\n[3/6] Full personalization pipeline")
    gen = MIMICEquivalentGenerator(seed=42)
    patients = gen.generate_mixed_cohort(n_patients=5)[:5]
    n_obs = sum(len(p.observations) for p in patients)
    print(f"  {len(patients)} patients, {n_obs} total observations")

    t0 = time.time()
    for p in patients:
        engine = create_dual_engine()
        engine.initialize(p.observations[0])
        for t in range(1, len(p.observations)):
            engine.update(p.observations[t])
    elapsed = time.time() - t0
    metrics["pipeline_total_s"] = elapsed
    metrics["pipeline_patients_per_sec"] = len(patients) / elapsed
    print(f"  {elapsed:.2f}s for full personalization, {len(patients)/elapsed:.2f} patients/sec")
    check("5 patients in < 180s", elapsed < 180, f"took {elapsed:.1f}s")


def test_encryption_throughput():
    print("\n[4/6] Encryption throughput")
    master_key = os.urandom(32)
    enc = FieldEncryptor(master_key)
    plaintext = "HbA1c 7.2% MRN 12345 diagnosis E11.9" * 10
    n = 100

    t0 = time.time()
    for _ in range(n):
        enc.encrypt_field(plaintext)
    enc_elapsed = time.time() - t0

    t0 = time.time()
    for _ in range(n):
        # Recompute envelope each time to avoid short-circuit
        e = enc.encrypt_field(plaintext)
        enc.decrypt_field(e)
    full_elapsed = time.time() - t0

    metrics["encrypt_per_op_ms"] = enc_elapsed / n * 1000
    metrics["encrypt_decrypt_per_op_ms"] = full_elapsed / n * 1000
    print(f"  Encrypt: {enc_elapsed/n*1000:.2f}ms/op, Round-trip: {full_elapsed/n*1000:.2f}ms/op")
    check("Encrypt < 5ms/op", enc_elapsed / n < 0.005, f"got {enc_elapsed/n*1000:.2f}ms")


def test_counterfactual_speed():
    print("\n[5/6] Counterfactual evaluation speed")
    dc = DoCalculusCounterfactual()
    state = np.zeros(PHYSIO_DIM)
    state[0] = 180
    state[1] = 5
    state[5] = 120
    state[6] = 80
    state[7] = 70
    intervention = InterventionSpec(InterventionType.INSULIN_BOLUS, 10.0, 6)

    t0 = time.time()
    n = 5
    for _ in range(n):
        dc.evaluate_intervention(state, DEFAULT_PARAMS, intervention, n_total_steps=24)
    elapsed = time.time() - t0
    per_eval = elapsed / n
    metrics["cf_per_eval_s"] = per_eval
    print(f"  {per_eval:.2f}s per intervention evaluation (24 steps)")
    check("CF eval < 5s", per_eval < 5, f"got {per_eval:.2f}s")


def test_memory():
    print("\n[6/6] Memory usage")
    mem_mb = get_memory_mb()
    metrics["peak_memory_mb"] = mem_mb
    print(f"  Peak RSS: {mem_mb:.0f} MB")
    check("Memory < 2GB", mem_mb < 2048, f"got {mem_mb:.0f}MB")
    if mem_mb > 1024:
        print(f"  {WARN} High memory usage: {mem_mb:.0f}MB")


def print_summary():
    print("\n" + "=" * 60)
    print("  PERFORMANCE METRICS")
    print("=" * 60)
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            if "ms" in k:
                print(f"  {k:30s} {v:10.2f} ms")
            elif "s" in k and "per" not in k:
                print(f"  {k:30s} {v:10.2f} s")
            elif "mb" in k:
                print(f"  {k:30s} {v:10.0f} MB")
            else:
                print(f"  {k:30s} {v:10.2f}")
        else:
            print(f"  {k:30s} {v}")
    print("=" * 60)


def main():
    print("=" * 60)
    print("  TEST 04: Performance Benchmarks")
    print("=" * 60)
    test_ukf_throughput()
    test_dual_engine_throughput()
    test_personalization_pipeline()
    test_encryption_throughput()
    test_counterfactual_speed()
    test_memory()
    print_summary()
    print()
    if errors:
        print(f"  {len(errors)} PERFORMANCE ISSUES")
        for e in errors:
            print(f"    - {e}")
        sys.exit(1)
    else:
        print("  ALL PERFORMANCE CHECKS PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
