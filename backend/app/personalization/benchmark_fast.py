"""
Lightweight benchmark runner — fast mode.

Runs the full Digital Twin Benchmark Framework with reduced sample sizes
for quick iteration during development. For full publication-quality results,
use run_all_benchmarks(n_patients=10+).
"""

import json
import time
from pathlib import Path

from app.personalization.benchmark_framework import run_all_benchmarks


def main():
    n_patients = 2
    print(f"Running benchmark with n_patients={n_patients} (fast mode)...")
    start = time.time()
    overall, dims = run_all_benchmarks(n_patients=n_patients, verbose=False)
    elapsed = time.time() - start

    report = {
        "framework_version": "1.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_patients": n_patients,
        "elapsed_seconds": elapsed,
        "overall_score": overall,
        "passing": all(d.passed for d in dims),
        "catastrophic": any(d.catastrophic for d in dims),
        "dimensions": [
            {
                "name": d.name,
                "score": d.score,
                "passed": d.passed,
                "gold": d.gold,
                "catastrophic": d.catastrophic,
                "sub_benchmarks": [
                    {
                        "name": s.name,
                        "score": s.score,
                        "passed": s.passed,
                        "failure": s.failure,
                        "failure_reasons": s.failure_reasons,
                    } for s in d.sub_results
                ],
            }
            for d in dims
        ],
    }

    out = "BENCHMARK_REPORT_FAST.json"
    Path(out).write_text(json.dumps(report, indent=2, default=str))
    print(f"Saved to {out}")
    print(f"\nOverall: {overall:.3f}  ({elapsed:.1f}s)")
    for d in dims:
        status = "GOLD" if d.gold else "PASS" if d.passed else "FAIL"
        if d.catastrophic:
            status = "CATASTROPHIC"
        print(f"  {status:13s} {d.name:30s}: {d.score:.3f}")


if __name__ == "__main__":
    main()
