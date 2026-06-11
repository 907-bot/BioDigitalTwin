"""
Benchmark report generator.

Runs the full Digital Twin Benchmark Framework and produces a structured
report file. Used to:
  1. Track progress over time
  2. Provide publication-grade evidence
  3. Document known weaknesses for FDA submission
"""

import json
import time
from dataclasses import asdict
from pathlib import Path

from app.personalization.benchmark_framework import (
    run_all_benchmarks,
    BenchmarkResult,
    DimensionResult,
)


def result_to_dict(obj):
    if isinstance(obj, (BenchmarkResult, DimensionResult)):
        d = {
            "name": obj.name,
            "score": obj.score,
        }
        if isinstance(obj, BenchmarkResult):
            d.update({
                "passed": obj.passed,
                "gold": obj.gold,
                "failure": obj.failure,
                "details": obj.details,
                "failure_reasons": obj.failure_reasons,
            })
        elif isinstance(obj, DimensionResult):
            d["sub_results"] = [result_to_dict(s) for s in obj.sub_results]
            d["passed"] = obj.passed
            d["gold"] = obj.gold
            d["catastrophic"] = obj.catastrophic
        return d
    return str(obj)


def generate_report(n_patients: int = 5, output_path: str = "BENCHMARK_REPORT.json"):
    print("=" * 70)
    print("DIGITAL TWIN BENCHMARK FRAMEWORK")
    print("=" * 70)
    print()
    start = time.time()
    overall, dimensions = run_all_benchmarks(n_patients=n_patients, verbose=False)
    elapsed = time.time() - start

    # Build structured report
    passing = all(d.passed for d in dimensions)
    gold = all(d.gold for d in dimensions)
    catastrophic = any(d.catastrophic for d in dimensions)

    if catastrophic:
        verdict = "FAIL (catastrophic failure triggered)"
    elif gold:
        verdict = "GOLD (publication-grade)"
    elif passing:
        verdict = "PASS (pre-publication threshold met)"
    else:
        verdict = "FAIL (below threshold)"

    report = {
        "framework_version": "1.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_patients": n_patients,
        "elapsed_seconds": elapsed,
        "overall_score": overall,
        "verdict": verdict,
        "passing": passing,
        "gold_standard": gold,
        "catastrophic_failure": catastrophic,
        "dimensions": [result_to_dict(d) for d in dimensions],
    }

    # Per-dimension summary
    summary = []
    for d in dimensions:
        summary.append({
            "name": d.name,
            "score": d.score,
            "passed": d.passed,
            "gold": d.gold,
            "catastrophic": d.catastrophic,
            "n_sub_benchmarks": len(d.sub_results),
            "sub_passed": sum(1 for s in d.sub_results if s.passed),
        })
    report["dimension_summary"] = summary

    # Save
    Path(output_path).write_text(json.dumps(report, indent=2, default=str))
    print(f"Report saved to {output_path}")
    print()

    # Print human-readable summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for s in summary:
        status = "GOLD" if s["gold"] else "PASS" if s["passed"] else "FAIL"
        if s["catastrophic"]:
            status = "CATASTROPHIC"
        print(f"  {status:13s} {s['name']:30s}: {s['score']:.3f}  "
              f"({s['sub_passed']}/{s['n_sub_benchmarks']} sub-tests pass)")
    print()
    print(f"OVERALL: {overall:.3f}  —  {verdict}")
    print(f"Elapsed: {elapsed:.1f}s")
    print("=" * 70)

    return report


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    output = sys.argv[2] if len(sys.argv) > 2 else "BENCHMARK_REPORT.json"
    generate_report(n_patients=n, output_path=output)
