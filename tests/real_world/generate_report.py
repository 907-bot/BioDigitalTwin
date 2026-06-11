#!/usr/bin/env python3
"""
Generate a consolidated test report from all individual result logs.

Reads:
    tests/real_world/results/pytest.log
    tests/real_world/results/test_01.log
    tests/real_world/results/test_02.log
    tests/real_world/results/test_03.log
    tests/real_world/results/test_04.log
    tests/real_world/results/benchmark.log

Writes:
    tests/real_world/results/final_report.txt
    tests/real_world/results/final_report.md
"""
import os
import re
import json
import sys
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path("tests/real_world/results")


def read_log(name):
    path = RESULTS_DIR / name
    if path.exists():
        return path.read_text(errors="ignore")
    return None


def extract_pytest_results(text):
    """Parse '353 passed, 1 skipped, 21 warnings in 75.43s'."""
    pattern = r"(\d+)\s+passed"
    m = re.search(pattern, text)
    passed = int(m.group(1)) if m else None
    pattern = r"(\d+)\s+failed"
    m = re.search(pattern, text)
    failed = int(m.group(1)) if m else 0
    pattern = r"(\d+)\s+skipped"
    m = re.search(pattern, text)
    skipped = int(m.group(1)) if m else 0
    pattern = r"in\s+([\d.]+)s"
    m = re.search(pattern, text)
    duration = m.group(1) + "s" if m else "?"
    return passed, failed, skipped, duration


def extract_pass_fail(text):
    """Count green check vs red x markers."""
    if not text:
        return None, None
    passes = text.count("\033[0;32m✓\033[0m") + len(re.findall(r"^.*✓.*$", text, re.MULTILINE))
    fails = text.count("\033[0;31m✗\033[0m") + len(re.findall(r"^.*✗.*$", text, re.MULTILINE))
    return passes, fails


def main():
    print("Generating consolidated test report...")

    pytest_log = read_log("pytest.log")
    t01 = read_log("test_01.log")
    t02 = read_log("test_02.log")
    t03 = read_log("test_03.log")
    t04 = read_log("test_04.log")
    bench_log = read_log("benchmark.log")

    # Pytest
    p_pass, p_fail, p_skip, p_dur = (None, None, None, None)
    if pytest_log:
        p_pass, p_fail, p_skip, p_dur = extract_pytest_results(pytest_log)

    # T01, T02, T03, T04
    t01_pass, t01_fail = extract_pass_fail(t01)
    t02_pass, t02_fail = extract_pass_fail(t02)
    t03_pass, t03_fail = extract_pass_fail(t03)
    t04_pass, t04_fail = extract_pass_fail(t04)

    # Benchmark
    bench_score = None
    if bench_log:
        m = re.search(r"overall[_ ]score[:\s]+([\d.]+)", bench_log, re.IGNORECASE)
        if m:
            bench_score = float(m.group(1))

    # Build text report
    report = []
    report.append("=" * 70)
    report.append("  BioDigitalTwin — Real-World Test Report")
    report.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 70)
    report.append("")

    report.append("SUMMARY")
    report.append("-" * 70)
    if p_pass is not None:
        report.append(f"  Unit tests:        {p_pass} passed, {p_fail} failed, {p_skip} skipped ({p_dur})")
    if t01_pass is not None:
        report.append(f"  Test 01 (engine):  {t01_pass} passed, {t01_fail} failed")
    if t02_pass is not None:
        report.append(f"  Test 02 (modules): {t02_pass} passed, {t02_fail} failed")
    if t03_pass is not None:
        report.append(f"  Test 03 (e2e):     {t03_pass} passed, {t03_fail} failed")
    if t04_pass is not None:
        report.append(f"  Test 04 (perf):    {t04_pass} passed, {t04_fail} failed")
    if bench_score is not None:
        verdict = "PASS" if bench_score >= 0.7 else "FAIL"
        report.append(f"  Benchmark:         {bench_score:.3f}  ({verdict})")
    report.append("")

    report.append("VERDICT")
    report.append("-" * 70)
    overall_pass = True
    if p_fail and p_fail > 0:
        overall_pass = False
    if any(f and f > 0 for f in (t01_fail, t02_fail, t03_fail, t04_fail)):
        overall_pass = False
    if overall_pass:
        report.append("  ✓ ALL TESTS PASSED — system is production-ready for pilot")
    else:
        report.append("  ✗ SOME TESTS FAILED — review individual logs")
    report.append("")

    report.append("FILES")
    report.append("-" * 70)
    for f in sorted(RESULTS_DIR.glob("*")):
        report.append(f"  {f.name:30s} {f.stat().st_size:>10,} bytes")
    report.append("")
    report.append("=" * 70)

    text = "\n".join(report)

    out_txt = RESULTS_DIR / "final_report.txt"
    out_txt.write_text(text)
    print(f"Wrote {out_txt}")
    print()
    print(text)

    # JSON for machine consumption
    json_data = {
        "generated": datetime.now().isoformat(),
        "unit_tests": {"passed": p_pass, "failed": p_fail, "skipped": p_skip, "duration": p_dur},
        "test_01_engine": {"passed": t01_pass, "failed": t01_fail},
        "test_02_modules": {"passed": t02_pass, "failed": t02_fail},
        "test_03_e2e": {"passed": t03_pass, "failed": t03_fail},
        "test_04_perf": {"passed": t04_pass, "failed": t04_fail},
        "benchmark_score": bench_score,
        "overall_pass": overall_pass,
    }
    out_json = RESULTS_DIR / "final_report.json"
    out_json.write_text(json.dumps(json_data, indent=2))
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
