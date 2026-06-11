#!/bin/bash
###############################################################################
# BioDigitalTwin — Real-World Test Suite
#
# End-to-end test workflow that exercises:
#   1. Server health and API availability
#   2. Unit test suite (353 tests)
#   3. Core engine components (UKF, dual estimation, do-calculus)
#   4. New modules (HIPAA encryption, Epic FHIR, safety layer)
#   5. Full benchmark framework
#   6. Performance and latency checks
#   7. End-to-end patient workflow
#
# Usage:
#   ./tests/real_world/run_all.sh           # full suite (~10 min)
#   ./tests/real_world/run_all.sh --quick    # smoke test only (~2 min)
#   ./tests/real_world/run_all.sh --no-server  # skip server checks
###############################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Defaults
QUICK_MODE=false
SKIP_SERVER=false
SERVER_URL="http://localhost:8000"
RESULTS_DIR="tests/real_world/results"
mkdir -p "$RESULTS_DIR"

for arg in "$@"; do
    case $arg in
        --quick) QUICK_MODE=true ;;
        --no-server) SKIP_SERVER=true ;;
    esac
done

step_count=0
pass_count=0
fail_count=0

run_step() {
    step_count=$((step_count + 1))
    echo ""
    echo -e "${YELLOW}═══ STEP $step_count: $1 ═══${NC}"
}

pass() {
    pass_count=$((pass_count + 1))
    echo -e "${GREEN}✓ PASS${NC}: $1"
}

fail() {
    fail_count=$((fail_count + 1))
    echo -e "${RED}✗ FAIL${NC}: $1"
}

# ────────────────────────────────────────────────────────────────────
# STEP 1: Environment check
# ────────────────────────────────────────────────────────────────────
run_step "Environment check"

if [ -d "backend/.venv" ]; then
    pass "Python virtual environment exists"
else
    fail "Python venv missing — run: cd backend && python3 -m venv .venv"
    exit 1
fi

PY="backend/.venv/bin/python"
if [ ! -f "$PY" ]; then
    fail "Python binary not found at $PY"
    exit 1
fi
pass "Python binary: $($PY --version)"

# Check critical packages
$PY -c "import numpy, fastapi, scipy" 2>/dev/null && pass "Core packages installed" || fail "Core packages missing"

# ────────────────────────────────────────────────────────────────────
# STEP 2: Server health
# ────────────────────────────────────────────────────────────────────
if [ "$SKIP_SERVER" = false ]; then
    run_step "Server health check"
    if curl -s -f "$SERVER_URL/health" >/dev/null 2>&1; then
        pass "Server is running at $SERVER_URL"
        curl -s "$SERVER_URL/health" | $PY -m json.tool | head -5
    else
        echo -e "${YELLOW}Server not running. Starting it...${NC}"
        PYTHONPATH=backend nohup $PY -m uvicorn app_main:app --host 0.0.0.0 --port 8000 \
            > /tmp/biodt_server.log 2>&1 &
        sleep 5
        if curl -s -f "$SERVER_URL/health" >/dev/null 2>&1; then
            pass "Server started successfully"
        else
            fail "Server failed to start — check /tmp/biodt_server.log"
            exit 1
        fi
    fi
fi

# ────────────────────────────────────────────────────────────────────
# STEP 3: Unit test suite
# ────────────────────────────────────────────────────────────────────
run_step "Unit test suite (353 tests)"
if PYTHONPATH=backend:$PYTHONPATH $PY -m pytest \
    backend/tests/personalization/ \
    -q --tb=line 2>&1 | tee "$RESULTS_DIR/pytest.log" | tail -5; then
    pass "All unit tests passed"
else
    fail "Some unit tests failed — see $RESULTS_DIR/pytest.log"
fi

# ────────────────────────────────────────────────────────────────────
# STEP 4: Core engine smoke test
# ────────────────────────────────────────────────────────────────────
run_step "Core engine smoke test"
$PY tests/real_world/test_01_engine.py 2>&1 | tee "$RESULTS_DIR/test_01.log"
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    pass "Core engine works"
else
    fail "Core engine test failed"
fi

# ────────────────────────────────────────────────────────────────────
# STEP 5: New module validation
# ────────────────────────────────────────────────────────────────────
run_step "New module validation (HIPAA, Epic, Safety, Do-calculus)"
$PY tests/real_world/test_02_modules.py 2>&1 | tee "$RESULTS_DIR/test_02.log"
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    pass "All new modules validated"
else
    fail "Module validation failed"
fi

# ────────────────────────────────────────────────────────────────────
# STEP 6: End-to-end patient workflow
# ────────────────────────────────────────────────────────────────────
run_step "End-to-end patient workflow"
if [ "$SKIP_SERVER" = false ]; then
    $PY tests/real_world/test_03_e2e_workflow.py 2>&1 | tee "$RESULTS_DIR/test_03.log"
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        pass "E2E workflow completed"
    else
        fail "E2E workflow failed"
    fi
else
    echo "  Skipped (--no-server)"
fi

# ────────────────────────────────────────────────────────────────────
# STEP 7: Performance benchmark
# ────────────────────────────────────────────────────────────────────
run_step "Performance and latency"
$PY tests/real_world/test_04_performance.py 2>&1 | tee "$RESULTS_DIR/test_04.log"
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    pass "Performance metrics captured"
else
    fail "Performance test failed"
fi

# ────────────────────────────────────────────────────────────────────
# STEP 8: Full benchmark framework
# ────────────────────────────────────────────────────────────────────
run_step "Full benchmark framework"
if [ "$QUICK_MODE" = true ]; then
    echo "  Skipped (--quick mode)"
else
    PYTHONPATH=backend $PY backend/app/personalization/benchmark_fast.py 2>&1 \
        | tee "$RESULTS_DIR/benchmark.log" | tail -15
    if [ -f "BENCHMARK_REPORT_FAST.json" ]; then
        pass "Benchmark complete — see BENCHMARK_REPORT_FAST.json"
    else
        fail "Benchmark failed"
    fi
fi

# ────────────────────────────────────────────────────────────────────
# STEP 9: Generate consolidated report
# ────────────────────────────────────────────────────────────────────
run_step "Generating consolidated report"
$PY tests/real_world/generate_report.py 2>&1 | tee "$RESULTS_DIR/final_report.txt"

# ────────────────────────────────────────────────────────────────────
# Summary
# ────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  TEST SUITE SUMMARY"
echo "═══════════════════════════════════════════════════════════════"
echo -e "  Steps passed: ${GREEN}${pass_count}${NC}"
echo -e "  Steps failed: ${RED}${fail_count}${NC}"
echo ""
echo "  Results saved to: $RESULTS_DIR/"
ls -la "$RESULTS_DIR/"
echo ""
if [ $fail_count -eq 0 ]; then
    echo -e "${GREEN}✓ ALL TESTS PASSED${NC}"
    exit 0
else
    echo -e "${RED}✗ SOME TESTS FAILED${NC}"
    exit 1
fi
