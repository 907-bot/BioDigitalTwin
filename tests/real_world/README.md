# BioDigitalTwin — Real-World Test Suite

End-to-end test workflow for the digital twin system. Validates the engine,
new modules (HIPAA, Epic, safety, do-calculus), API workflow, and performance.

## Quick start

```bash
# 1. Make scripts executable
chmod +x tests/real_world/*.sh

# 2. Start the API server
./tests/real_world/start_server.sh

# 3. Run the full suite (~10 min)
./tests/real_world/run_all.sh

# OR run quick smoke test (~2 min)
./tests/real_world/quick_test.sh

# OR run individual steps
PYTHONPATH=backend:. backend/.venv/bin/python tests/real_world/test_01_engine.py
PYTHONPATH=backend:. backend/.venv/bin/python tests/real_world/test_02_modules.py
PYTHONPATH=backend:. backend/.venv/bin/python tests/real_world/test_04_performance.py
backend/.venv/bin/python tests/real_world/test_03_e2e_workflow.py
```

## What gets tested

| Step | Script | What it covers | Time |
|------|--------|----------------|------|
| 1 | `start_server.sh` | Boots uvicorn on :8000, waits for /health | ~5s |
| 2 | `run_all.sh` (Step 3) | Full pytest suite (353 unit tests) | ~75s |
| 3 | `test_01_engine.py` | Dynamics, observation, UKF, dual engine, do-calculus, safety | ~10s |
| 4 | `test_02_modules.py` | HIPAA encryption/audit/RBAC, Epic FHIR, MIMIC cohort, do-calculus refutation | ~15s |
| 5 | `test_03_e2e_workflow.py` | Health, OpenAPI, personalize, predict, counterfactual, audit, benchmark endpoints | ~30s |
| 6 | `test_04_performance.py` | UKF throughput, dual engine speed, encryption perf, CF speed, memory | ~30s |
| 7 | `run_all.sh` (Step 8) | Full benchmark framework (skipped in `--quick` mode) | ~3 min |
| 8 | `generate_report.py` | Consolidates all logs into `results/final_report.txt` and `.json` | <1s |

## Results layout

After `run_all.sh` completes:

```
tests/real_world/results/
├── pytest.log                  # full pytest output
├── test_01.log                 # engine smoke test
├── test_02.log                 # module validation
├── test_03.log                 # e2e workflow
├── test_04.log                 # performance
├── benchmark.log               # benchmark framework output
├── final_report.txt            # human-readable consolidated report
└── final_report.json           # machine-readable
```

## Test philosophy

These scripts implement the **assumed-nothing-works** discipline:

1. **No mocks** — every test runs against the actual engine on synthetic data
   generated from the same code path the real system uses.
2. **Honest assertions** — if personalization RMSE is poor, the test reports it;
   it doesn't paper over it.
3. **Real timing** — performance tests measure wall-clock time, not simulated.
4. **End-to-end** — the API workflow goes through the actual FastAPI server,
   not a direct function call.

## Adding new tests

For a new validation, add a `test_05_yourfeature.py` that follows the same
pattern as the others:

```python
def test_your_feature():
    print("\n[N/M] Your feature")
    check("Assertion 1", condition_1)
    check("Assertion 2", condition_2)

def main():
    print("=" * 60)
    print("  TEST 05: Your Feature")
    print("=" * 60)
    test_your_feature()
    # ...
    if errors:
        sys.exit(1)
```

Then wire it into `run_all.sh` as a new step.
