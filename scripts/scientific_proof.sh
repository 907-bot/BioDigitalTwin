#!/bin/bash
###############################################################################
# BioDigitalTwin — Milestone 1: Scientific Proof Runner
#
# Usage:
#   ./scripts/scientific_proof.sh                 # Full run (~20 min)
#   ./scripts/scientific_proof.sh --quick          # Quick smoke test (~3 min)
#
# Output:
#   scientific_proof/results.md          # Publishable markdown report
#   scientific_proof/results.json        # Machine-readable results
#   scientific_proof/forecasting_*.csv   # Per-patient data
#   scientific_proof/*_results.json      # Individual pillar results
###############################################################################

set -e
cd "$(dirname "$0")/.."

PY="backend/.venv/bin/python"
MODE="${1:---quick}"  # default to quick for demo

echo "============================================"
echo "  BioDigitalTwin — Scientific Proof"
echo "============================================"
echo ""

time PYTHONPATH=backend $PY scripts/scientific_proof.py $MODE

echo ""
echo "============================================"
echo "  Done. Open the report:"
echo "    open scientific_proof/results.md"
echo "============================================"
