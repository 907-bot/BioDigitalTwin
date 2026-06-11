#!/bin/bash
###############################################################################
# Quick smoke test — runs in under 2 minutes.
# Skips the full benchmark.
###############################################################################
exec "$(dirname "$0")/run_all.sh" --quick "$@"
