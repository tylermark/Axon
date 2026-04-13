#!/usr/bin/env bash
# Autonomous training monitor — thin wrapper.
# Usage: ./scripts/monitor.sh once   (single cycle, JSON output)
#        ./scripts/monitor.sh watch  (blocking daemon)
set -euo pipefail
python -m src.monitor "$@"
