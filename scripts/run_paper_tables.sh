#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPUS="0"
RESULTS_ROOT="results"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --results-root)
      RESULTS_ROOT="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

# Tables 1 and 2 share qm9_single. Tables 3, 4, and 5 each have one suite.
for SUITE in qm9_single qm9_mop docking_mop qm9_struct_cmop; do
  bash "$SCRIPT_DIR/run_suite.sh" \
    --suite "$SUITE" \
    --gpus "$GPUS" \
    --results-root "$RESULTS_ROOT" \
    --skip-completed \
    --plot \
    --summarize \
    "${EXTRA_ARGS[@]}"
done
