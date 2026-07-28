#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The real-world-style QM9 CMOP is not one of Tables 1-5. Table ablations are
# already included in their corresponding paper-table suites.
bash "$SCRIPT_DIR/run_suite.sh" \
  --suite qm9_cmop \
  --skip-completed \
  --plot \
  --summarize \
  "$@"
