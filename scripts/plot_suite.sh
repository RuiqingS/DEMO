#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/_activate_env.sh"
cd "$REPO_DIR"
python -m demo_runtime.plotting "$@"
