#!/usr/bin/env bash

# Shared non-interactive Conda activation for WSL Ubuntu 18.04.
ENV_NAME="${DEMO_CONDA_ENV:-cgm310}"

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
elif [[ -n "${CONDA_EXE:-}" ]]; then
  CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
else
  CONDA_BASE=""
  for CANDIDATE in \
    "$HOME/miniconda3" \
    "$HOME/anaconda3" \
    "$HOME/miniforge3"; do
    if [[ -f "$CANDIDATE/etc/profile.d/conda.sh" ]]; then
      CONDA_BASE="$CANDIDATE"
      break
    fi
  done
fi

if [[ -z "${CONDA_BASE:-}" ]] || [[ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  echo "Conda initialization script was not found." >&2
  exit 2
fi

# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
