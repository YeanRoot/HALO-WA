#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RLINF_ROOT="${RLINF_ROOT:-$PROJECT_ROOT/rlinf}"

if [[ "${USE_RLINF_VENV:-1}" != "0" && -f "$RLINF_ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$RLINF_ROOT/.venv/bin/activate"
elif [[ "${SKIP_CONDA:-0}" != "1" ]]; then
  if [[ -n "${CONDA_SH:-}" ]]; then
    # shellcheck disable=SC1090
    source "$CONDA_SH"
    conda activate "${CONDA_ENV:-halo-wa}"
  elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-halo-wa}"
  else
    echo "No conda activation found; using the current Python environment." >&2
  fi
fi

# shellcheck disable=SC1091
source "$SCRIPT_DIR/setup_env.sh"

cd "$RLINF_ROOT/examples/embodiment"
exec python train_embodied_agent_gigawa.py \
  --config-path ./config \
  --config-name online_rl_bell_ansyc \
  "$@"
