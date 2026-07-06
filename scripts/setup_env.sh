#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export PROJECT_ROOT
export RLINF_ROOT="${RLINF_ROOT:-$PROJECT_ROOT/rlinf}"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-$PROJECT_ROOT/data/artifacts}"

export REPO_PATH="${REPO_PATH:-$RLINF_ROOT}"
export ROBOTWIN_PATH="${ROBOTWIN_PATH:-$ARTIFACT_ROOT/RoboTwin-RLinf_support}"
export PYTHONPATH="$ROBOTWIN_PATH:$REPO_PATH:${PYTHONPATH:-}"

export GIGA_WA_TRANSFORMER_PATH="${GIGA_WA_TRANSFORMER_PATH:-$ARTIFACT_ROOT/checkpoints/checkpoint_epoch_2_step_100000/transformer}"
export GIGA_WA_ROOT="${GIGA_WA_ROOT:-$ARTIFACT_ROOT/wan-casual-cj}"
export WAN_BASE_MODEL_DIR="${WAN_BASE_MODEL_DIR:-$ARTIFACT_ROOT/models/Wan2.2-TI2V-5B-Diffusers}"
export GIGA_NORM_JSON="${GIGA_NORM_JSON:-$ARTIFACT_ROOT/norm_stats_delta.json}"
export RLINF_OUTPUT_ROOT="${RLINF_OUTPUT_ROOT:-$PROJECT_ROOT/outputs}"

export HAMMER_DEMO_BUFFER_PATH="${HAMMER_DEMO_BUFFER_PATH:-$ARTIFACT_ROOT/demo_buffers/hammer/mergeall_original}"
export BELL_DEMO_BUFFER_PATH="${BELL_DEMO_BUFFER_PATH:-$ARTIFACT_ROOT/demo_buffers/bell/mergeall_all_rank1}"

if [[ "${HALO_WA_CHECK_PATHS:-1}" == "1" ]]; then
  missing=0
  for path in \
    "$REPO_PATH" \
    "$ROBOTWIN_PATH" \
    "$GIGA_WA_TRANSFORMER_PATH" \
    "$GIGA_WA_ROOT" \
    "$WAN_BASE_MODEL_DIR" \
    "$GIGA_NORM_JSON" \
    "$HAMMER_DEMO_BUFFER_PATH" \
    "$BELL_DEMO_BUFFER_PATH"; do
    if [[ ! -e "$path" ]]; then
      echo "Missing required path: $path" >&2
      missing=1
    fi
  done

  if [[ "$missing" != "0" ]]; then
    cat >&2 <<EOF

Download the required artifacts first:

  bash scripts/download_artifacts.sh

Or set ARTIFACT_ROOT to the directory where the artifacts are already extracted.
EOF
    return 2 2>/dev/null || exit 2
  fi
fi
