#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

REPO=${REPO:-YeanRoot/HALO-WA}
TAG=${TAG:-artifacts-v1}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-$PROJECT_ROOT/data/artifacts}
DOWNLOAD_DIR=${DOWNLOAD_DIR:-$ARTIFACT_ROOT/.downloads/$TAG}
RELEASE_API="https://api.github.com/repos/$REPO/releases/tags/$TAG"

mkdir -p "$DOWNLOAD_DIR" "$ARTIFACT_ROOT" "$PROJECT_ROOT/data"

echo "Fetching release metadata for $REPO@$TAG"
python3 - "$RELEASE_API" "$DOWNLOAD_DIR/assets.tsv" <<'PY'
import json
import sys
import urllib.request

api_url, output = sys.argv[1], sys.argv[2]
req = urllib.request.Request(api_url, headers={"Accept": "application/vnd.github+json"})
with urllib.request.urlopen(req) as resp:
    release = json.load(resp)

assets = []
for asset in release.get("assets", []):
    name = asset["name"]
    if name == "SHA256SUMS" or name.startswith("halo-wa-artifacts-v1.tar.part-"):
        assets.append((name, asset["browser_download_url"]))

if not assets:
    raise SystemExit(f"No HALO-WA artifact assets found at {api_url}")

assets.sort()
with open(output, "w", encoding="utf-8") as f:
    for name, url in assets:
        f.write(f"{name}\t{url}\n")
PY

while IFS=$'\t' read -r name url; do
  out="$DOWNLOAD_DIR/$name"
  if [ -s "$out" ]; then
    echo "Already downloaded $name"
  else
    echo "Downloading $name"
    curl -L --fail --retry 5 --retry-delay 5 -o "$out" "$url"
  fi
done < "$DOWNLOAD_DIR/assets.tsv"

echo "Verifying checksums"
(cd "$DOWNLOAD_DIR" && sha256sum -c SHA256SUMS)

echo "Extracting artifacts into $ARTIFACT_ROOT"
cat "$DOWNLOAD_DIR"/halo-wa-artifacts-v1.tar.part-* \
  | tar -C "$ARTIFACT_ROOT" --strip-components=1 -xf -

cat <<EOF

Artifacts are ready under:
  $ARTIFACT_ROOT

Use these environment variables before running the GigaWA RoboTwin configs from
the reorganized HALO-WA repository:

export PROJECT_ROOT=$PROJECT_ROOT
export RLINF_ROOT=\$PROJECT_ROOT/rlinf
export REPO_PATH=\$RLINF_ROOT
export ROBOTWIN_PATH=$ARTIFACT_ROOT/RoboTwin-RLinf_support
export PYTHONPATH=\$ROBOTWIN_PATH:\$REPO_PATH:\$PYTHONPATH
export GIGA_WA_TRANSFORMER_PATH=$ARTIFACT_ROOT/checkpoints/checkpoint_epoch_2_step_100000/transformer
export GIGA_WA_ROOT=$ARTIFACT_ROOT/wan-casual-cj
export WAN_BASE_MODEL_DIR=$ARTIFACT_ROOT/models/Wan2.2-TI2V-5B-Diffusers
export GIGA_NORM_JSON=$ARTIFACT_ROOT/norm_stats_delta.json
export RLINF_OUTPUT_ROOT=\$PROJECT_ROOT/outputs
export HAMMER_DEMO_BUFFER_PATH=$ARTIFACT_ROOT/demo_buffers/hammer/mergeall_original
export BELL_DEMO_BUFFER_PATH=$ARTIFACT_ROOT/demo_buffers/bell/mergeall_all_rank1
EOF
