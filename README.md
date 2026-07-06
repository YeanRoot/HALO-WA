# HALO-WA

HALO-WA packages the GigaWA RoboTwin online RL experiments as a small project
wrapper around the RLinf-based training code. The top-level repository is kept
focused on setup, data download, and the two supported task entrypoints.

## Repository Layout

```text
HALO-WA/
|-- rlinf/                 # RLinf-based training code and configs
|-- data/                  # Artifact links and optional local downloads
|-- scripts/               # Download and task launch scripts
|-- environment.yml        # Public conda environment exported from pi-rl
|-- environment.pi-rl-export.yml  # Reference export from the dev server
`-- README.md
```

The training code lives under `rlinf/`. Most users should only need the
root-level scripts and the configs documented below.

## System Requirements

The experiments were smoke-tested on Linux with NVIDIA A800 GPUs, CUDA, and the
`pi-rl` environment used for development. A fresh machine should have:

- Linux with an NVIDIA driver and CUDA toolkit available on `PATH`.
- `nvcc`, required by the RoboTwin/SAPIEN dependency build.
- Vulkan runtime libraries visible to SAPIEN.
- Python 3.11.

On Ubuntu-like systems, install the common system packages first:

```bash
sudo apt-get update
sudo apt-get install -y \
  git git-lfs curl wget build-essential cmake ninja-build pkg-config \
  libvulkan1 vulkan-tools mesa-vulkan-drivers mesa-utils
```

If SAPIEN reports Vulkan ICD warnings, make sure the NVIDIA ICD file is present,
for example `/etc/vulkan/icd.d/nvidia_icd.json`, and that your driver install is
complete.

## Environment Setup

The repository includes a sanitized export of the development conda environment
used for smoke tests. `environment.yml` is the public install file; the reference
`environment.pi-rl-export.yml` records the original `pi-rl` export after removing
its machine-specific prefix. Editable packages that lived under private paths on
the development server are installed from public source by the RLinf installer.

If you already have an equivalent conda environment such as the development
`pi-rl` environment, you can activate it directly with your local conda path:

```bash
source /path/to/conda/etc/profile.d/conda.sh
conda activate pi-rl
```

For a fresh setup, first create the public conda environment:

```bash
conda env create -f environment.yml
conda activate halo-wa
```

Then let the RLinf installer create `rlinf/.venv` and install the source-built
robotics dependencies used by these experiments:

```bash
cd rlinf

# Wan/GigaWA world-model stack used by the online RL configs.
bash requirements/install.sh embodied --model openvla-oft --env wan --install-rlinf --no-root

# RoboTwin/OpenPI/SAPIEN stack used by the simulator side.
bash requirements/install.sh embodied --model openpi --env robotwin --install-rlinf --no-root

source .venv/bin/activate
cd ..
```

Use `--use-mirror` on the two installer commands if GitHub/PyPI access is slow
from your machine. The RoboTwin step builds CUDA extensions, so `nvcc` must be
available. Drop `--no-root` if you want the installer to attempt system
dependency installation for you.

A quick sanity check after installation:

```bash
python - <<'CHECK'
import diffusers, hydra, ray, sapien, torch
print('torch', torch.__version__)
print('cuda available', torch.cuda.is_available())
print('ray', ray.__version__)
print('sapien', sapien.__version__)
CHECK
```

## Download Data And Checkpoints

Large assets are stored in the GitHub Release
[`artifacts-v1`](https://github.com/YeanRoot/HALO-WA/releases/tag/artifacts-v1)
instead of the git repository. The release contains the Wan base model, GigaWA
checkpoint, RoboTwin support tree, normalization stats, and the hammer/bell demo
buffers.

Download and extract them with:

```bash
bash scripts/download_artifacts.sh
```

By default this extracts to `data/artifacts/`. To use another location:

```bash
ARTIFACT_ROOT=/path/to/halo-wa-artifacts bash scripts/download_artifacts.sh
```

The downloader verifies `SHA256SUMS` before extraction.

## Run The Tasks

The launch scripts set the expected `REPO_PATH`, `ROBOTWIN_PATH`, `PYTHONPATH`,
GigaWA checkpoint paths, and demo buffer paths for the new repository layout.

Hammer:

```bash
bash scripts/run_hammer.sh
```

Bell:

```bash
bash scripts/run_bell.sh
```

Both scripts forward additional Hydra overrides to
`train_embodied_agent_gigawa.py`. For example:

```bash
bash scripts/run_bell.sh runner.resume_dir=/path/to/checkpoints/global_step_220
```

If you want to use a specific conda environment instead of `rlinf/.venv`,
set `USE_RLINF_VENV=0` and provide the conda information:

```bash
CONDA_SH=/path/to/conda/etc/profile.d/conda.sh \
CONDA_ENV=pi-rl \
USE_RLINF_VENV=0 \
bash scripts/run_hammer.sh
```

To skip environment activation entirely and use the current shell:

```bash
SKIP_CONDA=1 USE_RLINF_VENV=0 bash scripts/run_hammer.sh
```

## Direct Commands

After sourcing `scripts/setup_env.sh`, the original training commands still work
from the embodiment example directory:

```bash
source scripts/setup_env.sh
cd rlinf/examples/embodiment

python train_embodied_agent_gigawa.py \
  --config-path ./config \
  --config-name online_rl_hammer_ansyc

python train_embodied_agent_gigawa.py \
  --config-path ./config \
  --config-name online_rl_bell_ansyc
```

## Validation

Before publishing this layout, both supported configs were smoke-tested with a
single training step on the development server:

- `online_rl_hammer_ansyc`: reached `Global Step: 1/1`.
- `online_rl_bell_ansyc`: reached `Global Step: 1/1`.
