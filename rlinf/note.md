# GigaWA RoboTwin Quickstart

This note documents the compact open-source RLinf branch for the GigaWA
RoboTwin online RL examples.

Legacy private experiment configs and generated result artifacts are intentionally
excluded from the compact release. The supported GigaWA RoboTwin entrypoints are
the hammer and bell configs shown below.

## Environment

```bash
source /path/to/conda/etc/profile.d/conda.sh
conda activate pi-rl

export REPO_PATH=/path/to/RLinf
export ROBOTWIN_PATH=/path/to/RoboTwin-RLinf_support
export PYTHONPATH=$ROBOTWIN_PATH:$REPO_PATH:$PYTHONPATH

cd $REPO_PATH/examples/embodiment
```

These configs have been tested with the `pi-rl` conda environment. If the
simulator reports Vulkan ICD warnings, install the Vulkan runtime/driver
packages for the machine and make sure the NVIDIA ICD JSON is visible to Vulkan.

## Required Artifacts

The easiest way to fetch the pretrained checkpoints, RoboTwin support tree, and
demo replay buffers is:

```bash
cd $REPO_PATH
bash scripts/download_artifacts.sh
```

By default this downloads the `artifacts-v1` GitHub Release assets and extracts
them into `$REPO_PATH/artifacts`. You can choose another location with:

```bash
ARTIFACT_ROOT=/path/to/halo-wa-artifacts bash scripts/download_artifacts.sh
```

Then set these paths before launching training:

```bash
export GIGA_WA_TRANSFORMER_PATH=$REPO_PATH/artifacts/checkpoints/checkpoint_epoch_2_step_100000/transformer
export GIGA_WA_ROOT=$REPO_PATH/artifacts/wan-casual-cj
export WAN_BASE_MODEL_DIR=$REPO_PATH/artifacts/models/Wan2.2-TI2V-5B-Diffusers
export GIGA_NORM_JSON=$REPO_PATH/artifacts/norm_stats_delta.json
export RLINF_OUTPUT_ROOT=$REPO_PATH/examples/results
```

The online RL configs also use demonstration replay buffers:

```bash
export HAMMER_DEMO_BUFFER_PATH=$REPO_PATH/artifacts/demo_buffers/hammer/mergeall_original
export BELL_DEMO_BUFFER_PATH=$REPO_PATH/artifacts/demo_buffers/bell/mergeall_all_rank1
```

For bell experiments that should continue from a checkpoint, pass the checkpoint
on the command line:

```bash
python train_embodied_agent_gigawa.py \
  --config-path ./config \
  --config-name online_rl_bell_ansyc \
  runner.resume_dir=/path/to/checkpoints/global_step_220
```

## Training

Hammer:

```bash
python train_embodied_agent_gigawa.py \
  --config-path ./config \
  --config-name online_rl_hammer_ansyc
```

Bell:

```bash
python train_embodied_agent_gigawa.py \
  --config-path ./config \
  --config-name online_rl_bell_ansyc
```

## Config Check

Use Hydra's config dump to verify environment variables before starting a full
training run:

```bash
python train_embodied_agent_gigawa.py \
  --config-path ./config \
  --config-name online_rl_hammer_ansyc \
  --cfg job

python train_embodied_agent_gigawa.py \
  --config-path ./config \
  --config-name online_rl_bell_ansyc \
  --cfg job
```
