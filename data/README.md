# Data Artifacts

The large HALO-WA assets are not stored in git. They are published as release
assets here:

https://github.com/YeanRoot/HALO-WA/releases/tag/artifacts-v1

Use the downloader from the repository root:

```bash
bash scripts/download_artifacts.sh
```

The default extraction directory is:

```text
data/artifacts/
```

Expected contents after extraction:

```text
data/artifacts/
|-- RoboTwin-RLinf_support/
|-- checkpoints/checkpoint_epoch_2_step_100000/transformer/
|-- demo_buffers/hammer/mergeall_original/
|-- demo_buffers/bell/mergeall_all_rank1/
|-- models/Wan2.2-TI2V-5B-Diffusers/
|-- norm_stats_delta.json
`-- wan-casual-cj/
```

The release is split into 35 tar parts plus `SHA256SUMS`, totaling about 69 GB.
The downloader verifies checksums before extracting.

To place data elsewhere:

```bash
ARTIFACT_ROOT=/path/to/halo-wa-artifacts bash scripts/download_artifacts.sh
```

The run scripts read `ARTIFACT_ROOT` too, so use the same variable when launching
training if you do not use the default `data/artifacts/` path.
