# Docker submission and verification

## Status: complete and tested

The final system needs two images because the locally trained models and the
upstream R2-V2 release require different pinned PyTorch/CUDA environments.

| Image | Purpose | Docker Hub digest |
|---|---|---|
| `choiclara9/gave2-kirby:prelim-7.95236-e2e-main` | Local Task 1/2/3 models, post-processing, packaging | `sha256:c1f3eac78ee3842b476cd9e17d04ebae5d38691a1b2da55e7196b1ef9aac5f5a` |
| `choiclara9/gave2-kirby:prelim-7.95236-e2e-r2v2` | Pinned upstream R2-V2 CFP-only AV inference | `sha256:b652ce33b8179c47894573f303294886283cbb547923364de555cbcd41e298e9` |

Both images are code/environment only: no official data, checkpoints,
predictions, submission ZIPs, or credentials are embedded.

## Required private mounts

The organizer receives the local checkpoint/optic-disc package through a
restricted link. After extracting it and running `fetch_r2v2.sh`, the mounted
paths must contain:

```text
DATA_DIR/
  training/biomarker/g_001.txt ... g_050.txt
  validation/images/g_051.png ... g_100.png
  validation/masks/g_051.png ... g_100.png
  validation/FFA_A/g_051.png ... g_100.png
  validation/FFA_AV/g_051.png ... g_100.png
WEIGHTS_DIR/
  task1/fold_0.pth ... fold_4.pth
  task2/fold_0.pth ... fold_2.pth
  task3/task2_e_topology_fold0.pth
  task3/task2_e_baseline_fold1.pth
  task3/task2_e_baseline_fold2.pth
  external/optic_disc/{config.json,preprocessor_config.json,pytorch_model.bin}
  external/r2v2/...  # created by material_review/fetch_r2v2.sh
```

## One-command raw-input reproduction

On a Linux host with Docker, NVIDIA Container Toolkit, and an NVIDIA GPU:

```bash
bash material_review/fetch_r2v2.sh WEIGHTS_DIR
bash material_review/run_end_to_end_docker.sh DATA_DIR WEIGHTS_DIR OUTPUT_DIR
```

Expected final path: `OUTPUT_DIR/main/kirby.zip`.

The launcher runs R2-V2 from raw CFP/ROI first, then runs the local container
from raw CFP/ROI/FFA and all selected local checkpoints. It validates the
150-entry ZIP and fails unless the final SHA-256 is exactly:

```text
4c19a97503668ef003c992edcbf76ade644c40903b683394347c0214db51306c
```

The completed RTX 2080 test is recorded in
[`material_review/DOCKER_TEST_REPORT.md`](../material_review/DOCKER_TEST_REPORT.md):
it regenerated 50 R2-V2 maps byte-for-byte, then regenerated the final ZIP
byte-for-byte from raw inputs. The test also verified 352 endpoint bridges,
812 endpoint pixels, and 527,170 additive final vein-support pixels.
