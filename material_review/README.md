# kirby GAVE2 preliminary material-review package

This directory is the reproducibility source for team **kirby**'s final
preliminary submission (rank 7, overall score 7.95236).

## Reproducibility levels

The package provides three independent checks. The primary path uses two
containers because the scored artifact used two pinned PyTorch environments:
kirby's local models used PyTorch 2.2/CUDA 12.1, while the upstream R2-V2
release specified PyTorch 2.8/CUDA 12.8. Keeping those environments separate
avoids numerical changes at threshold boundaries.

1. **Raw-input end-to-end inference (primary)**: `pipeline.py run-all` starts
   with the official CFP, ROI, early/late FFA, training biomarker labels and
   every selected checkpoint. It executes all Task-1/2/3 models, optic-disc
   inference, fixed post-processing, biomarker extraction/calibration, compact
   encoding, ZIP packaging and validation. The command must reproduce SHA-256
   `4c19a97503668ef003c992edcbf76ade644c40903b683394347c0214db51306c`.
2. **Exact scored-artifact reconstruction (fast audit)**: `pipeline.py
   reproduce` combines two hash-pinned intermediate artifacts with the fixed,
   additive Task-2 vein operation and produces the same SHA-256.
3. **Local model-component inference**: `pipeline.py infer-components` is a
   shorter diagnostic for the five local CFP-only Task-1 folds and three
   multimodal Task-2 folds.

The challenge data, pretrained weights, local checkpoints, and generated
outputs are not committed to the public repository. They are mounted at run
time and verified against `weights_manifest.json` before inference.

## Expected mounted files

```text
/data/
  training/biomarker/g_001.txt ... g_050.txt
  validation/images/g_051.png ... g_100.png
  validation/masks/g_051.png ... g_100.png
  validation/FFA_A/g_051.png ... g_100.png
  validation/FFA_AV/g_051.png ... g_100.png
/weights/
  task1/fold_0.pth ... fold_4.pth
  task2/fold_0.pth ... fold_2.pth
  task3/task2_e_topology_fold0.pth
  task3/task2_e_baseline_fold1.pth
  task3/task2_e_baseline_fold2.pth
  external/r2v2/av.pth
  external/r2v2/av_config.json
  external/r2v2/source/{infer,model,preprocessing,transformations}.py
  external/optic_disc/{config,preprocessor_config}.json
  external/optic_disc/pytorch_model.bin
/artifacts/
  reproducible_base.zip
  veinskel10_source.zip
```

Hashes and provenance are listed in `weights_manifest.json` and
`artifacts_manifest.json`.

After extracting the restricted archive, obtain the non-redistributed R2-V2
release files with:

```bash
bash material_review/fetch_r2v2.sh "$PWD/review_inputs/weights"
```

## Docker quick start

Build both images from the repository root:

```bash
docker build -f material_review/Dockerfile \
  -t choiclara9/gave2-kirby:prelim-7.95236-e2e-main .
docker build -f material_review/Dockerfile.r2v2 \
  -t choiclara9/gave2-kirby:prelim-7.95236-e2e-r2v2 .
```

Verify the GPU/runtime:

```bash
docker run --rm --gpus all \
  choiclara9/gave2-kirby:prelim-7.95236-e2e-main system-check
```

Run every selected model from official raw inputs and produce the final ZIP:

```bash
bash material_review/run_end_to_end_docker.sh \
  "$PWD/data" "$PWD/material_review/private_inputs/weights" \
  "$PWD/review_output"
```

The first container regenerates R2-V2 A/V predictions from CFP in its upstream
PyTorch-2.8 environment. The main container then regenerates every local model,
Task-3 optic-disc mask and biomarker, applies all fixed operations, and writes
`review_output/main/kirby.zip`. The script exits non-zero unless this file
matches the scored artifact hash and passes all structural checks. For a fast
artifact-only audit:

```bash
docker run --rm --gpus all \
  -v "$PWD/review_inputs/artifacts:/artifacts:ro" \
  -v "$PWD/review_output:/output" \
  choiclara9/gave2-kirby:prelim-7.95236-e2e-main reproduce
```

The command exits non-zero unless the generated artifact has the expected
SHA-256 and all 150 entries pass structural, image, text, and CRC checks.

## Weight distribution policy

The public Docker images contain code only. Locally trained checkpoints,
the Apache-2.0 optic-disc checkpoint, and two fast-audit artifacts are provided
to the organizers via a restricted download link. R2-V2 is downloaded from its
official repository/release with `fetch_r2v2.sh` and verified by SHA-256 because
the upstream repository does not state redistribution terms. This avoids
embedding protected data or weights in a public image while keeping the primary
raw-input route fully reproducible.

## Draft status

English spellings of the members' names must be confirmed before the final
email package is sent. The generated report therefore carries a visible draft
note until `material_review/report_data.json` is updated.
