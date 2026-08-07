#!/usr/bin/env bash
set -euo pipefail

python /weights/external/r2v2/source/infer.py \
  --cfp_path /data/validation/images \
  --masks_path /data/validation/masks \
  --model_type av \
  --weights_path /weights/external/r2v2 \
  --save_path /output/r2v2 \
  --tta \
  --use-gave-format \
  "$@"
