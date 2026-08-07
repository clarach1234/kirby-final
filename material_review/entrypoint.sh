#!/usr/bin/env bash
set -euo pipefail

command_name="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi

case "${command_name}" in
  help)
    cat <<'EOF'
kirby GAVE2 reproducibility container

Commands:
  system-check       Print Python/PyTorch/CUDA/GPU information.
  verify-weights     Verify mounted /weights against the checkpoint manifest.
  infer-components  Run local Task-1/Task-2 folds from /data and /weights.
  run-all           Run raw official inputs through all models to /output/kirby.zip.
  reproduce         Rebuild /output/kirby.zip from mounted /artifacts inputs.
  validate          Validate a submission (default /output/kirby.zip).

Additional arguments are forwarded to material_review/pipeline.py.
EOF
    ;;
  system-check)
    python /workspace/material_review/pipeline.py system-check "$@"
    ;;
  verify-weights)
    python /workspace/material_review/pipeline.py verify-weights \
      --weights-root /weights "$@"
    ;;
  infer-components)
    python /workspace/material_review/pipeline.py infer-components \
      --data-root /data --weights-root /weights --output-root /output/components "$@"
    ;;
  run-all)
    r2_args=()
    if [[ -d /external/r2v2/av ]]; then
      r2_args=(--r2-av-source /external/r2v2/av)
    fi
    python /workspace/material_review/pipeline.py run-all \
      --data-root /data --weights-root /weights \
      --work-root /output/run_all_work --output /output/kirby.zip \
      "${r2_args[@]}" "$@"
    ;;
  reproduce)
    python /workspace/material_review/pipeline.py reproduce \
      --base /artifacts/reproducible_base.zip \
      --veinskel /artifacts/veinskel10_source.zip \
      --output /output/kirby.zip "$@"
    ;;
  validate)
    python /workspace/material_review/pipeline.py validate \
      --submission /output/kirby.zip "$@"
    ;;
  *)
    exec "${command_name}" "$@"
    ;;
esac
