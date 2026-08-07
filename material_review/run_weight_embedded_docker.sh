#!/usr/bin/env bash
# Reproduce the final preliminary ZIP using official data mounted at /data and
# images that already contain every selected model weight.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 DATA_DIR OUTPUT_DIR" >&2
  exit 2
fi

data_dir="$(realpath "$1")"
output_dir="$(realpath -m "$2")"
if [[ ! -d "${data_dir}" ]]; then
  echo "DATA_DIR must be an existing official GAVE2 data directory" >&2
  exit 2
fi
if [[ -e "${output_dir}/main/kirby.zip" ]]; then
  echo "Refusing to overwrite ${output_dir}/main/kirby.zip; use a new OUTPUT_DIR" >&2
  exit 2
fi
mkdir -p "${output_dir}/r2v2" "${output_dir}/main"

r2_image="${KIRBY_R2_WEIGHT_IMAGE:-choiclara9/gave2-kirby:prelim-7.95236-weights-r2v2}"
main_image="${KIRBY_MAIN_WEIGHT_IMAGE:-choiclara9/gave2-kirby:prelim-7.95236-weights-main}"

docker run --rm --gpus all \
  -v "${data_dir}:/data:ro" \
  -v "${output_dir}/r2v2:/output" \
  "${r2_image}"

docker run --rm --gpus all \
  -v "${data_dir}:/data:ro" \
  -v "${output_dir}/r2v2/r2v2:/external/r2v2:ro" \
  -v "${output_dir}/main:/output" \
  "${main_image}" run-all

echo "Final submission: ${output_dir}/main/kirby.zip"
sha256sum "${output_dir}/main/kirby.zip"
