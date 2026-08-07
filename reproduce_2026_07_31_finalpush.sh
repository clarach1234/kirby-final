#!/usr/bin/env bash
# Deterministically assemble the exact highest-scoring 2026-07-31 artifact.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
gave2_python="${GAVE2_PYTHON:-python3}"
base_zip="${1:-${repo_dir}/submission_experiments/kirby_repro_t1R2kirbyG040fold0i53160_t2endpointRB_DcG_t3bestfield_compact_metric_equivalent.zip}"
veinskel_zip="${2:-${repo_dir}/submission_experiments/kirby_scorepush_t1eq025_t2Db_veinskel10_t3E_fdstrong.zip}"
output_zip="${3:-${repo_dir}/submission_experiments/kirby_finalpush_t1reproR2G040_t2currentR_DcG_veinskel10OR_t3bestfield_compact_metric_equivalent.zip}"
manifest="${output_zip%.zip}_manifest.json"

base_sha="e00619d313ab7520c24ec87fd97538a2f41145f11c28d8457a4d20461c14dda1"
veinskel_sha="9b668305a1eb5f45854210753981adde607ee0368eeab00f0edbefe02bf116e8"
final_sha="4c19a97503668ef003c992edcbf76ade644c40903b683394347c0214db51306c"

printf '%s  %s\n' "${base_sha}" "${base_zip}" | sha256sum --check
printf '%s  %s\n' "${veinskel_sha}" "${veinskel_zip}" | sha256sum --check

if ! "${gave2_python}" -c 'import numpy, PIL' 2>/dev/null; then
  echo "GAVE2_PYTHON must point to an environment with numpy and Pillow" >&2
  echo "Example: GAVE2_PYTHON=/path/to/cmrrwnet/bin/python $0" >&2
  exit 1
fi

"${gave2_python}" "${repo_dir}/build_t2_veinskel10_or_submission.py" \
  --base-submission "${base_zip}" \
  --veinskel10-submission "${veinskel_zip}" \
  --output "${output_zip}" \
  --manifest "${manifest}"

printf '%s  %s\n' "${final_sha}" "${output_zip}" | sha256sum --check
echo "Exact finalpush artifact reproduced: ${output_zip}"
