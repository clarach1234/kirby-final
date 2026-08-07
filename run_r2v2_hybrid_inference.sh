#!/usr/bin/env bash
# Reproduce the R2-V2 AV + kirby vessel-channel GAVE2 hybrid.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
r2v2_work_root="${R2V2_WORK_ROOT:-${repo_dir}/r2v2_external}"
r2v2_repo="${r2v2_work_root}/R2-V2"
r2v2_venv="${r2v2_work_root}/venv"
r2v2_weights="${r2v2_work_root}/weights"
r2v2_predictions="${repo_dir}/r2v2_validation_predictions"
python_seed="${PYTHON_SEED:-python3.12}"

r2v2_commit="7f6a8ea7a51782b1e0f89723a9ec137ba0a29913"
av_weight_sha256="74d425afb714384cb3f4d5db9cc852c1ea6d7552e46c866e29a3777db12b9d80"
av_config_sha256="8c4bb170f0f4df5cc21ce6929ac1e6e738c82404fe420310181974f572beff54"

mkdir -p "${r2v2_work_root}" "${r2v2_weights}"

if [[ ! -d "${r2v2_repo}/.git" ]]; then
    git clone https://github.com/j-morano/R2-V2.git "${r2v2_repo}"
fi
git -C "${r2v2_repo}" fetch --all --tags
git -C "${r2v2_repo}" checkout --detach "${r2v2_commit}"

if [[ ! -f "${r2v2_weights}/av.pth" ]]; then
    curl -fL \
        https://github.com/j-morano/R2-V2/releases/download/v1/av.pth \
        -o "${r2v2_weights}/av.pth"
fi
if [[ ! -f "${r2v2_weights}/av_config.json" ]]; then
    curl -fL \
        https://github.com/j-morano/R2-V2/releases/download/v1/av_config.json \
        -o "${r2v2_weights}/av_config.json"
fi

printf '%s  %s\n' "${av_weight_sha256}" "${r2v2_weights}/av.pth" \
    | sha256sum --check
printf '%s  %s\n' "${av_config_sha256}" "${r2v2_weights}/av_config.json" \
    | sha256sum --check

if [[ ! -x "${r2v2_venv}/bin/python" ]]; then
    "${python_seed}" -m venv "${r2v2_venv}"
    "${r2v2_venv}/bin/pip" install -r "${r2v2_repo}/requirements.txt"
fi

"${r2v2_venv}/bin/python" "${r2v2_repo}/infer.py" \
    --cfp_path "${repo_dir}/data/validation/images" \
    --masks_path "${repo_dir}/data/validation/masks" \
    --model_type av \
    --weights_path "${r2v2_weights}" \
    --save_path "${r2v2_predictions}" \
    --tta \
    --use-gave-format

cd "${repo_dir}"
"${r2v2_venv}/bin/python" build_r2v2_hybrid_submission.py
