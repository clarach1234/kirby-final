#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_dir}"

python_bin="/home/bispl/miniconda3/envs/cmrrwnet/bin/python"
version="task2_topoft_fold2_15ep"
epochs="${TOPOFT_EPOCHS:-15}"
wandb_project="${WANDB_PROJECT:-gave2-challenge}"
wandb_entity="${WANDB_ENTITY:-teamkirby}"
wandb_mode="${WANDB_MODE:-online}"

baseline_checkpoint="__training/first_submit/GAVE_pair/4_folds/CMRRWNet_1it_lr1e-04_RRLoss-BCE3Loss_bc16/2/generator_best.pth"
experiment_dir="__training/${version}/GAVE_pair/4_folds/CMRRWNet_1it_lr1e-05_RRLoss-OfficialScoreLoss_bc16/2"

if [[ ! -f "${baseline_checkpoint}" ]]; then
    echo "Baseline fold 2 checkpoint not found: ${baseline_checkpoint}"
    echo "Train or resume Task 2 fold 2 to completion first."
    exit 1
fi

if pgrep -f 'python.*train/train.py' >/dev/null; then
    echo "Another training process is already running. Pause it first" \
         "(this fine-tune needs the whole GPU on an 8 GB card)."
    exit 1
fi

mkdir -p "${experiment_dir}"
if [[ ! -f "${experiment_dir}/generator_best.pth" ]]; then
    cp "${baseline_checkpoint}" "${experiment_dir}/generator_best.pth"
fi

exec "${python_bin}" -u train/train.py \
    --data_folder . \
    --dataset GAVE_pair \
    --model CMRRWNet \
    --in_channels 5 \
    --base_channels 16 \
    --num_iterations 1 \
    --num_folds 4 \
    --active_folds 2 \
    --n_proc 1 \
    --gpu_id 0 \
    --criterion RRLoss \
    --base_criterion OfficialScoreLoss \
    --learning_rate 1e-5 \
    --optimizer Adam \
    --weight_decay 0 \
    --amp \
    --num_epochs "${epochs}" \
    --scheduler_patience 2147483647 \
    --stopping_patience 100 \
    --version "${version}" \
    --wandb_project "${wandb_project}" \
    --wandb_entity "${wandb_entity}" \
    --wandb_mode "${wandb_mode}" \
    --wandb_task Task2 \
    --wandb_tags topology-aware fold2 15ep remote \
    --wandb_log_model
