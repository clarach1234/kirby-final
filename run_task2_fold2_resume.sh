#!/usr/bin/env bash
# Reproduces the resume-to-completion of Task 2 fold 2's baseline checkpoint.
# This is the exact recipe that took the Kaggle-interrupted checkpoint
# (iteration ~47880) to its final, submitted state (iteration 55008, best
# iteration 51408, validation loss 0.385, stopped naturally via
# stopping_patience) on 2026-07-26. The resulting checkpoint is used, as-is,
# in Task 2 candidates A, B, C, and E (see SUBMISSIONS_2026-07-26.md), and is
# also the starting point for run_task2_topology_finetune_fold2.sh.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_dir}"

python_bin="/home/bispl/miniconda3/envs/cmrrwnet/bin/python"
version="first_submit"
epochs="${TASK2_EPOCHS:-2500}"
wandb_project="${WANDB_PROJECT:-gave2-challenge}"
wandb_entity="${WANDB_ENTITY:-teamkirby}"
wandb_mode="${WANDB_MODE:-online}"

experiment_dir="__training/${version}/GAVE_pair/4_folds/CMRRWNet_1it_lr1e-04_RRLoss-BCE3Loss_bc16/2"
if [[ ! -f "${experiment_dir}/generator_best.pth" ]]; then
    echo "No existing checkpoint at ${experiment_dir}/generator_best.pth."
    echo "Restore the interrupted Kaggle checkpoint there first (or start"
    echo "fresh if none exists), then re-run this script."
    exit 1
fi

if pgrep -f 'python.*train/train.py' >/dev/null; then
    echo "Another training process is already running. Pause it first" \
         "(this fold needs the whole GPU on an 8 GB card)."
    exit 1
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
    --base_criterion BCE3Loss \
    --learning_rate 1e-4 \
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
    --wandb_tags cmrrwnet fold2 resume remote \
    --wandb_log_model
