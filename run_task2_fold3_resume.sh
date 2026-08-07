#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_dir}"

python_bin="/home/bispl/miniconda3/envs/cmrrwnet/bin/python"
version="first_submit"
epochs="${TASK2_EPOCHS:-2000}"
stopping_patience="${TASK2_STOPPING_PATIENCE:-100}"
wandb_project="${WANDB_PROJECT:-gave2-challenge}"
wandb_entity="${WANDB_ENTITY:-teamkirby}"
wandb_mode="${WANDB_MODE:-online}"

if pgrep -f 'python.*train/train.py' >/dev/null; then
    echo "Another training process is already running."
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
    --active_folds 3 \
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
    --stopping_patience "${stopping_patience}" \
    --wandb_project "${wandb_project}" \
    --wandb_entity "${wandb_entity}" \
    --wandb_mode "${wandb_mode}" \
    --wandb_task Task2 \
    --wandb_tags cmrrwnet fold3 resume remote \
    --wandb_log_model \
    --version "${version}"
