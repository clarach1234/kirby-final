#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_dir}"

python_bin="/home/bispl/miniconda3/envs/cmrrwnet/bin/python"
version="${TASK1_VERSION:-task1_rrwnet_fold1_bc16}"
# fold0 needed 1200 epochs (best at 1191) to reach its submitted checkpoint;
# budget headroom above that rather than the script's historical 160 default.
epochs="${TASK1_EPOCHS:-1400}"
stopping_patience="${TASK1_STOPPING_PATIENCE:-100}"
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
    --model RRWNet \
    --in_channels 3 \
    --base_channels 16 \
    --num_iterations 5 \
    --num_folds 5 \
    --active_folds 1 \
    --balanced_folds \
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
    --wandb_task Task1 \
    --wandb_tags rrwnet cfp-only fold1 \
    --wandb_log_model \
    --version "${version}"
