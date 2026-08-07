import argparse
import os
import socket


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='GAVE_pair', help='Dataset to use')
parser.add_argument('--num_iterations', type=int, default=5, help='Number of iterations')
parser.add_argument('--criterion', type=str, default='RRLoss', help='Criterion to use')
parser.add_argument('--base_criterion', type=str, default='BCE3Loss', help='Base criterion to use')
parser.add_argument('--model', type=str, default='CMRRWNet', help='Model to use')
parser.add_argument('--num_folds', type=int, default=4, help='Number of folds for cross-validation')
parser.add_argument('--active_folds', type=int, nargs='+', default=None, help='Fold indices to run; default runs every fold')
parser.add_argument('--balanced_folds', action='store_true', help='Distribute remainder images instead of using the legacy split')
parser.add_argument('--learning_rate', type=float, default=1e-04, help='Learning rate for training')
parser.add_argument('--num_epochs', type=int, default=None, help='Number of epochs for training')
parser.add_argument('--scheduler_patience', type=int, default=2147483647, help='Validation epochs before reducing LR (disabled by default for baseline compatibility)')
parser.add_argument('--stopping_patience', type=int, default=100, help='Validation epochs without improvement before stopping')
parser.add_argument('--optimizer', choices=['Adam', 'AdamW'], default='Adam', help='Optimizer to use')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (mainly for AdamW)')
parser.add_argument('--amp', action='store_true', help='Use automatic mixed precision on CUDA')
parser.add_argument('--wandb_project', default=os.environ.get('WANDB_PROJECT'), help='W&B project; unset disables tracking')
parser.add_argument('--wandb_entity', default=os.environ.get('WANDB_ENTITY'), help='W&B username or team entity')
parser.add_argument('--wandb_mode', choices=['online', 'offline', 'disabled'], default=os.environ.get('WANDB_MODE', 'online'), help='W&B sync mode')
parser.add_argument('--wandb_task', default=os.environ.get('WANDB_TASK'), help='Task label such as Task1 or Task2; inferred from input channels when omitted')
parser.add_argument('--wandb_group', default=os.environ.get('WANDB_RUN_GROUP'), help='Optional W&B run group; defaults to --version')
parser.add_argument('--wandb_run_name', default=os.environ.get('WANDB_RUN_NAME'), help='Optional run-name prefix; fold is appended automatically')
parser.add_argument('--wandb_tags', nargs='*', default=None, help='Additional W&B tags')
parser.add_argument('--wandb_notes', default=os.environ.get('WANDB_NOTES'), help='Optional W&B run notes')
parser.add_argument('--wandb_log_model', action='store_true', help='Upload the best checkpoint as a W&B artifact after training')
parser.add_argument('--base_channels', type=int, default=64, help='Number of base channels for the model')
parser.add_argument('--in_channels', type=int, default=5, help='Number of input channels for the model')
parser.add_argument('--out_channels', type=int, default=3, help='Number of output channels for the model')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
parser.add_argument('--n_proc', type=int, default=1, help='Number of processes to use')
parser.add_argument('--data_folder', type=str, default='./', help='Folder containing the data')
parser.add_argument('--version', type=str, default='Journal_paper', help='Version of the experiment')
parser.add_argument('--seed', type=int, default=77, help='Random seed for reproducibility')
args = parser.parse_args()


### Configuration arguments

num_folds = args.num_folds
balanced_folds = args.balanced_folds

active_folds = args.active_folds if args.active_folds is not None else list(range(num_folds))
invalid_folds = [fold for fold in active_folds if fold < 0 or fold >= num_folds]
if invalid_folds:
    parser.error(
        f'--active_folds contains invalid indices {invalid_folds}; '
        f'valid range is 0..{num_folds - 1}'
    )

learning_rate = args.learning_rate
num_epochs = args.num_epochs
scheduler_patience = args.scheduler_patience
stopping_patience = args.stopping_patience
optimizer = args.optimizer
weight_decay = args.weight_decay
amp = args.amp
wandb_project = args.wandb_project
wandb_entity = args.wandb_entity
wandb_mode = args.wandb_mode
wandb_task = args.wandb_task
wandb_group = args.wandb_group
wandb_run_name = args.wandb_run_name
wandb_tags = args.wandb_tags
wandb_notes = args.wandb_notes
wandb_log_model = args.wandb_log_model

dataset = args.dataset

model = args.model
in_channels = args.in_channels
out_channels = args.out_channels
if socket.gethostname() == 'hemingway':
    # Reduce the model size for local testing
    args.base_channels = 16
base_channels = args.base_channels
num_iterations = args.num_iterations

criterion = args.criterion
base_criterion = args.base_criterion

n_proc = args.n_proc
gpu_id = args.gpu_id

training_folder = f'./__training/{args.version}/{dataset}'

seed = args.seed

if dataset == 'RITE-train':
    images = [
        33, 24, 36, 30, 25, 29, 40, 21, 37, 34, 35, 32, 27, 39, 26, 38, 28, 23,
        31, 22
    ]
    data = {
        'data_folder': args.data_folder,
        'target': {
            'path': 'RITE/train/av3',
            'pattern': '[0-9]+[.]png'
        },
        'original': {
            'path': 'RITE/train/enhanced',
            'pattern': '[0-9]+[.]png'
        },
        'mask': {
            'path': 'RITE/train/enhanced_masks',
            'pattern': '[0-9]+[.]png'
        }
    }
elif dataset == 'HRF-Karlsson-w1024':
    images = [
        '06_dr', '06_g', '06_h', '07_dr', '07_g', '07_h', '08_dr', '08_g',
        '08_h', '09_dr', '09_g', '09_h', '10_dr', '10_g', '10_h', '11_dr',
        '11_g', '11_h', '12_dr', '12_g', '12_h', '13_dr', '13_g', '13_h',
        '14_dr', '14_g', '14_h', '15_dr', '15_g', '15_h',
    ]
    data = {
        'data_folder': args.data_folder,
        'target': {
            'path': f'HRF_AVLabel_191219/train_karlsson_w1024/av3',
            'pattern': '[0-9]+_.+[.]png'
        },
        'original': {
            'path': f'HRF_AVLabel_191219/train_karlsson_w1024/enhanced',
            'pattern': '[0-9]+_.+[.]png'
        },
        'mask': {
            'path': f'HRF_AVLabel_191219/train_karlsson_w1024/enhanced_masks',
            'pattern': '[0-9]+_.+[.]png'
        }
    }
elif dataset == 'GAVE_pair':
    images = [f"g_{i:03d}" for i in range(1, 51)]
    data = {
        'data_folder': args.data_folder,
        'target': {
            'path': 'data/training/av',
            'pattern': 'g_[0-9]+[.]png'
        },
        'original': {
            'path': 'data/training/images',
            'pattern': 'g_[0-9]+[.]png'
        },
        'mask': {
            'path': 'data/training/masks',
            'pattern': 'g_[0-9]+[.]png'
        },
        'a': {
            'path': 'data/training/FFA_A',
            'pattern': 'g_[0-9]+[.]png'
        },
        'av': {
            'path': 'data/training/FFA_AV',
            'pattern': 'g_[0-9]+[.]png'
        }
    }
else:
    raise ValueError('dataset not supported')
