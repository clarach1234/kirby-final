"""Optional experiment tracking kept separate from the training core."""

import os
from pathlib import Path


def _task_name(config):
    if config.wandb_task:
        return config.wandb_task
    return 'Task1' if config.in_channels == 3 else 'Task2'


def init_wandb(config, fold, train_images, validation_images, run_dir):
    """Create one W&B run per fold, or return None when tracking is disabled."""
    if not config.wandb_project or config.wandb_mode == 'disabled':
        return None

    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            'W&B tracking was requested but wandb is not installed. '
            'Install it with: python -m pip install "wandb>=0.19.10"'
        ) from error

    run_config = dict(vars(config.args))
    run_config.update({
        'task': _task_name(config),
        'fold': fold,
        'train_size': len(train_images),
        'validation_size': len(validation_images),
        'train_images': list(train_images),
        'validation_images': list(validation_images),
    })

    return wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        group=config.wandb_group or config.args.version,
        name=f'{config.wandb_run_name or config.args.version}-fold-{fold}',
        job_type='cross-validation',
        tags=tuple(dict.fromkeys([
            'GAVE2',
            _task_name(config),
            config.model,
            *(config.wandb_tags or []),
        ])),
        notes=config.wandb_notes,
        config=run_config,
        dir=os.path.abspath(run_dir),
        mode=config.wandb_mode,
        reinit='finish_previous',
        save_code=True,
    )


def log_wandb_artifacts(tracker, run_dir, log_model=False):
    """Attach reproducibility files and optionally the best model."""
    if tracker is None:
        return

    import wandb

    run_dir = Path(run_dir)
    metadata = wandb.Artifact(
        name=f'{tracker.name}-metadata',
        type='run-metadata',
    )
    for filename in (
        'config.json',
        'train_loss.csv',
        'test_loss.csv',
        'best_loss.csv',
        'scheduler.csv',
        'learning_curves.svg',
    ):
        path = run_dir / filename
        if path.is_file():
            metadata.add_file(str(path), name=filename)
    tracker.log_artifact(metadata)

    best_model = run_dir / 'generator_best.pth'
    if log_model and best_model.is_file():
        model_artifact = wandb.Artifact(
            name=f'{tracker.name}-best-model',
            type='model',
            metadata={'source': str(best_model)},
        )
        model_artifact.add_file(
            str(best_model),
            name='generator_best.pth',
        )
        tracker.log_artifact(model_artifact)
