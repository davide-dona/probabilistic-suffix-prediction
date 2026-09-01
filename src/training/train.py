from dataclasses import asdict

import torch
import wandb
from torch import optim
from torch.utils.data import DataLoader

from src import paths
from src.configs.schema import (
    EarlyStoppingConfig,
    LossConfig,
    OptimizerConfig,
    TrainingConfig,
)
from src.datasets.codec import DatasetCodec
from src.git import current_git_state
from src.identity import WANDB_PROJECT, RunIdentity, wandb_id
from src.logs.continuations import ContinuationIndex
from src.logs.keys import Split
from src.model import TransformerCVAE, save_checkpoint
from src.training.early_stopping import EarlyStopper
from src.training.kl import linear_warmup_weight
from src.training.loss import Loss, compute_loss
from src.training.validation import validate, validate_generation

# The device's own stream, which dropout and the latent sampling draw from, lives behind a
# namespace of its own; `cpu` has none beside the global one.
# Both halves take the device the run is on: on a multi-GPU machine `torch.cuda`'s zero-argument
# form would capture the current device's stream rather than the pinned one, so a run on `cuda:1`
# would checkpoint and restore `cuda:0`. `torch.mps` has one device and so takes no argument.
DEVICE_RNG = {
    'cuda': (torch.cuda.get_rng_state, torch.cuda.set_rng_state),
    'mps': (
        lambda _device: torch.mps.get_rng_state(),
        lambda state, _device: torch.mps.set_rng_state(state),
    ),
}


def rng_state(generator: torch.Generator, device: torch.device) -> dict:
    """
    Every random stream a run draws from, so a resumed one carries on where it left off.

    Args:
        generator: The generator the training loader shuffles with.
        device: The device the model runs on.
    Returns:
        The states, keyed for `restore_rng_state`.
    """
    capture = DEVICE_RNG.get(device.type)
    return {
        'cpu': torch.get_rng_state(),
        'dataloader': generator.get_state(),
        'device': capture[0](device) if capture is not None else None,
    }


def restore_rng_state(state: dict, *, generator: torch.Generator, device: torch.device) -> None:
    """
    Put back what `rng_state` captured, on the device the run trained on.

    Args:
        state: What `rng_state` returned.
        generator: The generator the training loader shuffles with.
        device: The device the model runs on.
    """
    torch.set_rng_state(state['cpu'])
    generator.set_state(state['dataloader'])

    restore = DEVICE_RNG.get(device.type)
    if restore is not None:
        restore[1](state['device'], device)


def train(
    *,
    model: TransformerCVAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    generation_loader: DataLoader,
    generation_samples: int,
    codec: DatasetCodec,
    dataset: str,
    run: RunIdentity,
    experiment_config: dict,
    generator: torch.Generator,
    resume: dict | None,
    loss_config: LossConfig,
    optimizer_config: OptimizerConfig,
    training: TrainingConfig,
    early_stopping_config: EarlyStoppingConfig,
) -> None:
    """
    Train a model on a dataset, logging to W&B and saving checkpoints.

    Every validation writes a checkpoint to `paths.LAST_CHECKPOINT`, whether or not the step is
    the best one, and a validation that improves on the best writes a second copy to
    `paths.BEST_CHECKPOINT`. Either file can be handed back as `resume`.

    Args:
        model: The model to train, already on `training.device`.
        train_loader: Batches to learn from.
        val_loader: Batches to score teacher-forced, every `training.val_every_n_steps` steps.
        generation_loader: Prefixes to generate suffixes for on the same cadence. A far smaller
            slice than `val_loader`, since a suffix costs one decoder pass per event.
        generation_samples: Suffixes to draw per prefix on that pass, normally
            `inference.validation_samples`, which is smaller than the
            `inference.evaluation_samples` a report is built from: a curve is read for its shape
            over steps rather than against a report's numbers.
        codec: The codec the splits were encoded through, passed on to the
            generation pass so its remaining times are scored in minutes.
        dataset: The log being trained on, naming the validation split's continuation index the
            selection score is read against.
        run: What every file this run writes is named after (see `src/paths.py`), and what its
            W&B run is identified by (`src.tracking.wandb_id`). One W&B run is one identity, so
            an identity reused across runs overlays their curves instead of listing them side by
            side; what makes its tag unique is the caller's business.
        experiment_config: The whole `ExperimentConfig`, dumped to plain data, written into
            every checkpoint so the run can be rebuilt and carried on from the file alone.
        generator: The generator the training loader shuffles with, whose state is saved and
            restored along with the rest.
        resume: A checkpoint to carry on from, read by `load_checkpoint`, or `None` to start
            from step zero. Its step, weights, optimizer state, early-stopping state and random
            streams are all restored.
        loss_config: The KL annealing schedule.
        optimizer_config: The optimizer hyperparameters.
        training: Step budget, validation cadence, gradient clipping and device.
        early_stopping_config: When to give up.
    """
    device = torch.device(training.device)

    # The validation split's continuations, which the selection score is measured against. Read
    # once here rather than per validation, and never the test split's: selecting against those
    # would fold the held-out set into which checkpoint is kept.
    continuations = ContinuationIndex(dataset=dataset, split=Split.VAL)

    optimizer = optim.Adam(
        model.parameters(), lr=optimizer_config.lr, weight_decay=optimizer_config.weight_decay
    )
    early_stopper = EarlyStopper(early_stopping_config)

    step = 0
    should_stop = False
    interval_totals = Loss()
    seen = 0

    # If resuming, restore the model, optimizer, early stopper and random streams to the state
    # they were in when the checkpoint was written.
    if resume is not None:
        model.load_state_dict(state_dict=resume['model_state_dict'])
        optimizer.load_state_dict(state_dict=resume['optimizer_state'])
        early_stopper.load_state_dict(resume['early_stopping_state'])
        # Restored after the loaders were built, so that the fixed validation and generation
        # subsets are still drawn from the seeded stream exactly as an unresumed run draws them.
        restore_rng_state(resume['rng_state'], generator=generator, device=device)
        step = resume['step']
        print(f'Resuming {run} from step {step}, best score {resume["selection_score"]:.4f}')

    # The commit training runs under, carried into the W&B config and every checkpoint, so a set
    # of weights never leaves anyone guessing what code produced it.
    git_state = current_git_state()

    # `resume='allow'` picks the W&B run back up if `wandb_id(run)` already exists (a carried-on
    # run) and starts a fresh one otherwise. Unlike TensorBoard's `purge_step`, W&B has no way to
    # truncate history that ran ahead of the last checkpoint: a crash between two validations,
    # followed by resume, can leave a few stray, overlapping points on the step axis until the
    # next validation. Cosmetic only, not a data-loss risk.
    wandb.init(
        project=WANDB_PROJECT,
        id=wandb_id(run),
        name=str(run),
        group=run.dataset,
        job_type=run.model,
        resume='allow',
        config=experiment_config | {'git_commit': git_state.commit, 'git_dirty': git_state.dirty},
    )
    print(f'Logging to {wandb.run.url}')

    try:
        while step < training.max_steps and not should_stop:
            for batch in train_loader:
                model.train()
                batch = batch.to(device)
                # Get the current KL weight for this step
                kl_weight = linear_warmup_weight(
                    step,
                    ramp_steps=loss_config.kl_annealing_ramp_steps,
                    start=loss_config.kl_annealing_start_weight,
                    stop=loss_config.kl_annealing_full_weight,
                )
                # Run a forward pass
                output = model(batch)

                # Compute the loss and propagate gradients
                loss, metrics, latent = compute_loss(
                    output,
                    batch,
                    pad_activity_index=model.pad_activity_index,
                    kl_weight=kl_weight,
                    free_bits=loss_config.free_bits,
                )
                optimizer.zero_grad()
                loss.backward()
                if training.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), training.grad_clip_norm)
                optimizer.step()

                # Update the running totals and log to W&B
                batch_size = batch.suffix.activities.size(0)
                interval_totals += metrics
                seen += batch_size
                step += 1
                (metrics / batch_size).log(step, prefix='train')
                (latent / batch_size).log(step, prefix='train')

                if step % training.val_every_n_steps == 0 or step >= training.max_steps:
                    train_metrics = interval_totals / seen
                    interval_totals, seen = Loss(), 0

                    # Score the model on the validation set and the generation set, and log
                    # the results.
                    val_metrics, val_latent = validate(
                        model,
                        val_loader,
                        kl_weight=kl_weight,
                        free_bits=loss_config.free_bits,
                        device=device,
                    )
                    val_metrics.log(step, prefix='val')
                    val_latent.log(step, prefix='val')

                    gen_metrics = validate_generation(
                        model,
                        generation_loader,
                        num_samples=generation_samples,
                        codec=codec,
                        index=continuations,
                        device=device,
                    )
                    gen_metrics.accuracy.log(step, prefix='gen')
                    gen_metrics.distribution.log(step, prefix='gen')

                    wandb.log({'kl_weight': kl_weight}, step=step)

                    # The one line of live feedback: enough to see a run is alive and heading down
                    print(
                        f'Step {step:>{len(str(training.max_steps))}}/{training.max_steps}  '
                        f'kl {kl_weight:.2f}  train {train_metrics.loss:.4f}  '
                        f'val {val_metrics.loss:.4f}  '
                        f'gen_dls {gen_metrics.accuracy.dls_mean:.4f} mean / '
                        f'{gen_metrics.accuracy.dls_point:.4f} point  '
                        f'emsc {gen_metrics.distribution.emsc:.4f}',
                        flush=True,
                    )
                    # The early stopper minimizes, and EMSC is a similarity, so it is the distance
                    # that is tracked.
                    selection_score = 1.0 - gen_metrics.distribution.emsc

                    # Read before `update` folds this score into it, since afterwards it can
                    # no longer tell an improvement from a step that just matched the best.
                    is_best = selection_score < early_stopper.min_validation_score
                    should_stop = early_stopper.update(selection_score)

                    # Save a checkpoint with the model, optimizer, early stopper and random
                    # streams in their current state.
                    checkpoint = dict(
                        experiment_config=experiment_config,
                        step=step,
                        selection_score=selection_score,
                        run=run,
                        optimizer_state=optimizer.state_dict(),
                        early_stopping_state=early_stopper.state_dict(),
                        rng_state=rng_state(generator, device),
                        git=asdict(git_state),
                    )
                    save_checkpoint(model, **checkpoint, path=paths.LAST_CHECKPOINT.prepare(run))
                    # If this validation improved on the best, save a second copy to the
                    # best-model directory, and version it as a W&B Artifact: `best` always
                    # points at the newest version of this run's own lineage.
                    if is_best:
                        path = save_checkpoint(
                            model, **checkpoint, path=paths.BEST_CHECKPOINT.prepare(run)
                        )
                        artifact = wandb.Artifact(
                            name=wandb_id(run),
                            type='checkpoint',
                            metadata={'step': step, 'selection_score': selection_score},
                        )
                        artifact.add_file(str(path))
                        wandb.log_artifact(artifact, aliases=['best'])
                        print(
                            f'New best model (step {step}, score {selection_score:.4f}) '
                            f'saved at {path}'
                        )

                if should_stop or step >= training.max_steps:
                    break

        # Only reached on a normal finish, not a crash, and while the W&B run is still open, so
        # the alert is one nobody has to be watching a terminal to get.
        reason = (
            f'no validation improvement for {early_stopping_config.patience} validations'
            if should_stop
            else 'reached max_steps'
        )
        print(f'Finished training after {step} steps ({reason})')
        wandb.alert(title=f'Training finished: {run}', text=f'{step} steps, {reason}.')
    finally:
        wandb.finish()
