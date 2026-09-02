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
from src.identity import WANDB_PROJECT, RunIdentity, experiment, wandb_artifact, wandb_id
from src.logs.continuations import ContinuationIndex
from src.logs.keys import Split
from src.model import SuffixModel, save_checkpoint
from src.training.early_stopping import EarlyStopper
from src.training.kl import linear_warmup_weight
from src.training.loss import Loss, compute_loss
from src.training.validation import validate, validate_generation


def train(
    *,
    model: SuffixModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    generation_loader: DataLoader,
    generation_samples: int,
    codec: DatasetCodec,
    dataset: str,
    run: RunIdentity,
    experiment_config: dict,
    loss_config: LossConfig,
    optimizer_config: OptimizerConfig,
    training: TrainingConfig,
    early_stopping_config: EarlyStoppingConfig,
) -> None:
    """
    Train a model on a dataset, logging to W&B and saving checkpoints.

    A validation that improves on the best selection score so far overwrites
    `paths.BEST_CHECKPOINT`; no other step is kept. A run that ends, however it ends, is over:
    there is no carrying one on, so nothing here writes the optimizer, early-stopping or random
    state a resume would have read.

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
        run: What every file this run writes is named after (see `src/paths.py`), what its W&B
            run is identified by (`src.identity.wandb_id`) and, minus its tag, which group and
            Artifact lineage it belongs to. One W&B run is one identity, so an identity reused
            across runs overlays their curves instead of listing them side by side; what makes
            its tag unique is the caller's business.
        experiment_config: The whole `ExperimentConfig`, dumped to plain data, written into the
            checkpoint so the model can be rebuilt from the file alone.
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

    # The step the best checkpoint on disk came from, kept so the Artifact this run leaves can
    # say which step it is without anyone downloading it.
    best_step = 0

    # `group` is the experiment, `dataset/model`, so runs of one model on one log sit together and
    # a run is one attempt at it; `job_type` says which stage of the pipeline this is, leaving room
    # for a later generate or evaluate stage on the same run. The tags repeat the two halves of the
    # group so either can be filtered on alone, which one group string cannot do. The commit is
    # W&B's own to record: it reads it off the working tree at `init`.
    wandb.init(
        project=WANDB_PROJECT,
        id=wandb_id(run),
        name=str(run),
        group=experiment(run),
        job_type='train',
        tags=[run.dataset, run.model],
        config=experiment_config,
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
                # Only a model with a latent has one to watch, and only it is charged a KL term.
                if latent is not None:
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
                    if val_latent is not None:
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

                    if val_latent is not None:
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

                    # Only an improvement is worth a file: the last step is never read back.
                    # The Artifact waits for the end of the run, so one run leaves one version
                    # rather than one per improvement.
                    if is_best:
                        best_step = step
                        path = save_checkpoint(
                            model,
                            experiment_config=experiment_config,
                            step=step,
                            selection_score=selection_score,
                            run=run,
                            path=paths.BEST_CHECKPOINT.prepare(run),
                        )
                        print(
                            f'New best model (step {step}, score {selection_score:.4f}) '
                            f'saved at {path}'
                        )

                if should_stop or step >= training.max_steps:
                    break

        # Everything below is only reached on a normal finish, not a crash, and while the W&B run
        # is still open. A run that dies leaves its best checkpoint on the machine that ran it and
        # nothing on W&B, which is the same thing a run that dies leaves behind anywhere else: it
        # cannot be carried on from either way.
        reason = (
            f'no validation improvement for {early_stopping_config.patience} validations'
            if should_stop
            else 'reached max_steps'
        )
        print(f'Finished training after {step} steps ({reason})')

        # One version per run, in the lineage its experiment shares, aliased with the run's own
        # tag. The file is uploaded as it sits: a checkpoint holds nothing a downloader would want
        # trimmed off it.
        artifact = wandb.Artifact(
            name=wandb_artifact(run),
            type='model',
            metadata={
                'run': str(run),
                'step': best_step,
                'selection_score': early_stopper.min_validation_score,
            },
        )
        artifact.add_file(str(paths.BEST_CHECKPOINT.path(run)), name='model.pt')
        wandb.log_artifact(artifact, aliases=[run.tag])

        # The alert is the one nobody has to be watching a terminal to get.
        wandb.alert(title=f'Training finished: {run}', text=f'{step} steps, {reason}.')
    finally:
        wandb.finish()
