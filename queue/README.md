# GPU queues

The file queues launch one job per GPU, defaulting to GPUs 0 and 1. Each worker masks its device
with `CUDA_VISIBLE_DEVICES`, so a job uses `training.device=cuda:0` inside that worker.

## Training

A `.txt` job holds one Hydra override argument per line. Blank lines and lines starting with `#`
are ignored. Arguments are passed directly, without shell evaluation.

```bash
mkdir -p queue/train
cat > queue/train/sepsis-cvae.txt <<'JOB'
dataset=sepsis
model=cvae
JOB
scripts/train_queue.sh -g 0,1
```

Common overrides can follow the GPU option, for example `scripts/train_queue.sh wandb.mode=offline`.
Job-specific overrides follow common overrides and take precedence. Jobs must select a dataset;
the model defaults to CVAE. Each process writes its artifacts to its own Hydra output directory.

## Generation

Copy checkpoints into the queue under distinct filenames. Successful jobs delete these copies,
so keep the original checkpoints elsewhere.

```bash
mkdir -p queue/generate
cp /path/to/best.pt queue/generate/sepsis-cvae.pt
scripts/generate_queue.sh -g 0,1 num_samples=100
```

The queue consumes the new checkpoint format. Existing legacy checkpoints require retraining
or use of the older code revision.

## Evaluation

```bash
scripts/evaluate_queue.sh workers=4
scripts/evaluate_queue.sh --force workers=4
```

Evaluation runs sequentially, using its own CPU process pool. It discovers `generations.parquet`
under `outputs/` and `pinned/`, skipping source hashes already recorded in a completed evaluation.
Use `--roots <directory> ...` for other roots; separate subsequent Hydra overrides with `--`.

## Job lifecycle

A GPU job is claimed by renaming it `.running`. Success deletes the queued job; failure renames
it `.failed`. Requeue a failed job by removing that suffix after inspecting its log. Logs are
under `outputs/queue/<stage>/`; the console prints starts, completions, and a final summary.
Interrupting the queue stops new claims. Inspect any `.running` files before requeuing them.
