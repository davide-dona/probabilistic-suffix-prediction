# Suffix Generation

Conditional suffix generation for predictive process monitoring.

This repository provides two Transformer architectures, a conditional variational autoencoder
and a Head-sampling Transformer, together with preprocessing, training, inference, evaluation,
and visualization pipelines.

## Install

**Requirements:** Python 3.13+, [uv](https://docs.astral.sh/uv/), and
[Git LFS](https://git-lfs.com/).

The datasets under `data/` are tracked with Git LFS. Pull them before preprocessing:

```bash
git lfs install
git lfs pull
uv sync --locked
```

Training logs to [W&B](https://wandb.ai/). Sign in once per machine:

```bash
uv run wandb login
```

## Run the pipeline

The pipeline stages below run in sequence, each reading an explicit artifact written by the
previous stage. Hydra composes the YAML files under `config/` and accepts overrides directly on
the command line.

Every invocation records its resolved configuration beside the stage artifacts described below.

### Run multiple jobs

Hydra multirun executes the Cartesian product of comma-separated values, one job at a time.
Use it to process a batch at any pipeline stage:

```bash
uv run python -m pipelines.preprocess --multirun dataset=sepsis,bpic13,bpic17,bpic19

uv run python -m pipelines.train --multirun \
  dataset=sepsis,bpic13,bpic17,bpic19 \
  model=transformer_cvae,head_sampling_transformer
```

Generation and evaluation can be queued in the same way by listing their input artifacts:

```bash
uv run python -m pipelines.generate --multirun \
  checkpoint=/path/to/first.pt,/path/to/second.pt device=cpu num_samples=100

uv run python -m pipelines.evaluate --multirun \
  generations=/path/to/first/generations.parquet,/path/to/second/generations.parquet workers=4
```

Each multirun job receives its own dataset, model, and run ID directory. Batch runs do not
transfer artifacts between stages automatically, so supply each stage's input artifact
explicitly.

### 1. Preprocessing

Run once per dataset:

```bash
uv run python -m pipelines.preprocess dataset=sepsis
```

The original log is read from `data/sepsis/original.csv`. The out-of-time splits, fitted codec,
and declarative model are written under `data/sepsis/` and reused by later stages. Invocation
records are written under `outputs/preprocess/sepsis/<timestamp>/`.

> [!WARNING]
> Training, generation, and evaluation stop if their required preprocessing artifacts are
> missing.

### 2. Training

Choose the dataset and architecture independently:

```bash
uv run python -m pipelines.train dataset=sepsis model=transformer_cvae
```

The available architectures are `transformer_cvae` and `head_sampling_transformer`. Training
writes the best validation checkpoint to
`outputs/train/<dataset>/<model>/<run-id>/best.pt`. Runs cannot be resumed, but an interrupted run
retains its last successfully saved best checkpoint.

Training curves are logged to the `suffix-generation` W&B project. On normal completion, the
selected checkpoint is also uploaded to W&B.

### 3. Sampler tuning

Tune a Head-sampling Transformer on the validation split before test generation:

```bash
uv run python -m pipelines.tune checkpoint=/path/to/best.pt device=cpu
```

The selected sampler is written to
`outputs/tune/<dataset>/<model>/<run-id>/tuning.json`. This stage does not apply to the
Transformer CVAE.

### 4. Inference

Generate suffixes for every prefix of the test split:

```bash
uv run python -m pipelines.generate checkpoint=/path/to/best.pt device=cpu num_samples=100
```

For a tuned Head-sampling Transformer, pass the tuning report explicitly:

```bash
uv run python -m pipelines.generate checkpoint=/path/to/best.pt \
  tuning=/path/to/tuning.json device=cpu num_samples=100
```

The generations are written to
`outputs/generate/<dataset>/<model>/<run-id>/generations.parquet`.

### 5. Evaluation

Evaluate a generations file:

```bash
uv run python -m pipelines.evaluate generations=/path/to/generations.parquet workers=4
```

The report and its per-prefix scores are written under
`outputs/evaluate/<dataset>/<model>/<run-id>/` as `evaluation.json` and
`prefix_scores.parquet`.

### 6. Visualization

Plot and tabulate one or more evaluation reports:

```bash
uv run python -m pipelines.visualize \
  'evaluations=[/path/to/first/evaluation.json,/path/to/second/evaluation.json]'
```

Keep every `evaluation.json` beside its `prefix_scores.parquet`. Figures are written as PDF under
`outputs/visualize/<date>/<time>/figures/`, and comparison tables as LaTeX under
`outputs/visualize/<date>/<time>/tables/`.

To visualize every report below one or more directories instead:

```bash
uv run python -m pipelines.visualize 'evaluations_dir=[outputs/evaluate,pinned]'
```

## Published checkpoints

Download all published checkpoints to `pretrained/<dataset>/<model>.pt`:

```bash
uv run python -m scripts.fetch
```

## Configuration

Datasets, models, training defaults, and runtime profiles live in the corresponding groups under
`config/`. Override individual settings with dotted keys:

```bash
uv run python -m pipelines.train dataset=bpic17 model=head_sampling_transformer \
  optimizer.lr=0.0005 training.device=cuda:0
```

Inspect the fully resolved configuration without starting a run:

```bash
uv run python -m pipelines.train dataset=sepsis model=transformer_cvae --cfg job --resolve
```

## Maintainer operations

### Publish a checkpoint

Once a run has been evaluated, propose its checkpoint as a published model:

```bash
uv run python -m scripts.publish -m /path/to/best.pt
```

This opens a pull request against the Hugging Face model repository.
