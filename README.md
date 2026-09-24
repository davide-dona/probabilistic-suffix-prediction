# Evaluating Possible Futures: Rethinking the Assessment of Probabilistic Suffix Prediction

This work proposes an **evaluation framework** for **probabilistic suffix prediction** and compares two new models with the current state of the art, **U-ED-LSTM**.
**SuTraN-VAE** samples a prefix-conditioned latent variable once per suffix.
**SuTraN-PH** samples activities and times from probabilistic output heads at each step.
[**U-ED-LSTM**](https://arxiv.org/abs/2505.21339) combines Monte Carlo dropout with autoregressive sampling from learned event distributions.

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

Run the stages in order: **preprocess → train → tune (SuTraN-PH only) → generate → evaluate → visualize**.
Each command records its resolved configuration with its outputs. Pass the artifact from one
stage explicitly to the next.

### 1. Preprocess

Prepare the dataset once:

```bash
uv run python -m pipelines.preprocess dataset=sepsis
```

**Input:** `data/sepsis/original.csv`

**Output:** out-of-time splits, fitted codec, and declarative model under `data/sepsis/`;
invocation records under `outputs/preprocess/sepsis/<timestamp>/`.

### 2. Train

Choose a dataset and model:

```bash
uv run python -m pipelines.train dataset=sepsis model=transformer_cvae
```

**Models:** `transformer_cvae` (SuTraN-VAE) or `head_sampling_transformer` (SuTraN-PH).

**Output:** best validation checkpoint at `outputs/train/<dataset>/<model>/<run-id>/best.pt`.
Training curves and completed checkpoints are logged to W&B. Interrupted runs retain their best
checkpoint but cannot be resumed.

### 3. Tune sampler (SuTraN-PH only)

Select sampling settings on the validation split:

```bash
uv run python -m pipelines.tune checkpoint=/path/to/best.pt device=cpu
```

**Output:** `outputs/tune/<dataset>/<model>/<run-id>/tuning.json`.

### 4. Generate

Sample suffixes for every test prefix:

```bash
uv run python -m pipelines.generate checkpoint=/path/to/best.pt device=cpu num_samples=100
```

For SuTraN-PH, pass the tuning report:

```bash
uv run python -m pipelines.generate checkpoint=/path/to/best.pt \
  tuning=/path/to/tuning.json device=cpu num_samples=100
```

**Output:** `outputs/generate/<dataset>/<model>/<run-id>/generations.parquet`.

### 5. Evaluate

Score the generated suffixes:

```bash
uv run python -m pipelines.evaluate generations=/path/to/generations.parquet
```

**Output:** `evaluation.json` and `prefix_scores.parquet` under
`outputs/evaluate/<dataset>/<model>/<run-id>/`.

### 6. Visualize

Compare evaluation reports:

```bash
uv run python -m pipelines.visualize \
  'evaluations=[/path/to/first/evaluation.json,/path/to/second/evaluation.json]'
```

Keep each `evaluation.json` beside its `prefix_scores.parquet`.

**Output:** PDF figures and LaTeX tables under `outputs/visualize/<date>/<time>/`.

To include all reports in a directory:

```bash
uv run python -m pipelines.visualize 'evaluations_dir=[outputs/evaluate,pinned]'
```

> [!TIP]
> **Run multiple jobs:** Add `--multirun` and comma-separated values to any stage, for example:
> `uv run python -m pipelines.train --multirun dataset=sepsis,bpic13 model=transformer_cvae,head_sampling_transformer`.
> Each combination runs separately. Pass checkpoints, generations, or reports explicitly to later stages.

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
