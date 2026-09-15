# Suffix Generation

Conditional suffix generation for predictive process monitoring. The repository compares a
Transformer CVAE with a Head-sampling Transformer.

## Install

Python 3.13+, uv, and Git LFS are required. Dataset CSVs are stored with Git LFS.

```bash
git lfs install
git lfs pull
uv sync --locked
uv run wandb login
```

## Configuration

Hydra composes plain YAML under `config/`. Choose a dataset and model independently:

```bash
uv run python -m pipelines.train dataset=sepsis model=transformer_cvae
uv run python -m pipelines.train dataset=bpic17 model=head_sampling_transformer optimizer.lr=0.0005
```

The groups are `dataset`, `model`, `training`, and `runtime`. The two architectures share
`model/backbone.yaml`; optimizer, stopping, and inference defaults live in
`training/default.yaml`. `runtime=cuda` uses CUDA and online W&B. Override individual settings
with dotted keys.
Unknown ordinary override keys are rejected; semantic checks catch invalid dimensions,
split fractions, budgets, and sampling parameters.

Inspect settings without running a pipeline:

```bash
uv run python -m pipelines.train dataset=sepsis model=transformer_cvae --cfg job --resolve
```

Hydra multirun uses the basic sequential launcher. Sweep every combination of comma-separated
values in one command:

```bash
uv run python -m pipelines.train --multirun dataset=sepsis,bpic13 \
  model=transformer_cvae,head_sampling_transformer
```

To use GPUs concurrently, start one multirun command per GPU in separate terminals. Each command
runs its own jobs sequentially on the specified device:

```bash
uv run python -m pipelines.train --multirun dataset=sepsis,bpic13 \
  model=transformer_cvae training.device=cuda:0
uv run python -m pipelines.train --multirun dataset=sepsis,bpic13 \
  model=head_sampling_transformer training.device=cuda:1
```

## Pipeline

Preprocess once per dataset. Existing prepared data can be reused if its preprocessing settings
have not changed. The codec, splits, continuation indices, and declarative model remain under
`data/<dataset>/`.

```bash
uv run python -m pipelines.preprocess dataset=sepsis
uv run python -m pipelines.train dataset=sepsis model=transformer_cvae
```

Each invocation writes to `outputs/<stage>/<date>/<time-with-microseconds>/`; multirun jobs receive
separate numbered directories. Hydra records the configuration and overrides in `.hydra/`.
`config.yaml` holds the fully resolved effective settings; `invocation.json` records the source
revision, dirty state, command, and Python version. Relative inputs remain relative to the
launch directory. Override `hydra.run.dir` for an explicit destination; an already reserved
invocation directory is rejected.

Training writes `best.pt` whenever validation DLS energy score improves. It contains weights, the
resolved training config, selection metric/direction, score, step, and W&B ID. W&B groups runs by
dataset/model and uploads the selected checkpoint on normal completion. Runs cannot be resumed;
interrupted runs retain their last successfully saved best checkpoint.

Use explicit input artifact paths for the subsequent stages:

```bash
uv run python -m pipelines.generate checkpoint=/path/to/best.pt device=cpu num_samples=100
uv run python -m pipelines.evaluate generations=/path/to/generations.parquet workers=4
uv run python -m pipelines.visualize 'evaluations=[/path/to/evaluation.json]'
```

Batch generation and evaluation use the same sequential multirun interface. List the artifacts
explicitly, and split commands by GPU when generating concurrently:

```bash
uv run python -m pipelines.generate --multirun \
  checkpoint=/path/to/first.pt,/path/to/second.pt device=cuda:0 num_samples=100
uv run python -m pipelines.evaluate --multirun \
  generations=/path/to/first/generations.parquet,/path/to/second/generations.parquet workers=4
```

Generation writes `generations.parquet`. Evaluation writes `evaluation.json` and
`prefix_scores.parquet`. Visualization writes PDFs in `figures/` and LaTeX tables in `tables/`
inside its own invocation directory. Compare explicit reports or use
`'evaluations_dir=[outputs/evaluate,pinned]'`; include only one run per dataset/model.
Move each evaluation report together with its `prefix_scores.parquet`.

Head-sampling Transformer tuning uses the validation split, before test generation:

```bash
uv run python -m pipelines.tune checkpoint=/path/to/head_sampling_transformer/best.pt device=cpu
uv run python -m pipelines.generate checkpoint=/path/to/head_sampling_transformer/best.pt \
  tuning=/path/to/tuning.json
```

Override the grid with `temperatures=[0.9,1.0,1.1]` and `top_ps=[0.95,1.0]`; `pairs` and `samples`
control the validation budget. Generation accepts an explicit
`'sampling={temperature:1.1,top_p:0.95}'` instead of a tuning report. Both stages accept
`num_workers=0`. Sampling overrides apply only to the Head-sampling Transformer.

Artifacts carry dataset/model labels and source content hashes. A tuning report must match the
exact checkpoint bytes, even after files are moved. There is no custom run identity or
path-derived model selection.

## Model selection

Checkpoint selection, early stopping, and sampler tuning minimize activity-sequence DLS energy
score:

```text
mean_i d(sample_i, truth) - 0.5 * mean_{i != j} d(sample_i, sample_j)
```

`d` is normalized Damerau-Levenshtein distance over the activity suffix and its terminal token.
The pairwise term uses `N(N-1)` and retains duplicate-draw multiplicities. Scores are first
reduced within each prefix and then averaged with every prefix receiving equal weight. The same
score is reported on test examples.

Finite-sample energy estimates can be negative; they are not clipped. Strict propriety is not
established for this sequence distance. Both sample-count defaults remain 100, with at least
10 required by the full metric bundle. Sampling and accelerator kernels can introduce variation;
the seed does not promise bitwise equivalence across hardware.

## Evaluation metrics

Reports separate point prediction, sample prediction, calibration, and conformance. “Timestamp
suffix” follows the task name used in the literature; internally it is represented precisely as
the inter-event duration before each predicted event. Timestamp-suffix MAE, CRPS, and coverage
are reduced within each prefix before prefixes are averaged equally. Remaining-time values are
predicted independently rather than reconstructed from inter-event durations.

## Existing artifacts

Legacy prepared datasets, checkpoints, tuning reports, generations, evaluation reports, and
per-prefix score files use different schemas and are not accepted by the new readers. Preserve
them with the code revision that produced them; no legacy reader or automatic conversion is
provided.

Rescoring an old best checkpoint cannot recover earlier training steps that were not saved.
Run preprocessing, training, optional sampler tuning, generation, and evaluation again before
comparing final results.

## Utilities

```bash
uv run python -m scripts.dataset_stats dataset=sepsis
uv run python -m scripts.fetch
uv run python -m scripts.publish -m /path/to/best.pt
scripts/sync_outputs.sh <ssh-host> <remote-repo-path>
```

Fetching downloads the published files as-is; older published checkpoints have the same
compatibility limitation as older local files. Publishing proposes the chosen checkpoint on the
Hugging Face Hub. Synchronization pulls the new `outputs/generate/` tree.
