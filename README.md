# C-VAE for Suffix Generation

A Conditional Variational Autoencoder (C-VAE) for generating suffixes based on given prefixes.

This repository provides a comprehensive implementation of the C-VAE architecture, along with training scripts, evaluation metrics, configuration files, and pre-trained models — built to support research and experimentation in **predictive process monitoring**.

---

## Install

**Requirements:** Python 3.13+, [uv](https://docs.astral.sh/uv/) and [Git LFS](https://git-lfs.com)

The datasets under `data/` (`original.csv` for each) are tracked with Git LFS, so a plain clone
only checks out pointer files. Install Git LFS once per machine and pull them before preprocessing:

```bash
git lfs install
git lfs pull
```

```bash
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv sync                     # installs the locked dependencies into .venv
```

---

## Reproducibility

The repository is designed to make experiment results fully reproducible. The four pipelines below run in sequence, each reading what the previous one wrote. Every command takes `-c`/`--config`, the name of the dataset's experiment config YAML in `config/datasets/` (e.g. `bpic17`). Training and generation, which build a model and a `DataLoader`, also take `-w`/`--hardware`, the name of a hardware profile YAML in `config/hardware/` (e.g. `mps`, `cuda-t4`); preprocessing and evaluation never read a hardware-dependent value, so they don't take it.

### 1. Preprocessing

Run once per dataset, before anything else:

```bash
python -m pipelines.preprocess -c <dataset>
```

The out-of-time splits as well as the fitted codec and the declarative model are written to `data/<dataset>/`.

> [!WARNING]
> Training and generation read these outputs and will stop with an error naming what's missing if the dataset hasn't been preprocessed yet.

Declarative model discovery is the slowest step and only evaluation reads its output, so it can be skipped with `--skip-discovery` when preprocessing for training or generation alone.

### 2. Training

Once the dataset is preprocessed, start a new run or resume one already started:

```bash
python -m pipelines.train -c <dataset> -w <hardware>
python -m pipelines.train -r <path-to-checkpoint>   # resume instead of starting fresh
```

Exactly one of `-c` (start a new run) or `-r`/`--resume` (carry on from a checkpoint, config included) is required. `-w`/`--hardware` is required alongside `-c`, and not used with `-r`, whose config is already resolved. A resumed run keeps its original name, so it continues writing to the same TensorBoard directory and the same files.

> [!NOTE]
> **Skip training:** pre-trained models are available on the Hugging Face model hub. Fetch every published model into `pretrained/` with:
> ```bash
> python -m scripts.fetch
> ```
> There is one file per model per log, at `pretrained/<name>/<model>.pt`. They are trimmed to what generation reads, so they can be generated from but not resumed from.

The training logs are written to `outputs/tensorboard/<name>/<model>/<timestamp>/`. To see the training curves, point TensorBoard at the root:

```bash
tensorboard --logdir outputs/tensorboard
```

Model checkpoints are written to two places:
 - `outputs/checkpoints/best/<name>/<model>/<timestamp>.pt`: a single file holding the run's last improvement, overwritten each time the selection score improves
 - `outputs/checkpoints/last/<name>/<model>/<timestamp>.pt` is overwritten at every validation step so the run can resume from where it left off.

### 3. Inference

After training, generate suffixes for the test set:

```bash
python -m pipelines.generate -c <dataset> -w <hardware> -m <path-to-model>
```

`-m`/`--model` points to the model to generate with, from `pretrained/`, `outputs/checkpoints/best/` or `outputs/checkpoints/last/`. 

The generated suffixes for every prefix of the test split are written to `outputs/generations/<name>/<model>/<timestamp>.parquet`, named after the run the checkpoint carries.

### 4. Evaluation

Reads the generated suffixes and writes an evaluation report:

```bash
python -m pipelines.evaluate -c <dataset> -g <path-to-generations> -j <number-of-jobs>
```

- `-g`/`--generations` points to the generations file to score, produced by `pipelines.generate`. 
- `-j`/`--workers` sets how many processes to score with, defaulting to one per available CPU. 

The resulting report is written to `outputs/eval/<name>/<model>/<timestamp>.json`.

### 5. Publishing

Once a run has been evaluated and is worth being the one others reach for, propose its best checkpoint as that model's published version:

```bash
python -m scripts.publish -m <path-to-best-checkpoint>
```

`-m`/`--model` points to the checkpoint to publish, from `outputs/checkpoints/best/`. Which run deserves the name is exactly the decision this step exists to record, so it is named rather than searched for.

The checkpoint is trimmed to what generation reads, dropping the optimizer, early-stopping and RNG state that only `--resume` needs, and uploaded to `<name>/<model>.pt` in the Hugging Face model repo. It goes up as a pull request, printed as a link, and only becomes what `python -m scripts.fetch` hands out once a maintainer merges it. Publishing a second run of the same model on the same log therefore proposes replacing the first: the published set holds one file per model per log, and the run's timestamp is deliberately not part of that name.

---

## Notebooks

---

## Visualization

Once a dataset has been evaluated, the results of one or more runs can be visualized and compared with:

```bash
python -m pipelines.visualize -e <path-to-report> [<path-to-report> ...]
```

- `-e`/`--evaluations` takes the paths to the evaluation reports to compare, from `pipelines.evaluate`; passing several overlays them on the same axes, which is also how models or datasets are compared. 
- `-l`/`--labels` renames each report's series in legends and tables. Two reports sharing a label are read as one model shown on two datasets. 
- `--dataset-labels bpic17=BPIC17` renames a dataset in the tables only, since figures are already split one per dataset directory. 
- `-f`/`--formats` picks the image format(s) to write (`pdf`, `svg`, `png`; default `pdf`).
- `--coverage` bounds the x-axis to the share of prefix pairs it must cover, cutting off the sparse tail of long prefixes — `1.0` draws every length.

The figures and comparison tables are written to `outputs/plots/`.

---

## Configs

A dataset config declares everything a run needs: where to find the raw log and how to read it, the model architecture, and every training and inference hyperparameter. All configs are validated against the models in `src/configs/schema/`, one module per section.

### Config inheritance

Two sections, `data` and `declare`, are hardware-independent: they're assembled from two layers, deep-merged in order, each taking precedence over the last:

1. `config/base.yaml` — hardware- and dataset-agnostic defaults, including all of `declare`.
2. `config/datasets/<dataset>.yaml` — the `-c`/`--config` dataset config, selected by name (e.g. `-c sepsis` loads `sepsis.yaml`). Owns the `data` section's dataset-specific keys (columns, splits, features) and any dataset-specific overrides, such as `sepsis.yaml`'s model, sized down for a log two orders of magnitude smaller than the bpic ones.

This is what `pipelines.preprocess` and `pipelines.evaluate` load, since neither reads anything beyond `data`/`declare`, and so neither needs `-w`/`--hardware`.

Every other section — `model`, `loss`, `optimizer`, `training`, `dataloader`, `early_stopping`, `inference` — adds a third layer in between:

2. `config/hardware/<hardware>.yaml` — the `-w`/`--hardware` profile, e.g. `mps.yaml` or `cuda-t4.yaml`. Owns everything that varies with the machine a run executes on: `training.device`, `dataloader.batch_size`, `dataloader.num_workers`, `inference.generation_rows_upper_bound`, and the batch-size-derived `optimizer.lr`, `training.max_steps`, `training.val_every_n_steps`, and `loss.kl_annealing_period_steps`.

This is what `pipelines.train` and `pipelines.generate` load, since both build a model and a `DataLoader` and so need `-w`/`--hardware`.

Nested dicts are merged key by key, so any layer can override a single field of a nested section without repeating the rest — some sections, like `training` and `inference`, get some of their keys from `base.yaml` and others from the hardware profile.

### Hardware profiles

Three profiles are checked in under `config/hardware/`:

- `mps.yaml` — local development on Apple-silicon GPUs.
- `cuda-t4.yaml` — an Azure T4 VM: larger batch size and worker count, and the learning rate, step count, and KL-annealing period scaled to match.
- `cuda-a6000.yaml`: a dual RTX A6000 workstation (48GB per GPU), batch size and every derived hyperparameter at twice the T4 profile, pinned to one of the two cards.

Add a new profile by dropping a `<name>.yaml` file into `config/hardware/` with the same keys; it becomes selectable as `-w <name>` immediately.

**Picking a GPU.** `training.device` takes `cuda:<n>` as well as a bare `cuda`, so a profile on a multi-GPU machine names the card it runs on rather than leaving it to whichever device happens to be current. To put a run on the other card, either edit that field or launch with `CUDA_VISIBLE_DEVICES=1`, which hides the first card and remaps the second to `cuda:0`; the second form is what lets two experiments run concurrently, one per GPU, without a second profile.

**`max_steps` is a ceiling, not the expected end of a run.** Early stopping is what normally ends one, and since `early_stopping.patience` counts validations rather than steps, it only means something relative to `max_steps / val_every_n_steps`. Every profile is therefore sized for 120 validations per run, so the checked-in patience of 30 catches a plateau a quarter of the way in and the same setting means the same thing on every machine. A profile that shrinks that ratio far enough makes early stopping unreachable and silently hands the decision back to `max_steps`.

### The `data` section

Names the raw log to build the run from — a CSV at `data/<name>/original.csv`. Its keys point to that CSV's columns: `case_key` for the case identifier, `activity_key` for the activity, `resource_key` for the resource, `timestamp_key` for the timestamp, and `event_features` for any other columns used as the model's categorical or numeric per-event inputs.

Preprocessing reads this file, and everything downstream builds on what it writes.