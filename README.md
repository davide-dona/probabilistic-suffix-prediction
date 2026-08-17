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

The repository is designed to make experiment results fully reproducible. The four pipelines below run in sequence, each reading what the previous one wrote.

Every argument that names a file is a path to it, so there is one form to learn and everything completes in a shell. Preprocessing and training take `-c`/`--config`, the dataset's experiment config YAML (e.g. `config/datasets/bpic17.yaml`). Training and generation, which build a model and a `DataLoader`, take `-w`/`--hardware`, a hardware profile YAML (e.g. `config/hardware/mps.yaml`). Generation and evaluation read no config file at all: a checkpoint carries the config of the run that wrote it, and a generations file carries its own run and dataset identity.

### 1. Preprocessing

Run once per dataset, before anything else:

```bash
python -m pipelines.preprocess -c config/datasets/<dataset>.yaml
```

The out-of-time splits as well as the fitted codec and the declarative model are written to `data/<dataset>/`.

> [!WARNING]
> Training and generation read these outputs and will stop with an error naming what's missing if the dataset hasn't been preprocessed yet.

Declarative model discovery is the slowest step and only evaluation reads its output, so it can be skipped with `--skip-discovery` when preprocessing for training or generation alone.

### 2. Training

Once the dataset is preprocessed, start a new run or resume one already started:

```bash
python -m pipelines.train -c config/datasets/<dataset>.yaml -w config/hardware/<hardware>.yaml
python -m pipelines.train -r <path-to-checkpoint>   # resume instead of starting fresh
```

Exactly one of `-c` (start a new run) or `-r`/`--resume` (carry on from a checkpoint, config included) is required. `-w`/`--hardware` is required alongside `-c`, and rejected with `-r`: a resumed run keeps the batch size, learning rate and annealing schedule it started with, all of which a profile carries. A resumed run also keeps its original name, so it continues writing to the same TensorBoard directory and the same files.

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
python -m pipelines.generate -m <path-to-checkpoint> -w config/hardware/<hardware>.yaml
```

- `-m`/`--checkpoint` points to the checkpoint to generate with, from `pretrained/`, `outputs/checkpoints/best/` or `outputs/checkpoints/last/`. No dataset config is passed alongside it: the checkpoint carries the config of the run that wrote it, so the model, the dataset and the sampling are already settled and cannot be made to disagree with it.
- `-w`/`--hardware` is the profile to generate under, replacing the one the run was trained with — a run trained on a workstation is routinely generated from on a laptop.
- `-n`/`--num-samples` overrides how many suffixes are drawn per prefix for this generation alone. Defaults to the run's own `inference.num_samples`.

The generated suffixes for every prefix of the test split are written to `outputs/generations/<name>/<model>/<timestamp>.parquet`, named after the run the checkpoint carries.

### 4. Evaluation

Reads the generated suffixes and writes an evaluation report:

```bash
python -m pipelines.evaluate -g <path-to-generations> -j <number-of-jobs>
```

- `-g`/`--generations` points to the generations file to score, produced by `pipelines.generate`. Its embedded run identity says which dataset's declarative model to check conformance against, so no config is needed.
- `-j`/`--workers` sets how many processes to score with, defaulting to one per available CPU. 

Conformance is checked the way the declarative models were discovered, with `declare.consider_vacuity` false: a constraint a trace never activates counts as violated. That is not a flag, since it is a property of the models under `data/*/declare/` rather than a choice made at scoring time.

The resulting report is written to `outputs/eval/<name>/<model>/<timestamp>.json`.

### 5. Publishing

Once a run has been evaluated and is worth being the one others reach for, propose its best checkpoint as that model's published version:

```bash
python -m scripts.publish -m <path-to-best-checkpoint>
```

`-m`/`--checkpoint` points to the checkpoint to publish, from `outputs/checkpoints/best/`. Which run deserves the name is exactly the decision this step exists to record, so it is named rather than searched for.

The checkpoint is trimmed to what generation reads, dropping the optimizer, early-stopping and RNG state that only `--resume` needs, and uploaded to `<name>/<model>.pt` in the Hugging Face model repo. It goes up as a pull request, printed as a link, and only becomes what `python -m scripts.fetch` hands out once a maintainer merges it. Publishing a second run of the same model on the same log therefore proposes replacing the first: the published set holds one file per model per log, and the run's timestamp is deliberately not part of that name.

---

## Notebooks

---

## Visualization

Once a dataset has been evaluated, the scores of one or more runs can be plotted and tabulated with:

```bash
python -m pipelines.visualize -e <path-to-report> [<path-to-report> ...]
python -m pipelines.visualize -E outputs/eval
python -m pipelines.visualize -E outputs/eval -G outputs/generations
```

- `-e`/`--evaluations` takes the paths to the evaluation reports to compare, from `pipelines.evaluate`; passing several overlays them on the same axes, which is also how models or datasets are compared. These draw the per-length metric figures and the comparison tables.
- `-E`/`--evaluations-dir` instead compares every report under a directory, at any depth: `outputs/eval` for a whole set of results, `outputs/eval/bpic17` for one dataset. Each report says which model and dataset it belongs to, so nothing has to be typed alongside it. Two runs of one model on one dataset are an error, since a figure cannot draw them apart, so keep the directory to the runs being reported. `-e` and `-E` cannot be combined, and one of them is required.
- `-g`/`--generations` and `-G`/`--generations-dir` add the distribution figure, drawn from the generated suffixes themselves rather than from the scores they earned. It is opt-in because it reads the generations and costs a minute or two per log; left out, every other figure is still drawn. The two cannot be combined.

The figures are written to `outputs/visual/figures/<dataset>/` as PDF, which is what the paper takes, and the comparison tables to `outputs/visual/tables/` as `tex`. With generations, each dataset also gets `distribution.pdf`, a UMAP of the suffixes each model generates against the ground truth, one panel per model over one shared embedding.

Nothing about how a run is named or drawn is typed on the command line. What a model, a dataset and a metric are called, the colour, marker and line style a model keeps in every figure, and the colour a dataset keeps, are declared in `src/visualization/labels/`, so a model is the same colour in every figure of every log and two runs of the pipeline produce the same page. A model, a dataset or a metric with nothing declared for it stops the run, naming what to add and where. The colours are the subset of the Okabe-Ito palette that stays furthest apart under simulated colour blindness, split so that a dataset takes none of the three the models hold, and every model also carries its own marker and line style, so the figures survive both a colourblind reader and a black-and-white print. The order the two are declared in is the order they are drawn and tabulated in: a model's line and column, and a dataset's rows. Two names sharing a style are one model, which is how the CVAE trained at a smaller size for a smaller log draws that log's CVAE line and fills its CVAE column rather than standing beside it as a second model. Which figures are drawn is the `FIGURES` catalogue in `src/visualization/figures.py`, each entry a group of related metrics against either prefix or suffix length.

---

## Configs

A dataset config declares everything a run needs: where to find the raw log and how to read it, the model architecture, and every training and inference hyperparameter. All configs are validated against the models in `src/configs/schema/`, one module per section.

### Config inheritance

Two sections, `data` and `declare`, are hardware-independent: they're assembled from two layers, deep-merged in order, each taking precedence over the last:

1. `config/base.yaml` — hardware- and dataset-agnostic defaults, including all of `declare`.
2. `config/datasets/<dataset>.yaml` — the `-c`/`--config` dataset config, named by path (e.g. `-c config/datasets/sepsis.yaml`). Owns the `data` section's dataset-specific keys (columns, splits, features) and any dataset-specific overrides, such as `sepsis.yaml`'s model, sized down for a log two orders of magnitude smaller than the bpic ones. Its filename means nothing beyond being what you type: `data.name` and `model.name` inside it are what name a run and its output directories.

This is what `pipelines.preprocess` loads, since it reads nothing beyond `data`/`declare` and so needs no `-w`/`--hardware`. `pipelines.evaluate` reads no config at all: the generations file it scores already says which run and dataset produced it.

Every other section — `model`, `loss`, `optimizer`, `training`, `dataloader`, `early_stopping`, `inference` — adds a third layer in between:

2. `config/hardware/<hardware>.yaml` — the `-w`/`--hardware` profile, e.g. `config/hardware/mps.yaml`. Owns everything that varies with the machine a run executes on: `training.device`, `dataloader.batch_size`, `dataloader.num_workers`, `inference.generation_rows_upper_bound`, and the batch-size-derived `optimizer.lr`, `training.max_steps`, `training.val_every_n_steps`, and `loss.kl_annealing_period_steps`.

This is what `pipelines.train` loads, since it builds a model and a `DataLoader` and so needs `-w`/`--hardware`. `pipelines.generate` needs a profile too, but merges it over the config stored inside the checkpoint rather than over a dataset config: the run's model and data are settled, and only the machine has changed.

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