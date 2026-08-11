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

The repository is designed to make experiment results fully reproducible. The four pipelines below run in sequence, each reading what the previous one wrote. Every command takes `-c`/`--config`, the path to the dataset's experiment config YAML in `config/`.

### 1. Preprocessing

Run once per dataset, before anything else:

```bash
python -m pipelines.preprocess -c config/<dataset>.yaml
```

The out-of-time splits as well as the fitted codec and the declarative model are written to `data/<dataset>/`.

> [!WARNING]
> Training and generation read these outputs and will stop with an error naming what's missing if the dataset hasn't been preprocessed yet.

### 2. Training

Once the dataset is preprocessed, start a new run or resume one already started:

```bash
python -m pipelines.train -c config/<dataset>.yaml
python -m pipelines.train -r <path-to-checkpoint>   # resume instead of starting fresh
```

Exactly one of `-c` (start a new run) or `-r`/`--resume` (carry on from a checkpoint, config included) is required. A resumed run keeps its original name, so it continues writing to the same TensorBoard directory and the same files.

> [!NOTE]
> **Skip training:** pre-trained models are available on the Hugging Face model hub. Fetch every published checkpoint into `best-models/` with:
> ```bash
> python -m scripts.fetch
> ```

The training logs are written to `outputs/tensorboard/<name>/<model>/<timestamp>/`. To see the training curves, point TensorBoard at the root:

```bash
tensorboard --logdir outputs/tensorboard
```

Model checkpoints are written to two places:
 - `best-models/<name>/<model>/<timestamp>.pt`: a single file holding the run's last improvement, overwritten each time validation loss improves
 - `outputs/checkpoints/<name>/<model>/<timestamp>.pt` is overwritten at every validation step so the run can resume from where it left off.

### 3. Inference

After training, generate suffixes for the test set:

```bash
python -m pipelines.generate -c config/<dataset>.yaml -m <path-to-model>
```

`-m`/`--model` points to the model to generate with, from either `best-models/` or `outputs/checkpoints/`. 

The generated suffixes for every prefix of the test split are written to `outputs/generations/<name>/<model>/<timestamp>.parquet`, named after the run the checkpoint carries.

### 4. Evaluation

Reads the generated suffixes and writes an evaluation report:

```bash
python -m pipelines.evaluate -c config/<dataset>.yaml -g <path-to-generations> -j <number-of-jobs>
```

- `-g`/`--generations` points to the generations file to score, produced by `pipelines.generate`. 
- `-j`/`--workers` sets how many processes to score with, defaulting to one per available CPU. 

The resulting report is written to `outputs/eval/<name>/<model>/<timestamp>.json`.

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

A dataset config declares everything a run needs: where to find the raw log and how to read it, the model architecture, and every training and inference hyperparameter. All configs are validated against `src/configs/schema.py`.

### Config inheritance

Every dataset config in `config/` (`sepsis.yaml`, `bpic17.yaml`, `bpic19.yaml`, ...) is deep-merged over the sibling `config/base.yaml`, which holds the model architecture and the shared training, optimizer, loss, and inference settings. A dataset config only needs to state its `data` section, `seed`, and the `declare` block if it deviates from the defaults, and nested dicts are merged key by key, so a config can override a single field of a nested section without repeating the rest.

### The `data` section

Names the raw log to build the run from — a CSV at `data/<name>/original.csv`. Its keys point to that CSV's columns: `case_key` for the case identifier, `activity_key` for the activity, `resource_key` for the resource, `timestamp_key` for the timestamp, and `event_features` for any other columns used as the model's categorical or numeric per-event inputs.

Preprocessing reads this file, and everything downstream builds on what it writes.