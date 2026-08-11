# C-VAE for Suffix Generation

A Conditional Variational Autoencoder (C-VAE) for generating suffixes based on given prefixes.

This repository provides a comprehensive implementation of the C-VAE architecture, along with training scripts, evaluation metrics, configuration files, and pre-trained models — built to support research and experimentation in **predictive process monitoring**.

---

## Install

**Requirements:** Python 3.13+ and [uv](https://docs.astral.sh/uv/)

```bash
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv sync                     # installs the locked dependencies into .venv
```

---

## Reproducibility

The repository is designed to make experiment results fully reproducible.
To reproduce the experiments, follow the steps below using the configuration files provided in `config/`.

---

### 1. Preprocessing

Run **once per dataset**, before anything else.

```bash
python -m pipelines.preprocess -c config/<dataset>.yaml
```

**Parameters**

| Flag | Meaning |
|---|---|
| `-c`, `--config` | Path to this dataset's experiment config YAML. |

**Outputs**

| Path | Contents |
|---|---|
| `data/<dataset>/processed/{train,val,test}.csv` | The out-of-time train/val/test splits. |
| `data/<dataset>/codec/dataset.json` | The vocabularies and normalization statistics fit on the train split, used to encode and decode every downstream run. |
| `data/<dataset>/declare/model.decl` | The declarative model discovered from the train split. |

> [!WARNING]
> Training and generation read these outputs and will stop with an error naming what's missing if the dataset hasn't been preprocessed yet.

---

### 2. Training

Once the dataset is preprocessed, start a new run or resume one already started:

```bash
python -m pipelines.train -c config/<dataset>.yaml
```

**Parameters**

| Flag | Meaning |
|---|---|
| `-c`, `--config` | Path to this experiment's config YAML, to start a new run. |
| `-r`, `--resume` | Path to a checkpoint to carry on from, its config included. The run keeps its name, so it writes to the same TensorBoard directory and the same files. |

Exactly one of `-c` / `-r` is required.

> [!NOTE]
> **Skip training:** Pre-trained models are available on the Hugging Face model hub. Fetch every published checkpoint into `best-models/` with:
> ```bash
> python -m scripts.fetch
> ```

**Outputs**

| Path | Contents |
|---|---|
| `outputs/tensorboard/<name>/<model>/<timestamp>/` | Loss and its terms under `train/` and `val/`, plus `kl_weight`. Point TensorBoard at the root to see every run as its own toggleable set of curves, grouped by dataset and model: `tensorboard --logdir outputs/tensorboard` |
| `best-models/<name>/<model>/<timestamp>.pt` | The run's result — a single file, overwritten each time validation loss improves. Contains the run's last improvement. |
| `outputs/checkpoints/<name>/<model>/<timestamp>.pt` | Checkpoints written (and overwritten) at every validation step, enabling the run to resume from the last checkpoint. |

---

### 3. Inference

After training, generate suffixes for the test set:

```bash
python -m pipelines.generate -c config/<dataset>.yaml -m <path-to-model>
```

**Parameters**

| Flag | Meaning |
|---|---|
| `-c`, `--config` | Path to this experiment's config YAML. |
| `-m`, `--model` | Path to the checkpoint to generate with, from `best-models/` or `outputs/checkpoints/`. |

**Outputs**

| Path | Contents |
|---|---|
| `outputs/generations/<name>/<model>/<timestamp>.parquet` | The generated suffixes for every prefix of the test split, named after the run the checkpoint carries. |

---

### 4. Evaluation

Reads the generated suffixes and writes an evaluation report:

```bash
python -m pipelines.evaluate -c config/<dataset>.yaml -g <path-to-generations> -j <number-of-jobs>
```

**Parameters**

| Flag | Meaning |
|---|---|
| `-c`, `--config` | Path to the experiment config the generations were written under. |
| `-g`, `--generations` | Path to the generations file to score, from `pipelines.generate`. |
| `-j`, `--workers` | How many processes to score with. Defaults to one per available CPU. |

**Outputs**

| Path | Contents |
|---|---|
| `outputs/eval/<name>/<model>/<timestamp>.json` | The evaluation report scoring that run's generations. |

---

## Notebooks

Two notebooks under `notebooks/` **read** what the pipelines wrote — they write nothing themselves. Each is driven by a constant set in its second cell.

| Notebook | Purpose | Set this constant |
|---|---|---|
| `data_exploration.ipynb` | A log and its splits — from the raw overview down to the distribution of every event attribute. | `DATASET` → any preprocessed dataset |
| `generations.ipynb` | One run's suffixes, read trace by trace, plus length, activity, diversity, and remaining-time diagnostics not covered by the report. | `RUN` → the run whose generations to open |

---

## Visualization

Once a dataset has been evaluated, turn one or more evaluation reports into the paper's figures and comparison tables:

```bash
python -m pipelines.visualize -e <path-to-report> [<path-to-report> ...]
```

Passing several reports overlays them on the same axes — this is also how models or datasets are compared.

**Parameters**

| Flag | Meaning |
|---|---|
| `-e`, `--evaluations` | Paths to the evaluation reports to compare, from `pipelines.evaluate`. |
| `-l`, `--labels` | Renames each report's series in legends and tables. Two reports sharing a label are read as one model shown on two datasets. |
| `--dataset-labels bpic17=BPIC17` | Renames a dataset in the **tables only** (figures are already split one per dataset directory). |
| `-f`, `--formats` | Picks image format(s) to write: `pdf`, `svg`, `png` (default: `pdf`). |
| `--coverage` | Bounds the x-axis to the share of prefix pairs it must cover, cutting off the sparse tail of long prefixes. `1.0` draws every length. |

**Outputs**

| Path | Contents |
|---|---|
| `outputs/plots/` | The paper's figures and comparison tables. |

---

## Configs

A dataset config declares everything a run needs:
- Where to find the raw log and how to read it
- The model architecture
- Every training and inference hyperparameter

All configs are validated against `src/configs/schema.py`.

### Config inheritance

Every dataset config in `config/` (`sepsis.yaml`, `bpic17.yaml`, `bpic19.yaml`, ...) is **deep-merged** over the sibling `config/base.yaml`, which holds the model architecture and the shared training, optimizer, loss, and inference settings.

- A dataset config only needs to state its `data` section, `seed`, and the `declare` block **if it deviates from the defaults**.
- Nested dicts are merged key by key, so a config can override a single field of a nested section without repeating the rest.

### The `data` section

Names the raw log to build the run from — a CSV at:

```
data/<name>/original.csv
```

| Config key | Column it names |
|---|---|
| `case_key` | Case identifier |
| `activity_key` | Activity |
| `resource_key` | Resource |
| `timestamp_key` | Timestamp |
| `event_features` | Any other columns used as the model's categorical/numeric per-event inputs |

Preprocessing reads this file, and everything downstream builds on what it writes.