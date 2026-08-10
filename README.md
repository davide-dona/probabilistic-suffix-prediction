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
python -m pipelines.preprocess -c <path-to-config>
```

**Writes:**
- Splits + dataset codec → `data/<name>/processed/`
- Discovered declarative model → `data/<name>/declare/model.decl`

> [!WARNING] 
> Training and generation read these outputs and will stop with an error naming what's missing if the dataset hasn't been preprocessed yet.

---

### 2. Training

Once the dataset is preprocessed:

```bash
python -m pipelines.train -c <path-to-config>
```

> [!NOTE]
> **Skip training:** Pre-trained models are available on the Hugging Face model hub. Fetch every published checkpoint into `best-models/` with:
> ```bash
> python -m scripts.fetch
> ```

#### Outputs

| Path | Contents |
|---|---|
| `outputs/tensorboard/<name>/<model>/<timestamp>/` | Loss and its terms under `train/` and `val/`, plus `kl_weight`. Point TensorBoard at the root to see every run as its own toggleable set of curves, grouped by dataset and model: `tensorboard --logdir outputs/tensorboard` |
| `best-models/<name>/<model>/<timestamp>.pt` | The run's result — a single file, overwritten each time validation loss improves. Contains the run's last improvement. |
| `checkpoints/<name>/<model>/<timestamp>.pt` | Checkpoints written (and overwritten) at every validation step, enabling the run to resume from the last checkpoint. |

---

### 3. Inference

After training, generate suffixes for the test set:

```bash
python -m pipelines.generate -c <path-to-config> -m <path-to-model>
```

---

### 4. Evaluation

Reads the generated suffixes and writes an evaluation report:

```bash
python -m pipelines.evaluate -c <path-to-config> -g <path-to-generations> -j <number-of-jobs>
```

**Output:** `outputs/evaluation/<name>/<model>/<timestamp>/report.json`

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

| Flag | Effect |
|---|---|
| `-l` | Renames each report's series in legends and tables. Two reports sharing a label are read as one model shown on two datasets. |
| `--dataset-labels bpic17=BPIC17` | Renames a dataset in the **tables only** (figures are already split one per dataset directory). |
| `-f` | Picks image format(s) to write: `pdf`, `svg`, `png` (default: `pdf`). |
| `--coverage` | Bounds the x-axis to the share of prefix pairs it must cover, cutting off the sparse tail of long prefixes. `1.0` draws every length. |

**Output:** `outputs/plots/`

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