# C-VAE for Suffix Generation
In this repository, we present a Conditional Variational Autoencoder (C-VAE) model designed for generating suffixes based on given prefixes.
We provide a comprehensive implementation of the C-VAE architecture, along with training scripts, evaluation metrics, configuration files, and pre-trained models to facilitate research and experimentation in the field of predictive process monitoring.


## Install
Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv sync                    # installs the locked dependencies into .venv
```

## Reproducibility
The repository is designed to ensure reproducibility of our results.

A run is identified by three fields the config declares rather than by any filename: the dataset's
`name`, the model's `name`, and a timestamp. Every artifact carries them inside it and is written
under `<name>/<model>/<timestamp>`, so a run's curves, checkpoints, generations and evaluation
report are all found under the same path, and one dataset's directory says which models were run
on it before any filename is read.

To reproduce the experiments, follow the steps below, using the provided configuration files in the `config/` directory.

### Preprocessing
Run once per dataset, before anything else. It writes the splits and dataset codec under `data/<name>/processed/` and the discovered declarative model to `data/<name>/declare/model.decl`:

```bash
python -m pipelines.preprocess -c <path-to-config>
```
Training and generation read those outputs and stop with an error naming what is missing if the dataset has not been preprocessed.

### Training
Once the dataset is preprocessed, train the model using the following command.
```bash
python -m pipelines.train -c <path-to-config>
```

> [!NOTE]
> Pre-trained models are available for download from the Hugging Face model hub instead of training
> from scratch. Every published checkpoint can be fetched into `best-models/` via:
> ```bash
> python -m scripts.fetch
> ```

#### Outputs

The training pipeline writes the following outputs:
- `outputs/tensorboard/<name>/<model>/<timestamp>/`: the loss and its terms under `train/` and `val/`, plus `kl_weight`.
Point TensorBoard at the root and every run shows up as its own toggleable set of curves, grouped by dataset and model:
  ```bash
  tensorboard --logdir outputs/tensorboard
  ```
- `best-models/<name>/<model>/<timestamp>.pt`: the run's result. One file, overwritten every time the validation loss improves, so the last improvement of the run is what is left in it.

- `checkpoints/<name>/<model>/<timestamp>.pt`: the run's checkpoints, written every validation step and overwritten every time. Allow to resume a run from the last checkpoint.

### Inference
After training, generate suffixes for the test set using the following command:
```bash
python -m pipelines.generate -c <path-to-config> -m <path-to-model>
```

### Evaluation
The evaluation pipeline reads the generated suffixes and writes the evaluation report to `outputs/evaluation/<name>/<model>/<timestamp>/report.json`. Run it with:
```bash
python -m pipelines.evaluate -c <path-to-config> -g <path-to-generations> -j <number-of-jobs>
```

### Notebooks
Two notebooks under `notebooks/` read what the pipelines wrote, and write nothing themselves. Each
is driven by a constant in its second cell:

- `data_exploration.ipynb`: a log and its splits, from the raw overview down to the distribution of
  every event attribute. Set `DATASET` to any preprocessed dataset.
- `generations.ipynb`: one run's suffixes, read trace by trace and then through the length,
  activity, diversity and remaining-time diagnostics the report does not carry. Set `RUN` to the
  run whose generations to open.

### Visualization
Once a dataset has been evaluated, `pipelines.visualize` turns one or more evaluation reports into
the paper's figures and comparison tables, written to `outputs/plots/`:
```bash
python -m pipelines.visualize -e <path-to-report> [<path-to-report> ...]
```
Passing several reports overlays them on the same axes, so this is also how models or datasets are
compared. `-l` renames each report's series in legends and tables (two reports sharing a label are
read as one model shown on two datasets); `--dataset-labels bpic17=BPIC17` renames a dataset in the
tables only, since figures are already split one per dataset directory. `-f` picks which image
formats to write (`pdf`, `svg`, `png`; defaults to `pdf`), and `--coverage` bounds the x-axis to the
share of prefix pairs it must cover, cutting off the sparse tail of long prefixes (`1.0` draws every
length).

### Configs
A dataset config declares everything a run needs: where to find the raw log and how to read it, the
model architecture, and every training and inference hyperparameter. It is validated against
`src/configs/schema.py`.

Every dataset config in `config/` (`sepsis.yaml`, `bpic17.yaml`, `bpic19.yaml`, ...) is deep-merged
over the sibling `config/base.yaml`, which holds the model architecture and the training,
optimizer, loss and inference settings shared across datasets. A dataset config only needs to state
its `data` section, `seed`, and the `declare` block if it deviates from the defaults; nested dicts
are merged key by key, so a config can override a single field of a nested section without
repeating the rest.

The `data` section names the raw log to build the run from: a CSV at `data/<name>/original.csv`
with the case, activity, resource and timestamp columns named in `case_key`, `activity_key`,
`resource_key` and `timestamp_key`, plus whichever other columns are listed under
`event_features` as the model's categorical or numeric per-event inputs. Preprocessing reads this
file and writes everything downstream.
