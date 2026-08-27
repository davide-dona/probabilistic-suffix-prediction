# C-VAE for Suffix Generation

A Conditional Variational Autoencoder (C-VAE) for generating suffixes based on given prefixes.

This repository provides a comprehensive implementation of the C-VAE architecture, along with training scripts, evaluation metrics, configuration files, and pre-trained models.

The whole project is built to support research and experimentation in **predictive process monitoring**.

---

## Install

**Requirements:** Python 3.13+, [uv](https://docs.astral.sh/uv/) and [Git LFS](https://git-lfs.com)

The datasets under `data/` are tracked with Git LFS. A plain clone
would only checks out pointer files. Install Git LFS once per machine and pull them before preprocessing:

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

The four pipelines below run in sequence, each reading what the previous one wrote. 

Preprocessing and training take `-c`/`--config`, the dataset's experiment config YAML (e.g. `config/datasets/bpic17.yaml`). 

Training and generation, which build a model and a `DataLoader`, take `-w`/`--hardware`, a hardware profile YAML (e.g. `config/hardware/mps.yaml`). 

### 1. Preprocessing

Run once per dataset, before anything else:

```bash
python -m pipelines.preprocess -c config/datasets/<dataset>.yaml
```

The out-of-time splits as well as the fitted codec and the declarative model are written to `data/<dataset>/`.

> [!WARNING]
> Training and generation read these outputs and will stop with an error naming what's missing if the dataset hasn't been preprocessed yet.

The continuation index and the declarative model are read by evaluation alone, and discovery is the slowest step of preprocessing by a wide margin. When preprocessing solely for training or generation, both can be skipped with `--skip-evaluation`.

### 2. Training

Once the dataset is preprocessed, start a new run or resume one already started:

```bash
python -m pipelines.train -c config/datasets/<dataset>.yaml -w config/hardware/<hardware>.yaml
python -m pipelines.train -r <path-to-checkpoint>   # resume instead of starting fresh
```

Exactly one of `-c` (start a new run) or `-r`/`--resume` (carry on from a checkpoint, config included) is required. `-w`/`--hardware` is required alongside `-c`, and rejected with `-r`. 

A resumed run also keeps its original name, so it continues writing to the same TensorBoard directory and the same files.

#### Running a batch on every GPU at once

Testing an experiment usually means training it on every dataset, then generating from every run
it produced. Both are batched the same way: copy the jobs into `queue/` and hand the whole folder
to the machine's GPUs.

```bash
cp config/datasets/bpic17.yaml config/datasets/bpic19.yaml queue/train/
scripts/train_queue.sh -w config/hardware/cuda-a6000.yaml      # -g 0,1 by default

cp outputs/checkpoints/best/bpic17/cvae/*.pt queue/generate/
scripts/generate_queue.sh -w config/hardware/cuda-a6000.yaml   # -n 100 for every job in the batch
```

A training job is a dataset config; a generation job is a copy of a best checkpoint, which carries
the config and the run identity of what wrote it. Both scripts are thin callers of
`scripts/lib/queue.sh`, which is the queue itself.

One job per GPU at a time, and a GPU that finishes picks up the next rather than waiting on the job
beside it. Each is launched with `CUDA_VISIBLE_DEVICES` masking in its own card, so the one profile
is used for both rather than a second one naming `cuda:1`.

Each job's output goes to `outputs/queue/<pipeline>/<job>-<timestamp>.log`, since two of them share
a terminal; the console gets a line per job and a summary at the end. A job that succeeded is
deleted from the queue, and one that failed is renamed `<job>.failed` and kept, to be re-queued by
dropping the suffix once the log has been read. See [`queue/README.md`](queue/README.md).

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

- `-m`/`--checkpoint` points to the checkpoint to generate with, from `pretrained/`, `outputs/checkpoints/best/` or `outputs/checkpoints/last/`. 
- `-w`/`--hardware` is the profile to generate under, replacing the one the run was trained with.
- `-n`/`--num-samples` overrides how many suffixes are drawn per prefix for this generation alone. Defaults to the run's own `inference.num_samples`.

The generated suffixes for every prefix of the test split are written to `outputs/generations/<name>/<model>/<timestamp>.parquet`, named after the run the checkpoint carries.

### 4. Evaluation

Reads the generated suffixes and writes an evaluation report:

```bash
python -m pipelines.evaluate -g <path-to-generations> -j <number-of-jobs>
```

- `-g`/`--generations` points to the generations file to score, produced by `pipelines.generate`.
- `-j`/`--workers` sets how many processes to score with, defaulting to one per available CPU. 

The resulting report is written to `outputs/eval/<name>/<model>/<timestamp>.json`, and the scores of each prefix behind it to the same path with a `.parquet` suffix.

### 5. Publishing

Once a run has been evaluated and is worth being the one others reach for, propose its best checkpoint as that model's published version:

```bash
python -m scripts.publish -m <path-to-best-checkpoint>
```

`-m`/`--checkpoint` points to the checkpoint to publish, from `outputs/checkpoints/best/`. Which run deserves the name is exactly the decision this step exists to record, so it is named rather than searched for.

The checkpoint is trimmed to what generation reads, dropping the optimizer, early-stopping and RNG state that only `--resume` needs, and uploaded to `<name>/<model>.pt` in the Hugging Face model repo.

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

- `-e`/`--evaluations` takes the paths to the evaluation reports to compare, from `pipelines.evaluate`; passing several overlays them on the same axes, which is also how models or datasets are compared.
- `-E`/`--evaluations-dir` instead compares every report under a directory, at any depth: `outputs/eval` for a whole set of results, `outputs/eval/bpic17` for one dataset.
- `-g`/`--generations` and `-G`/`--generations-dir` add the distribution figure.

The figures are written to `outputs/visual/figures/<dataset>/` as PDF, which is what the paper takes, and the comparison tables to `outputs/visual/tables/` as `tex`. With generations, each dataset also gets `distribution.pdf`, a UMAP of the suffixes each model generates against the ground truth, one panel per model over one shared embedding.

---

## Configs

A dataset config declares everything a run needs: where to find the raw log and how to read it, the model architecture, and every training and inference hyperparameter. All configs are validated against the models in `src/configs/schema/`, one module per section.

### Config inheritance

Fields can be overridden between files. Three layers are deep-merged in order, each taking precedence over the last:

1. `config/base.yaml` — default config, independent of any dataset or hardware.
2. `config/hardware/<hardware>.yaml` — the `-w`/`--hardware` profile, e.g. `config/hardware/mps.yaml`. Owns everything that varies with the machine a run executes on.
3. `config/datasets/<dataset>.yaml` — the `-c`/`--config` dataset config, e.g. `config/datasets/sepsis.yaml`. Owns the raw log and any dataset-specific overrides, such as `sepsis.yaml`'s model, sized down for a log two orders of magnitude smaller than the bpic ones.
