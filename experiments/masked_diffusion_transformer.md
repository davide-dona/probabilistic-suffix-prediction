# Length-first masked diffusion Transformer

## Hypothesis

Predicting suffix length from the encoded prefix and then denoising the whole fixed-length suffix
in parallel improves activity-sequence distribution fidelity over the Transformer CVAE and the
Head-sampling Transformer.

## Intervention

The experiment replaces the autoregressive suffix generator with a two-stage generator:

1. A categorical head predicts the number of suffix events.
2. A bidirectional Transformer denoises activity masks, standardized inter-event times, and a
   standardized remaining-time token while cross-attending to the same prefix encoder.

The architecture keeps the shared backbone dimensions, prefix event representation, optimizer,
training budget, seed, data splits, checkpoint-selection score, generation sample count, and
evaluation code fixed. The model is trained from scratch. The activity process uses absorbing
mask diffusion; the two time channels use the same cosine signal schedule with Gaussian noise.

## Measurements

Use test `energy_score_dls` as the primary measurement. Inspect point DLS, suffix-length and time
CRPS, calibration gaps, conformance, parameter count, and generation duration as diagnostics.
Compare the one seed-42 diffusion run on each configured dataset with the existing seed-42 CVAE
and Head-sampling Transformer reports. The visualization pipeline aligns identical prefix keys and
uses the paired case bootstrap for table emphasis.

## Interpretation

Davide will inspect the four datasets individually. No automated adoption threshold is part of
the experiment. A gain on only short logs suggests the fixed reverse-process budget or denoiser
capacity does not scale to BPIC17. Better activity fidelity with worse time CRPS points to loss
balance or continuous sampling, not to the length or discrete activity process. Better point DLS
without better energy score indicates insufficient sample diversity or calibration.

The intervention itself changes neither preprocessing nor scoring, so current-format processed
datasets and existing baseline evaluation reports remain semantically valid. The local ignored
codec files predate the `cycle_time` to `inter_event_time` naming migration and must be regenerated
before training. Results condition on one fitted run per architecture and do not measure
training-seed variability.

## Run order

```bash
uv sync --locked

uv run python -m pipelines.preprocess --multirun \
  dataset=sepsis,bpic13,bpic17,bpic19

uv run python -m pipelines.train --multirun \
  dataset=sepsis,bpic13,bpic17,bpic19 \
  model=masked_diffusion_transformer
```

For each printed `best.pt`, generate and evaluate with the same sample budget:

```bash
uv run python -m pipelines.generate checkpoint=/absolute/path/to/best.pt \
  device=cuda:0 num_samples=100
uv run python -m pipelines.evaluate \
  generations=/absolute/path/to/generations.parquet workers=4
```

Finally, pass the four new `evaluation.json` files and the existing CVAE and Transformer reports
to the visualization pipeline:

```bash
uv run python -m pipelines.visualize \
  'evaluations=[/absolute/path/to/report-1/evaluation.json,/absolute/path/to/report-2/evaluation.json]'
```
