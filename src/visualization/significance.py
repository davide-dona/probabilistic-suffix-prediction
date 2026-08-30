from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.evaluation.prefix_scores import read_prefix_scores
from src.evaluation.scores import METRICS
from src.identity import RunIdentity, group_by_model, read_run_identity
from src.scalar_metrics import Direction

# What a difference has to clear to be called real, and how many resamples it is read against.
# The resolution of an uncorrected p is `1 / RESAMPLES`, so this many leaves room for the
# correction to divide it by the handful of models a row compares.
ALPHA = 0.05
RESAMPLES = 10_000
# How many resamples are drawn at once. One draw is a row of `[b, cases]`, so this trades the
# memory that matrix takes against the number of passes over the per-case sums.
CHUNK = 250
# What the resampling is reproducible under, so two runs of the pipeline emphasize the same cells.
SEED = 24

# What `test_significance` returns, and what a table's emphasis is read from. One row per model per
# metric per log. `p_value` is the corrected p against that row's reference and is NaN for the
# reference itself; `best` is whether the cell is bold.
SIGNIFICANCE_COLUMNS = ('dataset', 'model', 'metric', 'p_value', 'best')

# The metrics a difference can be tested on: one with no better direction has no best value to be
# indistinguishable from, which is every spread and every property of the log.
_TESTED = tuple(
    key for key, metric in METRICS.entries.items() if metric.direction is not Direction.NONE
)
# How each metric is turned into one where more is better, so a comparison reads the same way
# whichever way the metric does. A gap is read on its distance from 0, both of its signs being a
# way of being wrong, so it is taken absolute and then negated. [num_metrics] each.
_SIGN = np.array(
    [1.0 if METRICS[key].direction is Direction.HIGHER else -1.0 for key in _TESTED],
    dtype=np.float64,
)
_ABSOLUTE = np.array([METRICS[key].direction is Direction.ZERO for key in _TESTED], dtype=bool)


def _oriented(values: np.ndarray) -> np.ndarray:
    """Rewrite a set of scores so that more is better on every metric.

    Args:
        values: Scores whose last axis is the metrics of `_TESTED`, in that order.
    Returns:
        The same shape, each metric negated where lower is better and taken absolute first where
        it is a gap. Applied to a mean rather than to a prefix's own score: the number a table
        prints is the mean gap, so its distance from 0 is what the test reads.
    """
    return _SIGN * np.where(_ABSOLUTE, np.abs(values), values)


def _score_files(reports: Sequence[Path]) -> dict[str, dict[str, Path]]:
    """Find the per-prefix scores beside each report and group them by the log they belong to.

    Args:
        reports: The evaluation reports being tabulated, from `pipelines.evaluate`.
    Returns:
        The scores file of each model, keyed by the log's own name.
    Raises:
        ValueError: If a report has no scores beside it, if one is not a scores file, or if one log
            is given two runs of the same model.
    """
    files = [(report, report.with_suffix('.parquet')) for report in reports]
    missing = [str(report) for report, scores in files if not scores.exists()]
    if missing:
        raise ValueError(
            'no per-prefix scores beside these reports, so which differences are real cannot be '
            'told:\n  ' + '\n  '.join(missing) + '\nScore them again with '
            '`python -m pipelines.evaluate`, which writes them beside the report.'
        )

    runs: list[tuple[RunIdentity, Path]] = []
    for _, scores in files:
        try:
            with pq.ParquetFile(scores) as parquet:
                runs.append((read_run_identity(parquet), scores))
        except (ValueError, TypeError, KeyError) as error:
            # An `OSError` is an unreadable disk, a real failure, and is left to surface as itself
            # rather than being relabelled as the wrong file type.
            raise ValueError(f'{scores} is not a per-prefix scores file: {error}') from error
    return group_by_model(runs)


def _aligned(dataset: str, files: dict[str, Path]) -> tuple[np.ndarray, pd.MultiIndex]:
    """Read every model of one log onto the prefixes all of them scored.

    Args:
        dataset: The log these runs belong to, for the errors below.
        files: The per-prefix scores of each of its models.
    Returns:
        Every model's score for every shared prefix, `[prefixes, models, metrics]` with the models
        in the order `files` gives them and the metrics in the order of `_TESTED`, and which
        prefixes those rows are, sorted so that a case's cuts are contiguous.
    Raises:
        ValueError: If a file scores one prefix twice, if it is missing a metric, or if the models
            of the log do not score exactly the same prefixes.
    """
    frames: dict[str, pd.DataFrame] = {}
    for model, file in files.items():
        frame = read_prefix_scores(file).set_index(['case_id', 'prefix_len'])
        if frame.index.has_duplicates:
            raise ValueError(f'{file} scores the same prefix twice, so it cannot be compared.')
        absent = [key for key in _TESTED if key not in frame.columns]
        if absent:
            raise ValueError(
                f'{file} carries no {", ".join(absent)}, so it predates the scores now reported. '
                'Score it again with `python -m pipelines.evaluate`.'
            )
        frames[model] = frame

    shared = None
    for frame in frames.values():
        shared = frame.index if shared is None else shared.intersection(frame.index)
    assert shared is not None

    # Every model has to have answered exactly the same prefixes, not merely to overlap. A cell is
    # a mean over the prefixes its own run scored, so models scored on different sets of them are
    # already not comparable by the numbers, whatever the emphasis says.
    differing = {model: len(frame) for model, frame in frames.items() if len(frame) != len(shared)}
    if differing:
        counted = ', '.join(f'{model} {count:,}' for model, count in differing.items())
        raise ValueError(
            f'the runs of {dataset} do not score the same prefixes: {counted}, against '
            f'{len(shared):,} in common. A table compares means over the same prefixes, so '
            'generate and score every model of a log from the same test split.'
        )

    # Sorted by case first, so a case's cuts sit together and the bootstrap can sum them with a
    # single reduction rather than a grouping.
    shared = shared.sort_values()
    # [prefixes, models, metrics]
    values = np.stack(
        [frames[model].loc[shared, list(_TESTED)].to_numpy(dtype=np.float64) for model in files],
        axis=1,
    )
    return values, shared


def _by_case(values: np.ndarray, prefixes: pd.MultiIndex) -> tuple[np.ndarray, np.ndarray]:
    """Reduce the prefixes to the cases they were cut from, which is what a resample draws.

    Args:
        values: Every model's score for every prefix, `[prefixes, models, metrics]`, its rows
            sorted so that a case's cuts are contiguous.
        prefixes: Which prefix each row is, in that same order.
    Returns:
        The sum of each case's cuts, `[cases, models, metrics]`, and how many cuts each case has,
        `[cases]`. Summed rather than averaged: every prefix weighs the same in the mean a report
        holds, so a long case weighs more, and the test has to read the same way.
    """
    cases = prefixes.get_level_values('case_id').to_numpy()
    starts = np.flatnonzero(np.r_[True, cases[1:] != cases[:-1]])
    # [cases, models, metrics] and [cases]
    return np.add.reduceat(values, starts, axis=0), np.diff(np.r_[starts, len(cases)])


def _bootstrap_p(case_sums: np.ndarray, case_cuts: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """The share of resamples in which each model's mean falls on the wrong side of the reference.

    A paired cluster bootstrap: the unit drawn is the case, with all of its cut points attached, so
    the dependence between the prefixes of one case is carried into the test. A resample's mean for
    one model is `sum(w * case_sums) / sum(w * case_cuts)`, the ratio estimator of exactly the mean
    the table prints.

    Args:
        case_sums: Each case's summed scores, `[cases, models, metrics]`.
        case_cuts: How many cuts each case has, `[cases]`.
        reference: Which model each metric is compared against, `[metrics]`.
    Returns:
        The uncorrected two-sided p of every model against the reference of its metric,
        `[models, metrics]`. The reference's own is 1.0, its difference from itself being 0.
    """
    cases, models, metrics = case_sums.shape
    flat = case_sums.reshape(cases, models * metrics)
    cuts = case_cuts.astype(np.float64)
    columns = np.arange(metrics)

    generator = np.random.default_rng(SEED)
    at_least, at_most = np.zeros((models, metrics)), np.zeros((models, metrics))

    drawn = 0
    while drawn < RESAMPLES:
        size = min(CHUNK, RESAMPLES - drawn)
        # Which case each of `cases` draws landed on, offset so that a resample's draws fall in
        # its own stretch of one flat count: `[size, cases]` counted in a single `bincount`, which
        # is an order of magnitude cheaper than a multinomial over this many cases.
        picks = generator.integers(low=0, high=cases, size=(size, cases))
        picks += (np.arange(size) * cases)[:, None]
        # How many times each case was drawn, [size, cases]. Drawing `cases` of them with
        # replacement is what a cluster bootstrap resamples.
        counts = (
            np.bincount(picks.ravel(), minlength=size * cases)
            .reshape(size, cases)
            .astype(np.float64)
        )
        # [size, models, metrics]
        means = ((counts @ flat) / (counts @ cuts)[:, None]).reshape(size, models, metrics)
        # How much better than its metric's reference each model came out, [size, models, metrics].
        against = means[:, reference, columns][:, None, :]  # [size, 1, metrics]
        better = _oriented(means) - _oriented(against)
        at_least += (better >= 0).sum(axis=0)
        at_most += (better <= 0).sum(axis=0)
        drawn += size

    # Two-sided, by inverting the bootstrap interval of the difference: a model that lands on both
    # sides of the reference is one the resamples cannot separate from it.
    return np.minimum(2.0 * np.minimum(at_least, at_most) / RESAMPLES, 1.0)


def _holm(p_values: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down adjustment of one family of p-values.

    A row compares every model against one reference at once, so its p-values are one family and
    an uncorrected threshold would call a difference real once every twenty comparisons by chance.

    Args:
        p_values: The uncorrected p of each comparison, in any order.
    Returns:
        The adjusted p of each, in the order given, each capped at 1.0 and at no less than the
        adjusted p of every stricter comparison.
    """
    order = np.argsort(p_values)
    remaining = len(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (remaining - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def test_significance(reports: Sequence[Path]) -> pd.DataFrame:
    """Tell, for every metric of every log, which models are indistinguishable from the best.

    The best value alone says a model won where the honest claim is usually that several of them
    are tied. Each log's models are compared over the prefixes all of them scored, by a paired
    bootstrap that resamples whole cases, and the best group is the best model plus everything the
    resamples cannot separate from it.

    Args:
        reports: The evaluation reports being tabulated. Each one's per-prefix scores are read from
            beside it, which is where `pipelines.evaluate` writes them.
    Returns:
        One row per model per metric per log, under `SIGNIFICANCE_COLUMNS`. Only metrics with a
        declared direction are tested, since one with no better value has no best to be tied with.
        A log holding a single model marks nothing, as a row with a single column does.
    Raises:
        ValueError: If a report has no per-prefix scores beside it, if one log is given two runs of
            the same model, or if the models of a log do not score the same prefixes.
    """
    rows: list[dict[str, object]] = []
    for dataset, files in _score_files(reports).items():
        models = list(files)
        if len(models) < 2:
            rows.extend(
                {
                    'dataset': dataset,
                    'model': models[0],
                    'metric': key,
                    'p_value': np.nan,
                    'best': False,
                }
                for key in _TESTED
            )
            continue

        values, prefixes = _aligned(dataset=dataset, files=files)
        case_sums, case_cuts = _by_case(values=values, prefixes=prefixes)

        # The mean the table prints: every prefix weighing the same, so a long case weighs more.
        observed = case_sums.sum(axis=0) / case_cuts.sum()  # [models, metrics]
        reference = _oriented(observed).argmax(axis=0)  # [metrics]

        uncorrected = _bootstrap_p(case_sums, case_cuts, reference)  # [models, metrics]
        for column, key in enumerate(_TESTED):
            best = reference[column]
            others = [index for index in range(len(models)) if index != best]
            adjusted = _holm(uncorrected[others, column])
            tied = {models[index] for index, p in zip(others, adjusted, strict=True) if p >= ALPHA}
            rows.extend(
                {
                    'dataset': dataset,
                    'model': models[index],
                    'metric': key,
                    'p_value': np.nan if index == best else float(adjusted[others.index(index)]),
                    'best': index == best or models[index] in tied,
                }
                for index in range(len(models))
            )

    frame = pd.DataFrame(rows, columns=list(SIGNIFICANCE_COLUMNS))
    return frame.astype({'p_value': 'float64', 'best': 'bool'})
