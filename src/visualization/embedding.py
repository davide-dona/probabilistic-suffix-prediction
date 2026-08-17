from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.identity import RunIdentity, group_by_model
from src.inference.generation_store import (
    PrefixKey,
    read_prefix_keys,
    read_run_identity,
    read_samples,
)
from src.suffixes import ActivityCodes, distances

# How many prefixes an embedding is built from. It costs a pairwise distance per pair of distinct
# suffixes, so this is what bounds it to seconds rather than minutes.
POINTS = 1500
# How many of a prefix's ten draws each run contributes. One draw over this many different
# prefixes already samples the model's marginal predictive distribution, and it leaves every cloud
# the same size, so a difference in spread is the models differing rather than the counts.
SAMPLES_PER_PREFIX = 1
# How much of the neighbourhood the layout preserves, and how tightly it may pack a cluster.
UMAP_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
# What the sampling and the layout are reproducible under. Seeding UMAP costs it its parallelism,
# which is the price of an embedding that comes out the same twice.
SEED = 24


class Source(StrEnum):
    """Which set of suffixes a point belongs to.

    A panel draws both: what actually followed the prefixes, and what one model wrote for them.
    """

    TRUTH = 'truth'
    GENERATED = 'generated'


# What `embed_suffixes` returns, and what the distribution figure is drawn from. The ground truth
# is repeated under each model rather than held once, so a panel is one selection of the frame.
EMBEDDING_COLUMNS = ('dataset', 'model', 'source', 'x', 'y')


def _group_runs(files: Sequence[Path]) -> dict[str, dict[str, Path]]:
    """Read which run wrote each generations file and group them by the log they belong to.

    Only the Parquet footers are touched, so this is cheap however large the files are.

    Args:
        files: The generations to embed, from `pipelines.generate`.
    Returns:
        The file of each model, keyed by the log's own name.
    Raises:
        ValueError: If a file is not a generations file, or if one log is given two runs of the
            same model, which would draw one panel over the other.
    """
    runs: list[tuple[RunIdentity, Path]] = []
    for file in files:
        try:
            with pq.ParquetFile(file) as parquet:
                runs.append((read_run_identity(parquet), file))
        except (ValueError, TypeError, KeyError) as error:
            # An `OSError` is an unreadable disk, a real failure, and is left to surface as itself
            # rather than being relabelled as the wrong file type.
            raise ValueError(f'{file} is not a generations file: {error}') from error
    return group_by_model(runs)


def _shared_prefixes(dataset: str, files: Iterable[Path]) -> set[PrefixKey]:
    """Draw `POINTS` of the prefixes every run of a log answered, or all of them where there are
    fewer.

    Only the prefixes every run answered are eligible: the runs of a log do not share a row order
    and a baseline generated elsewhere need not have answered exactly the same cuts, so anything
    less would compare clouds built from different questions.

    Args:
        dataset: The log these runs belong to, for the error where they share nothing.
        files: The generations of each of its runs.
    Returns:
        The prefixes to build every cloud of this log from, at most `POINTS` of them.
    Raises:
        ValueError: If the runs answer no prefix in common.
    """
    shared = set.intersection(*(read_prefix_keys(file) for file in files))
    if not shared:
        raise ValueError(
            f'the runs of {dataset} answer no prefix in common, so their generations cannot be '
            'compared. Generate them from the same test split.'
        )
    if len(shared) <= POINTS:
        return shared

    # Sorted first, so the draw depends on the seed alone rather than on set iteration order.
    ordered = sorted(shared)
    chosen = np.random.default_rng(SEED).choice(a=len(ordered), size=POINTS, replace=False)
    return {ordered[index] for index in chosen}


@dataclass(frozen=True)
class _Answer:
    """What one prefix contributes to one model's panel: the suffix that truly followed it, and
    the draws taken from what that model wrote for it."""

    truth: str
    generated: tuple[str, ...]


def _read_cloud(file: Path, keys: set[PrefixKey], codes: ActivityCodes) -> dict[PrefixKey, _Answer]:
    """Read what one run answered, out of its generations.

    Keyed by prefix rather than returned as two lists, so the caller can hold every run to the
    prefixes all of them answered: a run that wrote nothing for a prefix simply has no entry for
    it here.

    Args:
        file: The generations of one run, from `pipelines.generate`.
        keys: Which prefixes to read, from `_shared_prefixes`.
        codes: The log's codes, shared with every other cloud of it so that one activity is one
            character throughout.
    Returns:
        What this run answered for each of `keys` it wrote at least one suffix for.
    """
    generator = np.random.default_rng(SEED)
    answers: dict[PrefixKey, _Answer] = {}
    for key, true_activities, samples in read_samples(file):
        if key not in keys or not samples:
            continue
        drawn = generator.choice(
            a=len(samples), size=min(SAMPLES_PER_PREFIX, len(samples)), replace=False
        )
        answers[key] = _Answer(
            truth=codes.encode(true_activities),
            generated=tuple(codes.encode(samples[index]) for index in drawn),
        )
    return answers


def _embed(clouds: Sequence[Sequence[str]]) -> tuple[np.ndarray, dict[str, int]]:
    """Lay every suffix of one log out in two dimensions, in one shared embedding.

    Fitting once over every cloud at once, the ground truth's included, is what makes the panels
    comparable: a point means the same place in every one of them.

    Distinct suffixes rather than one row per point: these logs repeat suffixes heavily, and a
    precomputed distance matrix full of zeros is both far larger than it needs to be and degenerate
    to embed, since a point's nearest neighbours are then all copies of itself.

    Args:
        clouds: The encoded suffixes of each cloud, the truth's among them.
    Returns:
        Where each distinct suffix sits, `[num_distinct, 2]`, and the row each of them is at.
    """
    # Imported here rather than at the top: UMAP drags in numba, which spends seconds warming up,
    # and `--help` or a mistyped path should not wait for it.
    from umap import UMAP

    distinct = sorted({sequence for cloud in clouds for sequence in cloud})

    # [num_distinct, num_distinct]
    matrix = distances(queries=distinct, choices=distinct, workers=-1)
    # UMAP reads a precomputed matrix as given, and the rounding can leave a diagonal a hair off
    # zero or a pair a hair apart from its mirror.
    np.fill_diagonal(a=matrix, val=0.0)
    matrix = np.minimum(matrix, matrix.T)

    reducer = UMAP(
        n_components=2,
        metric='precomputed',
        n_neighbors=min(UMAP_NEIGHBORS, max(len(distinct) - 1, 2)),
        min_dist=UMAP_MIN_DIST,
        random_state=SEED,
    )
    coordinates = np.asarray(reducer.fit_transform(matrix), dtype=np.float64)
    return coordinates, {sequence: row for row, sequence in enumerate(distinct)}


def embed_suffixes(files: Sequence[Path]) -> pd.DataFrame:
    """Lay out what a set of runs generated, and what truly happened, in two dimensions.

    One embedding is fitted per log over every cloud of it at once, so the models of a log can be
    read against each other and against the truth on one pair of axes. Reading the generations,
    measuring the suffixes against each other and fitting the layout each take long enough that a
    caller should announce this before it starts.

    Args:
        files: The generations to compare, from `pipelines.generate`. Each says which run wrote
            it, so they may come from any number of logs and models.
    Returns:
        One row per point, under `EMBEDDING_COLUMNS`. A model carries both its own cloud and the
        log's ground truth, so a panel is a selection rather than a join.
    Raises:
        ValueError: If a file is not a generations file, if one log is given two runs of the same
            model, or if the runs of a log answer no prefix in common.
    """
    rows: list[dict[str, object]] = []
    for dataset, runs in _group_runs(files).items():
        keys = _shared_prefixes(dataset=dataset, files=runs.values())
        # One instance for the whole log, so one activity means one character across the truth and
        # every cloud, which is what lets them share an embedding.
        codes = ActivityCodes()
        clouds = {
            model: _read_cloud(file=file, keys=keys, codes=codes) for model, file in runs.items()
        }

        # Narrowed once more, to the prefixes every run actually wrote a suffix for: a run that
        # wrote nothing for one is the only reason a cloud can come up short of `keys`, and a panel
        # drawn against a truth cloud holding points its own cloud has no answer for would read as
        # the model missing a region it was never asked about.
        answered = sorted(set.intersection(*(set(cloud) for cloud in clouds.values())))
        if not answered:
            raise ValueError(
                f'every prefix of {dataset} is left unanswered by at least one of its runs, so '
                'their generations cannot be compared. Generate them again.'
            )

        # Every run answers the same prefixes of the same split and so carries the same truth for
        # each of them; the first of them answers for all.
        first = clouds[next(iter(clouds))]
        truth = [first[key].truth for key in answered]
        generated = {
            model: [sequence for key in answered for sequence in cloud[key].generated]
            for model, cloud in clouds.items()
        }
        coordinates, positions = _embed([truth, *generated.values()])

        for model, cloud in generated.items():
            for source, sequences in ((Source.TRUTH, truth), (Source.GENERATED, cloud)):
                # One row per point rather than per distinct suffix, so the repeats that carry a
                # cloud's density are what lands on the page.
                points = coordinates[[positions[sequence] for sequence in sequences]]  # [points, 2]
                rows.extend(
                    {'dataset': dataset, 'model': model, 'source': source, 'x': x, 'y': y}
                    for x, y in points.tolist()
                )

    return pd.DataFrame(rows, columns=list(EMBEDDING_COLUMNS))
