import json
from collections.abc import Collection, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src import paths
from src.logs.keys import (
    ACTIVITY_KEY,
    CASE_KEY,
    EVENT_DELTA_KEY,
    REMAINING_TIME_KEY,
    UNK_TOKEN,
    Split,
)
from src.suffixes import ActivityCodes, spread

# The activity names the encoded prefixes and suffixes are read back through, in code order.
# Held in the file's own metadata, so nothing else has to be read to make sense of it.
_VOCABULARY = b'activities'

_SCHEMA = pa.schema(
    [
        ('prefix', pa.large_string()),
        ('suffixes', pa.list_(pa.large_string())),
        ('weights', pa.list_(pa.int32())),
        ('remaining_times', pa.list_(pa.float32())),
        ('dispersion', pa.float32()),
        # The waits observed before each activity, pooled over every occurrence of the prefix:
        # one character of `wait_activities` per distinct activity, on the scale `suffixes` is
        # held on, and its waits in `waits`. Reduced here rather than kept per occurrence because
        # the pool is what a score reads, the way `dispersion` is already reduced at index time.
        ('wait_activities', pa.large_string()),
        ('waits', pa.list_(pa.list_(pa.float32()))),
    ]
)

_COMPRESSION = 'zstd'


@dataclass(frozen=True, slots=True)
class References:
    """Every continuation one prefix was observed to take, as the empirical distribution the
    suffixes generated for that prefix are measured against.

    Two of a prefix's occurrences that ran the same activities are one entry of `suffixes` with a
    weight of two, since they are one outcome the process produced twice. The remaining times are
    kept per occurrence instead: two cases can run the same activities and take different times
    over them, so collapsing them would drop a difference the log actually holds.
    """

    # The distinct continuations, encoded onto the scale `ContinuationIndex.encode` reads on
    suffixes: tuple[str, ...]
    # How many occurrences each of them accounts for, in the same order
    weights: np.ndarray
    # Minutes left at the cut, one per occurrence
    remaining_times: np.ndarray
    # The waits observed before each activity, keyed by the character `suffixes` spells it with.
    # A wait belongs to the activity it precedes rather than to the position it fell at, so the
    # occurrences are pooled under their activities instead of being kept as runs.
    waits: dict[str, np.ndarray]
    # How far apart two of the occurrences are, the term an energy score subtracts for the
    # reference set's own spread
    dispersion: float

    @property
    def occurrences(self) -> float:
        """How many times the prefix was observed, which is the mass `weights` spreads over."""
        return float(self.weights.sum())


class ContinuationIndex:
    """Every continuation one split of a log was observed to take after each of its prefixes.

    Read by the scoring pool, one instance per worker, from what `build_index` wrote. A prefix is
    keyed by the activities it runs alone, so two cases that ran the same activities share one
    reference distribution.
    """

    def __init__(self, dataset: str, split: Split) -> None:
        """
        Args:
            dataset: The dataset whose index to read, from where preprocessing wrote it.
            split: Which split's continuations to read. Evaluation scores against `TEST`, and
                training selects checkpoints against `VAL`.
        """
        table = pq.read_table(paths.CONTINUATIONS.require(dataset=dataset, split=split))

        self._codes = ActivityCodes.of(json.loads(table.schema.metadata[_VOCABULARY]))
        self._rows = {prefix: row for row, prefix in enumerate(table.column('prefix').to_pylist())}
        self._suffixes = table.column('suffixes').to_pylist()
        self._weights = table.column('weights').to_pylist()
        self._remaining_times = table.column('remaining_times').to_pylist()
        self._wait_activities = table.column('wait_activities').to_pylist()
        self._waits = table.column('waits').to_pylist()
        self._dispersion = table.column('dispersion').to_numpy()

    @property
    def vocabulary(self) -> tuple[str, ...]:
        """The activity names this index codes through, in code order.

        Both this and the generations file are seeded from `codec.activity.names`, so the two agree
        by construction and a caller compares them to catch a file built against an older
        preprocessing before it scores a single prefix.
        """
        return self._codes.vocabulary

    def references(self, prefix: str) -> References:
        """Every continuation observed after one prefix.

        Args:
            prefix: The prefix's activities, one character each, in order, as the generations file
                carries them and on the same scale this index is held on.
        Returns:
            The prefix's observed continuations, their weights, the minutes left at each of its
            occurrences and the spread of the set.
        Raises:
            KeyError: If the split never ran that prefix, which means the generations were written
                against a different preprocessing of this dataset than the index was built from.
        """
        row = self._rows.get(prefix)
        if row is None:
            raise KeyError(
                f'a prefix of {len(prefix)} events is in the generations but not in the '
                'continuation index: the two were built from different preprocessings of this '
                'dataset. Rerun pipelines.preprocess, then pipelines.generate.'
            )
        return References(
            suffixes=tuple(self._suffixes[row]),
            weights=np.asarray(self._weights[row], dtype=np.float64),
            remaining_times=np.asarray(self._remaining_times[row], dtype=np.float64),
            waits={
                activity: np.asarray(observed, dtype=np.float64)
                for activity, observed in zip(
                    self._wait_activities[row], self._waits[row], strict=True
                )
            },
            dispersion=float(self._dispersion[row]),
        )


def build_index(
    rows: pd.DataFrame,
    *,
    dataset: str,
    split: Split,
    vocabulary: Collection[str],
    names: Sequence[str],
) -> tuple[int, int]:
    """Index every continuation one held-out split takes, and write it beside the splits.

    The reference distribution a generated suffix is compared against is one held-out split and
    nothing else: an out-of-time sample of `p(suffix | prefix)`, so a model that memorized the
    train split gains nothing from it and the reference is not drawn from the older regime the
    split's own separation exists to keep apart. Every cut point of every case contributes,
    `min_prefix_len` included, since that bound governs what a model may be asked rather than what
    the log was observed to do.

    Both held-out splits are indexed, and which one is read is the reader's choice: evaluation
    scores against the test split, while training selects checkpoints against the validation
    split, since selecting on the test split's continuations would fold the held-out set into
    what gets kept.

    The split is read through the train split's vocabulary, an activity missing from it becoming
    UNK, which is the only name generation can give it: a model can emit no activity it was never
    shown, and the generations name a prefix and its ground truth the same way. Keying the index
    on the raw names instead would leave every prefix of an out-of-time activity unlookupable,
    and every reference suffix holding one unmatchable by construction.

    Args:
        rows: The split to index, as preprocessing holds it, sorted by case and by timestamp.
        dataset: The dataset the split came from, naming where the index goes.
        split: Which split `rows` is, naming the file written.
        vocabulary: The activity names the train split holds, from `codec.activity.vocab`.
        names: Every name the activity channel can decode to, in row order, from
            `codec.activity.names`. Seeds the codebook, so a suffix is spelled here exactly as a
            generations file spells it and neither side ever codes a name on the fly.
    Returns:
        How many distinct prefixes were indexed, and how many occurrences they cover.
    """
    known = set(vocabulary)
    seen = rows.assign(
        **{
            ACTIVITY_KEY: rows[ACTIVITY_KEY].where(
                cond=rows[ACTIVITY_KEY].isin(known), other=UNK_TOKEN
            )
        }
    )

    codes = ActivityCodes.of(names)
    cases = [
        (
            codes.encode(events[ACTIVITY_KEY]),
            events[REMAINING_TIME_KEY].tolist(),
            events[EVENT_DELTA_KEY].tolist(),
        )
        for _, events in seen.groupby(CASE_KEY, sort=False)
    ]

    # Every cut point of every case, grouped under the prefix it leaves behind. A prefix's
    # continuations are counted as they arrive, so a suffix the log took twice is one entry
    # weighing two; its remaining times are kept one per occurrence, and the wait before each of
    # its events is pooled under that event's own activity. The first event of a case carries no
    # wait, and it is never in a suffix either, since a cut leaves at least one event behind it.
    grouped: dict[str, tuple[dict[str, int], list[float], dict[str, list[float]]]] = {}
    for activities, remaining_times, deltas in cases:
        for cut in range(1, len(activities)):
            observed, minutes, waits = grouped.setdefault(activities[:cut], ({}, [], {}))
            suffix = activities[cut:]
            observed[suffix] = observed.get(suffix, 0) + 1
            minutes.append(remaining_times[cut - 1])
            for activity, wait in zip(suffix, deltas[cut:], strict=True):
                waits.setdefault(activity, []).append(wait)

    table = pa.table(
        {
            'prefix': list(grouped),
            'suffixes': [list(observed) for observed, _, _ in grouped.values()],
            'weights': [list(observed.values()) for observed, _, _ in grouped.values()],
            'remaining_times': [minutes for _, minutes, _ in grouped.values()],
            'dispersion': [
                spread(tuple(observed), weights=list(observed.values()))
                for observed, _, _ in grouped.values()
            ],
            'wait_activities': [''.join(waits) for _, _, waits in grouped.values()],
            'waits': [list(waits.values()) for _, _, waits in grouped.values()],
        },
        schema=_SCHEMA.with_metadata({_VOCABULARY: json.dumps(codes.vocabulary)}),
    )

    path = paths.CONTINUATIONS.prepare(dataset=dataset, split=split)
    pq.write_table(table, path, compression=_COMPRESSION)
    return len(grouped), sum(len(minutes) for _, minutes, _ in grouped.values())
