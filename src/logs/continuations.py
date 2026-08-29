import json
from collections.abc import Collection, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src import paths
from src.logs.keys import ACTIVITY_KEY, CASE_KEY, REMAINING_TIME_KEY, UNK_TOKEN
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
    # How far apart two of the occurrences are, the term an energy score subtracts for the
    # reference set's own spread
    dispersion: float

    @property
    def occurrences(self) -> float:
        """How many times the prefix was observed, which is the mass `weights` spreads over."""
        return float(self.weights.sum())


class ContinuationIndex:
    """Every continuation a log's test split was observed to take after each of its prefixes.

    Read by the scoring pool, one instance per worker, from what `build_index` wrote. A prefix is
    keyed by the activities it runs alone, so two cases that ran the same activities share one
    reference distribution.
    """

    def __init__(self, dataset: str) -> None:
        """
        Args:
            dataset: The dataset whose index to read, from where preprocessing wrote it.
        """
        paths.require_continuations(dataset)
        table = pq.read_table(paths.continuation_path(dataset))

        self._codes = ActivityCodes.of(json.loads(table.schema.metadata[_VOCABULARY]))
        self._rows = {prefix: row for row, prefix in enumerate(table.column('prefix').to_pylist())}
        self._suffixes = table.column('suffixes').to_pylist()
        self._weights = table.column('weights').to_pylist()
        self._remaining_times = table.column('remaining_times').to_pylist()
        self._dispersion = table.column('dispersion').to_numpy()

    def encode(self, activities: Sequence[str]) -> str:
        """Encode one generated suffix onto the scale this index's references are held on."""
        return self._codes.encode(activities)

    def references(self, prefix_activities: Sequence[str]) -> References:
        """Every continuation observed after one prefix.

        Args:
            prefix_activities: The prefix's activity names, in order, as the generations file
                carries them.
        Returns:
            The prefix's observed continuations, their weights, the minutes left at each of its
            occurrences and the spread of the set.
        Raises:
            KeyError: If the split never ran that prefix, which means the generations were written
                against a different preprocessing of this dataset than the index was built from.
        """
        row = self._rows.get(self.encode(prefix_activities))
        if row is None:
            raise KeyError(
                f'prefix {list(prefix_activities)} is in the generations but not in the '
                'continuation index: the two were built from different preprocessings of this '
                'dataset. Rerun pipelines.preprocess, then pipelines.generate.'
            )
        return References(
            suffixes=tuple(self._suffixes[row]),
            weights=np.asarray(self._weights[row], dtype=np.float64),
            remaining_times=np.asarray(self._remaining_times[row], dtype=np.float64),
            dispersion=float(self._dispersion[row]),
        )


def build_index(
    test: pd.DataFrame, *, dataset: str, vocabulary: Collection[str]
) -> tuple[int, int]:
    """Index every continuation a log's test split takes, and write it beside the split.

    The reference distribution a generated one is compared against is the test split and nothing
    else: it is a held-out, out-of-time sample of `p(suffix | prefix)`, so a model that memorized
    the train split gains nothing from it and the reference is not drawn from the older regime the
    split's own separation exists to keep apart. Every cut point of every case contributes,
    `min_prefix_len` included, since that bound governs what a model may be asked rather than what
    the log was observed to do.

    The split is read through the train split's vocabulary, an activity missing from it becoming
    UNK, which is the only name generation can give it: a model can emit no activity it was never
    shown, and the generations name a prefix and its ground truth the same way. Keying the index
    on the raw names instead would leave every prefix of an out-of-time activity unlookupable,
    and every reference suffix holding one unmatchable by construction.

    Args:
        test: The test split, as preprocessing holds it, sorted by case and by timestamp.
        dataset: The dataset the split came from, naming where the index goes.
        vocabulary: The activity names the train split holds, from `codec.activity.vocab`.
    Returns:
        How many distinct prefixes were indexed, and how many occurrences they cover.
    """
    known = set(vocabulary)
    seen = test.assign(
        **{
            ACTIVITY_KEY: test[ACTIVITY_KEY].where(
                cond=test[ACTIVITY_KEY].isin(known), other=UNK_TOKEN
            )
        }
    )

    codes = ActivityCodes()
    cases = [
        (codes.encode(events[ACTIVITY_KEY]), events[REMAINING_TIME_KEY].tolist())
        for _, events in seen.groupby(CASE_KEY, sort=False)
    ]

    # Every cut point of every case, grouped under the prefix it leaves behind. A prefix's
    # continuations are counted as they arrive, so a suffix the log took twice is one entry
    # weighing two; its remaining times are kept one per occurrence.
    grouped: dict[str, tuple[dict[str, int], list[float]]] = {}
    for activities, remaining_times in cases:
        for cut in range(1, len(activities)):
            observed, minutes = grouped.setdefault(activities[:cut], ({}, []))
            suffix = activities[cut:]
            observed[suffix] = observed.get(suffix, 0) + 1
            minutes.append(remaining_times[cut - 1])

    table = pa.table(
        {
            'prefix': list(grouped),
            'suffixes': [list(observed) for observed, _ in grouped.values()],
            'weights': [list(observed.values()) for observed, _ in grouped.values()],
            'remaining_times': [minutes for _, minutes in grouped.values()],
            'dispersion': [
                spread(tuple(observed), weights=list(observed.values()))
                for observed, _ in grouped.values()
            ],
        },
        schema=_SCHEMA.with_metadata({_VOCABULARY: json.dumps(codes.vocabulary)}),
    )

    path = paths.continuation_path(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression=_COMPRESSION)
    return len(grouped), sum(len(minutes) for _, minutes in grouped.values())
