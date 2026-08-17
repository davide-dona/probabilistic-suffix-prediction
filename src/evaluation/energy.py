from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

from src.inference.generation_store import read_suffixes
from src.suffixes import ActivityCodes, distances

# How many rows of the distance matrix are held at once.
# Bounds the memory use, making it computable for large logs.
BLOCK = 512


@dataclass(frozen=True, slots=True)
class PooledSuffixes:
    """Two sets of suffixes, pooled into the distinct sequences and how often each set holds them.

    These logs repeat suffixes heavily, so measuring the distinct sequences once and weighting each
    pair by how many suffix pairs it stands for is what makes the sums below affordable.
    """

    sequences: list[str]
    counts: np.ndarray  # [num_distinct, 2]: column 0 counts generated suffixes, column 1 true ones

    @classmethod
    def of(cls, generated: Iterable[str], truth: Iterable[str]) -> Self:
        """Collect two sets of encoded suffixes into the distinct sequences and their frequencies.

        Args:
            generated: The suffixes the model wrote, from `ActivityCodes.encode`.
            truth: The suffixes that actually followed the same prefixes, encoded by the same codes.
        Returns:
            Each sequence either set holds, once and in sorted order, beside how many suffixes of
            each set were it. A sequence only one set holds carries a count of 0 for the other.
        """
        generated_counts = Counter(generated)
        truth_counts = Counter(truth)
        sequences = sorted(generated_counts.keys() | truth_counts.keys())
        return cls(
            sequences=sequences,
            counts=np.array(
                [[generated_counts[sequence], truth_counts[sequence]] for sequence in sequences],
                dtype=np.float32,
            ),
        )


def _pair_sums(suffixes: PooledSuffixes) -> tuple[float, float, float]:
    """Sum the distance over every ordered pair of suffixes, within each set and between them.

    Args:
        suffixes: The two pooled sets, over one vocabulary.
    Returns:
        The sum between the two sets, the sum within the generated set and the sum within the true
        one, in that order. Self-pairs are included, and contribute 0.
    """
    num_distinct = len(suffixes.sequences)
    counts = suffixes.counts
    between = within_generated = within_truth = 0.0

    # What the walk actually visits, for the bar to count down: the pairs of the upper triangle,
    # block-diagonal squares whole.
    total_pairs = sum(
        min(BLOCK, num_distinct - start) * (num_distinct - start)
        for start in range(0, num_distinct, BLOCK)
    )
    with tqdm(
        total=total_pairs,
        desc=f'Measuring {num_distinct:,} distinct suffixes against each other',
        unit='pair',
        unit_scale=True,
    ) as progress:
        for start in range(0, num_distinct, BLOCK):
            stop = min(start + BLOCK, num_distinct)
            # [block, num_distinct - start]
            block_distances = distances(
                queries=suffixes.sequences[start:stop],
                choices=suffixes.sequences[start:],
                workers=-1,
            )

            block = stop - start
            # Both weighted sums of each row, in one pass over the block. [block, 2]
            tail = block_distances @ counts[start:]
            # The same over the block against itself, which the tail counts a second time.
            square = block_distances[:, :block] @ counts[start:stop]
            rows = counts[start:stop]

            tail_generated, tail_truth = tail[:, 0], tail[:, 1]
            square_generated, square_truth = square[:, 0], square[:, 1]
            rows_generated, rows_truth = rows[:, 0], rows[:, 1]

            # A pair of two different blocks appears once and stands for both of its orders, so it
            # is doubled; a pair inside the block is already in the tail both ways round, so the
            # block's own square comes off once. The two sets weigh a pair differently in each
            # order, which is why the between sum takes the tail from each side rather than
            # doubling one.
            between += float(
                rows_generated @ tail_truth
                + rows_truth @ tail_generated
                - rows_generated @ square_truth
            )
            within_generated += float(
                2.0 * (rows_generated @ tail_generated) - rows_generated @ square_generated
            )
            within_truth += float(2.0 * (rows_truth @ tail_truth) - rows_truth @ square_truth)
            progress.update(block * (num_distinct - start))

    return between, within_generated, within_truth


def energy_distance(suffixes: PooledSuffixes) -> float:
    """How far a set of generated suffixes sits from a set of real ones, lower being better.

    The mean distance between a generated suffix and a real one, discounted by the spread each set
    already covers on its own, so that a model is not rewarded for producing one popular variant
    over and over nor for scattering. 0.0 where both sets are drawn from the same distribution,
    rising towards 1.0.

    Each set's own spread is the mean over its ordered pairs of distinct draws, which is the
    unbiased estimate: the self-pairs the sums count contribute 0, so excluding them is the
    denominator alone.

    This is `energy_score` with the truth a whole sample rather than the single suffix one prefix
    happened to be followed by, over the same distance, so the two read on one scale. It inherits
    the same caveat: the discount stays below the distance, and so the value stays at or above 0.0,
    only where `1 - DLS` obeys the triangle inequality, which normalizing by the longer of two
    lengths does not guarantee. Measured against a log's own suffixes split in two, where the true
    value is 0.0, it comes out within 0.001 of it.

    Args:
        suffixes: The suffixes the model wrote and the ones that truly followed those prefixes,
            pooled together.
    Returns:
        The distance between the two sets.
    """
    generated, truth = (float(total) for total in suffixes.counts.sum(axis=0))
    between, within_generated, within_truth = _pair_sums(suffixes)

    # A set of fewer than two suffixes has no pair of distinct draws, and so no spread.
    spread_generated = within_generated / (generated * (generated - 1.0)) if generated >= 2 else 0.0
    spread_truth = within_truth / (truth * (truth - 1.0)) if truth >= 2 else 0.0

    return 2.0 * between / (generated * truth) - spread_generated - spread_truth


def score_distribution(generations_file: Path) -> float:
    """Measure a run's generated suffixes against the real ones over the whole split.

    Every prefix contributes one generated suffix and the one that truly followed it, so the two
    sets answer the same questions and are the same size. Nothing is subsampled: the distance is
    the exact value over the split, and reproducible without a seed.

    The term measuring the truth against itself depends on the log alone rather than on the model,
    and is recomputed for each run rather than cached, since a cache would be another artifact to
    invalidate for a few minutes of the largest log.

    Args:
        generations_file: The generations to measure, from `python -m pipelines.generate`.
    Returns:
        The energy distance between what the run generated and what the log holds.
    """
    codes = ActivityCodes()
    generated: list[str] = []
    truth: list[str] = []
    with pq.ParquetFile(generations_file) as parquet:
        with tqdm(
            total=parquet.metadata.num_rows,
            desc='Reading the suffixes of the split',
            unit='prefix',
        ) as progress:
            for true_activities, sample_activities in read_suffixes(parquet):
                truth.append(codes.encode(true_activities))
                generated.append(codes.encode(sample_activities))
                progress.update(1)

    return energy_distance(PooledSuffixes.of(generated=generated, truth=truth))
