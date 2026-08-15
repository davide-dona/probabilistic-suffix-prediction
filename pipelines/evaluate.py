import argparse
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

from src import paths
from src.cli import existing_file
from src.evaluation.metrics import EvaluationMetrics, ScoredPrefix, score_prefixes
from src.evaluation.report import EvaluationReport
from src.inference.generation_store import read_generation_block, read_run_identity
from src.logs.declare import ConformanceChecker

# Conformance checker configuration.
# If True, the checker will consider not satisfying a constraint as a failure.
CONSIDER_VACUITY = False


@dataclass(frozen=True)
class _Worker:
    """What one pool process holds for the whole of its life, rather than per block."""

    parquet: pq.ParquetFile
    checker: ConformanceChecker


# Set by `_init_worker` in each pool process, and read by `_score_block` there. Left
# unassigned so that scoring outside a pool fails loudly rather than on a silent `None`.
_worker: _Worker


def _init_worker(generations_file: Path, dataset: str) -> None:
    """Open the file and prepare the declarative model once for this process.

    Args:
        generations_file: The generations every task of this process reads from.
        dataset: The dataset whose declarative model conformance is checked against.
    """
    global _worker
    _worker = _Worker(
        parquet=pq.ParquetFile(generations_file),
        checker=ConformanceChecker(dataset, consider_vacuity=CONSIDER_VACUITY),
    )


def _score_block(block: int) -> list[ScoredPrefix]:
    """Score every prefix of one block of the generations file, in the order they were written.

    Args:
        block: Which block of the file `_init_worker` opened to score.
    Returns:
        One entry per prefix of the block, in the order it was written.
    """
    generations = read_generation_block(parquet=_worker.parquet, block=block)
    return score_prefixes(generations=generations, checker=_worker.checker)


def _score_in_parallel(
    generations_file: Path,
    *,
    dataset: str,
    workers: int | None,
) -> Iterator[ScoredPrefix]:
    """Score a generations file across a pool of processes, a block at a time.

    A prefix's score depends on nothing but the prefix, and the conformance checks that dominate
    it are pure Python that holds the GIL. Each prefix is scored down to a handful of floats and
    its generation dropped in the worker, so a split of hundreds of thousands of them never brings
    more than a block's objects back here.

    Args:
        generations_file: The generations to score, from `python -m pipelines.generate`. Passed as
            a path rather than as read prefixes, since each worker opens the file itself and only
            the scores it computes cross back.
        dataset: The dataset the prefixes were cut from, naming the declarative model to check
            conformance against.
        workers: How many processes to score with, or `None` for one per available CPU.
    Yields:
        Each prefix's scores, in the order the file holds them. The pool is shut down once they
        have all been drawn.
    """
    # Open the file once to get its block count and total prefix count
    with pq.ParquetFile(generations_file) as parquet:
        blocks, prefixes = parquet.num_row_groups, parquet.metadata.num_rows

    with (
        ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(generations_file, dataset),
        ) as executor,
        tqdm(total=prefixes, desc='Scoring', unit='prefix') as progress,
    ):
        # `map` yields the blocks in the order they were submitted however they finished, so the
        # scores arrive in the order the file holds, and the means over them stay reproducible.
        for block in executor.map(_score_block, range(blocks)):
            progress.update(len(block))
            yield from block


def run(generations_file: Path, workers: int | None) -> None:
    """Score a run's generated suffixes and write the result under `outputs/eval/`.

    Args:
        generations_file: The generations to score, from `python -m pipelines.generate`. It says
            which run and dataset wrote it, so the report is named after that run and the
            declarative model is looked up under that dataset.
        workers: How many processes to score with, or `None` for one per available CPU.
    """
    with pq.ParquetFile(generations_file) as parquet:
        run = read_run_identity(parquet)

    dataset = run.dataset
    paths.require_dataset(dataset)
    paths.require_declare_model(dataset)

    print(f'Scoring the suffixes generated for each prefix of {generations_file}...', flush=True)

    # Compute the metrics of the generation, folding each prefix's scores in as the pool hands
    # them back.
    metrics = EvaluationMetrics.aggregate(
        _score_in_parallel(generations_file, dataset=dataset, workers=workers)
    )

    # The report is named after the run the generations carry, so it sits under `outputs/eval/`
    # exactly where they sit under `outputs/generations/`.
    report = EvaluationReport(run=run, metrics=metrics)
    path = report.write(paths.evaluation_path(run))
    print(
        f'Scored {metrics.pairs} prefixes over {metrics.cases} cases. '
        f'Wrote evaluation report to {path}'
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a run's generated test-split suffixes against the ground truth."
    )
    parser.add_argument(
        '-g',
        '--generations',
        type=existing_file,
        metavar='GENERATIONS',
        required=True,
        help='Path to the generations file to score, from `pipelines.generate`.',
    )
    parser.add_argument(
        '-j',
        '--workers',
        type=int,
        default=None,
        metavar='N',
        help='How many processes to score with. Defaults to one per available CPU.',
    )
    args = parser.parse_args()

    run(args.generations, args.workers)


if __name__ == '__main__':
    main()
