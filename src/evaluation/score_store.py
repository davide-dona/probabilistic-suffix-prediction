from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.evaluation.metrics import METRICS
from src.evaluation.reports import _group_by_model
from src.evaluation.scoring import PrefixSummary
from src.inference.generation_store import PrefixKey
from src.runs.artifacts import read_metadata, with_metadata

BLOCK = 16_384
PREFIX_SCORE_KEYS = ('case_id', 'prefix_len', 'suffix_len')
_PREFIX_SCORE_SCHEMA = pa.schema(
    [
        ('case_id', pa.large_string()),
        ('prefix_len', pa.int64()),
        ('suffix_len', pa.int64()),
        *((key, pa.float64()) for key in METRICS.entries),
    ]
)


def stream_prefix_scores(
    summaries: Iterable[PrefixSummary],
    keys: Sequence[PrefixKey],
    *,
    path: Path,
    metadata: dict[str, str],
) -> Iterator[PrefixSummary]:
    """Write per-prefix scores while yielding the original summaries.

    Args:
        summaries: Scores in the same prefix order as keys, consumed once.
        keys: Case identifier and prefix length for each summary.
        path: Parquet destination in an existing directory, replaced after a complete write.
        metadata: Artifact provenance stored with the score schema.

    Yields:
        Each input summary unchanged. Consume the iterator fully to flush all buffered rows
        and finish the file; closing an unfinished iterator removes the temporary file.

    Raises:
        ValueError: If the summary and key counts differ. Failures during writing remove
            the temporary file before propagating the error.
    """
    columns: dict[str, list[str | int | float]] = {name: [] for name in _PREFIX_SCORE_SCHEMA.names}

    def flush(writer: pq.ParquetWriter) -> None:
        if not columns['case_id']:
            return
        writer.write_table(pa.Table.from_pydict(mapping=columns, schema=_PREFIX_SCORE_SCHEMA))
        for values in columns.values():
            values.clear()

    temporary = path.with_suffix('.parquet.tmp')
    try:
        with pq.ParquetWriter(
            where=temporary, schema=with_metadata(_PREFIX_SCORE_SCHEMA, metadata)
        ) as writer:
            for summary, (case_id, prefix_len) in zip(summaries, keys, strict=True):
                columns['case_id'].append(case_id)
                columns['prefix_len'].append(prefix_len)
                columns['suffix_len'].append(summary.suffix_len)
                for name, value in summary.scores.flatten().items():
                    columns[name].append(value)
                if len(columns['case_id']) >= BLOCK:
                    flush(writer)
                yield summary
            flush(writer)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def require_columns(path: Path, columns: Sequence[str]) -> None:
    """Require the requested columns in a score file's schema.

    Args:
        path: Parquet score file to inspect.
        columns: Required column names; additional columns are allowed.

    Raises:
        ValueError: If any requested column is missing.
    """
    available = set(pq.read_schema(where=path).names)
    absent = [key for key in columns if key not in available]
    if absent:
        raise ValueError(f'{path} is missing required score columns: {", ".join(absent)}.')


def read_prefix_scores(path: Path, *, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Read report columns by default, or explicitly selected per-prefix score columns.

    Args:
        path: Parquet score file to read.
        columns: Requested columns in output order. Defaults to prefix identifiers, lengths,
            and registered metrics.

    Returns:
        Selected columns as a dataframe, retaining the stored row order.

    Raises:
        ValueError: If a requested column is missing.
    """
    wanted = list((*PREFIX_SCORE_KEYS, *METRICS.entries) if columns is None else columns)
    require_columns(path, wanted)
    return pq.read_table(source=path, columns=wanted).to_pandas()


def score_files(reports: Sequence[Path]) -> dict[str, dict[str, Path]]:
    """Find score files beside reports and group them by dataset.

    Args:
        reports: Report paths with an adjacent prefix_scores.parquet file for each run.

    Returns:
        Dataset names mapped to model names and their score paths, validated for report
        columns and artifact provenance.

    Raises:
        ValueError: If score files are missing or invalid, or a dataset repeats a model.
    """
    files = [(report, report.with_name('prefix_scores.parquet')) for report in reports]
    missing = [str(report) for report, scores in files if not scores.exists()]
    if missing:
        raise ValueError(
            "no per-prefix scores beside these reports, so how far a model's means could be off "
            'cannot be read:\n  '
            + '\n  '.join(missing)
            + '\nScore them again with `python -m pipelines.evaluate`, which writes them beside '
            'the report.'
        )
    runs: list[tuple[dict[str, str], Path]] = []
    for _, scores in files:
        try:
            require_columns(scores, (*PREFIX_SCORE_KEYS, *METRICS.entries))
            with pq.ParquetFile(scores) as parquet:
                runs.append((read_metadata(parquet), scores))
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(f'{scores} is not a per-prefix scores file: {error}') from error
    return _group_by_model(runs)
