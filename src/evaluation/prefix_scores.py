from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.evaluation.scores import METRICS
from src.evaluation.summary import PrefixSummary, flatten_scores
from src.identity import RunIdentity, stamped
from src.inference.generation_store import PrefixKey

# How many prefixes one row group holds. A split of a quarter of a million is then a handful of
# blocks, and nothing beyond one block is ever buffered.
BLOCK = 16_384

# Which prefix a row scores, and the two lengths its scores are read against. The same key the
# generations are written under, so the rows of two runs of one log are matched on it.
PREFIX_SCORE_KEYS = ('case_id', 'prefix_len', 'suffix_len')

# The schema of the Parquet file that holds a run's per-prefix scores, one row per prefix. The
# metric columns are built from `METRICS` rather than listed, so a score added to a family reaches
# this file with no change here.
_SCHEMA = pa.schema(
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
    run: RunIdentity,
) -> Iterator[PrefixSummary]:
    """Write each prefix's scores as they pass through, and yield them on unchanged.

    A tee rather than a second pass: the pool hands its scores back once, and both the summary
    they are averaged into and this file are written from that single stream, so no list of a
    quarter of a million summaries is ever held.

    Args:
        summaries: Each prefix's scores, in the order the generations file holds them.
        keys: Which prefix each of them answers, in that same order, from `read_prefix_keys`.
            Paired positionally rather than carried on a `PrefixSummary`, which holds floats alone
            so that a worker never sends the log's own strings back.
        path: Where to write, from `paths.prefix_scores_path`. Overwritten if it already exists.
        run: The run these scores belong to, stamped into the file so a reader knows what it holds
            rather than guessing at it from the path.
    Yields:
        Every summary given, unchanged and in order.
    Raises:
        ValueError: If the scores and the keys are of different lengths, which would pair a
            prefix's scores to another prefix.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: dict[str, list] = {name: [] for name in _SCHEMA.names}

    def flush(writer: pq.ParquetWriter) -> None:
        if not columns['case_id']:
            return
        writer.write_table(pa.Table.from_pydict(mapping=columns, schema=_SCHEMA))
        for values in columns.values():
            values.clear()

    with pq.ParquetWriter(where=path, schema=stamped(_SCHEMA, run)) as writer:
        for summary, (case_id, prefix_len) in zip(summaries, keys, strict=True):
            columns['case_id'].append(case_id)
            columns['prefix_len'].append(prefix_len)
            columns['suffix_len'].append(summary.suffix_len)
            for name, value in flatten_scores(summary).items():
                columns[name].append(value)
            if len(columns['case_id']) >= BLOCK:
                flush(writer)
            yield summary
        flush(writer)


def read_prefix_scores(path: Path) -> pd.DataFrame:
    """Read a run's per-prefix scores back.

    Args:
        path: The file `stream_prefix_scores` wrote, from `paths.prefix_scores_path`.
    Returns:
        One row per prefix, under `PREFIX_SCORE_KEYS` and one float column per metric of
        `METRICS`.
    """
    return pq.read_table(source=path).to_pandas()
