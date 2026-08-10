import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.identity import RunIdentity
from src.inference.generation import DecodedEvents, Generation

# The schema metadata key the writing run's identity is stored under, so a generations file says
# what produced it rather than leaving that to be read off the path it happens to sit at.
_RUN_KEY = b'run'

# One run of activity names, the shape every activity column of the schema is built from.
_ACTIVITIES = pa.list_(pa.field(name='element', type=pa.string()))

# The schema of the Parquet file that holds a run's generations. One row per prefix: the samples
# nest inside it, so nothing describing the prefix is written once per sample.
_SCHEMA = pa.schema(
    [
        ('case_id', pa.large_string()),
        ('prefix_len', pa.int64()),
        # The events before the cut, which a constraint over the whole trace is checked against.
        ('prefix_activities', _ACTIVITIES),
        # One entry per draw of z, in the order they were drawn: `hit_rate_at_k` reads the first k.
        ('generated_activities', pa.list_(pa.field(name='element', type=_ACTIVITIES))),
        ('generated_remaining_time_minutes', pa.list_(pa.field(name='element', type=pa.float64()))),
        # The suffix written from the mean of `p(z | prefix)`: the model's single answer, drawn once
        # per prefix and the only column comparable against a model that does not sample.
        ('point_activities', _ACTIVITIES),
        ('point_remaining_time_minutes', pa.float64()),
        ('true_activities', _ACTIVITIES),
        ('true_remaining_time_minutes', pa.float64()),
    ]
)


def open_generations(path: Path, run: RunIdentity) -> pq.ParquetWriter:
    """Open a Parquet file for writing generations, creating its parent directories if needed.

    Args:
        path: The file to write, from `paths.generations_path`. Overwritten if it already exists.
        run: The run these generations come from, stamped into the file so evaluation can read it
            back instead of guessing at it.
    Returns:
        A writer bound to the generations schema, to be used as a context manager: closing it is
        what writes the file's footer.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = _SCHEMA.with_metadata({_RUN_KEY: json.dumps(asdict(run)).encode()})
    return pq.ParquetWriter(where=path, schema=schema)


def read_run_identity(parquet: pq.ParquetFile) -> RunIdentity:
    """Read back which run wrote a generations file.

    Args:
        parquet: The generations file, already open.
    Returns:
        The identity `open_generations` stamped into it.
    Raises:
        ValueError: If the file carries none, and so predates the identity it should name itself by.
    """
    metadata = parquet.schema_arrow.metadata or {}
    if _RUN_KEY not in metadata:
        raise ValueError(
            'this generations file does not say which run wrote it. Generate it again with '
            '`python -m pipelines.generate`.'
        )
    return RunIdentity.from_json(metadata[_RUN_KEY])


def table_from_generations(generations: list[Generation]) -> pa.Table:
    """Lay one batch's generations out as a table, one row per prefix.

    Args:
        generations: The model's answers, in the order `generate_batch` returned them.
    Returns:
        The rows as one table, ready to be written as one block of the file.
    """
    # Dicts here because they are what arrow's constructor takes, keyed by the schema's own names.
    rows = [
        {
            'case_id': generation.case_id,
            'prefix_len': generation.prefix_len,
            'prefix_activities': generation.prefix_activities,
            'generated_activities': [sample.activities for sample in generation.samples],
            'generated_remaining_time_minutes': [
                sample.remaining_time_minutes for sample in generation.samples
            ],
            'point_activities': generation.point.activities,
            'point_remaining_time_minutes': generation.point.remaining_time_minutes,
            'true_activities': generation.truth.activities,
            'true_remaining_time_minutes': generation.truth.remaining_time_minutes,
        }
        for generation in generations
    ]
    return pa.Table.from_pylist(mapping=rows, schema=_SCHEMA)


def read_generation_block(parquet: pq.ParquetFile, block: int) -> list[Generation]:
    """Read one block of a generations file back.

    A block is what one call to `table_from_generations` wrote, held as a Parquet row group, and
    this is the inverse over a single one. A prefix cannot straddle a block, since a row holds one,
    which is what makes a block an independent unit of work: what this costs to read and to score
    is set by the batch a run wrote rather than by the size of the split.

    Args:
        parquet: The generations file to read from, already open.
        block: Which of its blocks to decode, in `range(parquet.num_row_groups)`.
    Returns:
        The generation for each prefix of the block, in the order they were written.
    """
    frame = parquet.read_row_group(block).to_pandas()
    return [_generation_from_row(row) for _, row in frame.iterrows()]


def _generation_from_row(row: pd.Series) -> Generation:
    """Read one prefix's generation back out of the row a generations file holds it as.

    What lets a written file be scored through the same `score_generation` a training run reports
    from.

    Args:
        row: One row of a generations file, holding a prefix and every suffix drawn for it.
    Returns:
        The same generation `generate_batch` produced. Parquet hands the activity columns back as
        arrays, so they are copied into lists: `DecodedEvents` promises lists, and the edit
        distance that reads them is quicker to index for it. Copying is also what lets the row
        group behind them be dropped once the prefix has been scored.
    """
    return Generation(
        case_id=str(row.case_id),
        prefix_activities=list(row.prefix_activities),
        samples=[
            DecodedEvents(
                activities=list(activities),
                remaining_time_minutes=float(remaining_time_minutes),
            )
            for activities, remaining_time_minutes in zip(
                row.generated_activities, row.generated_remaining_time_minutes, strict=True
            )
        ],
        point=DecodedEvents(
            activities=list(row.point_activities),
            remaining_time_minutes=float(row.point_remaining_time_minutes),
        ),
        truth=DecodedEvents(
            activities=list(row.true_activities),
            remaining_time_minutes=float(row.true_remaining_time_minutes),
        ),
    )
