from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.identity import RunIdentity, stamped
from src.inference.generation import DecodedEvents, Generation

# Which prefix a row answers, and so what the rows of two runs of one log are matched on. A cut is
# a case and a length, and the pair is unique within a file.
type PrefixKey = tuple[str, int]

# One run of activity names, the shape every activity column of the schema is built from.
_ACTIVITIES = pa.list_(pa.field(name='element', type=pa.string()))

# One run's wait until each of its activities. Timestamps are these accumulated, so they are
# not written a second time.
_TIMES_TO_NEXT = pa.list_(pa.field(name='element', type=pa.float64()))

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
        (
            'generated_time_to_next_minutes',
            pa.list_(pa.field(name='element', type=_TIMES_TO_NEXT)),
        ),
        ('generated_remaining_time_minutes', pa.list_(pa.field(name='element', type=pa.float64()))),
        # The suffix written from the mean of `p(z | prefix)`: the model's single answer, drawn once
        # per prefix and the only column comparable against a model that does not sample.
        ('point_activities', _ACTIVITIES),
        ('point_time_to_next_minutes', _TIMES_TO_NEXT),
        ('point_remaining_time_minutes', pa.float64()),
        ('true_activities', _ACTIVITIES),
        ('true_time_to_next_minutes', _TIMES_TO_NEXT),
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
    return pq.ParquetWriter(where=path, schema=stamped(_SCHEMA, run))


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
            'generated_time_to_next_minutes': [
                sample.time_to_next_minutes for sample in generation.samples
            ],
            'generated_remaining_time_minutes': [
                sample.remaining_time_minutes for sample in generation.samples
            ],
            'point_activities': generation.point.activities,
            'point_time_to_next_minutes': generation.point.time_to_next_minutes,
            'point_remaining_time_minutes': generation.point.remaining_time_minutes,
            'true_activities': generation.truth.activities,
            'true_time_to_next_minutes': generation.truth.time_to_next_minutes,
            'true_remaining_time_minutes': generation.truth.remaining_time_minutes,
        }
        for generation in generations
    ]
    return pa.Table.from_pylist(mapping=rows, schema=_SCHEMA)


def read_suffixes(path: Path) -> Iterator[tuple[list[str], list[str]]]:
    """Walk a generations file for the pair of suffixes each prefix is summarized by.

    One of three readers over this file, each reading the columns its caller needs and no more:
    a walk that decoded every draw would cost this one ten times what it costs, on a file of a
    quarter of a million rows.

    Args:
        path: The generations file, opened and closed here.
    Yields:
        The true suffix and the first generated suffix of each prefix, in the order the file holds
        them, as activity names.
    """
    with pq.ParquetFile(path) as parquet:
        for batch in parquet.iter_batches(columns=['true_activities', 'generated_activities']):
            truths = batch.column('true_activities').to_pylist()
            samples = pc.list_element(batch.column('generated_activities'), 0).to_pylist()
            yield from zip(truths, samples, strict=True)


def read_prefix_keys(path: Path) -> list[PrefixKey]:
    """Read which prefixes a generations file answers, without decoding a single suffix.

    Args:
        path: The generations file, opened and closed here.
    Returns:
        The key of every row, in the order the file holds them, which is the order a walk of its
        blocks scores them in. Only the two columns that identify a prefix are read, which is cheap
        even on a file of a quarter of a million rows.
    """
    table = pq.read_table(source=path, columns=['case_id', 'prefix_len'])
    return list(
        zip(
            table.column('case_id').to_pylist(),
            table.column('prefix_len').to_pylist(),
            strict=True,
        )
    )


def read_samples(path: Path) -> Iterator[tuple[PrefixKey, list[str], list[list[str]]]]:
    """Walk a generations file for the prefix, the true suffix and every draw of each row.

    Args:
        path: The generations file, opened and closed here.
    Yields:
        Which prefix each row answers, the suffix that truly followed it and every suffix generated
        for it, in the order the file holds them, as activity names.
    """
    columns = ['case_id', 'prefix_len', 'true_activities', 'generated_activities']
    with pq.ParquetFile(path) as parquet:
        for batch in parquet.iter_batches(columns=columns):
            keys = zip(
                batch.column('case_id').to_pylist(),
                batch.column('prefix_len').to_pylist(),
                strict=True,
            )
            truths = batch.column('true_activities').to_pylist()
            samples = batch.column('generated_activities').to_pylist()
            yield from zip(keys, truths, samples, strict=True)


def read_generation_block(parquet: pq.ParquetFile, block: int) -> list[Generation]:
    """Read one block of a generations file back.

    A block is what one call to `table_from_generations` wrote, held as a Parquet row group, and
    this is the inverse over a single one. A prefix cannot cross a block, since a row holds one,
    which is what makes a block an independent unit of work: what this costs to read and to score
    is set by the batch a run wrote rather than by the size of the split.

    Args:
        parquet: The generations file to read from, already open.
        block: Which of its blocks to decode, in `range(parquet.num_row_groups)`.
    Returns:
        The generation for each prefix of the block, in the order they were written, exactly as
        `generate_batch` produced them. Read a column at a time: Arrow converts a whole column to
        Python in one call, and the lists that come back are the plain lists `DecodedEvents`
        promises, so nothing is copied a second time and the block itself is dropped as soon as
        this returns.
    """
    table = parquet.read_row_group(block)
    columns = {name: table.column(name).to_pylist() for name in table.schema.names}

    generations = []
    for position in range(table.num_rows):
        generations.append(
            Generation(
                case_id=columns['case_id'][position],
                prefix_activities=columns['prefix_activities'][position],
                samples=[
                    DecodedEvents(
                        activities=activities,
                        time_to_next_minutes=time_to_next_minutes,
                        remaining_time_minutes=remaining_time_minutes,
                    )
                    for activities, time_to_next_minutes, remaining_time_minutes in zip(
                        columns['generated_activities'][position],
                        columns['generated_time_to_next_minutes'][position],
                        columns['generated_remaining_time_minutes'][position],
                        strict=True,
                    )
                ],
                point=DecodedEvents(
                    activities=columns['point_activities'][position],
                    time_to_next_minutes=columns['point_time_to_next_minutes'][position],
                    remaining_time_minutes=columns['point_remaining_time_minutes'][position],
                ),
                truth=DecodedEvents(
                    activities=columns['true_activities'][position],
                    time_to_next_minutes=columns['true_time_to_next_minutes'][position],
                    remaining_time_minutes=columns['true_remaining_time_minutes'][position],
                ),
            )
        )
    return generations
