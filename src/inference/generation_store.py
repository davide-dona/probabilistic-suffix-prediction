import json
from collections.abc import Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.configs.schema import SamplingConfig
from src.identity import RunIdentity, stamped
from src.inference.generation import DecodedEvents, Draws, Generation

# Which prefix a row answers, and so what the rows of two runs of one log are matched on. A cut is
# a case and a length, and the pair is unique within a file.
type PrefixKey = tuple[str, int]

# The activity names the suffixes of this file are spelled on, in code order, held in the file's own
# metadata so nothing else has to be read to make sense of it. The same key and the same idea as
# `src/logs/continuations.py`, and both are seeded from `codec.activity.names`, so a suffix written
# here is the string the continuation index holds it as.
_VOCABULARY = b'activities'

# How the activity head was read, for a file whose model drew from it, and absent for one whose
# model read it at its mode. A run's identity does not settle this: the sampler is chosen after
# training and can be changed without the weights moving, so two files of one run are told apart
# by nothing else.
_SAMPLING = b'sampling'

# One run of activities, one character each. A suffix is a string rather than a list of names: the
# names live once in the metadata above, and an edit distance reads a string directly.
_SUFFIX = pa.string()

# One run's wait until each of its activities. Timestamps are these accumulated, so they are
# not written a second time. float32 because that is what the model emits: `denormalize` widens to
# float64 on the way out, and storing that width would double the largest column of the file to
# carry digits the decoder never produced.
_TIMES_TO_NEXT = pa.list_(pa.field(name='element', type=pa.float32()))

# The schema of the Parquet file that holds a run's generations. One row per prefix: the samples
# nest inside it, so nothing describing the prefix is written once per sample.
_SCHEMA = pa.schema(
    [
        ('case_id', pa.large_string()),
        ('prefix_len', pa.int64()),
        # The events before the cut, which a constraint over the whole trace is checked against.
        ('prefix_activities', _SUFFIX),
        # The distinct suffixes drawn for this prefix, each written once however many draws landed
        # on it, and which of them each draw took, in the order they were drawn. The decoder is
        # deterministic given `z`, so a repeated suffix is one answer the model gave twice;
        # `hit_rate_at_k` reads the first k of `generated_draws`, and a mean over the draws is the
        # weighted mean over the distinct suffixes.
        ('generated_suffixes', pa.list_(pa.field(name='element', type=_SUFFIX))),
        ('generated_draws', pa.list_(pa.field(name='element', type=pa.int16()))),
        # Still one entry per draw, in draw order. Two draws of one suffix came from different `z`
        # and the decoder wrote each its own times, so these do not fold the way the activities do.
        (
            'generated_time_to_next_minutes',
            pa.list_(pa.field(name='element', type=_TIMES_TO_NEXT)),
        ),
        (
            'generated_remaining_time_minutes',
            pa.list_(pa.field(name='element', type=pa.float32())),
        ),
        # The suffix written from the mean of `p(z | prefix)`: the model's single answer, drawn once
        # per prefix and the only column comparable against a model that does not sample.
        ('point_activities', _SUFFIX),
        ('point_time_to_next_minutes', _TIMES_TO_NEXT),
        ('point_remaining_time_minutes', pa.float32()),
        ('true_activities', _SUFFIX),
        ('true_time_to_next_minutes', _TIMES_TO_NEXT),
        ('true_remaining_time_minutes', pa.float32()),
    ]
)

_COMPRESSION = 'zstd'

# Worth the write time: generation is GPU-bound over hours, where the whole file costs under a
# minute more to compress at this level and comes out a tenth smaller than at the default.
_COMPRESSION_LEVEL = 9

# The float columns, named as Parquet names their leaves. Byte-stream-split splits a float into its
# four byte planes before compressing, so the exponents of a column line up and zstd has something
# repetitive to find; on the waits, which are continuous and share nothing as whole values, it is
# the difference between compressing and not.
_FLOAT_LEAVES = [
    'generated_time_to_next_minutes.list.element.list.element',
    'generated_remaining_time_minutes.list.element',
    'point_time_to_next_minutes.list.element',
    'point_remaining_time_minutes',
    'true_time_to_next_minutes.list.element',
    'true_remaining_time_minutes',
]


def open_generations(
    path: Path,
    run: RunIdentity,
    *,
    vocabulary: Sequence[str],
    sampling: SamplingConfig | None,
) -> pq.ParquetWriter:
    """Open a Parquet file for writing generations.

    Args:
        path: The file to write, its directory already made, from `paths.GENERATIONS.prepare`.
            Overwritten if it already exists.
        run: The run these generations come from, stamped into the file so evaluation can read it
            back instead of guessing at it.
        vocabulary: The activity names the suffixes are spelled on, in code order, from
            `ActivityCodes.vocabulary`. Written into the file so it says what its own characters
            mean.
        sampling: How the activity head was read, for a model that draws from it, or None for one
            that reads it at its mode. Written in for the same reason the vocabulary is: the
            sampler is chosen after training, so the run's identity alone does not say which one
            produced this file.
    Returns:
        A writer bound to the generations schema, to be used as a context manager: closing it is
        what writes the file's footer.
    """
    metadata = {_VOCABULARY: json.dumps(list(vocabulary))}
    if sampling is not None:
        metadata[_SAMPLING] = sampling.model_dump_json()
    schema = _SCHEMA.with_metadata(metadata)
    return pq.ParquetWriter(
        where=path,
        schema=stamped(schema, run),
        compression=_COMPRESSION,
        compression_level=_COMPRESSION_LEVEL,
        # Dictionary encoding takes precedence over byte-stream-split wherever it is left on, and a
        # column of continuous waits has no dictionary worth building, so the two are set together.
        use_dictionary=False,
        use_byte_stream_split=_FLOAT_LEAVES,
    )


def read_vocabulary(parquet: pq.ParquetFile) -> tuple[str, ...]:
    """Read the activity names a generations file spells its suffixes on.

    Args:
        parquet: The file, already open.
    Returns:
        The names in code order, which is what `ActivityCodes.of` seeds back into a codebook and
        what a reader compares against the continuation index's own.
    Raises:
        ValueError: If the file carries none, and so predates the vocabulary it should name itself
            by.
    """
    metadata = parquet.schema_arrow.metadata or {}
    if _VOCABULARY not in metadata:
        raise ValueError(
            'this file does not say what its activity codes mean. Write it again with the pipeline '
            'that produces it.'
        )
    return tuple(json.loads(metadata[_VOCABULARY]))


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
            'generated_suffixes': list(generation.samples.suffixes),
            'generated_draws': list(generation.samples.taken),
            'generated_time_to_next_minutes': [
                events.time_to_next_minutes for events in generation.samples.events
            ],
            'generated_remaining_time_minutes': [
                events.remaining_time_minutes for events in generation.samples.events
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
        suffixes = columns['generated_suffixes'][position]
        taken = columns['generated_draws'][position]
        generations.append(
            Generation(
                case_id=columns['case_id'][position],
                prefix_activities=columns['prefix_activities'][position],
                samples=Draws(
                    suffixes=tuple(suffixes),
                    taken=tuple(taken),
                    events=[
                        DecodedEvents(
                            activities=suffixes[index],
                            time_to_next_minutes=time_to_next_minutes,
                            remaining_time_minutes=remaining_time_minutes,
                        )
                        for index, time_to_next_minutes, remaining_time_minutes in zip(
                            taken,
                            columns['generated_time_to_next_minutes'][position],
                            columns['generated_remaining_time_minutes'][position],
                            strict=True,
                        )
                    ],
                ),
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
