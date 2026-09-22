import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Self

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import TypeAdapter, ValidationError

from src.evaluation.metrics import METRICS, PreparedPrefix
from src.evaluation.metrics.definitions import MetricGroup
from src.inference.generation import Generation
from src.inference.generation_store import PrefixKey
from src.logs.declare import ConformanceChecker
from src.runs.artifacts import read_metadata, with_metadata
from src.runs.identity import RunIdentity

GROUPS = tuple(MetricGroup)


@dataclass(frozen=True)
class ScoreGroups:
    """Every metric value for a prefix or an aggregate, grouped by evaluation question."""

    activity: dict[str, float]
    suffix_length: dict[str, float]
    time: dict[str, float]
    conformance: dict[str, float]

    @classmethod
    def of(cls, values: dict[str, float]) -> 'ScoreGroups':
        """Group a complete mapping of registered metric values."""
        expected = set(METRICS.entries)
        missing = expected - set(values)
        extra = set(values) - expected
        if missing or extra:
            raise ValueError(
                'metric values differ from the registry: '
                f'missing {sorted(missing)}, extra {sorted(extra)}.'
            )
        grouped = {
            group: {
                key: values[key] for key, metric in METRICS.entries.items() if metric.group is group
            }
            for group in GROUPS
        }
        return cls(**grouped)

    @classmethod
    def mean(cls, values: Sequence['ScoreGroups']) -> 'ScoreGroups':
        """Average complete score mappings, field by field."""
        return cls.of(
            {
                key: sum(value.flatten()[key] for value in values) / len(values) if values else 0.0
                for key in METRICS.entries
            }
        )

    def flatten(self) -> dict[str, float]:
        """Return all values in registry declaration order."""
        groups = {
            MetricGroup.ACTIVITY: self.activity,
            MetricGroup.SUFFIX_LENGTH: self.suffix_length,
            MetricGroup.TIME: self.time,
            MetricGroup.CONFORMANCE: self.conformance,
        }
        return {key: groups[metric.group][key] for key, metric in METRICS.entries.items()}


@dataclass(frozen=True)
class PrefixSummary:
    """Scores for one generated prefix, returned by a worker."""

    prefix_len: int
    suffix_len: int
    scores: ScoreGroups

    @classmethod
    def of(cls, generation: Generation, *, checker: ConformanceChecker) -> 'PrefixSummary':
        """Score one generated suffix against truth and constraints."""
        context = PreparedPrefix.of(generation, checker=checker)
        return cls(
            prefix_len=generation.prefix_len,
            suffix_len=len(generation.truth),
            scores=ScoreGroups.of(
                {key: metric.compute(context) for key, metric in METRICS.entries.items()}
            ),
        )


@dataclass(frozen=True)
class LengthSummary:
    """Mean scores for prefixes with one shared length."""

    length: int
    prefixes: int
    activity: dict[str, float]
    suffix_length: dict[str, float]
    time: dict[str, float]
    conformance: dict[str, float]

    @classmethod
    def of(cls, prefixes: Sequence[PrefixSummary], *, length: int) -> 'LengthSummary':
        """Aggregate scores for prefixes with a common length."""
        scores = ScoreGroups.mean([prefix.scores for prefix in prefixes])
        return cls(
            length=length,
            prefixes=len(prefixes),
            activity=scores.activity,
            suffix_length=scores.suffix_length,
            time=scores.time,
            conformance=scores.conformance,
        )


def _by_length(buckets: dict[int, list[PrefixSummary]]) -> list[LengthSummary]:
    """Summarize length buckets in ascending order."""
    return [LengthSummary.of(buckets[length], length=length) for length in sorted(buckets)]


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate evaluation scores for one run."""

    prefixes: int
    activity: dict[str, float]
    suffix_length: dict[str, float]
    time: dict[str, float]
    conformance: dict[str, float]
    by_prefix_length: list[LengthSummary]
    by_suffix_length: list[LengthSummary]

    @classmethod
    def of(cls, prefixes: Iterable[PrefixSummary]) -> 'EvaluationSummary':
        """Aggregate prefix scores overall and by prefix and suffix length."""
        prefix_buckets: dict[int, list[PrefixSummary]] = {}
        suffix_buckets: dict[int, list[PrefixSummary]] = {}
        for prefix in prefixes:
            prefix_buckets.setdefault(prefix.prefix_len, []).append(prefix)
            suffix_buckets.setdefault(prefix.suffix_len, []).append(prefix)
        every_prefix = [prefix for bucket in prefix_buckets.values() for prefix in bucket]
        scores = ScoreGroups.mean([prefix.scores for prefix in every_prefix])
        return cls(
            prefixes=len(every_prefix),
            activity=scores.activity,
            suffix_length=scores.suffix_length,
            time=scores.time,
            conformance=scores.conformance,
            by_prefix_length=_by_length(prefix_buckets),
            by_suffix_length=_by_length(suffix_buckets),
        )


type Summarized = PrefixSummary | LengthSummary | EvaluationSummary


def flatten_scores(summary: Summarized) -> dict[str, float]:
    """Flatten a summary's grouped scores into a registry-ordered mapping."""
    if isinstance(summary, PrefixSummary):
        return summary.scores.flatten()
    return ScoreGroups(
        activity=summary.activity,
        suffix_length=summary.suffix_length,
        time=summary.time,
        conformance=summary.conformance,
    ).flatten()


@dataclass(frozen=True)
class EvaluationReport:
    """Evaluation results and source artifact provenance."""

    metadata: dict[str, str]
    summary: EvaluationSummary

    @classmethod
    def read(cls, path: str | Path) -> Self:
        """Read and validate a JSON evaluation report."""
        path = Path(path)
        report = _REPORT_ADAPTER.validate_python(json.loads(path.read_bytes()))
        RunIdentity.from_metadata(report.metadata)
        return report

    def write(self, path: str | Path) -> Path:
        """Write the report as JSON."""
        path = Path(path)
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(asdict(self), indent=4))
        temporary.replace(path)
        return path


_REPORT_ADAPTER = TypeAdapter(EvaluationReport)


class Axis(StrEnum):
    """Metric aggregation levels."""

    OVERALL = 'overall'
    PREFIX = 'prefix'
    SUFFIX = 'suffix'


REPORT_COLUMNS = ('dataset', 'model', 'axis', 'length', 'prefixes', 'metric', 'value')


def _group_by_model(reports: Iterable[tuple[dict[str, str], Path]]) -> dict[str, dict[str, Path]]:
    """Group one evaluation artifact per model under each dataset, rejecting duplicates."""
    grouped: dict[str, dict[str, Path]] = {}
    for metadata, path in reports:
        dataset, model = metadata['dataset'], metadata['model']
        models = grouped.setdefault(dataset, {})
        if model in models:
            raise ValueError(f'{dataset} has two reports for {model}: {models[model]}, {path}')
        models[model] = path
    return grouped


def _report_rows(report: EvaluationReport) -> list[dict[str, object]]:
    """Flatten a report into one row per metric and breakdown."""
    metadata, summary = report.metadata, report.summary
    identity = {'dataset': metadata['dataset'], 'model': metadata['model']}
    rows: list[dict[str, object]] = [
        identity
        | {
            'axis': Axis.OVERALL,
            'length': None,
            'prefixes': summary.prefixes,
            'metric': metric,
            'value': value,
        }
        for metric, value in flatten_scores(summary).items()
    ]
    breakdowns = ((Axis.PREFIX, summary.by_prefix_length), (Axis.SUFFIX, summary.by_suffix_length))
    rows.extend(
        identity
        | {
            'axis': axis,
            'length': entry.length,
            'prefixes': entry.prefixes,
            'metric': metric,
            'value': value,
        }
        for axis, breakdown in breakdowns
        for entry in breakdown
        for metric, value in flatten_scores(entry).items()
    )
    return rows


def read_reports(files: Sequence[Path]) -> pd.DataFrame:
    """Load reports into the dataframe consumed by tables and figures."""
    reports: list[tuple[Path, EvaluationReport]] = []
    for file in files:
        try:
            reports.append((file, EvaluationReport.read(file)))
        except ValidationError as error:
            raise ValueError(f'{file} is not an evaluation report: {error}') from error
    _group_by_model((report.metadata, file) for file, report in reports)
    rows = [row for _, report in reports for row in _report_rows(report)]
    frame = pd.DataFrame(rows, columns=list(REPORT_COLUMNS))
    return frame.astype({'length': 'Int64', 'prefixes': 'Int64', 'value': 'float64'})


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
    """Write per-prefix scores while yielding the original summaries."""
    columns: dict[str, list] = {name: [] for name in _PREFIX_SCORE_SCHEMA.names}

    def flush(writer: pq.ParquetWriter) -> None:
        if not columns['case_id']:
            return
        writer.write_table(pa.Table.from_pydict(mapping=columns, schema=_PREFIX_SCORE_SCHEMA))
        for values in columns.values():
            values.clear()

    temporary = path.with_suffix('.parquet.tmp')
    with pq.ParquetWriter(
        where=temporary, schema=with_metadata(_PREFIX_SCORE_SCHEMA, metadata)
    ) as writer:
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
    temporary.replace(path)


def require_columns(path: Path, columns: Sequence[str]) -> None:
    """Require the requested columns in a score file's schema."""
    absent = [key for key in columns if key not in pq.read_schema(where=path).names]
    if absent:
        raise ValueError(f'{path} is missing required score columns: {", ".join(absent)}.')


def read_prefix_scores(path: Path, *, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Read all or selected per-prefix score columns."""
    require_columns(path, columns or (*PREFIX_SCORE_KEYS, *METRICS.entries))
    wanted = None if columns is None else list(columns)
    return pq.read_table(source=path, columns=wanted).to_pandas()


def score_files(reports: Sequence[Path]) -> dict[str, dict[str, Path]]:
    """Find score files beside reports and group them by dataset."""
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
