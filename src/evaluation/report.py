import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

from pydantic import TypeAdapter

from src.evaluation.metrics import EvaluationMetrics
from src.identity import RunIdentity


@dataclass(frozen=True)
class EvaluationReport:
    """Everything one evaluation produced, under the identity of the run it scored."""

    run: RunIdentity
    metrics: EvaluationMetrics

    @classmethod
    def read(cls, path: str | Path) -> Self:
        """Read a report back from the JSON `write` produced.

        Args:
            path: The report to read, e.g. from `paths.evaluation_path`.
        Returns:
            The report, validated against the current schema so a report written before a metric
            was added or renamed fails here rather than on a missing key further down.
        """
        return _ADAPTER.validate_json(Path(path).read_bytes())

    def write(self, path: str | Path) -> Path:
        """
        Write the report as JSON, creating parent directories.

        Args:
            path: Where to write, e.g. the generations file's own path with a `.json` suffix.
        Returns:
            The path written to.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=4))
        return path


# Built once, since a `TypeAdapter` compiles the schema it validates against.
_ADAPTER = TypeAdapter(EvaluationReport)
