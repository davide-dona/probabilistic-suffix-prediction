from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from pydantic import TypeAdapter


@dataclass(frozen=True)
class RunIdentity:
    """One model's one run on one dataset, which every output artifact is named after.

    Carried inside the checkpoint, the generations and the evaluation report rather than spelled
    into their paths, so a file always knows what produced it however it has been moved.
    """

    dataset: str
    model: str
    tag: str

    def __str__(self) -> str:
        """What a message calls this run, e.g. `sepsis/cvae/20260809-143043`.

        Its directories mirror this, but they are `src/paths.py`'s to lay out: this is a name.
        """
        return f'{self.dataset}/{self.model}/{self.tag}'

    @classmethod
    def from_dict(cls, data: dict) -> RunIdentity:
        """Rebuild a run's identity from the plain data an artifact stores it as.

        Args:
            data: What `dataclasses.asdict` produced, as read back from a checkpoint.
        Returns:
            The identity those fields describe.
        Raises:
            pydantic.ValidationError: If the fields are missing or of the wrong type.
        """
        return _RUN_ADAPTER.validate_python(data)

    @classmethod
    def from_json(cls, data: bytes | str) -> RunIdentity:
        """Rebuild a run's identity from the JSON an artifact stores it as.

        Args:
            data: The JSON `json.dumps(dataclasses.asdict(run))` produced, as read back from a
                generations file's metadata.
        Returns:
            The identity those fields describe.
        Raises:
            pydantic.ValidationError: If the JSON is malformed or does not describe a run.
        """
        return _RUN_ADAPTER.validate_json(data)


# Built once, since a `TypeAdapter` compiles the schema it validates against.
_RUN_ADAPTER = TypeAdapter(RunIdentity)


def group_by_model(runs: Iterable[tuple[RunIdentity, Path]]) -> dict[str, dict[str, Path]]:
    """Group a set of run artifacts by the log they belong to, one run of each model per log.

    What a figure or a table is built from is a set of files nobody necessarily typed one by one,
    since a directory can be swept for them, and a log holding two runs of one model would draw one
    line, panel or column over another without saying so.

    Args:
        runs: Which run wrote each file, in the order they were named. How that was read is the
            caller's: a report says so in its JSON, a generations file in its Parquet footer.
    Returns:
        The file of each model, keyed by the log's own name, both in the order they were given.
    Raises:
        ValueError: If one log is given two runs of the same model.
    """
    grouped: dict[str, dict[str, Path]] = {}
    for run, file in runs:
        models = grouped.setdefault(run.dataset, {})
        already = models.get(run.model)
        if already is not None:
            raise ValueError(
                f'{run.dataset} is given two runs of {run.model}:\n  {already}\n  {file}\n'
                'Name only the run you want, or move the stale one out of the swept directory.'
            )
        models[run.model] = file
    return grouped
