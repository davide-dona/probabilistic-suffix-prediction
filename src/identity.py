from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import TypeAdapter


@dataclass(frozen=True)
class DatasetIdentity:
    """Which log, prepared which way: what one preprocessed dataset is.

    A log can be described more than once, since a split strategy and a feature set are choices
    rather than properties of the raw events. Both descriptions read the same `original.csv` and
    keep their own splits, codec and declarative model, so they are the same `name` under
    different `variant`s.
    """

    name: str
    variant: str

    def __str__(self) -> str:
        """What a message calls this dataset, e.g. `sepsis/temporal`.

        Its directories mirror this, but they are `src/paths.py`'s to lay out: this is a name.
        """
        return f'{self.name}/{self.variant}'


@dataclass(frozen=True)
class RunIdentity:
    """One model's one run on one dataset, which every output artifact is named after.

    Carried inside the checkpoint, the generations and the evaluation report rather than spelled
    into their paths, so a file always knows what produced it however it has been moved.
    """

    dataset: DatasetIdentity
    model: str
    tag: str

    def __str__(self) -> str:
        """What a message calls this run, e.g. `sepsis/cvae/temporal/20260809-143043`.

        Its directories mirror this, but they are `src/paths.py`'s to lay out: this is a name.
        """
        return f'{self.dataset.name}/{self.model}/{self.dataset.variant}/{self.tag}'

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


def require_same_dataset(run: RunIdentity, dataset: DatasetIdentity, *, artifact: Path) -> None:
    """Check an artifact was produced for the dataset a config describes.

    Args:
        run: The run the artifact says wrote it.
        dataset: The dataset the config in hand describes.
        artifact: The file being read, named in the error so it says which one to replace.
    Raises:
        ValueError: If the two datasets differ, since everything downstream would then be read
            through the wrong codec, splits and declarative model.
    """
    if run.dataset != dataset:
        raise ValueError(
            f'{artifact} was produced for {run.dataset}, but this config describes {dataset}. '
            'Name the config it belongs to.'
        )


# Built once, since a `TypeAdapter` compiles the schema it validates against.
_RUN_ADAPTER = TypeAdapter(RunIdentity)
