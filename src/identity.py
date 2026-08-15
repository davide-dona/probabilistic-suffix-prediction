from __future__ import annotations

from dataclasses import dataclass

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
