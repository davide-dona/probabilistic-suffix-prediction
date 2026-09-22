import re
from dataclasses import asdict, dataclass
from typing import Self

# Regular expressions for validating run identity components.
_DATASET = re.compile(r'[a-z0-9][a-z0-9-]*')
_MODEL = re.compile(r'[a-z0-9][a-z0-9_]*')
_RUN_ID = re.compile(r'\d{8}-\d{6}-\d{6}')


def _validated(value: object, pattern: re.Pattern[str], field: str) -> str:
    """Require a string that matches the given pattern."""
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f'Invalid run identity {field}: {value!r}')
    return value


def validate_dataset(value: object) -> str:
    """Require a dataset name that can safely form one path component."""
    return _validated(value, _DATASET, 'dataset')


def validate_run_id(value: object) -> str:
    """Require the timestamp-based identifier used for one invocation."""
    return _validated(value, _RUN_ID, 'run_id')


@dataclass(frozen=True)
class RunIdentity:
    """The dataset, model, and invocation that identify one training run.

    Its string form is the relative path ``dataset/model/run_id`` used beneath stage output
    directories. It deliberately excludes a checkpoint hash: several checkpoints or downstream
    artifacts can belong to the same training run.
    """

    dataset: str
    model: str
    run_id: str

    def __post_init__(self) -> None:
        validate_dataset(self.dataset)
        _validated(self.model, _MODEL, 'model')
        validate_run_id(self.run_id)

    def __str__(self) -> str:
        return f'{self.dataset}/{self.model}/{self.run_id}'

    def as_dict(self) -> dict[str, str]:
        """Return the representation embedded in checkpoints and configuration files."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Construct an identity from a checkpoint or configuration mapping."""
        if not isinstance(value, dict) or set(value) != {'dataset', 'model', 'run_id'}:
            raise ValueError('Invalid run identity')
        return cls(dataset=value['dataset'], model=value['model'], run_id=value['run_id'])

    @classmethod
    def from_metadata(cls, metadata: dict[str, str]) -> Self:
        """Read the run portion of artifact metadata, allowing unrelated metadata keys."""
        return cls(
            dataset=metadata.get('dataset', ''),
            model=metadata.get('model', ''),
            run_id=metadata.get('run_id', ''),
        )
