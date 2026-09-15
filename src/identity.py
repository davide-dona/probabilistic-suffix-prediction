import re
from dataclasses import asdict, dataclass
from typing import Self

_DATASET = re.compile(r'[a-z0-9][a-z0-9-]*')
_MODEL = re.compile(r'[a-z0-9][a-z0-9_]*')
_RUN_ID = re.compile(r'\d{8}-\d{6}-\d{6}')


def _validated(value: object, pattern: re.Pattern[str], field: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f'Invalid run identity {field}: {value!r}')
    return value


def validate_dataset(value: object) -> str:
    return _validated(value, _DATASET, 'dataset')


def validate_run_id(value: object) -> str:
    return _validated(value, _RUN_ID, 'run_id')


@dataclass(frozen=True)
class RunIdentity:
    """The stable identity shared by every artifact derived from one training run."""

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
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if not isinstance(value, dict) or set(value) != {'dataset', 'model', 'run_id'}:
            raise ValueError('Invalid run identity')
        return cls(dataset=value['dataset'], model=value['model'], run_id=value['run_id'])

    @classmethod
    def from_metadata(cls, metadata: dict[str, str]) -> Self:
        return cls(
            dataset=metadata.get('dataset', ''),
            model=metadata.get('model', ''),
            run_id=metadata.get('run_id', ''),
        )
