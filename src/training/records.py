from collections.abc import Mapping, Sequence
from dataclasses import asdict, fields
from typing import Self

import wandb


class ScalarRecord:
    """Support aggregation of dataclass training records."""

    __slots__ = ()

    @classmethod
    def mean(cls, values: Sequence[Self]) -> Self:
        """Average records field by field, returning zeros for an empty sequence."""
        return cls(
            **{
                entry.name: sum(getattr(value, entry.name) for value in values) / len(values)
                if values
                else 0.0
                for entry in fields(cls)
            }
        )

    def __add__(self, other: Self) -> Self:
        """Add corresponding fields, as required for batch totals."""
        return type(self)(
            **{
                entry.name: getattr(self, entry.name) + getattr(other, entry.name)
                for entry in fields(self)
            }
        )

    def __truediv__(self, divisor: float) -> Self:
        """Divide every field by the same population size."""
        return type(self)(
            **{entry.name: getattr(self, entry.name) / divisor for entry in fields(self)}
        )


def log_records(records: Mapping[str, object], *, step: int) -> None:
    """Log dataclass records under caller-selected namespaces."""
    payload = {
        f'{namespace}/{key}': value
        for namespace, record in records.items()
        for key, value in asdict(record).items()
    }
    wandb.log(payload, step=step)
