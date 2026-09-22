from collections.abc import Sequence
from dataclasses import fields
from typing import Self


def mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean, or zero for an empty sequence."""
    return sum(values) / len(values) if values else 0.0


class ScalarRecord:
    """Support aggregation without coupling computed values to storage or logging."""

    __slots__ = ()

    @classmethod
    def mean(cls, values: Sequence[Self]) -> Self:
        """Average records field by field, returning zeros for an empty sequence."""
        return cls(
            **{
                entry.name: mean([getattr(value, entry.name) for value in values])
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
