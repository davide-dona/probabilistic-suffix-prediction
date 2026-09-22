from dataclasses import dataclass, field, fields
from enum import StrEnum
from typing import Any


class Unit(StrEnum):
    """The physical unit and ordinary display range of a scalar value."""

    SHARE = 'share'
    SCORE = 'score'
    COUNT = 'count'
    DAYS = 'days'
    EVENTS = 'events'

    @property
    def symbol(self) -> str:
        """Return the suffix written after a metric label."""
        return '' if self in (Unit.SHARE, Unit.SCORE, Unit.COUNT) else str(self)

    @property
    def bounds(self) -> tuple[float | None, float | None]:
        """Return fixed display bounds, leaving data-dependent sides open."""
        if self is Unit.SHARE:
            return 0.0, 1.0
        if self is Unit.SCORE:
            return None, None
        return 0.0, None


class Direction(StrEnum):
    """Which values are preferred when comparing models."""

    HIGHER = 'higher'
    LOWER = 'lower'
    ZERO = 'zero'
    NONE = 'none'


class Owner(StrEnum):
    """Whether a value belongs to a model or the observed log."""

    MODEL = 'model'
    LOG = 'log'


@dataclass(frozen=True, slots=True)
class Metric:
    """The field name, unit, direction, and owner of one reported score."""

    key: str
    unit: Unit
    direction: Direction
    owner: Owner = Owner.MODEL


_DECLARATION = 'metric'


def metric(
    *,
    unit: Unit,
    direction: Direction = Direction.NONE,
    owner: Owner = Owner.MODEL,
) -> Any:
    """Declare the semantic properties of a reported score beside its calculation.

    Args:
        unit: Physical unit and display range.
        direction: How model values are ranked.
        owner: Whether the value belongs to a model or observed log.
    Returns:
        A dataclass field carrying the declaration.
    """
    return field(metadata={_DECLARATION: (unit, direction, owner)})


def metrics_of(cls: type) -> tuple[Metric, ...]:
    """Read the declared score fields of a dataclass in their field order.

    Args:
        cls: The score dataclass to inspect.
    Returns:
        One metric per declared field.
    """
    result = []
    for entry in fields(cls):
        if _DECLARATION not in entry.metadata:
            continue
        unit, direction, owner = entry.metadata[_DECLARATION]
        result.append(
            Metric(
                key=entry.name,
                unit=unit,
                direction=direction,
                owner=owner,
            )
        )
    return tuple(result)
