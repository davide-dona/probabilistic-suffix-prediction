from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum


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


class MetricGroup(StrEnum):
    """The evaluation question a metric answers."""

    ACTIVITY = 'activity'
    SUFFIX_LENGTH = 'suffix_length'
    TIME = 'time'
    CONFORMANCE = 'conformance'


@dataclass(frozen=True, slots=True)
class Metric:
    """One evaluation value and the function that computes it for a prefix."""

    key: str
    label: str
    group: MetricGroup
    unit: Unit
    direction: Direction
    compute: Callable[..., float]
    owner: Owner = Owner.MODEL
