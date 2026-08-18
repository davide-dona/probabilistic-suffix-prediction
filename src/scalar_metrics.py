from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, fields
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


def mean(values: Sequence[float]) -> float:
    """The mean of `values`, or 0.0 if there are none."""
    return sum(values) / len(values) if values else 0.0


class Unit(StrEnum):
    """What a metric's number is: what is written after its name, and the range it is read over.

    The two are one question. A share is a share whether it is drawn or tabulated, and it is
    exactly what carries no unit and spans `[0, 1]`; a count of days is what carries `days` and
    starts at 0 with no ceiling.
    """

    SHARE = 'share'  # A fraction of something, in [0, 1]
    SCORE = 'score'  # Dimensionless with no fixed range, e.g. a distance between two sets
    DAYS = 'days'
    EVENTS = 'events'

    @property
    def symbol(self) -> str:
        """What is written after the metric's name, or `''` for a dimensionless number."""
        return '' if self in (Unit.SHARE, Unit.SCORE) else str(self)

    @property
    def bounds(self) -> tuple[float | None, float | None]:
        """The range a value of this unit is read over, either end `None` where the data sets it.

        A share spans the whole of `[0, 1]`, so a panel of one shows where a model sits within
        what the metric can be rather than within the range these models happen to cover. An
        absolute error starts at 0 and has no ceiling. A score is bounded at neither end.
        """
        if self is Unit.SHARE:
            return 0.0, 1.0
        if self is Unit.SCORE:
            return None, None
        return 0.0, None


@dataclass(frozen=True)
class Metric:
    """What one of the numbers a report carries is: the name it goes by, its unit, and which way
    is better.

    What it is called on a page is not here: a table holding one estimator throughout names a
    metric more shortly than a figure drawing that estimator against another, so the name belongs
    to the appearance rather than to the metric.
    """

    key: str  # The field it was declared on, which is the name it has in a report and a frame
    unit: Unit
    # `None` for a metric with no best value, e.g. a diversity or a length.
    higher_is_better: bool | None = None


# Where a field's declaration is kept. Read through `metrics_of` rather than directly.
_DECLARATION = 'metric'


def metric(*, unit: Unit, higher_is_better: bool | None = None) -> Any:
    """Declare a float field as one of the numbers a report carries.

    Args:
        unit: What the number is, which gives it both its printed unit and its range.
        higher_is_better: Which way is better, or `None` for a metric with no best value.
    Returns:
        A `dataclasses.field` carrying the declaration. The metric's key is the field's own name,
        so it is never written twice and cannot drift from what a report holds.
    """
    return field(metadata={_DECLARATION: (unit, higher_is_better)})


def metrics_of(cls: type) -> tuple[Metric, ...]:
    """Every field of a dataclass declared with `metric`, in declaration order.

    Args:
        cls: The dataclass to read. A field declared without `metric` is not one of the numbers a
            report carries and is skipped, which is what lets a class hold both.
    Returns:
        One `Metric` per declared field, keyed by the field's name.
    """
    declared = []
    for entry in fields(cls):
        if _DECLARATION not in entry.metadata:
            continue
        unit, higher_is_better = entry.metadata[_DECLARATION]
        declared.append(Metric(key=entry.name, unit=unit, higher_is_better=higher_is_better))
    return tuple(declared)


class ScalarMetrics:
    """Named scalars of one pass, aggregated field by field.

    Two aggregations, because a pass comes in two shapes. A loss is accumulated with `+` as the
    batches arrive and divided once by the traces it summed over, so a trace weighs the same
    however the batches fell. A per-prefix score is already reduced when it is produced, so it is
    averaged with `mean` over the prefixes themselves.
    """

    # Subclasses declare `slots=True`, which only saves anything if this carries no `__dict__`.
    __slots__ = ()

    @classmethod
    def mean(cls, values: Sequence[Self]) -> Self:
        """Average a set of metrics field by field.

        Args:
            values: The metrics to average, one per unit of the pass.
        Returns:
            The mean of every field, all 0.0 if there are none.
        """
        return cls(
            **{
                field.name: mean([getattr(value, field.name) for value in values])
                for field in fields(cls)
            }
        )

    @classmethod
    def metrics(cls) -> tuple[Metric, ...]:
        """Every field of a family of scores, each being one of the numbers a report carries.

        Asked only of a family a report is written from. `Loss` is logged rather than reported, so
        it declares nothing and never calls this.

        Returns:
            One `Metric` per field, in declaration order.
        Raises:
            ValueError: If a field was not declared with `metric`, naming it. A score added to a
                family and left undeclared would reach every report and no figure.
        """
        declared = metrics_of(cls)
        keys = {declaration.key for declaration in declared}
        undeclared = [entry.name for entry in fields(cls) if entry.name not in keys]
        if undeclared:
            raise ValueError(
                f'{cls.__name__} declares no unit or direction for {", ".join(undeclared)}. '
                f'Give each of them a `metric(unit=...)` default, since every field of a family of '
                f'scores is one of the numbers a report carries.'
            )
        return declared

    def __add__(self, other: Self) -> Self:
        return type(self)(
            **{
                field.name: getattr(self, field.name) + getattr(other, field.name)
                for field in fields(self)
            }
        )

    def __truediv__(self, divisor: float) -> Self:
        return type(self)(
            **{field.name: getattr(self, field.name) / divisor for field in fields(self)}
        )

    def log(self, writer: 'SummaryWriter', step: int, *, prefix: str) -> None:
        """
        Write every field to TensorBoard under a shared prefix.

        Args:
            writer: The TensorBoard writer to log to.
            step: The step these metrics belong to.
            prefix: Namespace to log under, e.g. `train` or `val`, so the two passes of a
                validation line up on the same chart.
        """
        for name, value in asdict(self).items():
            writer.add_scalar(f'{prefix}/{name}', value, step)
