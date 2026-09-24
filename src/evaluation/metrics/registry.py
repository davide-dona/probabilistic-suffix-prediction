from collections.abc import Callable

from src.evaluation.metrics.metadata import Direction, Metric, MetricGroup, Owner, Unit


class MetricRegistry:
    """Ordered declarations of every evaluation metric."""

    def __init__(self) -> None:
        self.entries: dict[str, Metric] = {}

    def register(
        self,
        key: str,
        *,
        label: str,
        group: MetricGroup,
        unit: Unit,
        publication_label: str | None = None,
        direction: Direction = Direction.NONE,
        owner: Owner = Owner.MODEL,
    ) -> Callable[[Callable[..., float]], Callable[..., float]]:
        """Register a prefix-scoring function under a stable metric key."""

        def decorator(compute: Callable[..., float]) -> Callable[..., float]:
            if key in self.entries:
                raise ValueError(f'metric {key!r} is registered more than once.')
            self.entries[key] = Metric(
                key=key,
                label=label,
                group=group,
                unit=unit,
                publication_label=publication_label,
                direction=direction,
                owner=owner,
                compute=compute,
            )
            return compute

        return decorator

    def __getitem__(self, key: str) -> Metric:
        if key not in self.entries:
            raise ValueError(f'no evaluation metric is registered as {key!r}.')
        return self.entries[key]


METRICS = MetricRegistry()
