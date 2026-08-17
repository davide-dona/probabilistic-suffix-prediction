from dataclasses import dataclass

from src.visualization.labels.registry import Registry


@dataclass(frozen=True)
class Metric:
    """What a metric is called wherever it is drawn or tabulated, and which way is better.

    Cosmetics alone: where a metric is read from is the frame's business, and which figure it
    appears in is the catalogue's.
    """

    label: str
    unit: str | None = None  # `days` or `events`; `None` for a score already in [0, 1]
    # `None` for a metric with no best value, e.g. a diversity or a length.
    higher_is_better: bool | None = None

    @property
    def is_score(self) -> bool:
        """Whether the metric is a share already in [0, 1], which is what carrying no unit says.

        An axis of one of these is drawn over the whole interval, so a panel shows where a model
        sits within what the metric can be rather than within the range these models happen to
        cover.
        """
        return self.unit is None

    @property
    def unit_suffix(self) -> str:
        """The bracketed unit appended to the metric's name, or `''` for a score in [0, 1]."""
        return '' if self.unit is None else f' [{self.unit}]'

    @property
    def direction_marker(self) -> str:
        """The arrow marking whether higher or lower is better, or `''` for a metric with no best
        value. It follows a non-breaking space, so a wrapped title never strands it on its own
        line."""
        if self.higher_is_better is None:
            return ''
        return '\N{NO-BREAK SPACE}↑' if self.higher_is_better else '\N{NO-BREAK SPACE}↓'

    @property
    def title(self) -> str:
        """A panel's title: the name and its arrow, the unit being the panel's own y-axis."""
        return f'{self.label}{self.direction_marker}'

    @property
    def axis_label(self) -> str:
        """The metric's axis label, its unit and which way is better appended where it has them."""
        return f'{self.label}{self.unit_suffix}{self.direction_marker}'

    def table_header(self, label: str) -> str:
        """The Metric cell of one of a table's blocks.

        Args:
            label: What that table calls the metric, its own single estimator making the qualifier
                in `label` redundant.
        Returns:
            The name, its unit and which way is better, the arrow written in math mode.
        """
        direction = ''
        if self.higher_is_better is not None:
            direction = r' $\uparrow$' if self.higher_is_better else r' $\downarrow$'
        return f'{label}{self.unit_suffix}{direction}'

    def format(self, value: float) -> str:
        """Format one of the metric's values for a table cell.

        Args:
            value: The value to format.
        Returns:
            The value at three decimals, the same for every metric, so a column of them lines up
            on the decimal point whatever it measures.
        """
        return f'{value:.3f}'


# What a metric is called, and what gives it a unit and a direction. Every metric a figure or a
# table names is declared here; one that is not stops the run.
METRICS = Registry[Metric](
    kind='metric',
    where='METRICS in src/visualization/labels/metrics.py',
    entries={
        'dls_point': Metric(
            label='DLS (point)',
            higher_is_better=True,
        ),
        'dls_mean': Metric(
            label='DLS (sample mean)',
            higher_is_better=True,
        ),
        'dls_best': Metric(
            label='DLS (best of k)',
            higher_is_better=True,
        ),
        'hit_rate_at_1': Metric(
            label='Hit rate @1',
            higher_is_better=True,
        ),
        'hit_rate_at_5': Metric(
            label='Hit rate @5',
            higher_is_better=True,
        ),
        'hit_rate_at_10': Metric(
            label='Hit rate @10',
            higher_is_better=True,
        ),
        'conformance_point': Metric(
            label='Conformance (point)',
            higher_is_better=True,
        ),
        'conformance_mean': Metric(
            label='Conformance (sample mean)',
            higher_is_better=True,
        ),
        'energy_distance': Metric(
            label='Energy distance',
            higher_is_better=False,
        ),
        'sample_diversity': Metric(
            label='Sample diversity',
        ),
        'unique_sample_rate': Metric(
            label='Unique sample rate',
        ),
        'remaining_time_ae_point_days': Metric(
            label='Remaining time MAE (point)',
            unit='days',
            higher_is_better=False,
        ),
        'remaining_time_ae_mean_days': Metric(
            label='Remaining time MAE (sample mean)',
            unit='days',
            higher_is_better=False,
        ),
        'length_ae_point': Metric(
            label='Suffix length MAE (point)',
            unit='events',
            higher_is_better=False,
        ),
        'length_ae_mean': Metric(
            label='Suffix length MAE (sample mean)',
            unit='events',
            higher_is_better=False,
        ),
        'suffix_length': Metric(label='True suffix length', unit='events'),
    },
)
