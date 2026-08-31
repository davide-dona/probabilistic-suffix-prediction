from dataclasses import dataclass

from src.scalar_metrics import Direction, Metric


@dataclass(frozen=True)
class MetricEntry:
    """One metric as it appears in one figure panel or one table column: what that appearance
    calls it, and how its values are written there.

    The metric itself carries no name, since one is not enough. A table naming a single estimator
    can name a metric shortly, where a figure drawing that estimator against another has to say
    which of them it is drawing. Everything below is that name with what the metric declares about
    itself appended, so two appearances differ in their wording alone.
    """

    metric: Metric
    label: str

    @property
    def key(self) -> str:
        """The metric's name in the `metric` column of `read_reports`, i.e. the rows to draw."""
        return self.metric.key

    @property
    def bounds(self) -> tuple[float | None, float | None]:
        """The y-limits a panel of this metric is drawn between, either end `None` for one the
        data sets."""
        return self.metric.unit.bounds

    @property
    def shares_scale(self) -> bool:
        """Whether every panel of this metric is drawn over the same range whatever log it covers.

        True of a share, which spans `[0, 1]` by what it is. A row of one prints its ticks in the
        leftmost panel alone, since repeating them says the same thing in the width of a panel.
        """
        return self.bounds == (0.0, 1.0)

    @property
    def _unit_suffix(self) -> str:
        """The bracketed unit appended to the name, or `''` for a dimensionless metric."""
        symbol = self.metric.unit.symbol
        return f' [{symbol}]' if symbol else ''

    @property
    def axis_label(self) -> str:
        """The label of an axis drawn along this metric, its unit and which way it reads
        appended where it has them."""
        return f'{self.label}{self._unit_suffix}{self._arrow}'

    @property
    def table_header(self) -> str:
        """The header of one of a table's columns: the name and which way it reads, the arrow
        written in math mode after a `~` so a wrapped name never strands it on its own line.

        The unit is not written here. A table's headers are unitless and the units are stated in
        its caption, which is what keeps a two-word name and a name carrying `[days]` the same
        height; `Table.note` is the sentence that caption has to carry. The name itself is a plain
        string the column wraps as it needs to, so a header is one line of LaTeX with nothing to
        read past.
        """
        return f'{self.label}{_TABLE_ARROWS[self.metric.direction]}'

    @property
    def _arrow(self) -> str:
        """The arrow marking which way the metric reads, or `''` for one with no best value.

        It follows a non-breaking space, so a wrapped label never strands it on its own line.
        """
        return _AXIS_ARROWS[self.metric.direction]

    def format(self, value: float) -> str:
        """Format one of the metric's values for a table cell.

        Args:
            value: The value to format.
        Returns:
            The value at three decimals, the same for every metric, so a column of them lines up
            on the decimal point whatever it measures.
        """
        return f'{value:.3f}'


# Which way a metric reads, in the two dialects a page writes it in. A gap is best at 0 rather than
# at either end, so it is marked with the target it is read against rather than with an arrow.
# Both dialects tie the mark to the name it follows: a non-breaking space in matplotlib's, a `~` in
# LaTeX's, so neither ever strands the mark on a line of its own.
_AXIS_ARROWS = {
    Direction.HIGHER: '\N{NO-BREAK SPACE}↑',
    Direction.LOWER: '\N{NO-BREAK SPACE}↓',
    Direction.ZERO: '\N{NO-BREAK SPACE}→0',
    Direction.NONE: '',
}
_TABLE_ARROWS = {
    Direction.HIGHER: r'~$\uparrow$',
    Direction.LOWER: r'~$\downarrow$',
    Direction.ZERO: r'~$\rightarrow 0$',
    Direction.NONE: '',
}
