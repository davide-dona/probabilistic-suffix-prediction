from dataclasses import dataclass


@dataclass(frozen=True)
class DecodedEvents:
    """Set of predicted events, in the log's own units. Any new addition
    to the model's output must be added here"""

    activities: list[str]
    remaining_time_minutes: float

    def __len__(self) -> int:
        return len(self.activities)


@dataclass(frozen=True)
class Generation:
    """One prefix's generated suffixes, the point prediction beside them, and the truth they were
    generated for.
    - case_id is used to identify the case in the log the prefix was cut from;
    - prefix_activities are added for convenience in the conformance report.
    """

    case_id: str  # which case of the log the prefix was cut from
    prefix_activities: list[str]  # the events before the cut, in order
    samples: list[DecodedEvents]  # one per draw of z
    point: DecodedEvents  # the suffix written from the mean of `p(z | prefix)`
    truth: DecodedEvents

    @property
    def prefix_len(self) -> int:
        """The cut point this answers: how many events ran before it."""
        return len(self.prefix_activities)
