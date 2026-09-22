from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

# Start of the Unicode private use area, where the activity codes are drawn from.
_FIRST_CODE = 0xE000

# Sequence boundaries must be distinct and below every activity code.
START_CODE = '\x02'
END_CODE = '\x03'
if START_CODE == END_CODE or max(ord(START_CODE), ord(END_CODE)) >= _FIRST_CODE:
    raise ValueError('Sequence boundaries overlap activity codes')


@dataclass(slots=True)
class ActivityCodec:
    """Encode activity names as one character for storage and sequence comparisons."""

    _codes: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_vocabulary(cls, activities: Sequence[str]) -> 'ActivityCodec':
        """Build a codebook from activity names already in code order.

        Args:
            activities: The names in the order their codes were assigned, as `vocabulary`
                returns them.
        Returns:
            A codebook giving each name its established code and assigning the next code to an
            unseen name.
        """
        return cls({activity: chr(_FIRST_CODE + code) for code, activity in enumerate(activities)})

    @property
    def vocabulary(self) -> tuple[str, ...]:
        """The activity names in code order, suitable for `from_vocabulary`."""
        return tuple(self._codes)

    @property
    def codes(self) -> Mapping[str, str]:
        """Each activity name to the character it is spelled with, read-only."""
        return MappingProxyType(self._codes)

    def encode(self, activities: Sequence[str]) -> str:
        """Encode one activity sequence, giving unseen names the next code point.

        Args:
            activities: The activity names, in order.
        Returns:
            One character per activity. An empty sequence produces an empty string.
        """
        return ''.join(
            self._codes.setdefault(activity, chr(_FIRST_CODE + len(self._codes)))
            for activity in activities
        )
