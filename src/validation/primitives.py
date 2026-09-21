import math
import re
from collections.abc import Sequence


def validate_number(
    value: object,
    name: str,
    *,
    minimum: float = 0,
    inclusive: bool = False,
    integer: bool = False,
) -> None:
    """Require a finite numeric value within the requested lower bound.

    Args:
        value: Value supplied by configuration or a command-line override.
        name: Field name included in the error message.
        minimum: Lower bound to enforce.
        inclusive: Whether the lower bound itself is valid.
        integer: Whether only integers are valid.
    Raises:
        ValueError: If the value has the wrong type, is non-finite, or is below the bound.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, int if integer else (int, float))
        or not math.isfinite(value)
        or (value < minimum if inclusive else value <= minimum)
    ):
        relation = '>=' if inclusive else '>'
        kind = 'integer' if integer else 'number'
        raise ValueError(f'{name} must be a finite {kind} {relation} {minimum}, got {value!r}')


def validate_identifier(value: object, name: str, pattern: str, description: str) -> None:
    """Require a string identifier matching a documented regular-expression pattern."""
    if not isinstance(value, str) or not re.fullmatch(pattern, value):
        raise ValueError(f'{name} must contain only {description}')


def validate_string_list(value: object, name: str) -> None:
    """Require a sequence of nonempty path-like strings, excluding a bare string."""
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f'{name} must be a list of nonempty strings')
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f'{name} must be a list of nonempty strings')
