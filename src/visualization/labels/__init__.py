from collections.abc import Iterable, Mapping

from src.visualization.labels.datasets import DATASETS
from src.visualization.labels.models import LOG_STYLE, MODELS, ModelStyle


def ordered(keys: Iterable[str], entries: Mapping[str, object], *, kind: str) -> list[str]:
    """Return present keys in declaration order, rejecting unknown keys.

    Args:
        keys: Names found in the report.
        entries: Declared names in display order.
        kind: Entry kind to name in an error.

    Returns:
        Present names in display order.

    Raises:
        ValueError: If a name has no display declaration.
    """
    present = set(keys)
    unknown = present - entries.keys()
    if unknown:
        raise ValueError(f'unknown {kind}: {", ".join(sorted(unknown))}')
    return [key for key in entries if key in present]


__all__ = [
    'DATASETS',
    'LOG_STYLE',
    'MODELS',
    'ModelStyle',
    'ordered',
]
