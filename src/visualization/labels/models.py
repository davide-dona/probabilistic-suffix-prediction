from dataclasses import dataclass

import pandas as pd

from src.registry import Registry


@dataclass(frozen=True)
class ModelStyle:
    """What a model is called and how it is drawn, the same way in every figure it appears in.

    A colour, a marker and a line style together, so a model can be told from the others when the
    paper is printed in black and white.
    """

    label: str
    color: str
    marker: str
    linestyle: str

# How a model is called and its stlye
MODELS = Registry[ModelStyle](
    kind='model',
    where='MODELS in src/visualization/labels/models.py',
    entries={
        'cvae': ModelStyle(label='CVAE', color='#2E8B57', marker='*', linestyle='-'),
        'cvae-small': ModelStyle(label='CVAE', color='#2E8B57', marker='*', linestyle='-'),
        'u-ed-lstm': ModelStyle(label='U-ED-LSTM', color='#E67300', marker='s', linestyle='--'),
        'sutran': ModelStyle(label='SuTraN', color='#6B3FA0', marker='D', linestyle=':')
    },
)


def _reported(model: str) -> str:
    """The name one run is drawn and tabulated under: the first of those sharing its style."""
    style = MODELS[model]
    return next(name for name, declared in MODELS.entries.items() if declared == style)


def reported_models(frame: pd.DataFrame) -> pd.DataFrame:
    """Rewrite a frame's models to the names they are reported under.

    Called on the way into every figure and table, so models sharing a style are one line and one
    column rather than two drawn alike.

    Args:
        frame: What a figure or a table is drawn from, from `read_reports` or `embed_suffixes`.
    Returns:
        A copy holding reported names alone.
    """
    return frame.assign(model=frame['model'].map(_reported))
