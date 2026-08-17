from dataclasses import dataclass

import pandas as pd

from src.visualization.labels.registry import Registry


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


# What a model is called and how it is drawn in the figures.
# A model trained under a name not listed here stops the run.
CVAE = ModelStyle(label='CVAE', color='#0072B2', marker='*', linestyle='-')
U_ED_LSTM = ModelStyle(label='U-ED-LSTM', color='#E69F00', marker='s', linestyle='--')
SUTRAN = ModelStyle(label='SuTraN', color='#009E73', marker='D', linestyle=':')
MODELS = Registry[ModelStyle](
    kind='model',
    where='MODELS in src/visualization/labels/models.py',
    entries={
        'cvae': CVAE,
        'cvae-small': CVAE,
        'u-ed-lstm': U_ED_LSTM,
        'sutran': SUTRAN,
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
