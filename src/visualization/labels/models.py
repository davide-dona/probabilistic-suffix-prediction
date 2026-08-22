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
    # A named style where one is free, and a dash pattern where they are all taken: there are
    # more models than the four names, and two models drawn alike are one model here.
    linestyle: str | tuple[int, tuple[int, ...]]


# How a model is called and its stlye
MODELS = Registry[ModelStyle](
    kind='model',
    where='MODELS in src/visualization/labels/models.py',
    entries={
        'cvae': ModelStyle(label='CVAE', color='#2E8B57', marker='*', linestyle='-'),
        'cvae-small': ModelStyle(label='CVAE', color='#2E8B57', marker='*', linestyle='-'),
        'cvae-filtered': ModelStyle(
            label='CVAE (filtered)', color='#C2185B', marker='o', linestyle='-.'
        ),
        'cvae-small-filtered': ModelStyle(
            label='CVAE (filtered)', color='#C2185B', marker='o', linestyle='-.'
        ),
        'cvae-cat': ModelStyle(
            label='CVAE (categorical)', color='#1F6FB2', marker='^', linestyle=(0, (5, 1))
        ),
        'cvae-small-cat': ModelStyle(
            label='CVAE (categorical)', color='#1F6FB2', marker='^', linestyle=(0, (5, 1))
        ),
        # The num_modes sweep. Each K is its own series, since what the sweep is about is how they
        # differ; the shade darkens with K so the ordering reads off the page.
        'cvae-cat-k4': ModelStyle(
            label='CVAE (categorical, K=4)', color='#9EC9E8', marker='^', linestyle=(0, (1, 1))
        ),
        'cvae-cat-k8': ModelStyle(
            label='CVAE (categorical, K=8)',
            color='#5AA3D5',
            marker='^',
            linestyle=(0, (3, 1, 1, 1)),
        ),
        'cvae-cat-k32': ModelStyle(
            label='CVAE (categorical, K=32)', color='#0B3D66', marker='^', linestyle=(0, (7, 2))
        ),
        'u-ed-lstm': ModelStyle(label='U-ED-LSTM', color='#E67300', marker='s', linestyle='--'),
        'sutran': ModelStyle(label='SuTraN', color='#6B3FA0', marker='D', linestyle=':'),
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
