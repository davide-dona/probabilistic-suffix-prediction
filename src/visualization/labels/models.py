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
        # A narrowed latent is its own series rather than a second name for the CVAE above, since
        # the width is what the run is about. The two never share a `ModelStyle` with each other
        # either: `_reported` folds names that do into one, which would draw a log's `cvae-z8`
        # and `cvae-small-z4` over each other after `group_by_model` had passed them as two.
        'cvae-z8': ModelStyle(label='CVAE (z=8)', color='#1F77B4', marker='^', linestyle='--'),
        # One width on two logs, so these share a style the way `cvae` and `cvae-small` do: a log
        # given either draws that log's z=4 line and fills its z=4 column.
        'cvae-z4': ModelStyle(label='CVAE (z=4)', color='#8C564B', marker='v', linestyle='--'),
        'cvae-small-z4': ModelStyle(
            label='CVAE (z=4)', color='#8C564B', marker='v', linestyle='--'
        ),
        # The arms of #60's KL price sweep, all at z=4: `w02` and `w01` are
        # `kl_annealing_full_weight`, `nfb` the same weight with the free-bits floor removed. Each
        # carries its own style rather than sharing `cvae-z4`'s, since what the sweep is about is
        # telling them apart; they come out again once #60 closes.
        'cvae-z4-w02': ModelStyle(
            label='CVAE (z=4, KL 0.2)', color='#D62728', marker='o', linestyle='--'
        ),
        'cvae-z4-w01': ModelStyle(
            label='CVAE (z=4, KL 0.1)', color='#9467BD', marker='P', linestyle='--'
        ),
        'cvae-z4-w02-nfb': ModelStyle(
            label='CVAE (z=4, KL 0.2, no free bits)',
            color='#17BECF',
            marker='X',
            linestyle=':',
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
