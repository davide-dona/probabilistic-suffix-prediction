from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd


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
CVAE = ModelStyle(label='CVAE', color='#0072B2', marker='o', linestyle='-')
U_ED_LSTM = ModelStyle(label='U-ED-LSTM', color='#E69F00', marker='s', linestyle='--')
SuTraN = ModelStyle(label='SuTraN', color='#009E73', marker='D', linestyle=':')
MODELS: dict[str, ModelStyle] = {
    'cvae': CVAE,
    'cvae-small': CVAE,
    'u-ed-lstm': U_ED_LSTM,
    'sutran': SuTraN,
}


def model_style(model: str) -> ModelStyle:
    """How to draw one model.

    Args:
        model: Its name, as `model.name` in its config.
    Returns:
        Its declared label, colour, marker and line style.
    Raises:
        ValueError: If nothing is declared for it, since a model drawn in a colour nobody chose
            would be a different colour in the next figure.
    """
    if model not in MODELS:
        raise ValueError(
            f'nothing declared for the model {model!r}. Add it to MODELS in '
            f'src/visualization/labels/models.py. The models declared are {", ".join(MODELS)}.'
        )
    return MODELS[model]


def ordered_models(models: Iterable[str]) -> list[str]:
    """Sort a set of models into the order they are drawn in, which `MODELS` declares.

    Args:
        models: The models present, in any order, as `reported_models` leaves them.
    Returns:
        Them alone, in the declared order, so a legend, a row of panels and a table's columns all
        read the same way.
    Raises:
        ValueError: If one of them is not declared.
    """
    present = set(models)
    for model in present:
        model_style(model)
    return [model for model in MODELS if model in present]


def reported(model: str) -> str:
    """The name one run is drawn and tabulated under: the first of those sharing its style."""
    style = model_style(model)
    return next(name for name, declared in MODELS.items() if declared == style)


def reported_models(frame: pd.DataFrame) -> pd.DataFrame:
    """Rewrite a frame's models to the names they are reported under.

    Called on the way into every figure and table, so models sharing a style are one line and one
    column rather than two drawn alike.

    Args:
        frame: What a figure or a table is drawn from, from `read_reports` or `embed_suffixes`.
    Returns:
        A copy holding reported names alone.
    """
    return frame.assign(model=frame['model'].map(reported))
