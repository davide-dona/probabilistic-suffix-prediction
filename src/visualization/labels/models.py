from dataclasses import dataclass


@dataclass(frozen=True)
class ModelStyle:
    """Label and visual style shared by a model's figures."""

    label: str
    color: str
    marker: str
    linestyle: str


# Style for observed log values.
LOG_STYLE = ModelStyle(label='Log', color='#737373', marker='o', linestyle='-.')

# Registered model labels and styles.
MODELS = {
    'transformer_cvae': ModelStyle(label='SuTraN-VAE', color='#3B7EA1', marker='*', linestyle='-'),
    'head_sampling_transformer': ModelStyle(
        label='SuTraN-PH', color='#A05A4B', marker='D', linestyle=':'
    ),
    'u_ed_lstm': ModelStyle(label='U-ED-LSTM', color='#7A4E97', marker='s', linestyle='--'),
}
