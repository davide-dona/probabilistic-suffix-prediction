from src.visualization.labels.models import ModelStyle

# How the log's own values are drawn, wherever a figure puts the log beside the models: the target
# they are read against rather than a competitor among them, so it keeps one recessive grey and a
# line style no model uses. Declared once and read by every figure that draws a metric owned by
# the log, so the log looks the same throughout. The grey is the one `distribution.TRUTH_CLOUD`
# already draws the ground truth in.
LOG_STYLE = ModelStyle(label='Log', color='#8A8A8A', marker='o', linestyle='-.')
