import tempfile
import unittest
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from src.evaluation.report import EvaluationReport, read_reports
from src.evaluation.scores import (
    CalibrationScores,
    ConformanceScores,
    PointPredictionScores,
    SamplePredictionScores,
)
from src.evaluation.summary import EvaluationSummary, PrefixSummary
from src.visualization.catalogue import FIGURES, TABLES
from src.visualization.figures import compose_figure
from src.visualization.tables import latex_table


def values(family: type, value: float = 0.25):
    return family(**{metric.key: value for metric in family.metrics()})


class VisualizationSmokeTests(unittest.TestCase):
    def test_all_tables_and_figures_render_from_new_schema(self) -> None:
        prefix = PrefixSummary(
            prefix_len=1,
            suffix_len=2,
            point=values(PointPredictionScores),
            sample=values(SamplePredictionScores),
            calibration=values(CalibrationScores, 0.0),
            conformance=values(ConformanceScores),
        )
        report = EvaluationReport(
            metadata={'dataset': 'sepsis', 'model': 'transformer_cvae'},
            summary=EvaluationSummary.of([prefix]),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = report.write(Path(directory) / 'evaluation.json')
            frame = read_reports([path])

        significance = pd.DataFrame(columns=['dataset', 'model', 'metric', 'best'])
        for table in TABLES:
            rendered = latex_table(frame, table, significance)
            self.assertIn(r'\begin{tabular', rendered)
        for plot in FIGURES:
            figure = compose_figure(frame, plot)
            self.assertGreater(len(figure.axes), 0)
            plt.close(figure)


if __name__ == '__main__':
    unittest.main()
