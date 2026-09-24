import tempfile
import unittest
from pathlib import Path

from src.evaluation import (
    EvaluationReport,
    EvaluationSummary,
    PrefixSummary,
    ScoreGroups,
    read_prefix_scores,
    read_reports,
    stream_prefix_scores,
)
from src.evaluation.metrics import METRICS
from src.inference.generation import DecodedEvents, Draws, Generation
from src.logs.declare.checker import Conformance
from src.visualization import FIGURES, TABLES, apply_style, compose_figure, latex_table


class EvaluationRefactorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata = {
            'dataset': 'sepsis',
            'model': 'transformer_cvae',
            'run_id': '20260924-120000-000001',
            'checkpoint_sha256': 'abc123',
        }

    def test_single_pass_aggregation_and_artifact_round_trip(self) -> None:
        first = PrefixSummary(
            prefix_len=1,
            suffix_len=2,
            scores=ScoreGroups.of(dict.fromkeys(METRICS.entries, 1.0)),
        )
        second = PrefixSummary(
            prefix_len=2,
            suffix_len=2,
            scores=ScoreGroups.of(dict.fromkeys(METRICS.entries, 3.0)),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scores_file = root / 'prefix_scores.parquet'
            streamed = stream_prefix_scores(
                iter((first, second)),
                [('case-a', 1), ('case-b', 2)],
                path=scores_file,
                metadata=self.metadata,
            )
            summary = EvaluationSummary.of(streamed)
            report_file = EvaluationReport(self.metadata, summary).write(root / 'evaluation.json')

            self.assertEqual(summary.prefixes, 2)
            self.assertEqual(summary.scores.activity['dls_sample_mean'], 2.0)
            self.assertEqual([bucket.length for bucket in summary.by_prefix_length], [1, 2])
            self.assertEqual(summary.by_suffix_length[0].prefixes, 2)
            self.assertEqual(
                EvaluationReport.read(report_file), EvaluationReport(self.metadata, summary)
            )
            scores = read_prefix_scores(scores_file)
            self.assertEqual(scores['case_id'].tolist(), ['case-a', 'case-b'])
            self.assertEqual(scores['dls_sample_mean'].tolist(), [1.0, 3.0])

            frame = read_reports([report_file])
            self.assertEqual(frame['model'].unique().tolist(), ['transformer_cvae'])
            self.assertIn('DLS mean', latex_table(frame, TABLES[0], frame.assign(best=False)))
            apply_style()
            figure = compose_figure(frame[frame['axis'].isin(FIGURES[0].breakdowns)], FIGURES[0])
            figure.savefig(root / 'conformance.pdf')
            self.assertTrue((root / 'conformance.pdf').exists())

    def test_interrupted_score_stream_removes_temporary_file(self) -> None:
        prefix = PrefixSummary(
            prefix_len=1,
            suffix_len=1,
            scores=ScoreGroups.of(dict.fromkeys(METRICS.entries, 0.0)),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'prefix_scores.parquet'
            stream = stream_prefix_scores(
                iter((prefix, prefix)),
                [('case-a', 1), ('case-b', 1)],
                path=path,
                metadata=self.metadata,
            )
            next(stream)
            stream.close()
            self.assertFalse(path.exists())
            self.assertFalse(path.with_suffix('.parquet.tmp').exists())

    def test_registered_metrics_score_a_prefix(self) -> None:
        class Checker:
            def check(self, trace: str) -> Conformance:
                return Conformance(satisfied=1, total=1)

        truth = DecodedEvents('A', [3.0], 3.0)
        generation = Generation(
            case_id='case-a',
            prefix_activities='B',
            samples=Draws.of([truth, truth]),
            point=truth,
            truth=truth,
        )
        summary = PrefixSummary.of(generation, checker=Checker())
        self.assertEqual(summary.scores.activity['dls_sample_mean'], 1.0)
        self.assertEqual(summary.scores.suffix_length['suffix_length_crps'], 0.0)


if __name__ == '__main__':
    unittest.main()
