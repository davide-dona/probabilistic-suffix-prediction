import unittest

import numpy as np
from rapidfuzz.distance import OSA, DamerauLevenshtein

from src.evaluation.scores import (
    METRICS,
    CalibrationScores,
    ConformanceScores,
    PointPredictionScores,
    SamplePredictionScores,
    ScoringContext,
)
from src.evaluation.summary import EvaluationSummary, PrefixSummary
from src.inference.generation import DecodedEvents, Draws, Generation
from src.suffixes import SuffixMetric, distances, energy_score, sequence_similarity
from src.visualization.catalogue import FIGURES, TABLES


def events(
    activities: str,
    inter_event_times: list[float],
    *,
    remaining_time: float = 0.0,
) -> DecodedEvents:
    return DecodedEvents(
        activities=activities,
        inter_event_time_minutes=inter_event_times,
        remaining_time_minutes=remaining_time,
    )


def generation(
    *,
    truth: DecodedEvents,
    point: DecodedEvents,
    samples: list[DecodedEvents] | None = None,
) -> Generation:
    return Generation(
        case_id='case',
        prefix_activities='P',
        samples=Draws.of(samples or [point]),
        point=point,
        truth=truth,
    )


def prefix_summary(item: Generation) -> PrefixSummary:
    context = ScoringContext.of(item)
    return PrefixSummary(
        prefix_len=item.prefix_len,
        suffix_len=len(item.truth),
        point=PointPredictionScores.of(context),
        sample=SamplePredictionScores.of(context),
        calibration=CalibrationScores.of(context),
        conformance=ConformanceScores(
            conformance_sample_mean=1.0,
            conformance_point=1.0,
            conformance_observed=1.0,
            full_conformance_sample_rate=1.0,
            full_conformance_observed=1.0,
        ),
    )


class DamerauLevenshteinTests(unittest.TestCase):
    def test_full_distance_differs_from_osa(self) -> None:
        predicted = (*'CA', '\x03')
        observed = (*'ABC', '\x03')
        self.assertGreater(
            DamerauLevenshtein.normalized_similarity(predicted, observed),
            OSA.normalized_similarity(predicted, observed),
        )
        self.assertEqual(
            sequence_similarity('CA', 'ABC'),
            DamerauLevenshtein.normalized_similarity(predicted, observed),
        )

    def test_terminal_token_is_part_of_dls_normalization(self) -> None:
        self.assertEqual(sequence_similarity('', ''), 1.0)
        self.assertEqual(sequence_similarity('', 'A'), 0.5)

    def test_batched_distances_match_scalar_similarity(self) -> None:
        sequences = ('CA', 'ABC', '')
        matrix = distances(sequences, sequences, metric=SuffixMetric.DLS, dtype=np.float64)
        for row, predicted in enumerate(sequences):
            for column, observed in enumerate(sequences):
                self.assertAlmostEqual(
                    matrix[row, column], 1.0 - sequence_similarity(predicted, observed)
                )

    def test_single_draw_energy_is_dls_distance(self) -> None:
        self.assertAlmostEqual(
            energy_score(('CA',), 'ABC', metric=SuffixMetric.DLS),
            1.0 - sequence_similarity('CA', 'ABC'),
        )


class TemporalEvaluationTests(unittest.TestCase):
    def test_short_predictions_are_zero_padded(self) -> None:
        item = generation(
            truth=events('AB', [10.0, 20.0]),
            point=events('A', [15.0]),
        )
        score = PointPredictionScores.of(ScoringContext.of(item))
        self.assertAlmostEqual(score.inter_event_time_ae_point_days, 12.5 / 1440.0)

    def test_long_predictions_are_truncated(self) -> None:
        item = generation(
            truth=events('A', [10.0]),
            point=events('ABC', [15.0, 1000.0, 1000.0]),
        )
        score = PointPredictionScores.of(ScoringContext.of(item))
        self.assertAlmostEqual(score.inter_event_time_ae_point_days, 5.0 / 1440.0)

    def test_empty_observed_suffix_has_zero_temporal_scores(self) -> None:
        item = generation(
            truth=events('', []),
            point=events('A', [10.0]),
            samples=[events('A', [10.0]), events('', [])],
        )
        context = ScoringContext.of(item)
        self.assertEqual(PointPredictionScores.of(context).inter_event_time_ae_point_days, 0.0)
        self.assertEqual(SamplePredictionScores.of(context).inter_event_time_crps_days, 0.0)

    def test_dataset_aggregation_weights_prefixes_equally(self) -> None:
        first = prefix_summary(
            generation(
                truth=events('A', [0.0]),
                point=events('A', [10.0]),
            )
        )
        second = prefix_summary(
            generation(
                truth=events('BCD', [0.0, 0.0, 0.0]),
                point=events('BCD', [0.0, 0.0, 0.0]),
            )
        )
        summary = EvaluationSummary.of([first, second])
        self.assertAlmostEqual(summary.point.inter_event_time_ae_point_days, 5.0 / 1440.0)
        self.assertNotAlmostEqual(summary.point.inter_event_time_ae_point_days, 2.5 / 1440.0)


class MetricCatalogueTests(unittest.TestCase):
    def test_registry_is_exactly_the_reported_metric_set(self) -> None:
        catalogued = {
            entry.key
            for table in TABLES
            for entry in table.columns
        } | {
            entry.key
            for figure in FIGURES
            for panel in figure.panels
            for entry in panel
        }
        self.assertEqual(len(METRICS.entries), 25)
        self.assertEqual(catalogued, set(METRICS.entries))


if __name__ == '__main__':
    unittest.main()
