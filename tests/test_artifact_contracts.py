import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from src.evaluation.report import EvaluationReport
from src.inference.generation import DecodedEvents, Draws, Generation
from src.inference.generation_store import Generations, GenerationWriter
from src.model.checkpoint import load_checkpoint, save_checkpoint


def events(activities: str, times: list[float]) -> DecodedEvents:
    return DecodedEvents(
        activities=activities,
        inter_event_time_minutes=times,
        remaining_time_minutes=float(sum(times)),
    )


class GenerationContractTests(unittest.TestCase):
    def test_generation_round_trip_uses_inter_event_time_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'generations.parquet'
            sample = events('AB', [1.0, 2.0])
            item = Generation(
                case_id='case-1',
                prefix_activities='P',
                samples=Draws.of([sample]),
                point=sample,
                truth=sample,
            )
            with GenerationWriter(
                path,
                {'dataset': 'test', 'model': 'transformer_cvae'},
                vocabulary=('A', 'B'),
                sampling=None,
            ) as writer:
                writer.write([item])
            with Generations(path) as generations:
                restored = generations.block(0)[0]
            self.assertEqual(restored, item)

    def test_legacy_generation_schema_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.parquet'
            table = pa.table({'generated_cycle_time_minutes': [[1.0]]})
            pq.write_table(table, path)
            with self.assertRaisesRegex(ValueError, 'Regenerate'):
                Generations(path)


class CheckpointContractTests(unittest.TestCase):
    def test_new_checkpoint_records_dls_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            save_checkpoint(
                torch.nn.Linear(in_features=1, out_features=1),
                config={'model': {'kind': 'transformer_cvae'}},
                step=1,
                selection_score=0.1,
                wandb_id=None,
                path=path,
            )
            checkpoint = load_checkpoint(path)
            self.assertEqual(checkpoint['selection_metric'], 'energy_score_dls')

    def test_legacy_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.pt'
            torch.save(
                {
                    'config': {'model': {'kind': 'cvae'}},
                    'model_state_dict': {},
                    'step': 1,
                    'selection_score': 0.1,
                    'selection_metric': 'energy_score',
                    'selection_direction': 'min',
                },
                path,
            )
            with self.assertRaisesRegex(ValueError, 'legacy'):
                load_checkpoint(path)


class ReportContractTests(unittest.TestCase):
    def test_legacy_accuracy_family_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.json'
            path.write_text(
                '{"metadata":{"dataset":"test","model":"cvae"},'
                '"summary":{"accuracy":{},"conformance":{}}}'
            )
            with self.assertRaisesRegex(ValueError, 'Score its generations again'):
                EvaluationReport.read(path)


if __name__ == '__main__':
    unittest.main()
