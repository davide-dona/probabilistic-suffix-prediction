import unittest
from pathlib import Path

from omegaconf import OmegaConf

from src.datasets.codec import (
    ACTIVITY_TOKENS,
    RESOURCE_TOKENS,
    CategoricalColumn,
    DatasetCodec,
    NumericColumn,
)
from src.model import (
    HeadSamplingTransformer,
    TransformerCVAE,
    build_model,
    model_from_checkpoint,
)
from src.validation import validate_model


def codec() -> DatasetCodec:
    return DatasetCodec(
        activity=CategoricalColumn(
            column='concept:name',
            vocab=('A', 'B'),
            special_tokens=ACTIVITY_TOKENS,
            offset=0,
        ),
        resource=CategoricalColumn(
            column='org:resource',
            vocab=('R',),
            special_tokens=RESOURCE_TOKENS,
            offset=0,
        ),
        inter_event_time=NumericColumn(
            column='inter_event_time', log=False, mean=0.0, std=1.0
        ),
        remaining_time=NumericColumn(column='rtime', log=False, mean=0.0, std=1.0),
        categorical_features=(),
        numeric_features=(),
        max_trace_length=4,
        dataset='test',
    )


def model_config(name: str):
    root = Path(__file__).parents[1] / 'config' / 'model'
    return OmegaConf.merge(
        OmegaConf.load(root / 'backbone.yaml'),
        OmegaConf.load(root / f'{name}.yaml'),
    )


class ModelNamingTests(unittest.TestCase):
    def assert_checkpoint_rebuilds(self, name: str, expected: type) -> None:
        config = model_config(name)
        model = build_model(config=config, codec=codec())
        checkpoint = {
            'config': {'model': OmegaConf.to_container(config, resolve=True)},
            'model_state_dict': model.state_dict(),
        }
        self.assertIsInstance(
            model_from_checkpoint(checkpoint=checkpoint, codec=codec()), expected
        )

    def test_transformer_cvae_selector_builds_named_class(self) -> None:
        config = model_config('transformer_cvae')
        validate_model(config)
        self.assertIsInstance(build_model(config=config, codec=codec()), TransformerCVAE)
        self.assert_checkpoint_rebuilds('transformer_cvae', TransformerCVAE)

    def test_head_sampling_selector_builds_named_class(self) -> None:
        config = model_config('head_sampling_transformer')
        validate_model(config)
        self.assertIsInstance(
            build_model(config=config, codec=codec()), HeadSamplingTransformer
        )
        self.assert_checkpoint_rebuilds(
            'head_sampling_transformer', HeadSamplingTransformer
        )


if __name__ == '__main__':
    unittest.main()
