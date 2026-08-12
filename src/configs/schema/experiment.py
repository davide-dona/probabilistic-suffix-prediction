from .base import StrictModel
from .data import DataConfig, DeclareConfig
from .inference import InferenceConfig
from .model import ModelConfig
from .training import (
    DataLoaderConfig,
    EarlyStoppingConfig,
    LossConfig,
    OptimizerConfig,
    TrainingConfig,
)


class ExperimentConfig(StrictModel):
    """Top-level config, the single object loaded from YAML.

    Pass sub-sections (e.g. `cfg.model.encoder`, `cfg.optimizer`) into functions rather than
    this whole object, so each function only depends on the parameters it actually uses.
    """

    seed: int

    data: DataConfig
    declare: DeclareConfig
    dataloader: DataLoaderConfig
    model: ModelConfig
    loss: LossConfig
    optimizer: OptimizerConfig
    training: TrainingConfig
    early_stopping: EarlyStoppingConfig
    inference: InferenceConfig


class DatasetConfig(StrictModel):
    """The hardware-independent parts of an experiment config: `data` and `declare` alone,
    merged from base.yaml and a dataset yaml with no hardware profile involved. Used by
    pipelines that never read a hardware-dependent value.
    """

    data: DataConfig
    declare: DeclareConfig
