"""Validation-only sampler selection and portable tuning reports."""

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

from omegaconf import DictConfig, OmegaConf
from pydantic import TypeAdapter, ValidationError


@dataclass(frozen=True)
class TuningPoint:
    sampling: dict[str, float]
    score: float
    conformance_sample_mean: float


@dataclass(frozen=True)
class SearchPass:
    pairs: int
    samples: int
    seed: int


@dataclass(frozen=True)
class TuningReport:
    checkpoint_sha256: str
    search: SearchPass
    chosen: dict[str, float]
    grid: tuple[TuningPoint, ...]
    selection_metric: str = 'energy_score_dls'
    selection_direction: str = 'min'

    @classmethod
    def of(cls, checkpoint_sha256: str, *, search: SearchPass, grid: Sequence[TuningPoint]) -> Self:
        if not grid or any(not math.isfinite(point.score) for point in grid):
            raise ValueError('Tuning requires a nonempty grid of finite energy scores')
        return cls(
            checkpoint_sha256=checkpoint_sha256,
            search=search,
            chosen=min(grid, key=lambda point: point.score).sampling,
            grid=tuple(grid),
        )

    @classmethod
    def read(cls, path: str | Path) -> Self:
        path = Path(path)
        payload = json.loads(path.read_bytes())
        if payload.get('selection_metric') != 'energy_score_dls':
            raise ValueError(
                f'{path} uses the legacy tuning schema. Tune the checkpoint again with '
                '`python -m pipelines.tune`.'
            )
        try:
            return _ADAPTER.validate_python(payload)
        except ValidationError as error:
            raise ValueError(
                f'{path} uses an incompatible tuning schema. Tune the checkpoint again with '
                '`python -m pipelines.tune`.'
            ) from error

    def write(self, path: Path) -> Path:
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(asdict(self), indent=2))
        temporary.replace(path)
        return path

    def sampling_for(self, checkpoint_sha256: str) -> DictConfig:
        if self.checkpoint_sha256 != checkpoint_sha256:
            raise ValueError('Tuning report belongs to a different checkpoint')
        return OmegaConf.create(self.chosen)


_ADAPTER = TypeAdapter(TuningReport)
