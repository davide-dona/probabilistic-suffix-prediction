from collections.abc import Sequence
from dataclasses import dataclass
from itertools import chain, islice, repeat
from typing import Self

import numpy as np

from src.inference.generation import Generation
from src.suffixes import sequence_similarity


@dataclass(frozen=True, slots=True)
class ScoringContext:
    """Decoded values shared by the score fields for one prefix."""

    generation: Generation
    similarities: tuple[float, ...]
    suffix_lengths: np.ndarray
    remaining_times: np.ndarray
    inter_event_times: np.ndarray
    true_suffix_length: np.ndarray
    true_remaining_time: np.ndarray
    true_inter_event_times: np.ndarray

    @classmethod
    def of(cls, generation: Generation) -> Self:
        """Prepare the shared draw and observation arrays for one prefix."""
        samples = generation.samples
        truth = generation.truth
        draws = len(samples)
        return cls(
            generation=generation,
            similarities=tuple(
                sequence_similarity(suffix, truth.activities) for suffix in samples.suffixes
            ),
            suffix_lengths=np.array(
                [[float(len(events))] for events in samples.events], dtype=np.float64
            ).reshape(draws, 1),
            remaining_times=np.array(
                [[events.remaining_time_minutes] for events in samples.events], dtype=np.float64
            ).reshape(draws, 1),
            inter_event_times=np.array(
                [
                    aligned_inter_event_times(events.inter_event_time_minutes, length=len(truth))
                    for events in samples.events
                ],
                dtype=np.float64,
            ).reshape(draws, len(truth)),
            true_suffix_length=np.array([float(len(truth))], dtype=np.float64),
            true_remaining_time=np.array([truth.remaining_time_minutes], dtype=np.float64),
            true_inter_event_times=np.array(truth.inter_event_time_minutes, dtype=np.float64),
        )


def aligned_inter_event_times(predicted: Sequence[float], *, length: int) -> list[float]:
    """Pad or truncate inter-event times to the observed suffix length."""
    return list(islice(chain(predicted, repeat(0.0)), length))
