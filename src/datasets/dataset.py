from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset

from src.datasets.codec import DatasetCodec
from src.logs import CASE_KEY, MIN_PREFIX_KEY, Split


class Events(NamedTuple):
    """The set of events composing a single trace, as the model reads it."""

    activities: torch.Tensor  # int64, [..., seq_len]
    resources: torch.Tensor  # int64, [..., seq_len]
    inter_event_times: torch.Tensor  # float32, standardized, [..., seq_len]
    categorical_attributes: torch.Tensor  # int64, [..., seq_len, num_categorical]
    numeric_attributes: torch.Tensor  # float32, standardized, [..., seq_len, num_numeric]
    numeric_attributes_present: (
        torch.Tensor
    )  # float32 0/1, 0.0 where the log had no value [..., seq_len, num_numeric]
    length: torch.Tensor  # int64, the real, unpadded length of the run, [...], 1D

    def _channels(self) -> dict[str, torch.Tensor]:
        """Return the per-event fields, excluding the scalar event count."""
        return {name: getattr(self, name) for name in self._fields if name != 'length'}

    def cut(self, index: slice | torch.Tensor) -> Events:
        """Slice every per-event field by index, updating `length`."""
        channels = {name: channel[index] for name, channel in self._channels().items()}
        return self._replace(
            **channels,
            length=torch.tensor(data=len(channels['activities']), dtype=torch.long),
        )

    def padded(self, to: int) -> Events:
        """Pad every per-event field to `to` positions, leaving `length` unchanged.

        Args:
            to: The width to pad to, `max_trace_length` for everything the model reads.
        Returns:
            The same events at the front of `to` positions, `length` unchanged.
        """
        return self._replace(
            **{
                name: torch.cat(
                    tensors=(
                        channel,
                        torch.zeros(
                            size=(to - channel.size(dim=0), *channel.shape[1:]), dtype=channel.dtype
                        ),
                    )
                )
                for name, channel in self._channels().items()
            }
        )

    def pad_mask(self) -> torch.Tensor:
        """Marks the positions that were padded out.
        Returns:
            `[batch_size, seq_len]`, True where the position holds padding.
        """
        positions = torch.arange(end=self.activities.size(dim=-1), device=self.length.device)
        return positions.unsqueeze(dim=0) >= self.length.unsqueeze(dim=1)

    def to(self, device: torch.device) -> Events:
        """Move every field to `device`."""
        return Events(*(field.to(device) for field in self))


class TraceCut(NamedTuple):
    """A single trace cut into a (prefix, suffix) pair, with the decoder's time targets aligned
    to the suffix positions they are read at."""

    # Which case of the log this was cut from. Allow to identify the original trace after
    # generation.
    case_id: str  # a `tuple[str, ...]` of `batch_size` of them once collated

    prefix: Events  # the condition: the events before the cut, no EOT
    suffix: Events  # what the decoder must produce: content, EOT, then padding

    # Standardized minutes until the event written at each suffix position, and until the case
    # ends. Both measure from the last prefix event at position 0.
    inter_event_times: torch.Tensor  # float32, [max_trace_length], batched [batch_size, ...]
    remaining_times: torch.Tensor  # float32, shaped like `inter_event_times`

    def to(self, device: torch.device) -> TraceCut:
        """Move every tensor field to `device`."""
        return TraceCut(
            case_id=self.case_id,
            prefix=self.prefix.to(device),
            suffix=self.suffix.to(device),
            inter_event_times=self.inter_event_times.to(device),
            remaining_times=self.remaining_times.to(device),
        )


@dataclass(frozen=True)
class _Case:
    """One complete encoded case, shared by every split trace cut from it."""

    case_id: str  # which case of the log this is
    events: Events  # the case's events, unpadded
    remaining_times: (
        torch.Tensor
    )  # standardized minutes from each event to the case's real ending, [len(events)]
    # The lower bound for the case cut points,
    # which the preprocessing step computed and stored in the log.
    min_prefix_len: int


class TraceDataset(Dataset):
    """A PyTorch Dataset of traces, each cut into a set of (prefix, suffix) pairs.

    A case of `n` events yields one data point per cut point `k`:
    - **prefix**: `events[:k]`
    - **suffix**: `events[k:]` followed by EOT.

    Prefixes and suffixes are both padded to `max_trace_length`.
    """

    def __init__(self, codec: DatasetCodec, *, split: Split):
        """Read one preprocessed split and cut every trace into all possible (prefix, suffix) pairs.

        Args:
            codec: The dataset codec the split was preprocessed against, which names where the
                split is as well as what its values are encoded through.
            split: Which of the three to read.
        """
        self.codec = codec
        self.max_len = codec.max_trace_length

        # Read the split and encode it whole: the same work done per event in `__getitem__`
        # would be repeated for every cut point of every case.
        split_dataset = codec.read_split(split)

        events = _encode_events(codec, split_dataset)
        remaining_times = torch.from_numpy(codec.remaining_time.encode(split_dataset))

        self._cases = _group_cases(
            split_dataset,
            events=events,
            remaining_times=remaining_times,
        )

        self._case_cuts: list[tuple[int, int]] = [
            (case_idx, k)
            for case_idx, case in enumerate(self._cases)
            for k in range(case.min_prefix_len, int(case.events.length))
        ]

    def _case_and_cut(self, i: int) -> tuple[_Case, int]:
        """Return the source case and cut point for the i-th split trace."""
        case_idx, k = self._case_cuts[i]
        return self._cases[case_idx], k

    def __len__(self) -> int:
        """Return the number of traces in this split."""
        return len(self._case_cuts)

    def __getitem__(self, i: int) -> TraceCut:
        """Return the i-th trace, both of its runs padded to `max_trace_length`."""
        case, k = self._case_and_cut(i)
        suffix_len = int(case.events.length) - k

        prefix = case.events.cut(slice(0, k)).padded(to=self.max_len)
        suffix = case.events.cut(slice(k, None)).padded(to=self.max_len)

        suffix.activities[suffix_len] = self.codec.activity.eot_index
        suffix.resources[suffix_len] = self.codec.resource.eot_index
        suffix = suffix._replace(length=suffix.length + 1)

        return TraceCut(
            case_id=case.case_id,
            prefix=prefix,
            suffix=suffix,
            inter_event_times=self._pad_target(case.events.inter_event_times[k : k + suffix_len]),
            remaining_times=self._pad_target(case.remaining_times[k - 1 : k + suffix_len - 1]),
        )

    def _pad_target(self, target: torch.Tensor) -> torch.Tensor:
        """Pad one of the decoder's time targets out to `max_trace_length`.

        Args:
            target: The target at each suffix position, `[suffix_len]`.
        Returns:
            `[max_trace_length]`, zero-filled past the suffix's content, which the loss masks out.
        """
        return F.pad(input=target, pad=(0, self.max_len - target.size(dim=0)))

    def length_sorted_indices(self) -> list[int]:
        """Order this split's traces by how many positions their suffix takes to decode.

        `Decoder.generate` runs a batch until every one of its rows has emitted EOT, so a batch
        of similarly-long suffixes finishes together; unsorted, almost every batch contains one
        early-cut, long-suffix straggler and decodes to nearly the cap regardless of its other
        rows. Sorting the whole split first is what lets that early exit actually save time.

        Returns:
            TraceCut indices, ascending by suffix length. Passed as a `DataLoader` sampler.
        """
        lengths = []
        for i in range(len(self)):
            case, k = self._case_and_cut(i)
            lengths.append(int(case.events.length) - k + 1)
        return sorted(range(len(self)), key=lengths.__getitem__)


def fixed_subset(dataset: Dataset, *, size: int, generator: torch.Generator) -> Dataset:
    """A random slice of `dataset`, or the whole of it if it is already no bigger.

    Drawn once, so every validation of a run reads the same traces and two of its points differ
    because the model moved rather than because the sample did.

    Args:
        dataset: The split to take from.
        size: How many items to keep.
        generator: The run's seeded generator.
    Returns:
        The slice, as a `Subset` the loaders can be built on directly.
    """
    # If the dataset is smaller than the requested size, just return it whole
    if len(dataset) <= size:
        return dataset
    # Otherwise, draw a random slice of the requested size and return it as a Subset
    indices = torch.randperm(n=len(dataset), generator=generator)[:size]
    return Subset(dataset=dataset, indices=indices.tolist())


def _encode_events(codec: DatasetCodec, log: pd.DataFrame) -> Events:
    """Map a run of raw events to the indices and normalized floats the model consumes.

    Args:
        codec: The dataset's codec, naming every channel read here and holding the vocabulary or
            range each is encoded through.
        log: The events as a preprocessed split holds them. The whole frame, not a selection of
            it: which columns each channel reads is the codec's answer, and this is where it is
            asked.

    Returns:
        The same events as vocabulary indices and normalized channels, unpadded, so every one of
        them counts towards `length`.
    """
    numeric_attributes, numeric_attributes_present = codec.encode_numeric_features(log)
    return Events(
        # `torch.tensor` rather than `from_numpy`: pandas hands back a read-only view of its
        # own block for some dtypes, which torch would wrap rather than copy.
        activities=torch.tensor(data=codec.activity.encode(log), dtype=torch.long),
        resources=torch.tensor(data=codec.resource.encode(log), dtype=torch.long),
        inter_event_times=torch.from_numpy(codec.inter_event_time.encode(log)),
        categorical_attributes=codec.encode_categorical_features(log),
        numeric_attributes=numeric_attributes,
        numeric_attributes_present=numeric_attributes_present,
        length=torch.tensor(data=len(log), dtype=torch.long),
    )


def _group_cases(
    split_dataset: pd.DataFrame,
    *,
    events: Events,
    remaining_times: torch.Tensor,
) -> list[_Case]:
    """Group a split's already-encoded events into per-case runs.

    Args:
        split_dataset: The split, from `_read_split`; only its case column and row order are
            read here, the values themselves already encoded into `events`.
        events: The split's events, encoded whole, indexed the same as `split_dataset`.
        remaining_times: The split's standardized remaining time, indexed the same way.
    Returns:
        One `_Case` per case of the split, each of them whole: preprocessing dropped the cases
        that do not fit `max_trace_length`, so nothing is cut short here.
    """
    cases = []
    for case_id, group in split_dataset.groupby(CASE_KEY, sort=False):
        # The case's rows as positions into the split-wide columns.
        positions = torch.from_numpy(split_dataset.index.get_indexer(group.index))
        # Constant over the case, so any of its rows answers for all of them.
        bounds = group.iloc[0]
        cases.append(
            _Case(
                case_id=str(case_id),
                events=events.cut(positions),
                remaining_times=remaining_times[positions],
                min_prefix_len=int(bounds[MIN_PREFIX_KEY]),
            )
        )
    return cases
