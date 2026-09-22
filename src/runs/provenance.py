from dataclasses import dataclass
from typing import Self

from src.runs.identity import RunIdentity


@dataclass(frozen=True)
class ArtifactProvenance:
    """The training run and exact checkpoint from which an artifact was derived.

    Artifact metadata may contain additional, artifact-specific keys. This value owns the four
    common keys and leaves those additional keys untouched when reading metadata.
    """

    run: RunIdentity
    checkpoint_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint_sha256, str) or not self.checkpoint_sha256:
            raise ValueError(
                f'Invalid artifact provenance checkpoint_sha256: {self.checkpoint_sha256!r}'
            )

    def as_metadata(self) -> dict[str, str]:
        """Return the metadata fields shared by every checkpoint-derived artifact."""
        return self.run.as_dict() | {'checkpoint_sha256': self.checkpoint_sha256}

    @classmethod
    def from_metadata(cls, metadata: object) -> Self:
        """Read and validate the shared provenance fields from artifact metadata."""
        if not isinstance(metadata, dict):
            raise ValueError('Invalid artifact provenance.')
        checkpoint_sha256 = metadata.get('checkpoint_sha256')
        return cls(
            run=RunIdentity.from_metadata(metadata),
            checkpoint_sha256=checkpoint_sha256,
        )
