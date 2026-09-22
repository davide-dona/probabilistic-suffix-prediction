import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.runs.provenance import ArtifactProvenance


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file's bytes."""
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def with_metadata(schema: pa.Schema, metadata: dict[str, str]) -> pa.Schema:
    """Attach validated provenance and artifact-specific metadata to a Parquet schema."""
    ArtifactProvenance.from_metadata(metadata)
    return schema.with_metadata(
        (schema.metadata or {}) | {b'provenance': json.dumps(metadata).encode()}
    )


def read_metadata(parquet: pq.ParquetFile) -> dict[str, str]:
    """Read validated provenance and artifact-specific metadata from a Parquet file."""
    raw = (parquet.schema_arrow.metadata or {}).get(b'provenance')
    if raw is None:
        raise ValueError('Missing artifact provenance; regenerate this file.')
    metadata = json.loads(raw)
    ArtifactProvenance.from_metadata(metadata)
    return metadata


def with_vocabulary(schema: pa.Schema, vocabulary: Sequence[str]) -> pa.Schema:
    """Attach the activity vocabulary used to encode a Parquet artifact."""
    return schema.with_metadata(
        (schema.metadata or {}) | {b'activities': json.dumps(list(vocabulary)).encode()}
    )


def read_vocabulary(schema: pa.Schema) -> tuple[str, ...]:
    """Read the activity vocabulary from an artifact schema."""
    raw = (schema.metadata or {}).get(b'activities')
    if raw is None:
        raise ValueError('Missing activity vocabulary; regenerate this file.')
    return tuple(json.loads(raw))
