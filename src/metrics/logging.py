from collections.abc import Mapping
from dataclasses import asdict

import wandb


def log_records(records: Mapping[str, object], *, step: int) -> None:
    """Log dataclass records under caller-selected namespaces.

    Args:
        records: Scalar records keyed by their W&B namespace.
        step: Training step assigned to the W&B event.
    """
    payload = {
        f'{namespace}/{key}': value
        for namespace, record in records.items()
        for key, value in asdict(record).items()
    }
    wandb.log(payload, step=step)
