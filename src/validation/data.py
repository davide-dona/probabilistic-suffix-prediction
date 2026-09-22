from omegaconf import DictConfig

from src.logs.keys import INTER_EVENT_TIME_KEY
from src.validation.primitives import validate_identifier, validate_number


def validate_data(data: DictConfig) -> None:
    """Validate the data section used to preprocess an event log.

    Raises:
        ValueError: If a split, percentile, feature relationship, separator, or dataset name is
            invalid.
    """
    validate_identifier(
        data.name,
        'data.name',
        r'[a-z0-9][a-z0-9-]*',
        'lowercase letters, digits, and hyphens',
    )

    splits = [data.train_split, data.val_split, data.test_split]
    for value in splits:
        validate_number(value, 'split fraction')
    if abs(sum(splits) - 1) > 1e-6:
        raise ValueError('train/val/test splits must sum to 1')

    for key in ('max_seq_len_percentile', 'max_case_duration_percentile'):
        validate_number(data[key], f'data.{key}')
        if data[key] > 100:
            raise ValueError(f'data.{key} must not exceed 100')

    if not set(data.log_scaled_features) <= set(data.event_features):
        raise ValueError('log_scaled_features must be event_features')
    if INTER_EVENT_TIME_KEY in data.event_features:
        raise ValueError('inter_event_time must not also be an event_feature')

    if not isinstance(data.separator, str) or not data.separator:
        raise ValueError('data.separator must be a nonempty string')


def validate_declare(config: DictConfig) -> None:
    """Validate the Declare-mining settings used during preprocessing.

    Raises:
        ValueError: If a support threshold, cardinality limit, or vacuity flag is invalid.
    """
    for key in ('min_support', 'itemsets_support'):
        validate_number(config[key], f'declare.{key}')
        if config[key] > 1:
            raise ValueError(f'declare.{key} must not exceed 1')

    validate_number(config.max_cardinality, 'declare.max_cardinality', integer=True)

    if not isinstance(config.consider_vacuity, bool):
        raise ValueError('declare.consider_vacuity must be boolean')
