from omegaconf import DictConfig

from src.validation.primitives import validate_identifier, validate_number


def validate_sampling(config: DictConfig) -> None:
    """Validate the temperature and nucleus-sampling parameters for output heads.

    Raises:
        ValueError: If the section has missing or extra keys, or either value is out of range.
    """
    if set(config) != {'temperature', 'top_p'}:
        raise ValueError('sampling must contain exactly temperature and top_p')

    validate_number(config.temperature, 'sampling.temperature')
    validate_number(config.top_p, 'sampling.top_p')

    if config.top_p > 1:
        raise ValueError('sampling.top_p must not exceed 1')


def validate_model(model: DictConfig) -> None:
    """Validate a configured model architecture and its architecture-specific settings.

    Raises:
        ValueError: If a dimension, probability, transformer layout, or model-kind-specific
            section is invalid.
    """
    validate_identifier(
        model.name,
        'model.name',
        r'[a-z0-9][a-z0-9_]*',
        'lowercase letters, digits, and underscores',
    )
    if model.kind not in ('transformer_cvae', 'head_sampling_transformer'):
        raise ValueError(f'Unknown model kind: {model.kind}')

    validate_number(model.d_model, 'model.d_model', integer=True)
    for key, value in model.embeddings.items():
        validate_number(value, f'model.embeddings.{key}', integer=True)

    for name in ('encoder', 'decoder'):
        section = model[name]

        for key in ('num_layers', 'num_heads', 'feedforward_dim'):
            validate_number(section[key], f'model.{name}.{key}', integer=True)
        if model.d_model % section.num_heads:
            raise ValueError(f'model.{name}.num_heads must divide model.d_model')

        for key in ('dropout', 'activity_dropout'):
            if key in section:
                validate_number(section[key], f'model.{name}.{key}', inclusive=True)
                if section[key] >= 1:
                    raise ValueError(f'model.{name}.{key} must be below 1')

    validate_number(model.decoder.head_hidden_dim, 'model.decoder.head_hidden_dim', integer=True)

    if model.kind == 'head_sampling_transformer':
        if any(key in model for key in ('prior', 'latent', 'loss')):
            raise ValueError(
                'head_sampling_transformer does not accept prior, latent, or loss settings'
            )
        validate_sampling(model.sampling)
        return

    if 'sampling' in model:
        raise ValueError('transformer_cvae does not accept sampling settings')

    validate_number(model.latent.latent_dim, 'model.latent.latent_dim', integer=True)
    for width in model.prior.hidden_dims:
        validate_number(width, 'model.prior.hidden_dims', integer=True)

    validate_number(model.prior.dropout, 'model.prior.dropout', inclusive=True)
    if model.prior.dropout >= 1:
        raise ValueError('model.prior.dropout must be below 1')

    validate_number(
        model.loss.kl_annealing_ramp_steps,
        'model.loss.kl_annealing_ramp_steps',
        integer=True,
    )

    for key in ('kl_annealing_start_weight', 'kl_annealing_full_weight', 'free_bits'):
        validate_number(model.loss[key], f'model.loss.{key}', inclusive=True)
