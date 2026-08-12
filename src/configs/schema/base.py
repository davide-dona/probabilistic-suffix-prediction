from pydantic import BaseModel, ConfigDict

# What a name that becomes a directory is allowed to be: lowercase, digits and hyphens, so it
# cannot be empty, escape its parent or nest a level nothing expects.
NAME_PATTERN = r'^[a-z0-9][a-z0-9-]*$'


class StrictModel(BaseModel):
    """Base model for all config sections: immutable and typo-proof."""

    model_config = ConfigDict(frozen=True, extra='forbid')
