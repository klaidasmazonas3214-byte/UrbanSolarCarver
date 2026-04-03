# Configuration

USC uses YAML files for all pipeline parameters. The configuration system supports:

- **YAML loading** with recursive merging of nested dictionaries.
- **CLI overrides** (`-o key=value`) that take precedence over file values, applied via dot-path notation.
- **Pydantic v2 validation** with strict bounds checking, type coercion, and informative error messages on invalid inputs.
- **Manifest schemas** for each pipeline stage, enabling re-entry at any stage.

See also: [Configuration guide](../getting-started/configuration.md) for user-facing parameter reference.

## Config loader

::: urbansolarcarver.load_config

## Schemas

::: urbansolarcarver.pydantic_schemas
