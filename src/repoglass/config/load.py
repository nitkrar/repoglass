"""Settings resolution through files, environment, and overrides."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from .schema import ENV_PREFIX, Settings

#: Settings that must never appear in the repository-level config,
#: because that file is meant to be committed so a team shares the same
#: classification rules. A comment recommending the environment
#: variable is not a control; this is.
_SECRET = frozenset({"embed_api_key"})


def _flatten(raw: dict) -> dict:
    """Accept both flat keys and grouped tables.

    `[retrieval] ranker = "rrf"` and a top-level `ranker = "rrf"` mean the
    same thing: grouping is presentation, not structure. Settings is flat,
    so a nested table cannot introduce a name that a flat key could not.
    """
    out: dict = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            out.update(_flatten(value))
        else:
            out[key] = value
    return out


def _read(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {}
    import tomllib

    return _flatten(tomllib.loads(path.read_text()))


def load(
    *,
    user_config: Path | None = None,
    repo_config: Path | None = None,
    overrides: dict[str, object] | None = None,
) -> Settings:
    """Resolve Settings from every source, later ones winning.

    User config, repository config, `REPOGLASS_*` environment, then
    `overrides`. An unknown key raises rather than being ignored, and a
    secret in the repository config is refused outright.
    """
    import os

    known = {f.name: f for f in fields(Settings)}
    merged: dict = {}
    merged.update(_read(user_config))

    from_repo = _read(repo_config)
    leaked = sorted(set(from_repo) & _SECRET)
    if leaked:
        raise ValueError(
            f"{', '.join(leaked)} must not be set in the repository config"
            f" ({repo_config}): that file is meant to be committed."
            f" Use the {ENV_PREFIX}{leaked[0].upper()} environment variable,"
            f" or the user-level config."
        )
    merged.update(from_repo)
    for name in known:
        env = os.environ.get(f"{ENV_PREFIX}{name.upper()}")
        if env is not None:
            merged[name] = env
    merged.update(overrides or {})

    unknown = sorted(set(merged) - set(known))
    if unknown:
        # Silently ignoring a typo means the setting never takes effect and
        # nothing says so.
        raise ValueError(f"unknown setting(s): {', '.join(unknown)}")

    coerced: dict = {}
    for name, value in merged.items():
        target = known[name].type
        if isinstance(value, list):
            # Nested too: TOML gives lists of lists, and a tuple
            # setting holding lists would never compare equal to its
            # default.
            value = tuple(tuple(x) if isinstance(x, list) else x for x in value)
        elif isinstance(value, str) and "int" in str(target):
            value = int(value)
        coerced[name] = value
    return Settings(**coerced)
