"""Render Settings as TOML."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from . import schema
from .schema import Settings

#: Which TOML table each setting is presented under. Grouping is
#: presentation only -- `load._flatten` accepts a flat key just the same.
_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("retrieval", ("ranker", "prose_min_words", "adaptive_alpha",
                   "alpha_symbol", "alpha_prose",
                   "saturation_decay", "rerank", "file_coherence", "stem_boost",
                   "definition_boost", "candidate_depth", "content",
                   "content_excluded", "index_excluded")),
    ("categories", ("doc_languages", "config_languages", "data_languages",
                    "test_markers")),
    ("embeddings", ("embed_backend", "embed_model", "embed_endpoint",
                    "embed_api_key", "embed_query_prefix", "embed_doc_prefix",
                    "embed_providers", "embed_onnx_file")),
    ("extraction", ("coverage", "window_chars", "lexical_mode",
                    "lexical_cap_chars", "lexical_enrich", "split_identifiers",
                    "distill_docs", "max_chunk_lines")),
    ("corpus", ("data_dir", "gitignore", "hard_exclude", "max_file_bytes")),
    ("refresh", ("refresh_mode", "rescan_after_seconds")),
)


def _emittable(name: str) -> bool:
    """Whether a setting may appear in a generated config.

    `load()` refuses a repository config that carries a secret, so
    emitting one produces a file this tool writes and then will not
    read. Skipping it here keeps `init` output loadable by
    construction rather than by a caller remembering.
    """
    from .load import _SECRET

    return name not in _SECRET


def as_toml(settings: Settings | None = None, *, comments: bool = True,
            active: bool = False) -> str:
    """Render settings as TOML, generated from the dataclass.

    Keys are commented out unless `active`, so the output can be saved
    as a starting config without pinning every default to whatever it
    happens to be today. `active=True` renders a run's resolved
    settings, for diagnostics.
    """
    s = settings or Settings()
    docs = {}
    if comments:
        src = Path(schema.__file__).read_text().splitlines()
        pending: list[str] = []
        for line in src:
            bare = line.strip()
            if bare.startswith("#:"):
                pending.append(bare[2:].strip())
            elif ":" in bare and pending and not bare.startswith("#"):
                docs[bare.split(":", 1)[0].strip()] = pending
                pending = []
            elif not bare.startswith("#"):
                pending = []

    known = {f.name for f in fields(s)}
    head = ("# repoglass settings as resolved for this run."
            if active else
            "# repoglass settings. Every key is optional. Uncomment only\n"
            "# what you want to change; the rest track the defaults.")
    out: list[str] = [head, ""]
    placed: set[str] = set()
    for table, names in _GROUPS:
        rows = [n for n in names if n in known and _emittable(n)]
        if not rows:
            continue
        out.append(f"[{table}]")
        for name in rows:
            placed.add(name)
            for line in docs.get(name, ()):
                out.append(f"# {line}")
            value = getattr(s, name)
            # TOML has no null, so an unset key must be absent rather
            # than empty-string.
            prefix = "" if active and value is not None else "# "
            out.append(f"{prefix}{name} = {_toml_value(value)}")
            out.append("")
    missing = sorted(n for n in known - placed if _emittable(n))
    if missing:
        out.append("# not yet grouped")
        for name in missing:
            value = getattr(s, name)
            out.append(f"{'' if active and value is not None else '# '}"
                       f"{name} = {_toml_value(value)}")
    return "\n".join(out).rstrip() + "\n"


def _toml_value(value) -> str:
    """One value as TOML. None renders as `""` -- TOML has no null."""
    if value is None:
        return '""'
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return '"' + str(value).replace('"', '\\"') + '"'
