"""Configuration: paths, settings, and their resolution."""

from .paths import Paths
from .load import load
from .render import as_toml
from .schema import (DATA_DIR_NAME, HOME_ENV,
                     IGNORE_FILE_NAME, MAX_CHUNK_CHARS, MIN_CHUNK_CHARS,
                     NEVER_INDEX_LANGS, RRF_K, Settings, VECTOR_DTYPE,
                     categories_rev)

__all__ = ["Paths", "Settings", "load", "as_toml", "RRF_K", "VECTOR_DTYPE", "HOME_ENV",
           "MIN_CHUNK_CHARS", "MAX_CHUNK_CHARS", "NEVER_INDEX_LANGS",
           "DATA_DIR_NAME", "IGNORE_FILE_NAME",
           "categories_rev"]
