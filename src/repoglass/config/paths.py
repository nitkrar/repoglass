"""All path derivation.

Nothing else in the codebase constructs a path. Two roots, both injectable,
so test isolation is a single construction.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .schema import DATA_DIR_NAME, HOME_ENV, IGNORE_FILE_NAME


@dataclass(frozen=True)
class Paths:
    root: Path              # the directory being indexed
    home: Path              # user-level; default ~/.repoglass
    data_dir: str | None = None   # see `data`

    @classmethod
    def for_root(cls, root: Path, *, home: Path | None = None) -> "Paths":
        """Resolve both roots, honouring REPOGLASS_HOME when home is not given."""
        if home is None:
            env = os.environ.get(HOME_ENV)
            home = Path(env).expanduser() if env else Path.home() / DATA_DIR_NAME
        return cls(root=Path(root).expanduser().resolve(), home=Path(home))

    @property
    def data(self) -> Path:
        """Where this repository's index lives.

        Central by default, under `home`, keyed by the resolved root so
        two checkouts of the same project do not share one index. The
        indexed tree is never written to, which is what allows a
        read-only checkout to be indexed at all.

        `data_dir` overrides: an absolute path is used as given, a
        relative one resolves against the root, so `.repoglass` puts
        the index back beside the code.
        """
        if self.data_dir is None:
            return self.home / "index" / _index_key(self.root)
        override = Path(self.data_dir).expanduser()
        return override if override.is_absolute() else self.root / override

    @property
    def db(self) -> Path:
        return self.data / "index.db"

    @property
    def repo_ignore(self) -> Path:
        return self.root / IGNORE_FILE_NAME

    @property
    def gitignore(self) -> Path:
        return self.root / ".gitignore"

    @property
    def repo_config(self) -> Path:
        return self.root / "repoglass.toml"

    @property
    def models(self) -> Path:
        return self.home / "models"

    @property
    def cache(self) -> Path:
        return self.home / "cache"

    @property
    def user_config(self) -> Path:
        return self.home / "repoglass.toml"


def _index_key(root: Path) -> str:
    """A stable directory name for one repository.

    The hash disambiguates same-named projects; the basename is there
    so the directory is recognisable when someone goes looking.
    """
    import hashlib

    resolved = str(Path(root).expanduser().resolve())
    digest = hashlib.blake2b(resolved.encode(), digest_size=4).hexdigest()
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in Path(resolved).name)
    return f"{safe or 'repo'}-{digest}"
