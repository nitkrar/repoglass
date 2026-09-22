# repoglass

Local code and prose search with exact symbol lookup.

repoglass builds a local index for one directory tree. `rpg search` does ranked
retrieval; `rpg defs` and `rpg refs` read the stored symbol table for exact
navigation.

## Quickstart

```bash
uv tool install repoglass        # or: pipx install repoglass
cd ~/your/repo
rpg search "how are retries handled"
```

Either installer puts `rpg` and `repoglass` on PATH and keeps them there
across upgrades. To work on repoglass itself, `pip install -e .` inside a
virtualenv leaves both commands in that virtualenv's `bin/`.

The first command that needs an index builds it. If the default model or grammar
bundle is not cached yet, that first run may download them; later runs are
local.

## Next commands

```bash
rpg defs Index
rpg refs humanise
rpg symbols --count-by lang
rpg status
repoglass --help
repoglass search --help
```

## Docs

- [docs/design.md](docs/design.md) records invariants that are not obvious from
  one source file.
- [docs/decisions-log.md](docs/decisions-log.md) records non-obvious decisions
  and absences.
- [docs/worklist.md](docs/worklist.md) lists current user-visible defects.
- `./.venv/bin/python benchmarks/run_corpus.py` prints current corpus scores.
- `python -m unittest discover -s tests` runs the test suite. It is stdlib
  `unittest`; there is no test dependency to install.

## Licence

MIT, see [LICENSE](LICENSE). Third-party material is attributed in
[src/repoglass/corpus/queries/NOTICE.md](src/repoglass/corpus/queries/NOTICE.md).
