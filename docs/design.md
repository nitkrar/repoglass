# Design

This doc records invariants that are not obvious from reading one source file.
Public signatures live in `src/repoglass/index.py`; CLI flags and payloads live
in `src/repoglass/cli.py`; non-obvious tradeoffs live in
[decisions-log.md](decisions-log.md).

## Scope

repoglass indexes one directory tree into one local SQLite database and answers
three question shapes:

- exact symbol navigation through `definitions()` and `references()`
- ranked retrieval through `search()`
- unranked symbol enumeration through `symbols()` and `symbol_counts()`

The indexed tree is read-only. Index data lives separately.

## Public surface

- `Index` is the library surface exported from `src/repoglass/__init__.py`.
- `definitions()` and `references()` are exact symbol-table lookups. They do
  not rank and they do not cap results.
- `search()` ranks and truncates. `Hit.score` is only comparable within one
  call.
- `symbols()` and `symbol_counts()` read the symbol table, so a definition may
  be enumerable even when it is too small to produce a retrievable chunk.
- Files without a tags query can still contribute retrieval chunks through the
  window path, but they do not produce definitions or references.

## Retrieval

`search()` runs three tiers: exact symbol match, FTS5/BM25, and vector
similarity. `Settings.ranker` merges those tier outputs; `rerank` is a post-pass
over the merged candidate pool rather than a fourth retriever.

## Refresh

- Staleness is tracked by comparing `(mtime_ns, size)` from discovery with the
  stored `file` rows.
- Deletions rely on `PRAGMA foreign_keys = ON` on every SQLite connection.
- Under `refresh_mode="auto"`, reads attempt a best-effort refresh. A never-built
  index blocks instead of returning an ambiguous empty result; a busy existing
  index serves stale data and records the skip.

## Non-goals

- Type or import resolution.
- Resident services, daemons, or MCP surfaces.
- Cross-repo queries.
- Editing or refactoring indexed source.
