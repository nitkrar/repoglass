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

Encoders, the store behind the keyword and vector tiers, rank fusion, window
chunking and the directory walk come from
[semsift](https://github.com/nitkrar/semsift). Each chunk is a semsift item in
the same SQLite file (`rg_*` tables), with the chunk's id and its file's
category, language and path as filterable fields; the chunk's text lives only
in its item. repoglass keeps its own `file`, `symbol` and `chunk` tables for
navigation, the exact tier and reranking, and maps its settings onto semsift in `embeddings.py`, `store.py`,
`search/`, `corpus/windows.py` and `corpus/discovery.py`.

When the model's outputs no longer match the stored vectors (semsift's canary
check), search skips the vector tier and logs a warning; `rpg index --force`
re-embeds every chunk.

## Refresh

- Staleness is tracked by comparing `(mtime_ns, size)` from discovery with the
  stored `file` rows.
- Deletions rely on `PRAGMA foreign_keys = ON` on every SQLite connection.
- Everything else that shaped the stored rows is the index identity in `meta`
  (D28). A change to any of it rebuilds the index.
- A refresh embeds any chunk still without a vector, so an embedding failure
  is retried without a file change.
- Under `refresh_mode="auto"`, reads attempt a best-effort refresh. A never-built
  index blocks instead of returning an ambiguous empty result; a busy existing
  index serves stale data and records the skip.

## Non-goals

- Type or import resolution.
- Resident services, daemons, or MCP surfaces.
- Cross-repo queries.
- Editing or refactoring indexed source.
