# sample.py — expected extraction

Counts are hand-verified against the file. Update both together.

## Definitions (7)

| # | name | kind | note |
|---|---|---|---|
| 1 | `top_level` | function | |
| 2 | `outer` | function | contains `inner` |
| 3 | `inner` | function | nested inside `outer` |
| 4 | `helper` | function | |
| 5 | `Thing` | class | contains `method`, `tiny` |
| 6 | `method` | function | |
| 7 | `tiny` | function | span is 32 chars, below MIN_CHUNK_CHARS (60) -- symbol, no chunk. Keep it comment-free: an inline comment pushes it over. |

## Cases this fixture exists to pin

- **innermost enclosing**: `helper()` on the `inner` line is inside both
  `inner` and `outer`. Its edge src must be `inner`.
- **module scope**: `top_level()` on the last line, and the `import os`, sit
  inside no definition. Their edge src must be NULL.
- **class nesting**: `helper()` inside `method` resolves to `method`, not `Thing`.
- **short span**: `tiny` must produce a symbol row and no chunk row.
