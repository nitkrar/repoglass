-- Index format 2: rows in position order, vectors independent of their
-- batch. `schema_rev` hashes this file, so bumping the number rebuilds
-- every index written under an earlier format.

CREATE TABLE meta (
  id             INTEGER PRIMARY KEY CHECK (id = 1),
  schema_rev     TEXT    NOT NULL,
  embed_model    TEXT    NOT NULL,
  embed_backend  TEXT    NOT NULL DEFAULT 'static',
  embed_dims     INTEGER NOT NULL,
  embed_variant  TEXT    NOT NULL DEFAULT '',
  embed_doc_prefix TEXT  NOT NULL DEFAULT '',
  embed_pooling  TEXT    NOT NULL DEFAULT '',
  coverage       TEXT    NOT NULL,
  extractor_rev  TEXT    NOT NULL,
  categories_rev TEXT    NOT NULL DEFAULT '',
  chunking_rev   TEXT    NOT NULL DEFAULT '',
  last_scan_at   REAL    NOT NULL,
  last_skip_at   REAL
);

CREATE TABLE file (
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  path      TEXT    NOT NULL UNIQUE,
  mtime_ns  INTEGER NOT NULL,
  size      INTEGER NOT NULL,
  lang      TEXT    NOT NULL,
  -- `path` with camelCase and separators split into words, so BM25 can
  -- match a query naming the file. Per file, not per chunk: the same
  -- string prefixed every chunk of a file when it was stored inline.
  path_words TEXT   NOT NULL DEFAULT '',
  -- docs / config / data / tests / code, decided by corpus.classify at
  -- walk time. Stored so a filtered query is an equality test any
  -- connection can run, rather than generated SQL calling a registered
  -- Python function.
  content_type TEXT NOT NULL DEFAULT 'code'
);

-- Deliberately NOT indexed. An index here makes the planner drive
-- `fts_search` from the category instead of from BM25 rank, probing
-- FTS once per candidate chunk rather than walking the ranked list,
-- which is dramatically slower. `file` has thousands of rows, so
-- scanning it costs nothing worth an index.

CREATE TABLE symbol (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  file_id    INTEGER NOT NULL REFERENCES file(id) ON DELETE CASCADE,
  name       TEXT    NOT NULL,
  tag        TEXT    NOT NULL,
  start_line INTEGER NOT NULL,
  end_line   INTEGER NOT NULL,
  -- A definition's header, cut at its body. NULL on a reference, and
  -- on a definition whose grammar gives no body to cut at.
  signature  TEXT,
  -- For a reference, the definition whose span contains it. NULL at
  -- module scope, and always NULL on a definition. Both endpoints are
  -- rows in this table, so "who calls what" is a self-join and needs
  -- no second table to hold it.
  enclosing_id INTEGER REFERENCES symbol(id) ON DELETE SET NULL
);
CREATE INDEX symbol_name ON symbol(name, tag);
CREATE INDEX symbol_file ON symbol(file_id);

CREATE TABLE chunk (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  file_id      INTEGER NOT NULL REFERENCES file(id)   ON DELETE CASCADE,
  -- NULL for a window chunk, which covers a span of the file that no
  -- definition owns. Definition chunks keep the link so the exact tier
  -- can go from a symbol name straight to its text.
  symbol_id    INTEGER          REFERENCES symbol(id) ON DELETE CASCADE,
  start_line   INTEGER NOT NULL,
  end_line     INTEGER NOT NULL,   -- capped span, may be shorter than the symbol's
  content_hash TEXT NOT NULL
);
CREATE INDEX chunk_file   ON chunk(file_id);
CREATE UNIQUE INDEX chunk_symbol ON chunk(symbol_id) WHERE symbol_id IS NOT NULL;

-- The chunk's text, vector and keyword index live in semsift's `rg_*`
-- tables in this database, one item per chunk with the chunk's id. The
-- keyword text is derived there from the file's path words and the text;
-- only a rendering the settings reshape further is stored.
