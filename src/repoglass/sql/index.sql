CREATE TABLE meta (
  id             INTEGER PRIMARY KEY CHECK (id = 1),
  schema_rev     TEXT    NOT NULL,
  embed_model    TEXT    NOT NULL,
  embed_backend  TEXT    NOT NULL DEFAULT 'static',
  embed_dims     INTEGER NOT NULL,
  coverage       TEXT    NOT NULL,
  extractor_rev  TEXT    NOT NULL,
  categories_rev TEXT    NOT NULL DEFAULT '',
  -- Set in the same transaction as any write that changes what FTS5
  -- should hold, and cleared by the rebuild. Chunks are committed
  -- before the rebuild runs, so without this a run that dies between
  -- the two leaves rows that are all present and an index that
  -- matches nothing -- and the next walk, seeing no file changed,
  -- never reaches the rebuild again.
  fts_dirty      INTEGER NOT NULL DEFAULT 0,
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
  text         TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  vec          BLOB,
  -- Normally NULL: the view derives the FTS text from the path and the
  -- span. Settings that reshape it further -- identifier splitting,
  -- semble's enriched header -- are not expressible in SQL, so those
  -- renderings are stored, and only those.
  lexical_override TEXT
);
CREATE INDEX chunk_file   ON chunk(file_id);
CREATE UNIQUE INDEX chunk_symbol ON chunk(symbol_id) WHERE symbol_id IS NOT NULL;

-- What FTS5 indexes: the humanised path followed by the chunk body.
-- A view rather than a column because the string is a concatenation of
-- two things already stored, and writing it down cost more than every
-- other column combined. Plain SQL, so the database stays readable by
-- anything that speaks SQLite.
CREATE VIEW chunk_lexical AS
  SELECT c.id AS rowid,
         COALESCE(c.lexical_override, f.path_words || char(10) || c.text)
           AS lexical_text
    FROM chunk c JOIN file f ON f.id = c.file_id;

CREATE VIRTUAL TABLE chunk_fts USING fts5(
  lexical_text, content='chunk_lexical', content_rowid='rowid',
  tokenize='unicode61'
);

