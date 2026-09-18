"""All SQL: schema, migrations, connection setup, queries.

Nothing outside this module issues SQL, so the schema can change without
rippling. One database per indexed directory.
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable, Sequence

from .config import Settings, categories_rev
from .models import Chunk, SearchMode, SourceFile, Symbol
from .text import humanise


#: Data, not a package -- same shape as `corpus/queries`.
SQL_DIR = Path(__file__).parent / "sql"


def schema_sql() -> str:
    """The DDL, read from `sql/index.sql` rather than written inline.

    One copy, openable by anything that reads SQL. The file must be
    declared as package data or an installed wheel omits it and the
    first open fails -- which is what `test_every_data_file_is_declared`
    is for, since an editable install reads the source tree and hides it.
    """
    return (SQL_DIR / "index.sql").read_text(encoding="utf-8")


#: The DDL as executed. Kept as a module attribute because the
#: revision is a hash of exactly this string.
SCHEMA = schema_sql()


def schema_rev() -> str:
    """Fingerprint of the DDL.

    Not a release version and not a migration chain -- there are no
    migrations. It answers one question at open time: do the tables on
    disk match the tables this code expects? If not, drop and rebuild.
    Without the check every query fails with `no such column` after any
    schema edit.

    Hashed rather than hand-maintained so it cannot be left un-bumped
    on a schema edit, which is exactly the failure it exists to catch.
    """
    import hashlib

    return hashlib.blake2b(SCHEMA.encode(), digest_size=8).hexdigest()


def connect(db_path: Path) -> sqlite3.Connection:
    """Open a connection with the required pragmas.

    `foreign_keys` is per-connection and OFF by default in Python. Without
    it every ON DELETE CASCADE in the schema is inert, which silently
    disables the entire deletion path.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # Written unconditionally because the index directory is only
    # sometimes inside the repository -- `data_dir` can put it there.
    # Ignoring its own directory keeps it out of `git status` without
    # editing a file the user owns.
    marker = db_path.parent / ".gitignore"
    if not marker.exists():
        marker.write_text("*\n")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


@dataclass(frozen=True)
class Identity:
    schema_rev: str
    embed_model: str
    #: Which embedder wrote the stored vectors. One model name can be
    #: served by two backends at the same width and still yield
    #: vectors that are not comparable, so the width cannot stand in
    #: for this.
    embed_backend: str
    embed_dims: int
    coverage: str
    extractor_rev: str
    #: Fingerprint of the category lists. `content_type` is written at
    #: walk time, so editing `doc_languages` or `test_markers` must
    #: reclassify; without this the stored value silently goes stale.
    categories_rev: str = ""


#: Fields that invalidate stored chunks and vectors.
_REINDEX_FIELDS = ("schema_rev", "embed_model", "embed_backend",
                   "embed_dims", "coverage", "extractor_rev",
                   "categories_rev")


class Store:
    def __init__(self, conn: sqlite3.Connection, settings: Settings) -> None:
        self.conn = conn
        self._settings = settings
        # (content, langs, include, exclude) -> (ids, packed blob).
        # See `vectors`.
        self._vector_cache: dict[str, tuple[list[int], bytes]] = {}
        #: Depth of nested `transaction()` blocks. Writers commit on
        #: their own at zero and defer above it.
        self._depth = 0

    @contextmanager
    def transaction(self):
        """Group writes so a failure leaves none of them.

        Every writer here commits when called on its own, which is
        what a caller doing one thing wants. Several writes that are
        only meaningful together -- a file's row and the chunks that
        make it searchable -- have to be one unit, or a crash between
        them leaves a row asserting that a file is indexed when it
        holds nothing. Nothing later notices: the source file is
        unchanged, so the walk sees neither an addition nor an edit.
        """
        self._depth += 1
        try:
            yield
        except BaseException:
            if self._depth == 1:
                self.conn.rollback()
                self._invalidate_vectors()
            raise
        finally:
            self._depth -= 1
        if self._depth == 0:
            self._commit()

    def _commit(self) -> None:
        """Commit, unless a `transaction()` block owns the decision."""
        if self._depth == 0:
            self.conn.commit()

    @classmethod
    def open(cls, db_path: Path, settings: Settings, *, extractor_rev: str = "") -> "Store":
        """Open the database, creating or rebuilding the schema as needed.

        `extractor_rev` is injected rather than imported: it is computed
        from the .scm set, which lives in corpus/, and store may not
        import corpus -- see the dependency rule. Inverting it here also
        keeps the store ignorant of how extraction works.
        """
        conn = connect(db_path)
        if _has_table(conn, "meta") and _stored_rev(conn) != schema_rev():
            # The tables themselves changed, so resetting rows is not enough.
            # At this corpus size rebuilding is cheaper than writing migrations.
            _drop_all(conn)
        if not _has_table(conn, "meta"):
            conn.executescript(SCHEMA)
            conn.execute(
                "INSERT INTO meta (id, schema_rev, embed_model,"
                " embed_backend, embed_dims, coverage, extractor_rev,"
                " categories_rev, last_scan_at)"
                " VALUES (1,?,?,?,?,?,?,?,0)",
                (schema_rev(), settings.embed_model, settings.embed_backend,
                 0, settings.coverage, extractor_rev,
                 categories_rev(settings)),
            )
            conn.commit()
        return cls(conn, settings)

    def identity(self) -> Identity | None:
        names = ", ".join(f.name for f in fields(Identity))
        row = self.conn.execute(
            f"SELECT {names} FROM meta WHERE id = 1"
        ).fetchone()
        return Identity(*row) if row else None

    def needs_reindex(self, current: Identity) -> bool:
        stored = self.identity()
        if stored is None:
            return True
        return any(
            getattr(stored, f) != getattr(current, f) for f in _REINDEX_FIELDS
        )

    def last_scan_at(self) -> float | None:
        row = self.conn.execute("SELECT last_scan_at FROM meta WHERE id=1").fetchone()
        return row[0] if row else None

    def mark_scanned(self) -> None:
        self.conn.execute("UPDATE meta SET last_scan_at=? WHERE id=1", (time.time(),))
        self._commit()

    def mark_skipped(self) -> None:
        """A refresh was wanted but the write lock was held."""
        self.conn.execute("UPDATE meta SET last_skip_at=? WHERE id=1", (time.time(),))
        self._commit()

    def upsert_files(self, files: Iterable[SourceFile]) -> None:
        # `path_words` is computed here rather than taken from the
        # caller: it must exist for every row the FTS view reads, and a
        # caller that forgot would lose path matching silently.
        self.conn.executemany(
            "INSERT INTO file (path, mtime_ns, size, lang, path_words,"
            "                  content_type)"
            " VALUES (?,?,?,?,?,?)"
            " ON CONFLICT(path) DO UPDATE SET"
            "   mtime_ns=excluded.mtime_ns, size=excluded.size,"
            "   lang=excluded.lang, content_type=excluded.content_type",
            [(f.path, f.mtime_ns, f.size, f.lang, humanise(f.path),
              f.content_type)
             for f in files],
        )
        self._commit()

    def delete_files(self, paths: Iterable[str]) -> int:
        cur = self.conn.executemany("DELETE FROM file WHERE path = ?",
                                    [(p,) for p in paths])
        self._invalidate_vectors()
        self._mark_fts_dirty()
        self._commit()
        return cur.rowcount

    def file_id(self, path: str) -> int | None:
        row = self.conn.execute("SELECT id FROM file WHERE path=?", (path,)).fetchone()
        return row[0] if row else None

    def replace_symbols(self, path: str, symbols: Sequence[Symbol]) -> None:
        fid = self.file_id(path)
        if fid is None:
            return
        self.conn.execute("DELETE FROM symbol WHERE file_id=?", (fid,))
        # Definitions first, so a reference can point at one. Ids are
        # issued here, so the link cannot be resolved by the caller.
        definitions = [s for s in symbols if s.tag == "def"]
        references = [s for s in symbols if s.tag != "def"]
        self.conn.executemany(
            "INSERT INTO symbol (file_id, name, tag, start_line, end_line,"
            " signature) VALUES (?,?,?,?,?,?)",
            [(fid, s.name, s.tag, s.start_line, s.end_line, s.signature)
             for s in definitions],
        )
        by_name: dict[str, list[tuple[int, int, int]]] = {}
        for sid, nm, first, last in self.conn.execute(
            "SELECT id, name, start_line, end_line FROM symbol"
            " WHERE file_id=? AND tag='def'", (fid,)
        ):
            by_name.setdefault(nm, []).append((first, last, sid))

        def enclosing_id(ref: Symbol) -> int | None:
            """The named definition whose span holds this reference.

            Matched on the span as well as the name: one file can
            define the same name twice -- a method on two classes --
            and their spans are disjoint, so containment picks the
            right one where the name alone could not.
            """
            for first, last, sid in by_name.get(ref.enclosing or "", ()):
                if first <= ref.start_line <= last:
                    return sid
            return None

        self.conn.executemany(
            "INSERT INTO symbol (file_id, name, tag, start_line, end_line,"
            " enclosing_id) VALUES (?,?,?,?,?,?)",
            [(fid, s.name, s.tag, s.start_line, s.end_line, enclosing_id(s))
             for s in references],
        )
        self._invalidate_vectors()
        self._commit()

    def upsert_chunks(self, path: str, chunks: Sequence[Chunk]) -> None:
        """Replace a file's chunks.

        A chunk carries its own id and need not correspond to a symbol.
        Definition chunks resolve `symbol_id` by (name, start_line) --
        that link is what lets the exact tier go from a matched symbol
        name straight to its text. A window chunk has no symbol and
        stores NULL.

        Rewritten wholesale per file rather than merged: symbol ids are
        reissued on every re-extraction, so any surviving row would
        point at a symbol that no longer exists.
        """
        fid = self.file_id(path)
        if fid is None:
            return
        ids = {
            (name, start): sid
            for sid, name, start in self.conn.execute(
                "SELECT id, name, start_line FROM symbol WHERE file_id=? AND tag='def'",
                (fid,),
            )
        }
        self.conn.execute("DELETE FROM chunk WHERE file_id=?", (fid,))
        rows = []
        claimed: set[int] = set()
        for c in chunks:
            sid = ids.get((c.name, c.start_line)) if c.name else None
            if sid is not None:
                # One chunk per definition. A tags query can capture the
                # same node under two capture names -- Kotlin `object`
                # matching both class and object patterns, for instance --
                # which yields two chunks for one symbol. The first wins;
                # the second would otherwise violate chunk_symbol and
                # abort the whole file's insert.
                if sid in claimed:
                    continue
                claimed.add(sid)
            rows.append((fid, sid, c.start_line, c.end_line, c.text,
                         c.content_hash, c.lexical_override))
        self.conn.executemany(
            "INSERT INTO chunk (file_id, symbol_id, start_line, end_line,"
            " text, content_hash, lexical_override) VALUES (?,?,?,?,?,?,?)",
            rows,
        )
        self._invalidate_vectors()
        self._mark_fts_dirty()
        self._commit()

    def set_vectors(self, vectors: Sequence[tuple[int, bytes]]) -> None:
        self.conn.executemany("UPDATE chunk SET vec=? WHERE id=?",
                              [(v, sid) for sid, v in vectors])
        self._invalidate_vectors()
        self._commit()

    def reset_content(self) -> None:
        """Drop every indexed row, keeping meta. Used when identity changes."""
        self.conn.execute("DELETE FROM file")
        self._invalidate_vectors()
        self._commit()
        self.rebuild_fts()

    def set_identity(self, current: "Identity") -> None:
        """Persist every field `needs_reindex` compares.

        Built from the dataclass rather than a hand-written column
        list. A field that is compared but not written can never
        converge: the stored value stays behind, so every refresh
        reindexes the whole corpus and the next one still thinks it
        must. Every `Identity` field is a `meta` column of the same
        name, which is what makes this safe.
        """
        names = [f.name for f in fields(Identity)]
        assignments = ", ".join(f"{name}=?" for name in names)
        self.conn.execute(
            f"UPDATE meta SET {assignments} WHERE id=1",
            tuple(getattr(current, name) for name in names),
        )
        self._commit()

    def fts_dirty(self) -> bool:
        """Whether a write has landed that the FTS index has not seen."""
        row = self.conn.execute("SELECT fts_dirty FROM meta WHERE id=1").fetchone()
        return bool(row and row[0])

    def _mark_fts_dirty(self) -> None:
        """Called by every writer that changes what FTS5 should hold.

        Inside the caller's transaction, so a rollback takes the flag
        with the rows it describes.
        """
        self.conn.execute("UPDATE meta SET fts_dirty=1 WHERE id=1")

    def rebuild_fts(self) -> None:
        """Wholesale rather than trigger-based.

        External-content FTS5 needs an order-sensitive delete-before-insert
        protocol that corrupts silently when violated; rebuild cannot.

        Clearing the flag is part of the same transaction as the
        rebuild: cleared first, a crash would lose the record that
        the work is still outstanding.
        """
        with self.transaction():
            self.conn.execute(
                "INSERT INTO chunk_fts(chunk_fts) VALUES('rebuild')")
            self.conn.execute("UPDATE meta SET fts_dirty=0 WHERE id=1")

    def known_files(self) -> dict[str, tuple[int, int]]:
        return {
            path: (mtime, size)
            for path, mtime, size in self.conn.execute(
                "SELECT path, mtime_ns, size FROM file"
            )
        }

    @staticmethod
    def _langs(lang) -> tuple[str, ...]:
        """One language, several, or nothing -> a tuple."""
        if lang is None:
            return ()
        return (lang,) if isinstance(lang, str) else tuple(lang)

    def _symbols(self, name: str, tag: str, lang, limit: int | None):
        sql = ("SELECT s.name, s.tag, f.path, s.start_line, s.end_line, f.lang,"
               " f.content_type, e.name, s.signature"
               " FROM symbol s JOIN file f ON f.id = s.file_id"
               " LEFT JOIN symbol e ON e.id = s.enclosing_id"
               " WHERE s.name = ? AND s.tag = ?")
        args: list = [name, tag]
        langs = self._langs(lang)
        if langs:
            sql += f" AND f.lang IN ({','.join('?' * len(langs))})"
            args.extend(langs)
        sql += " ORDER BY f.path, s.start_line"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [
            Symbol(name=n, tag=t, path=p, start_line=a, end_line=b, lang=lg,
                   content_type=ct, enclosing=enc, signature=sig)
            for n, t, p, a, b, lg, ct, enc, sig in self.conn.execute(sql, args)
        ]

    def definitions(self, name: str, *, lang=None) -> list[Symbol]:
        return self._symbols(name, "def", lang, None)

    def references(self, name: str, *, lang=None) -> list[Symbol]:
        return self._symbols(name, "ref", lang, None)

    #: What `count_by` may group on. A column each, so the grouping is
    #: the database's rather than a tally built in Python over rows we
    #: would have to fetch in full first.
    COUNT_BY = {"name": "s.name", "lang": "f.lang", "file": "f.path",
                "tag": "s.tag", "content": "f.content_type"}

    def _symbol_filter(self, pattern, tag, lang, content, include,
                       exclude) -> tuple[str, list]:
        """The WHERE shared by enumeration and counting.

        Unlike `_symbols`, the name is a GLOB and is optional: the
        question here is "which symbols are there", and requiring a
        name would make it another spelling of `defs`.
        """
        sql, args = "", []
        if pattern:
            sql += " AND s.name GLOB ?"
            args.append(pattern)
        if tag:
            sql += " AND s.tag = ?"
            args.append(tag)
        sql += (self._lang_clause(lang) + self._mode_clause(content)
                + self._path_clause(include) + self._exclude_clause(exclude))
        return sql, args

    def symbol_rows(self, pattern=None, *, tag=None, lang=None, content=None,
                    include=None, exclude=None, limit=None) -> list[Symbol]:
        """Every matching symbol, in path order. Not ranked, not scored.

        Reads the symbol table, not the chunk table, so a definition
        below MIN_CHUNK_CHARS is counted here though no search can
        return it.
        """
        where, args = self._symbol_filter(pattern, tag, lang, content,
                                          include, exclude)
        sql = ("SELECT s.name, s.tag, f.path, s.start_line, s.end_line,"
               " f.lang, f.content_type, e.name, s.signature"
               " FROM symbol s JOIN file f ON f.id = s.file_id"
               " LEFT JOIN symbol e ON e.id = s.enclosing_id"
               " WHERE 1=1" + where + " ORDER BY f.path, s.start_line")
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [
            Symbol(name=n, tag=t, path=p, start_line=a, end_line=b, lang=lg,
                   content_type=ct, enclosing=enc, signature=sig)
            for n, t, p, a, b, lg, ct, enc, sig in self.conn.execute(sql, args)
        ]

    def symbol_counts(self, by: str, pattern=None, *, tag=None, lang=None,
                      content=None, include=None, exclude=None,
                      limit=None) -> tuple[list[tuple[str, int]], int, int]:
        """Matching symbols grouped by one column, largest group first.

        Returns the groups plus the totals over *everything* matched,
        not over what `limit` kept. Summing the returned groups under a
        limit gives a number that looks like a total and is not one,
        and the caller cannot see that it was truncated.
        """
        column = self.COUNT_BY[by]
        where, args = self._symbol_filter(pattern, tag, lang, content,
                                          include, exclude)
        frm = (" FROM symbol s JOIN file f ON f.id = s.file_id"
               " WHERE 1=1" + where)
        groups, total = self.conn.execute(
            f"SELECT COUNT(DISTINCT {column}), COUNT(*)" + frm, args
        ).fetchone()
        sql = (f"SELECT {column}, COUNT(*) c" + frm +
               f" GROUP BY {column} ORDER BY c DESC, {column} ASC")
        if limit:
            sql += f" LIMIT {int(limit)}"
        rows = [(str(g), n) for g, n in self.conn.execute(sql, args)]
        return rows, groups, total

    def _lang_clause(self, lang) -> str:
        """SQL narrowing `f` to the requested languages.

        A sibling of `_mode_clause`, applied at the same three places.
        Pre-filtering, not post-filtering: restricting the candidate
        pool means `k` results are `k` results in that language, rather
        than `k` retrieved and then thinned to whatever survived.
        """
        langs = self._langs(lang)
        if not langs:
            return ""
        marks = ", ".join("'" + x.replace("'", "''") + "'" for x in langs)
        return f" AND f.lang IN ({marks})"

    def _exclude_clause(self, exclude) -> str:
        """The inverse of `_path_clause`. Applied at the same sites.

        Excluding wins over including: `--include 'src/*' --exclude
        '*_test.py'` gives source files that are not tests, which is
        what the pair reads as. Both are ANDed, so that falls out.
        """
        pats = self._langs(exclude)
        if not pats:
            return ""
        terms = " AND ".join("f.path NOT GLOB '" + x.replace("'", "''") + "'"
                             for x in pats)
        return f" AND ({terms})"

    def _path_clause(self, include) -> str:
        """SQL narrowing `f` to paths matching any of these globs.

        SQLite's GLOB, so `*` crosses `/` and `src/*` is recursive --
        the forgiving reading, and the one a caller typing "under src"
        means. gitignore-style globs would need `**` for that; we do
        not implement that dialect.

        Third sibling of `_mode_clause` and `_lang_clause`, applied at
        the same three query sites so the filter narrows the candidate
        pool rather than thinning results after the fact.
        """
        pats = self._langs(include)          # same one-or-many shape
        if not pats:
            return ""
        terms = " OR ".join("f.path GLOB '" + x.replace("'", "''") + "'"
                            for x in pats)
        return f" AND ({terms})"

    def _mode_clause(self, content) -> str:
        """SQL narrowing `f` to the requested categories.

        Empty or None means no restriction: "everything" is the absence
        of a filter, not a category. Several categories OR together, so
        `("code", "docs")` is expressible -- which a single-valued enum
        could not say without also admitting tests.

        The category was decided at walk time by corpus.classify and
        stored, so this is a plain `IN` test any connection can run,
        with no Python function a connection must first have registered.
        """
        wanted = normalise_content(content, self._settings)
        if not wanted:
            return ""
        marks = ", ".join("'" + w.replace("'", "''") + "'"
                          for w in sorted(wanted))
        return f" AND f.content_type IN ({marks})"

    def fts_search(self, query: str, *, limit: int, mode: SearchMode,
                   lang=None, include=None, exclude=None
                   ) -> list[tuple[int, float]]:
        """Returns (chunk_id, bm25_score). bm25() is negative, ascending-best.

        Joined through `chunk.file_id` rather than through `symbol`: a
        window chunk has no symbol, and an inner join via symbol would
        silently drop every one of them.
        """
        sql = (
            "SELECT t.rowid, bm25(chunk_fts) FROM chunk_fts t"
            " JOIN chunk c ON c.id = t.rowid"
            " JOIN file f ON f.id = c.file_id"
            " WHERE chunk_fts MATCH ?" + self._mode_clause(mode)
            + self._lang_clause(lang)
            + self._path_clause(include)
            + self._exclude_clause(exclude) +
            " ORDER BY bm25(chunk_fts) LIMIT ?"
        )
        try:
            return list(self.conn.execute(sql, (query, limit)))
        except sqlite3.OperationalError:
            return []    # malformed MATCH expression

    def pending_vectors(self) -> list[tuple[int, str]]:
        return list(self.conn.execute(
            "SELECT id, text FROM chunk WHERE vec IS NULL"
        ))

    def set_embed_dims(self, dims: int) -> None:
        self.conn.execute("UPDATE meta SET embed_dims=? WHERE id=1", (dims,))
        self._commit()

    def vectors(self, *, mode: SearchMode, lang=None, include=None,
                exclude=None) -> tuple[list[int], bytes]:
        """Every stored vector for `mode`, as ids plus one packed blob.

        Cached in memory per filter combination: uncached, re-reading
        the whole `vec` column dominates the tier pass and dwarfs the
        dot product it feeds. SQLite stays the store of record; this is
        only a read-through cache.

        Invalidated by `_invalidate_vectors`, called from every writer
        that can change the set: `set_vectors`, `upsert_chunks`,
        `delete_files`, `replace_symbols` (its cascade drops chunks) and
        `reset_content`.
        """
        # Keyed by the normalised categories, not the caller's argument:
        # `mode` may arrive as a list (argparse nargs="+"), which is
        # unhashable, and ("code",) must hit the same entry as "code".
        key = (normalise_content(mode, self._settings), self._langs(lang),
               self._langs(include), self._langs(exclude))
        hit = self._vector_cache.get(key)
        if hit is not None:
            return hit
        sql = ("SELECT c.id, c.vec FROM chunk c"
               " JOIN file f ON f.id = c.file_id"
               " WHERE c.vec IS NOT NULL" + self._mode_clause(mode)
               + self._lang_clause(lang)
               + self._path_clause(include)
               + self._exclude_clause(exclude) +
               " ORDER BY c.id")
        ids: list[int] = []
        blobs: list[bytes] = []
        for cid, vec in self.conn.execute(sql):
            ids.append(cid)
            blobs.append(vec)
        loaded = (ids, b"".join(blobs))
        self._vector_cache[key] = loaded
        return loaded

    def hits(self, ids: Sequence[int]) -> dict[int, tuple]:
        """Everything a result needs, in one statement.

        chunk id -> (path, start_line, end_line, name, text, signature).
        `name` is '' for a window chunk, and `signature` is NULL unless
        the chunk holds a definition whose grammar gave a header.

        One query rather than three per hit: the caller has the whole
        ranked list before it builds any of them. The body comes from
        here too -- the span, the line numbers and the text were all
        recorded together, and reading the body back off disk pairs
        indexed line numbers with current content.
        """
        if not ids:
            return {}
        marks = ",".join("?" for _ in ids)
        return {
            cid: (path, a, b, name or "", text, sig)
            for cid, path, a, b, name, text, sig in self.conn.execute(
                f"SELECT c.id, f.path, c.start_line, c.end_line, s.name,"
                f"       c.text, s.signature"
                f"  FROM chunk c"
                f"  JOIN file f ON f.id = c.file_id"
                f"  LEFT JOIN symbol s ON s.id = c.symbol_id"
                f" WHERE c.id IN ({marks})", tuple(ids)
            )
        }

    def chunk_rows(self, ids: Sequence[int]) -> dict[int, tuple[str, str]]:
        if not ids:
            return {}
        marks = ",".join("?" for _ in ids)
        return {
            sid: (path, text)
            for sid, path, text in self.conn.execute(
                f"SELECT c.id, f.path, c.text FROM chunk c"
                f" JOIN file f ON f.id = c.file_id"
                f" WHERE c.id IN ({marks})", tuple(ids)
            )
        }

    def chunk_paths(self, ids: Sequence[int]) -> dict[int, str]:
        if not ids:
            return {}
        marks = ",".join("?" for _ in ids)
        return {
            sid: path
            for sid, path in self.conn.execute(
                f"SELECT c.id, f.path FROM chunk c JOIN file f ON f.id=c.file_id"
                f" WHERE c.id IN ({marks})", tuple(ids)
            )
        }

    def non_candidate_rows(
        self, names: set[str], *, mode: SearchMode, limit: int = 200
    ) -> list[tuple[int, str, str]]:
        """Chunks whose file path contains one of these names.

        A substring test on the path, not a symbol lookup: it reaches
        chunks that retrieval never scored. Names under three
        characters are dropped, since as a substring they match most
        paths.
        """
        stems = {n.lower() for n in names if len(n) >= 3}
        if not stems:
            return []
        where = " OR ".join("lower(f.path) LIKE ?" for _ in stems)
        args = tuple(f"%{s}%" for s in stems)
        sql = (f"SELECT c.id, f.path, c.text FROM chunk c"
               f" JOIN file f ON f.id = c.file_id"
               f" WHERE ({where})" + self._mode_clause(mode)
               + f" LIMIT {int(limit)}")
        return list(self.conn.execute(sql, args))

    def _invalidate_vectors(self) -> None:
        """Drop the cached matrices. Cheap; they reload on next read."""
        self._vector_cache.clear()

    def chunk_span(self, chunk_id: int) -> tuple[str, int, int] | None:
        """The CHUNK's span, not the symbol's.

        A definition chunk is capped at max_chunk_lines, so reading back
        the symbol's span would return source that was never indexed. A
        window chunk has no symbol to read back from at all.
        """
        row = self.conn.execute(
            "SELECT f.path, c.start_line, c.end_line FROM chunk c"
            " JOIN file f ON f.id = c.file_id WHERE c.id = ?",
            (chunk_id,),
        ).fetchone()
        return tuple(row) if row else None

    def chunked_definitions(
        self, name: str, *, mode: SearchMode, lang=None, include=None,
        exclude=None
    ) -> list[int]:
        """CHUNK ids for definitions that have a chunk and pass the filter.

        The exact tier must obey both: a symbol below MIN_CHUNK_CHARS is
        navigable but not retrievable, and mode filtering applies to every
        tier, not only the ones that go through SQL.

        Returns chunk ids, not symbol ids -- every tier speaks in chunk
        ids so their outputs can be fused.
        """
        # Two ways a chunk can hold a definition. Usually the symbol
        # link is explicit. A definition too small to be chunked has no
        # chunk of its own, so under hybrid coverage the match is the
        # window whose span contains its first line; without that
        # branch the exact tier simply misses it.
        sql = ("SELECT DISTINCT c.id FROM symbol s"
               " JOIN file f ON f.id = s.file_id"
               " JOIN chunk c ON c.file_id = f.id AND ("
               "   c.symbol_id = s.id"
               "   OR (c.symbol_id IS NULL"
               "       AND s.start_line BETWEEN c.start_line AND c.end_line))"
               " WHERE s.name=? AND s.tag='def'" + self._mode_clause(mode)
               + self._lang_clause(lang)
               + self._path_clause(include)
               + self._exclude_clause(exclude))
        args: list = [name]
        sql += " ORDER BY f.path, s.start_line"
        return [r[0] for r in self.conn.execute(sql, args)]

    def status_counts(self) -> tuple[int, int, int]:
        row = self.conn.execute(
            "SELECT (SELECT count(*) FROM file), (SELECT count(*) FROM chunk),"
            " (SELECT count(*) FROM symbol)"
        ).fetchone()
        return tuple(row) if row else (0, 0, 0)

    def top_languages(self, *, limit: int = 8) -> list[tuple[str, int]]:
        return list(self.conn.execute(
            "SELECT lang, count(*) FROM file GROUP BY lang ORDER BY 2 DESC"
            f" LIMIT {int(limit)}"
        ))


def _stored_rev(conn: sqlite3.Connection) -> str:
    try:
        row = conn.execute("SELECT schema_rev FROM meta WHERE id=1").fetchone()
    except sqlite3.OperationalError:
        return -1
    return row[0] if row else -1


def _drop_all(conn: sqlite3.Connection) -> None:
    """Drop every object this schema owns, in dependency order."""
    conn.execute("PRAGMA foreign_keys = OFF")
    # Views first: one of them reads the tables below, and dropping a
    # table out from under it is what makes the order matter.
    for name, kind in conn.execute(
        "SELECT name, type FROM sqlite_master"
        " WHERE type IN ('view','table','index') AND name NOT LIKE 'sqlite_%'"
        " ORDER BY CASE type WHEN 'view' THEN 0 WHEN 'index' THEN 1 ELSE 2 END"
    ).fetchall():
        try:
            conn.execute(f"DROP {kind.upper()} IF EXISTS {name}")
        except sqlite3.OperationalError:
            pass
    conn.commit()
    conn.execute("PRAGMA foreign_keys = ON")


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


_CONTENT_TYPES = ("code", "tests", "docs", "config", "data")


#: The one category `"all"` never covers. Everything else is reachable
#: without naming it; `data` is bulk that does not answer questions
#: about code, so a caller who omits the filter -- or does not know
#: there is one -- cannot be handed spreadsheet rows.
_NEVER_IMPLICIT = ("data",)


def normalise_content(content, settings: Settings | None = None) -> tuple[str, ...]:
    """A category, several, or nothing -> the categories to return.

    Always an explicit tuple; there is no "no filter" state. `"all"`
    means every category except `_NEVER_IMPLICIT`, and omitting the
    filter means every category except `content_excluded` too.
    """
    excluded = settings.content_excluded if settings else ("config", "data")
    default = tuple(c for c in _CONTENT_TYPES if c not in excluded)
    everything = tuple(c for c in _CONTENT_TYPES if c not in _NEVER_IMPLICIT)

    if content is None:
        return default
    if isinstance(content, str):
        content = (content,)
    content = tuple(content)
    if not content:
        return default
    named = tuple(c for c in content if c != "all")
    unknown = sorted(set(named) - set(_CONTENT_TYPES))
    if unknown:
        raise ValueError(
            f"unknown content type(s): {', '.join(unknown)};"
            f" expected any of {', '.join(_CONTENT_TYPES)}"
        )
    if len(named) != len(content):          # "all" was present
        return tuple(sorted(set(everything) | set(named)))
    return tuple(sorted(set(named)))
