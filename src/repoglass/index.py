"""The facade. The only public surface.

Every argument that shadows a Settings field defaults to None, meaning
"take the resolved setting". A literal default would make an omitted
argument indistinguishable from an explicit one.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from .config import Paths, Settings, categories_rev
from .config import load as load_settings
from .corpus import discovery, extract, languages
from .embeddings import build as build_embedder
from .embeddings import pack
from .models import Hit, LanguageFilter, RefreshReport, SearchMode, Symbol
from .search import fuse, lexical, rank, vector
from .store import Store, normalise_content


class Index:
    def __init__(self, store: Store, settings: Settings, paths: Paths) -> None:
        self._store = store
        self._settings = settings
        self._paths = paths
        self._embedder = None
        self._embedder_built = False

    @classmethod
    def open(
        cls,
        root: Path,
        settings: Settings | None = None,
        *,
        paths: Paths | None = None,
    ) -> "Index":
        """Open or create the index for a directory.

        The root need not be a git repository.
        """
        paths = paths or Paths.for_root(Path(root))
        settings = settings or load_settings(
            user_config=paths.user_config, repo_config=paths.repo_config
        )
        # `data_dir` is a setting, but the config files that carry it
        # are found through `paths`. Resolve paths first, then rebuild
        # it once the setting is known.
        if settings.data_dir != paths.data_dir:
            paths = replace(paths, data_dir=settings.data_dir)
        store = Store.open(
            paths.db, settings, extractor_rev=languages.extractor_rev()
        )
        return cls(store, settings, paths)

    def _embed(self):
        if not self._embedder_built:
            self._embedder = build_embedder(self._settings, self._paths)
            self._embedder_built = True
        return self._embedder

    def refresh(self, *, force: bool = False) -> RefreshReport:
        """Bring the index up to date.

        Compared by inequality on (mtime_ns, size), not ordering: mtime
        moves backwards on branch checkout, cp -p and archive extraction.
        """
        started = time.perf_counter()
        self._reindex_if_identity_changed()

        walked = {f.path: f for f in discovery.walk(self._paths, self._settings)}
        indexed = self._store.known_files()

        added = {p for p in walked if p not in indexed}
        deleted = [p for p in indexed if p not in walked]
        changed = {
            p for p in walked.keys() & indexed.keys()
            if (walked[p].mtime_ns, walked[p].size) != indexed[p]
        }
        if force:
            changed |= walked.keys() - added

        if deleted:
            self._store.delete_files(deleted)
        touched = sorted(added | changed)
        if touched or deleted:
            for path in touched:
                file = walked[path]
                try:
                    source = (self._paths.root / path).read_text(errors="ignore")
                except OSError:
                    # No row, so the next walk sees it as new and tries
                    # again. A row would claim it had been indexed.
                    continue
                # One transaction per file. The row says the file is
                # indexed; the chunks are what make that true, and a
                # failure between them is invisible afterwards --
                # the source has not changed, so no later walk
                # reports it as added or edited.
                with self._store.transaction():
                    self._store.upsert_files([file])
                    self._index_file(file, source)
            self._embed_pending()
        # Outside the `touched` branch: a run that wrote chunks and
        # died before the rebuild leaves every row correct, so no
        # later walk reports a change, and only the flag remembers
        # that the lexical index is behind.
        if self._store.fts_dirty():
            self._store.rebuild_fts()
        self._store.mark_scanned()

        return RefreshReport(
            added=len(added),
            changed=len(changed),
            deleted=len(deleted),
            elapsed_s=time.perf_counter() - started,
        )

    def _current_identity(self):
        from .store import Identity

        stored = self._store.identity()
        # `dims` is the only thing wanted from the embedder, and
        # constructing one imports the whole model stack. The stored
        # width already answers it whenever the embedder that wrote it
        # is the one configured now -- or when there is no embedder at
        # all, where there is nothing to construct and nothing to learn.
        no_embedder = self._settings.embed_backend == "none"
        settled = (stored is not None
                   and stored.embed_model == self._settings.embed_model
                   and stored.embed_backend == self._settings.embed_backend
                   and stored.embed_dims > 0)
        if stored is not None and (no_embedder or settled):
            dims = stored.embed_dims
        else:
            embedder = self._embed()
            dims = (embedder.dims if embedder
                    else (stored.embed_dims if stored else 0))
        return Identity(
            schema_rev=stored.schema_rev if stored else "",
            embed_model=self._settings.embed_model,
            embed_backend=self._settings.embed_backend,
            embed_dims=dims,
            coverage=self._settings.coverage,
            extractor_rev=languages.extractor_rev(),
            categories_rev=categories_rev(self._settings),
        )

    def _reindex_if_identity_changed(self) -> None:
        """Drop every indexed row when anything that shaped it changed.

        Vectors from a different model are not comparable, and a changed
        .scm shifts spans. Reusing rows across either produces mixed-width
        blobs that crash the reshape in vector.search.
        """
        current = self._current_identity()
        if self._store.needs_reindex(current):
            self._store.reset_content()
            self._store.set_identity(current)

    def _index_file(self, file, source: str) -> None:
        result = extract.extract(file, source, self._settings)
        self._store.replace_symbols(file.path, result.symbols)
        self._store.upsert_chunks(file.path, result.chunks)

    def _embed_pending(self) -> None:
        embedder = self._embed()
        if embedder is None:
            return
        rows = self._store.pending_vectors()
        if not rows:
            return
        vectors = embedder.encode([t for _, t in rows])
        self._store.set_vectors(
            [(sid, pack(v)) for (sid, _), v in zip(rows, vectors)]
        )
        self._store.set_embed_dims(embedder.dims)

    def _maybe_refresh(self) -> None:
        """Best-effort refresh before a read, under refresh_mode='auto'.

        On lock contention, records the skip and serves existing data.

        Exception: an index that has never completed a build blocks on the
        writer instead of returning [], since an empty result would be
        indistinguishable from "no matches".
        """
        if self._settings.refresh_mode != "auto":
            return
        last = self._store.last_scan_at() or 0.0
        never_built = last == 0.0
        if not never_built and self._settings.rescan_after_seconds > 0:
            if time.time() - last < self._settings.rescan_after_seconds:
                return
        try:
            self.refresh()
        except sqlite3.OperationalError:
            if never_built:
                raise
            self._store.mark_skipped()

    def definitions(self, name: str, *, lang: LanguageFilter = None) -> list[Symbol]:
        self._maybe_refresh()
        return self._store.definitions(name, lang=lang)

    def references(self, name: str, *, lang: LanguageFilter = None) -> list[Symbol]:
        """Every reference, unbounded, ordered by path then line.

        No limit: these are exact matches with no ranking, so a cap
        truncates alphabetically by path and silently hides real
        answers -- it would return `benchmarks/...` and never reach
        `src/...`. Narrow with `lang`, which is a meaningful subset,
        not an arbitrary prefix.

        Empty for a language without reference captures -- a capability
        gap, not an absence of matches. See languages.capability().
        """
        self._maybe_refresh()
        return self._store.references(name, lang=lang)

    #: Dimensions `symbol_counts` can group on. Re-exported from the
    #: store so the CLI, which may not import `store`, still offers
    #: exactly the set the query supports.
    COUNT_BY = Store.COUNT_BY

    def symbols(self, pattern: str | None = None, *, tag: str | None = None,
                lang: LanguageFilter = None, content: SearchMode = None,
                include=None, exclude=None,
                limit: int | None = None) -> list[Symbol]:
        """Every symbol matching a name glob and the usual filters.

        The complement of `search`, which ranks and truncates and so
        cannot answer "how many" or "is there one at all". An empty
        result here means the index holds none, which a ranked list
        cannot say.
        """
        self._maybe_refresh()
        return self._store.symbol_rows(pattern, tag=tag, lang=lang,
                                       content=content, include=include,
                                       exclude=exclude, limit=limit)

    def symbol_counts(self, by: str, pattern: str | None = None, *,
                      tag: str | None = None, lang: LanguageFilter = None,
                      content: SearchMode = None, include=None, exclude=None,
                      limit: int | None = None
                      ) -> tuple[list[tuple[str, int]], int, int]:
        """The same set, grouped and counted, largest group first.

        With the group and symbol totals over everything matched, so a
        `limit` is visibly a truncation rather than a smaller total.
        """
        self._maybe_refresh()
        return self._store.symbol_counts(by, pattern, tag=tag, lang=lang,
                                         content=content, include=include,
                                         exclude=exclude, limit=limit)

    def _require_indexed(self, content: SearchMode) -> None:
        """Refuse a category this index does not hold.

        Returning an empty list would be indistinguishable from "no
        matches", so a caller could not tell a missing category from a
        missing answer. `index_excluded` is part of `categories_rev`, so
        the index always matches the setting consulted here.
        """
        from .store import normalise_content

        if content is None:
            return
        # Validate spelling first. A typo is a ValueError about the
        # name, not a LookupError advising a reindex to add "cod".
        normalise_content(content)
        named = {content} if isinstance(content, str) else set(content)
        # Only what the caller spelled out can be missing. Omitting the
        # filter, or asking for "all", means "whatever this index holds"
        # and narrows silently rather than failing.
        named.discard("all")
        if not named:
            return
        missing = sorted(named & set(self._settings.index_excluded))
        if missing:
            names = ", ".join(missing)
            raise LookupError(
                f"content={names} is not in this index: index_excluded ="
                f" {', '.join(self._settings.index_excluded)}."
                f" To include it, drop {names} from index_excluded in"
                f" repoglass.toml and reindex with refresh(force=True)."
            )

    def _tiers(self, query: str, *, k: int, mode: SearchMode,
               lang: LanguageFilter = None, include=None,
               exclude=None) -> list[fuse.RankedList]:
        lists: list[fuse.RankedList] = []
        # Chunk-backed and mode-filtered: a symbol below MIN_CHUNK_CHARS is
        # navigable but not retrievable, and mode applies to every tier.
        hits = [
            (sid, 1.0)
            for sid in self._store.chunked_definitions(query, mode=mode,
                                                       lang=lang, include=include,
                                                       exclude=exclude)
        ]
        if hits:
            lists.append(fuse.RankedList("exact", tuple(hits)))
        # The lexical tier runs for prose queries too: BM25 wins enough
        # of them outright that skipping it costs more than the
        # dilution it causes.
        kw = lexical.keyword(self._store, query, lang=lang, include=include,
                             exclude=exclude,
                             limit=self._settings.candidate_depth, mode=mode)
        if kw:
            lists.append(fuse.RankedList("lexical", tuple(kw)))
        embedder = self._embed()
        if embedder is not None:
            # encode_query, not encode: bge and e5 want an instruction on
            # the query side and nothing on the document side. Using the
            # document path here costs recall and fails silently.
            qv = embedder.encode_query([query])[0]
            vec = vector.search(self._store, qv, lang=lang, include=include,
                                exclude=exclude,
                                limit=self._settings.candidate_depth, mode=mode)
            if vec:
                lists.append(fuse.RankedList("vector", tuple(vec)))
        return lists

    @staticmethod
    def _evidence(lists) -> dict[int, tuple[tuple[str, float], ...]]:
        """chunk id -> the raw score each tier gave it."""
        by_id: dict[int, list[tuple[str, float]]] = {}
        for ranked in lists:
            for sid, raw in ranked.items:
                by_id.setdefault(sid, []).append((ranked.tier, raw))
        return {k: tuple(v) for k, v in by_id.items()}

    @staticmethod
    def _opening_line(text: str) -> str | None:
        """Stand-in header for a chunk that holds no definition.

        A window is a gap-filler over lines no definition owns, so it
        has no header to cut. Without a stand-in, `--code signature`
        gives a path and a line range and no indication of what is
        there, which is not enough to decide whether to open the file.
        """
        for line in text.splitlines():
            if stripped := line.strip():
                return stripped[:extract.MAX_SIGNATURE_CHARS]
        return None

    def _to_hits(self, scored: Sequence[tuple[int, float]], limit: int,
                 evidence: dict[int, tuple[tuple[str, float], ...]] | None = None
                 ) -> list[Hit]:
        evidence = evidence or {}
        rows = self._store.hits([sid for sid, _ in scored[:limit]])
        out: list[Hit] = []
        for sid, score in scored[:limit]:
            row = rows.get(sid)
            if row is None:
                continue
            path, start, end, name, text, signature = row
            out.append(
                Hit(
                    path=path,
                    start_line=start,
                    end_line=end,
                    name=name,
                    score=score,
                    code=text,
                    signature=signature or self._opening_line(text),
                    tiers=evidence.get(sid, ()),
                )
            )
        return out

    def search(
        self, query: str, *, k: int = 10, content: SearchMode | None = None,
        mode: SearchMode | None = None, lang: LanguageFilter = None,
        include=None, exclude=None,
    ) -> list[Hit]:
        """Exact, then FTS5/BM25, then semantic; merged per Settings.ranker.

        `content` selects which files may be returned: code, tests, docs,
        config, data, or all. Omitting it returns code, tests and docs;
        `"all"` adds config. `data` is returned only when named, and
        only if `index_excluded` did not drop it. `mode` is a deprecated
        alias for `content`.
        """
        self._maybe_refresh()
        content = content if content is not None else (mode if mode is not None
                                                       else self._settings.content)
        self._require_indexed(content)
        lists = self._tiers(query, k=k, mode=content, lang=lang,
                            include=include, exclude=exclude)
        if not lists:
            return []
        is_prose = lexical.looks_like_prose(
            query, min_words=self._settings.prose_min_words
        )
        evidence = self._evidence(lists)
        scored = fuse.merge(lists, self._settings, is_prose=is_prose)
        if not self._settings.rerank:
            return self._to_hits(scored, k, evidence)

        # The reranker works on chunk text, not just ids: the
        # definition boost has to read the source to see whether a
        # chunk actually defines the queried name.
        cands = self._candidates(scored)
        ranked = rank.rerank(
            cands, query, self._settings,
            load_non_candidates=self._non_candidate_loader(content),
            # Path priors are about code layout; asking for tests or docs
            # and then penalising them would be perverse.
            penalise_paths=normalise_content(content) in ((), ("code",)),
            limit=k,
        )
        # Rescale to [0,1]. The layer multiplies by boosts -- a definition
        # match is 3x -- so its scores leave the range `Hit.score`
        # promises. Dividing by the top score keeps the order and the
        # documented contract.
        top = max((c.score for c in ranked), default=0.0) or 1.0
        return self._to_hits([(c.symbol_id, c.score / top) for c in ranked],
                             k, evidence)

    def _candidates(self, scored: Sequence[tuple[int, float]]) -> list[rank.Candidate]:
        """Attach path and indexed text to each scored symbol id."""
        ids = [sid for sid, _ in scored]
        if not ids:
            return []
        rows = self._store.chunk_rows(ids)
        return [
            rank.Candidate(symbol_id=sid, path=rows[sid][0],
                           text=rows[sid][1], score=score)
            for sid, score in scored if sid in rows
        ]

    def _non_candidate_loader(self, content):
        """Chunks retrieval missed, whose file stem matches the query name.

        A full table scan per query is too expensive, so the stem
        filter is pushed into SQL and only matching rows are read.
        """
        def load(names: set[str]) -> list[rank.Candidate]:
            return [
                rank.Candidate(symbol_id=sid, path=path, text=text, score=0.0)
                for sid, path, text in self._store.non_candidate_rows(
                    names, mode=content
                )
            ]
        return load

    def _relative(self, path: Path | str) -> str:
        """Repo-relative, / separated.

        Absolute paths are natural for a public API, so accept both.
        """
        p = Path(path)
        if p.is_absolute():
            try:
                p = p.resolve().relative_to(self._paths.root.resolve())
            except ValueError:
                return str(path).replace("\\", "/")
        return str(p).replace("\\", "/")
