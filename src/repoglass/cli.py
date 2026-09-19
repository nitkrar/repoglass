"""Command line entry point.

JSON on stdout by default, because the caller is usually a program.
`--text` is for reading with your eyes. Anything that is not the
answer -- progress, warnings, errors -- goes to stderr, so stdout stays
pipeable in both formats.

Exit codes:
    0  it worked, including "no results"
    1  something failed
    2  the request was not answerable: unknown category, or a category
       this index does not hold

Every retrieval setting comes from `load()`. There are no per-knob
flags: a flag default silently overrides the setting it shadows, so
what ran stops matching what was asked for, and a CLI is where that is
hardest to notice.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

from .config import Paths, as_toml, load
from .index import Index
from .models import Hit


def _version() -> str:
    from . import __version__

    return __version__


def _open(args) -> Index:
    """Open the index, announcing a first build on stderr.

    A first `search` on a large repository indexes it, which can take
    minutes. Silence there reads as a hang, and the notice cannot go to
    stdout without corrupting the JSON.
    """
    root = Path(args.repo).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(_fail(f"not a directory: {root}"))
    try:
        settings = load(repo_config=args.config) if args.config else None
    except ValueError as exc:
        # A config can outlive the settings it names -- `init` writes
        # resolved values, so a key removed later leaves a file this
        # tool refuses. That is a message, not a traceback.
        raise SystemExit(_fail(f"{exc}\n"
                               "  regenerate with: repoglass init --force", 2))
    paths = Paths.for_root(root)
    if not paths.db.exists():
        print(f"repoglass: building the index for {root} "
              f"(first run; this is not repeated)", file=sys.stderr)
    return Index.open(root, settings, paths=paths)


def _fail(message: str, code: int = 1) -> int:
    print(f"repoglass: {message}", file=sys.stderr)
    return code


def _emit(payload: dict, args) -> None:
    if not args.text:
        json.dump(payload, sys.stdout, indent=None)
        sys.stdout.write("\n")
    else:
        _emit_text(payload)


def _emit_text(payload: dict) -> None:
    """The same payload as JSON, rendered for eyes.

    Both formats render the one payload dict, so the two cannot drift
    into disagreeing about what a result contains.
    """
    for hit in payload.get("results", []):
        head = f"{hit['path']}:{hit['start_line']}-{hit['end_line']}"
        name = f"  {hit['name']}" if hit.get("name") else ""
        tiers = "  ".join(f"{k}={v}" for k, v in hit.get("tiers", {}).items())
        print(f"{head}{name}")
        print(f"    score={hit['score']}" + (f"  [{tiers}]" if tiers else ""))
        if hit.get("signature"):
            print(f"    {hit['signature']}")
        if hit.get("code"):
            print(hit["code"].rstrip())
        print()
    for sym in payload.get("symbols", []):
        where = f"  in {sym['enclosing']}" if sym.get("enclosing") else ""
        print(f"{sym['path']}:{sym['start_line']}  {sym['name']}"
              f"  ({sym['tag']}, {sym['content_type']}){where}")
        if sym.get("signature"):
            print(f"    {sym['signature']}")
    if "counts" in payload:
        width = max((len(g) for g in payload["counts"]), default=0)
        for group, n in payload["counts"].items():
            print(f"{group:<{width}}  {n}")
        shown = (f"showing {payload['shown']} of " if
                 payload.get("shown", payload["groups"]) < payload["groups"]
                 else "")
        print(f"# {shown}{payload['groups']} group(s),"
              f" {payload['total']} symbol(s) by {payload['count_by']}")
    if "count" in payload:
        split = payload.get("by_content_type") or {}
        detail = (": " + ", ".join(f"{k} {v}" for k, v in split.items())
                  if len(split) > 1 else "")
        print(f"# {payload['count']} result(s){detail}")
    if "status" in payload:
        for key, value in payload["status"].items():
            print(f"{key:<18}{value}")
    if "cleared" in payload:
        for line in payload["cleared"]:
            print(line)


#: Tier -> the metric it reports. `score` is relative to the top hit
#: and so says nothing absolute; these do. What each one means and
#: which way it runs is in `search --help`, not in every response: it
#: is the same four lines every time and the caller is usually a
#: program that already knows.
_METRIC = {"lexical": "bm25", "vector": "cosine", "exact": "exact"}


def _hit(h: Hit, *, code: str) -> dict:
    out = {"path": h.path, "start_line": h.start_line,
           "end_line": h.end_line, "name": h.name, "score": round(h.score, 4),
           # Raw per-tier scores, keyed by the metric rather than the
           # tier, because -7.86 and 0.327 are uninterpretable without
           # knowing which is which and which way each runs.
           "tiers": {_METRIC.get(name, name): round(v, 4)
                     for name, v in h.tiers}}
    if code == "full":
        out["code"] = h.code
    elif code == "signature" and h.signature is not None:
        # Still absent rather than empty on a chunk with no text to
        # take a line from: an empty string would read as a header
        # that happens to be blank.
        out["signature"] = h.signature
    return out


def cmd_search(args) -> int:
    index = _open(args)
    try:
        query = " ".join(args.query)
        hits = index.search(query, k=args.k, content=args.content,
                            lang=args.lang, include=args.include,
                            exclude=args.exclude)
    except LookupError as exc:
        return _fail(str(exc), 2)
    except ValueError as exc:
        return _fail(str(exc), 2)
    if (bad := _check_lang(index, args.lang)) is not None:
        return bad
    results = [_hit(h, code=args.code) for h in hits]
    _emit({"query": query, "count": len(results), "results": results}, args)
    return 0


def _check_lang(index, lang: str | None) -> int | None:
    """Refuse a language this index does not hold.

    Called only after a query came back empty, for two reasons. A typo
    otherwise returns nothing and exits 0, reading as "no definitions
    in that language" -- the confusion `_require_indexed` exists to
    prevent for categories. And checking beforehand reads the file
    table before the first refresh has populated it, so on a new index
    every language looks absent.

    The valid set is what this index holds, not every name grep_ast
    knows, because that is the set that can return anything.
    """
    if lang is None:
        return None
    have = {name for name, _ in index._store.top_languages(limit=1000)}
    # Nothing indexed at all: the caller's language is not wrong, the
    # index is merely empty.
    if not have:
        return None
    missing = [x for x in ((lang,) if isinstance(lang, str) else lang)
               if x not in have]
    if not missing:
        return None
    lang = missing[0]
    # A suggestion beats a list: the caller mistyped one word and
    # should not have to scan six to find it. difflib is stdlib.
    import difflib

    near = difflib.get_close_matches(lang, sorted(have), n=1, cutoff=0.6)
    hint = (f"did you mean {near[0]!r}?" if near
            else f"this index has: {', '.join(sorted(have))}")
    # Phrased as a bad argument, not a bad index. "no 'pyton' files in
    # this index" reads as though something needs rebuilding; nothing
    # does.
    return _fail(f"--lang {lang!r} matches no indexed language, {hint}", 2)


def cmd_defs(args) -> int:
    index = _open(args)
    syms = index.definitions(args.name, lang=args.lang)
    # After the query, so the refresh has run -- but not only when the
    # result is empty: `-l python pyton` matches on python and would
    # otherwise drop the typo silently.
    if (bad := _check_lang(index, args.lang)) is not None:
        return bad
    rows = [_symbol(s) for s in syms]
    _emit({"name": args.name, "count": len(rows),
           "by_content_type": _by_content(rows, lambda r: r["content_type"]),
           "symbols": rows}, args)
    return 0


def cmd_refs(args) -> int:
    index = _open(args)
    syms = index.references(args.name, lang=args.lang)
    if (bad := _check_lang(index, args.lang)) is not None:
        return bad
    rows = [_symbol(s) for s in syms]
    _emit({"name": args.name, "count": len(rows),
           "by_content_type": _by_content(rows, lambda r: r["content_type"]),
           "symbols": rows}, args)
    return 0


def _by_content(items, key) -> dict[str, int]:
    """How many results fell in each category, most first.

    A name can have far more references in tests than in production
    code. The per-row category makes that knowable; this makes it
    visible without the caller tallying it.
    """
    import collections

    counted = collections.Counter(key(i) for i in items)
    return dict(counted.most_common())


def _symbol(s) -> dict:
    row = {"name": s.name, "tag": s.tag, "path": s.path,
           "start_line": s.start_line, "end_line": s.end_line,
           "lang": s.lang, "content_type": s.content_type}
    # Each is absent rather than null when it does not apply: a
    # definition has no enclosing definition, a reference has no
    # signature, and module scope has neither.
    if s.enclosing is not None:
        row["enclosing"] = s.enclosing
    if s.signature is not None:
        row["signature"] = s.signature
    return row


def cmd_symbols(args) -> int:
    index = _open(args)
    if args.count_by:
        counts, groups, total = index.symbol_counts(
            args.count_by, args.pattern, tag=args.tag, lang=args.lang,
            content=args.content, include=args.include, exclude=args.exclude,
            limit=args.limit)
        if not counts and (bad := _check_lang(index, args.lang)) is not None:
            return bad
        _emit({"count_by": args.count_by, "groups": groups, "total": total,
               "shown": len(counts), "counts": dict(counts)}, args)
        return 0
    syms = index.symbols(args.pattern, tag=args.tag, lang=args.lang,
                         content=args.content, include=args.include,
                         exclude=args.exclude, limit=args.limit)
    if not syms and (bad := _check_lang(index, args.lang)) is not None:
        return bad
    rows = [_symbol(s) for s in syms]
    _emit({"pattern": args.pattern, "count": len(rows),
           "by_content_type": _by_content(rows, lambda r: r["content_type"]),
           "symbols": rows}, args)
    return 0


def cmd_index(args) -> int:
    index = _open(args)
    report = index.refresh(force=args.force)
    _emit({"status": {"added": report.added, "changed": report.changed,
                      "deleted": report.deleted,
                      "seconds": round(report.elapsed_s, 2)}}, args)
    return 0


def cmd_status(args) -> int:
    index = _open(args)
    store = index._store
    files, chunks, symbols = store.status_counts()
    langs = store.top_languages(limit=8)
    db = index._paths.db
    size = sum(f.stat().st_size for f in db.parent.rglob("*") if f.is_file())
    _emit({"status": {
        "root": str(index._paths.root),
        "index": str(db.parent),
        "files": files, "chunks": chunks, "symbols": symbols,
        "size_mb": round(size / 1e6, 1),
        "languages": ", ".join(f"{name} {n}" for name, n in langs),
    }}, args)
    return 0


def cmd_clear(args) -> int:
    """Remove this directory's index. `--all` removes every index.

    Defaults to the one index the caller is standing in, named by
    `--repo`: a destructive command should do the smallest thing its
    name allows.
    """
    paths = Paths.for_root(Path(args.repo).expanduser().resolve())
    if args.config:
        paths = replace(paths, data_dir=load(repo_config=args.config).data_dir)
    if args.all:
        root_dir = paths.home / "index"
        targets = sorted(e for e in root_dir.iterdir() if e.is_dir()) \
            if root_dir.is_dir() else []
    else:
        targets = [paths.data]
    if not targets:
        _emit({"cleared": ["nothing to clear"]}, args)
        return 0
    removed: list[str] = []
    failed = False
    for entry in targets:
        if not entry.is_dir():
            continue
        size = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
        if args.dry_run:
            removed.append(f"would remove {entry.name}  {size / 1e6:.1f} MB")
            continue
        # Reported after the fact, not before. `ignore_errors=True` plus
        # an optimistic message prints "removed" for a directory that is
        # still there -- rmtree refuses a symlink, for one.
        shutil.rmtree(entry, ignore_errors=True)
        if entry.exists():
            failed = True
            removed.append(f"could NOT remove {entry}")
        else:
            removed.append(f"removed {entry.name}  {size / 1e6:.1f} MB")
    _emit({"cleared": removed or ["nothing matched"]}, args)
    return 1 if failed else 0


def _resolved(args):
    """The settings this invocation would actually use."""
    return load(repo_config=args.config) if args.config else load(
        repo_config=Path(args.repo).expanduser().resolve() / "repoglass.toml")


def cmd_init(args) -> int:
    """Write a config file holding the resolved settings.

    Resolved rather than commented-out, so the file shows what is
    actually in force. The cost is that it then pins every setting: a
    later change to a default will not reach a repository that has one
    of these. The header says so, because nothing else will.
    """
    target = Path(args.repo).expanduser().resolve() / "repoglass.toml"
    if target.exists() and not args.force:
        return _fail(f"{target} exists; pass --force to overwrite")
    header = (
        "# Written by `repoglass init`. Values are the ones resolved at\n"
        "# the time of writing, so this file PINS them: a later change to\n"
        "# a repoglass default will not reach this repository while the\n"
        "# key is present. Delete any key you would rather have track the\n"
        "# default, and regenerate with `repoglass init --force`.\n\n"
    )
    target.write_text(header + as_toml(_resolved(args), active=True))
    _emit({"status": {"wrote": str(target)}}, args)
    return 0


def cmd_config(args) -> int:
    """Print the resolved settings. Writes nothing.

    TOML rather than JSON, unlike every other command here: the whole
    point is to show configuration in the form configuration takes, so
    the output can be diffed against a real file or pasted into one.
    """
    sys.stdout.write(as_toml(_resolved(args), active=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="repoglass",
        description="Local code and prose index: symbols, lexical and "
                    "semantic retrieval.",
        epilog=(
            "every subcommand also takes:\n"
            "  -r, --repo PATH  the directory to index and search"
            " (default: .). Result\n"
            "                   paths are relative to it, and its index is"
            " keyed by the\n"
            "                   resolved path, so two checkouts of one"
            " project do not\n"
            "                   share an index. Use it to query another"
            " repository\n"
            "                   without changing directory.\n"
            "  --config PATH    a TOML file overriding the resolved"
            " settings\n"
            "  --text           human-readable output instead of json"
            " (`config`\n"
            "                   always emits TOML)\n"
            "\nrun `repoglass <command> --help` for a command's own"
            " options."),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-V", "--version", action="version",
                        version=_version())
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        # `--repo`, though it need not be a git repository: with
        # `--include` taking path globs, a flag called `--path` meaning
        # something else entirely is the worse confusion.
        p.add_argument("-r", "--repo", default=".",
                       help="directory to index and search (default: .);"
                            " its index is keyed by the resolved path")
        p.add_argument("--config", type=Path,
                       help="TOML overriding the resolved settings")
        # A boolean, not --format json|text. Two formats do not need an
        # enum, and `--format json` is noise when json is the default.
        # If a third format ever lands, this becomes --format again.
        p.add_argument("--text", action="store_true",
                       help="human-readable output instead of json")

    s = sub.add_parser(
        "search", help="search the index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "output (JSON unless --text):\n"
            "  query      the query as searched\n"
            "  results[]  path, start_line, end_line, name, score, tiers,\n"
            "             and code unless --no-code\n"
            "    score    the fused rank divided by the top hit. Always\n"
            "             1.0 at rank 1, so it orders results but does\n"
            "             NOT measure how good any of them are.\n"
            "    tiers    raw score from each tier that matched, keyed by\n"
            "             metric. A result matched by two tiers is better\n"
            "             corroborated than one matched by a single tier.\n"
            "             bm25   SQLite FTS5 bm25(); negative, and more\n"
            "                    negative is better.\n"
            "             cosine cosine similarity in [-1, 1]; higher is\n"
            "                    better.\n"
            "             exact  1.0 when the query literally matched a\n"
            "                    symbol name.\n"
            "\nno results is success (exit 0). Exit 2 means the request\n"
            "could not be answered: an unknown category, or one this\n"
            "index does not hold."))
    # Several words, joined. Quoting works too, but `rpg search how are
    # chunks bounded` failing on the unquoted form is a bad first
    # experience and query is the only positional, so nothing is lost.
    s.add_argument("query", nargs="+")
    s.add_argument("-k", type=int, default=10, help="results (default: 10)")
    s.add_argument("--content", nargs="+",
                   help="code, tests, docs, config, data, or all")
    s.add_argument("-l", "--lang", nargs="+",
                   help="only results in these languages, e.g. python go")
    s.add_argument("--include", nargs="+", metavar="GLOB",
                   help="only paths matching these globs, e.g. 'src/*'."
                        " * crosses / , so src/* is recursive")
    # No short forms: -i is --ignore-case in grep and ripgrep, -e is
    # --regexp in grep. GNU grep ships --include/--exclude without
    # shorts for the same reason.
    s.add_argument("--exclude", nargs="+", metavar="GLOB",
                   help="skip paths matching these globs; wins over"
                        " --include")
    s.add_argument("--code", choices=("full", "signature", "none"),
                   default="full",
                   help="how much of each result to return: the whole"
                        " span, one line, or neither. that line is the"
                        " definition's header, or the span's first"
                        " non-blank line where it holds no definition")
    # The shorter way to say the common case, and one value of --code
    # rather than a second axis.
    s.add_argument("--no-code", action="store_const", const="none",
                   dest="code", help="same as --code none")
    common(s)
    s.set_defaults(func=cmd_search)

    d = sub.add_parser("defs", help="where a name is defined (all of them)")
    d.add_argument("name")
    # -l/--lang follows ast-grep, the closest tool in this space
    # (tree-sitter, structural). ripgrep spells the adjacent idea
    # -t/--type, but that is extension-based file typing, and -t is
    # already --text here.
    d.add_argument("-l", "--lang", nargs="+",
                   help="only definitions in these languages, e.g. python go")
    common(d)
    d.set_defaults(func=cmd_defs)

    r = sub.add_parser("refs", help="where a name is used (all of them)")
    r.add_argument("name")
    r.add_argument("-l", "--lang", nargs="+",
                   help="only references in these languages, e.g. python go")
    common(r)
    r.set_defaults(func=cmd_refs)

    y = sub.add_parser(
        "symbols", help="list or count symbols, complete and unranked",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "the complement of `search`, which ranks and truncates.\n"
            "use this to ask how many, which ones, or whether any.\n"
            "\nexamples:\n"
            "  rpg symbols 'test_*' --tag def\n"
            "  rpg symbols --count-by lang\n"
            "  rpg symbols --tag ref --count-by name --limit 10\n"
            "  rpg symbols --count-by file --content tests\n"
            "\nreads the symbol table, so a definition too small to be\n"
            "chunked is listed here though no search can return it.\n"
            "count 0 means the index holds none -- which an empty\n"
            "ranked list does not say."))
    y.add_argument("pattern", nargs="?",
                   help="glob on the symbol name, e.g. 'handle_*'."
                        " omit for all of them")
    y.add_argument("--tag", choices=("def", "ref"),
                   help="definitions or references; omit for both")
    y.add_argument("--count-by", dest="count_by",
                   choices=tuple(Index.COUNT_BY),
                   help="group and count instead of listing")
    y.add_argument("--limit", type=int,
                   help="cap the rows, or the groups under --count-by")
    y.add_argument("--content", nargs="+",
                   help="code, tests, docs, config, data, or all")
    y.add_argument("-l", "--lang", nargs="+",
                   help="only symbols in these languages, e.g. python go")
    y.add_argument("--include", nargs="+", metavar="GLOB",
                   help="only paths matching these globs")
    y.add_argument("--exclude", nargs="+", metavar="GLOB",
                   help="skip paths matching these globs; wins over --include")
    common(y)
    y.set_defaults(func=cmd_symbols)

    i = sub.add_parser("index", help="build or update the index")
    i.add_argument("--force", action="store_true",
                   help="re-extract every file, not just changed ones")
    common(i)
    i.set_defaults(func=cmd_index)

    st = sub.add_parser("status", help="what this index contains")
    common(st)
    st.set_defaults(func=cmd_status)

    n = sub.add_parser("init", help="write repoglass.toml for this repo")
    n.add_argument("--force", action="store_true",
                   help="overwrite an existing file")
    common(n)
    n.set_defaults(func=cmd_init)

    g = sub.add_parser("config", help="print the resolved settings")
    common(g)
    g.set_defaults(func=cmd_config)

    c = sub.add_parser("clear", help="remove index directories")
    c.add_argument("--all", action="store_true",
                   help="every index under the repoglass home, not just"
                        " this one")
    c.add_argument("--dry-run", action="store_true")
    common(c)
    c.set_defaults(func=cmd_clear)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except BrokenPipeError:
        return 0                      # `| head` closed the pipe
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
