"""Configuration schema and fixed values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ENV_PREFIX = "REPOGLASS_"

# Fixed values. Not in Settings because changing them invalidates
# indexes on disk or has no sensible per-repository answer.
RRF_K = 60
MIN_CHUNK_CHARS = 60          # shorter spans are navigable, not retrievable
#: Upper bound on one chunk's text. `max_chunk_lines` says nothing
#: about a minified file, where every definition on the one long line
#: takes that whole line as its text. Over this a definition is
#: navigable but not retrievable, as under MIN_CHUNK_CHARS.
MAX_CHUNK_CHARS = 20_000
#: Languages refused outright. "Has a grammar" is all that admits a
#: file, and `pem` has one. Not a judgement about usefulness: this is
#: material that must not enter a search index at all.
NEVER_INDEX_LANGS = frozenset({"pem"})
DATA_DIR_NAME = ".repoglass"
IGNORE_FILE_NAME = ".repoglassignore"
#: On-disk width of one vector component. Components of a
#: unit-normalised vector lie in [-1, 1], which float16 resolves finer
#: than the score gaps ranking turns on, at half the bytes. Changing it
#: makes every stored vector unreadable.
VECTOR_DTYPE = "float16"
#: Overrides `Paths.home`. Named here beside ENV_PREFIX so the two
#: environment names are declared together.
HOME_ENV = "REPOGLASS_HOME"

@dataclass(frozen=True)
class Settings:
    ranker: Literal["rrf", "none"] = "rrf"
    prose_min_words: int = 4
    #: Weight the dense tier against the lexical tier by query shape
    #: instead of letting unweighted RRF give both an equal say.
    adaptive_alpha: bool = True
    #: Dense share for an identifier-shaped query, and for a sentence.
    #: The lexical tier takes the remainder; `exact` is never weighted.
    alpha_symbol: float = 0.3
    alpha_prose: float = 0.5
    #: Each further chunk from an already-represented file is multiplied
    #: by this. 1.0 disables. Stops one large file filling every slot.
    #: Applied inside `rerank`, so it does nothing while that is off.
    saturation_decay: float = 0.5
    #: Post-retrieval re-ranking. Off leaves raw fusion order.
    #: `saturation_decay` and `candidate_depth` only take effect with
    #: it on.
    rerank: bool = True
    #: Weights inside that layer, as multiples of the top fused score.
    file_coherence: float = 0.2
    stem_boost: float = 1.0
    definition_boost: float = 3.0
    #: Candidates pulled per tier before fusion. Fixed, not a multiple
    #: of k, or the top 3 would depend on how many results the caller
    #: asked for. Also the ceiling on how many distinct files a search
    #: can return. Raising it only helps with `rerank` on.
    candidate_depth: int = 30
    #: Distil prose chunks instead of embedding them verbatim. Code is
    #: always embedded raw. Off by default: suppressing a category is
    #: the content filter's job.
    distill_docs: bool = False
    #: What a caller asked for, when the caller is a config file rather
    #: than an argument. Empty means "use the default", which is every
    #: category except `content_excluded`.
    content: tuple[str, ...] = ()
    #: Categories a search excludes unless the caller names them.
    #: Stated as exclusions rather than inclusions so a category added
    #: later is visible by default instead of silently dropped. Config
    #: stays in: a question about how something is wired is often
    #: answered by a manifest or a dockerfile.
    content_excluded: tuple[str, ...] = ("data",)
    #: Categories that never enter the index. Opting one back in is a
    #: reindex -- which is what makes excluding it safe to default.
    index_excluded: tuple[str, ...] = ("data",)

    # Tried in order: docs, config, data, tests, code. Resolved once at
    # walk time and stored on the file row, so editing any list below
    # reclassifies -- these feed `categories_rev`, part of the index
    # identity.
    #: Languages that count as documentation. Matched against the
    #: stored `file.lang`, which comes from the extension.
    doc_languages: tuple[str, ...] = (
        "markdown", "markdown_inline", "html", "rst", "asciidoc", "latex",
    )
    #: Languages that count as configuration. None has a tags query, so
    #: these are retrievable as windows and carry no symbols.
    config_languages: tuple[str, ...] = (
        "toml", "yaml", "json", "ini", "xml", "dockerfile", "make",
        "cmake", "hcl", "terraform", "properties", "editorconfig",
        "gitignore", "gitattributes", "gomod", "gosum", "requirements",
        "pymanifest",
    )
    #: Tabular data. Its own category rather than config, because a
    #: question about configuration should not return spreadsheet rows.
    data_languages: tuple[str, ...] = ("csv", "tsv", "psv")
    #: Plain substrings, matched case-insensitively against the whole
    #: relative path. Words like 'attestation' and 'latest' collide;
    #: narrow the marker ('test_', '/tests/') if that bites.
    test_markers: tuple[str, ...] = ("test", "spec")

    #: 'static'  model2vec, no extra dependency, effectively instant.
    #: 'onnx'    a real transformer via onnxruntime. Much slower to
    #:           index, and costs an optional dependency. Pair with
    #:           embed_model 'BAAI/bge-small-en-v1.5'.
    #: 'http'    an OpenAI-compatible /v1/embeddings server.
    #: 'none'    no vector tier; exact and lexical only.
    embed_backend: Literal["static", "onnx", "http", "none"] = "static"
    #: Code-specialised static model. Changing it invalidates every
    #: stored vector, so it forces a reindex.
    embed_model: str = "minishlab/potion-code-16M-v2"
    #: Empty means unset. No tri-state here -- it is only ever tested
    #: for truthiness, unlike the prefixes below.
    embed_endpoint: str = ""
    #: Secret. Prefer the REPOGLASS_EMBED_API_KEY environment variable
    #: over writing it into a config file that lives in the repo.
    #: Empty means unset.
    embed_api_key: str = ""
    #: None = use whatever the model family expects (bge and e5 want an
    #: instruction on queries only). '' = deliberately none.
    embed_query_prefix: str | None = None
    #: Document-side marker. e5 and nomic want one; most models do not.
    #: None = whatever the model family expects, '' = deliberately none.
    embed_doc_prefix: str | None = None
    #: onnxruntime execution provider.
    #:
    #:   webgpu  the plugin provider, via `repoglass[webgpu]`. Takes the
    #:           whole graph on Metal, Vulkan or D3D12.
    #:   cpu     always present, and the fallback when the plugin is not
    #:           installed.
    #:   auto    CoreML or CUDA where the build offers them, then CPU.
    #:           Avoid on Apple silicon: CoreML claims only part of the
    #:           graph and splits the rest across dozens of partitions,
    #:           which is slower than cpu and costs far more memory.
    embed_providers: str = "webgpu"
    #: Path of the graph inside the HF repo. Layout is not standardised:
    #: bge and nomic use onnx/model.onnx, e5-small-v2 puts model.onnx at
    #: the root, and quantised variants sit beside them.
    embed_onnx_file: str = "onnx/model.onnx"

    #: How much of each file ends up in a chunk.
    #:
    #:   definition  only tags-query definitions, leaving module-level
    #:               code in no chunk and so unreachable by search
    #:   hybrid      plus windows over the lines no definition owns
    #:
    #: Both keep the definition chunk and its symbol link, so the exact
    #: tier can go from a name straight to its text. Changing this
    #: forces a reindex.
    coverage: Literal["definition", "hybrid"] = "hybrid"
    #: Target -- not a cap -- for a window, in bytes. An indivisible node
    #: larger than this is emitted whole.
    window_chars: int = 750
    lexical_mode: Literal["full", "capped", "contentless"] = "full"
    lexical_cap_chars: int = 2_000      # lexical_mode="capped" only
    #: Build the FTS input as span + stem + stem + dirs instead of
    #: humanised-path + span. Changes the index, so it forces a
    #: reindex via extractor identity.
    lexical_enrich: bool = False
    #: Append or inline the sub-words of compound identifiers, so
    #: "handler" can match `HandlerStack` -- FTS5's unicode61 tokenizer
    #: does not split camelCase while `tokenize_query` does.
    #:
    #: Off by default: the gap it closes is one the exact-symbol tier
    #: already covers.
    split_identifiers: Literal["off", "inline", "append"] = "off"
    max_chunk_lines: int = 200

    #: Where the index is written. Unset means a central directory
    #: under `~/.repoglass/index`, keyed by the repository path, so the
    #: indexed tree is never written to and a read-only checkout can be
    #: indexed. A relative path resolves against the repository, so
    #: ".repoglass" puts the index beside the code instead.
    data_dir: str | None = None

    #: Honour the repository's .gitignore. A good default heuristic for
    #: "not my code", but only a heuristic: git-ignored is not the same
    #: claim as not-worth-searching. A `!` pattern in .repoglassignore
    #: overrides this per file.
    gitignore: bool = True
    hard_exclude: tuple[str, ...] = (
        ".git", ".repoglass", ".venv", "venv", "node_modules",
        "__pycache__", ".build", "dist", "target",
        ".pytest_cache", ".mypy_cache",
    )
    #: Skip files larger than this. `hard_exclude` catches the usual
    #: homes for vendored and generated code by directory name, but a
    #: bundle committed anywhere else -- an editor plugin, a saved web
    #: page -- is indistinguishable from source by extension alone.
    max_file_bytes: int = 1_000_000

    refresh_mode: Literal["auto", "manual"] = "auto"
    #: Minimum gap between staleness walks triggered by a read. 0 walks
    #: on every call, which on a large repo spends most of the query
    #: re-stat-ing files that have not changed. Edits are still picked
    #: up, just at most this often.
    rescan_after_seconds: int = 5


def categories_rev(settings: Settings) -> str:
    """Fingerprint of the lists that decide a file's content type.

    The type is resolved at walk time and stored, so editing any of
    these must reclassify. Part of the index identity for the same
    reason `extractor_rev` is: the stored value is derived from inputs
    that can change under it.
    """
    import hashlib

    parts = (settings.doc_languages, settings.config_languages,
             settings.data_languages, settings.test_markers,
             tuple(sorted(settings.index_excluded)))
    joined = chr(10).join(chr(31).join(p) for p in parts)
    return hashlib.blake2b(joined.encode(), digest_size=8).hexdigest()
