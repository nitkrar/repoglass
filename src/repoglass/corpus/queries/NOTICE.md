# Vendored tree-sitter tag queries

`*-tags.scm` files here are adapted from upstream tree-sitter grammar
repositories, by way of [Aider](https://github.com/Aider-AI/aider)'s
`aider/queries/` collection. `manifest.json` records which source set
each query came from. The sets target different grammar versions and
are not interchangeable.

Upstream licences, per Aider's own attribution:

* tree-sitter/tree-sitter-{c,c-sharp,cpp,go,java,javascript,ocaml,php,python,ql,ruby,rust,typescript} — MIT
* Wilfred/tree-sitter-elisp — MIT
* elm-tooling/tree-sitter-elm — MIT
* r-lib/tree-sitter-r — MIT
* starelmanma/tree-sitter-fortran — MIT
* elixir-lang/tree-sitter-elixir — Apache License 2.0

Files adapted from the `tree-sitter-language-pack` set derive from the
repositories listed at
https://github.com/Goldziher/tree-sitter-language-pack/blob/main/sources/language_definitions.json

## MinishLab/semble

MIT License, Copyright (c) 2025 MinishLab.
https://github.com/MinishLab/semble

This project includes logic derived from semble in its search ranking
and ignore-walk implementations, and uses the MIT-licensed
minishlab/potion-code-16M-v2 model and model2vec.
