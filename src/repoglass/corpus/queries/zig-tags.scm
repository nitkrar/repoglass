; Written against tree-sitter-zig as bundled in
; tree-sitter-language-pack; Aider ships no zig query. This grammar
; uses PascalCase node names and wraps top-level items in `Decl`.
(Decl
  (FnProto
    function: (IDENTIFIER) @name.definition.function)) @definition.function

(Decl
  (VarDecl
    (IDENTIFIER) @name.definition.constant)) @definition.constant
