; Written against tree-sitter-haskell as bundled in
; tree-sitter-language-pack; Aider ships no haskell query.
; A top-level binding is `function`, whose `variable` child is the name.
(declarations
  (function
    name: (variable) @name.definition.function) @definition.function)

(declarations
  (signature
    name: (variable) @name.definition.function) @definition.function)

(declarations
  (data_type
    name: (name) @name.definition.type) @definition.type)

(declarations
  (newtype
    name: (name) @name.definition.type) @definition.type)

(declarations
  (class
    (name) @name.definition.interface) @definition.interface)
