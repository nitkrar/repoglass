; Reference captures for Lua, beyond lua-tags.scm's calls.

; A field read: `count` in `w.count`. Calls are matched upstream.
(dot_index_expression
  field: (identifier) @name.reference.member) @reference.member

; A table named before a dot: `Widget` in `Widget.make()`. A lowercase
; table is a value or a module alias such as `w` or `vim`, which names
; no definition.
((dot_index_expression
  table: (identifier) @name.reference.type) @reference.type
  (#match? @name.reference.type "^[A-Z]"))
