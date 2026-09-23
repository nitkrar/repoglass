; Method definitions

(
  (comment)* @doc
  .
  [
    (method
      name: (_) @name.definition.method) @definition.method
    (singleton_method
      name: (_) @name.definition.method) @definition.method
  ]
  (#strip! @doc "^#\\s*")
  (#select-adjacent! @doc @definition.method)
)

(alias
  name: (_) @name.definition.method) @definition.method

(setter
  (identifier) @ignore)

; Class definitions

(
  (comment)* @doc
  .
  [
    (class
      name: [
        (constant) @name.definition.class
        (scope_resolution
          name: (_) @name.definition.class)
      ]) @definition.class
    (singleton_class
      value: [
        (constant) @name.definition.class
        (scope_resolution
          name: (_) @name.definition.class)
      ]) @definition.class
  ]
  (#strip! @doc "^#\\s*")
  (#select-adjacent! @doc @definition.class)
)

; Module definitions

(
  (module
    name: [
      (constant) @name.definition.module
      (scope_resolution
        name: (_) @name.definition.module)
    ]) @definition.module
)

; Calls

(call method: (identifier) @name.reference.call) @reference.call

; A bare identifier is a paren-less call or a local read, and the
; grammar does not say which. Upstream separates them with
; `(#is-not? local)`, which needs the scope tracking that comes with
; locals.scm; nothing here provides it, so the predicate never fired
; and every parameter and local became a reference. Matching only what
; the grammar already calls a call loses the paren-less ones and keeps
; the result navigable, which is the point of a reference.
(
  (constant) @name.reference.call @reference.call
  (#not-match? @name.reference.call "^(lambda|load|require|require_relative|__FILE__|__LINE__)$")
)
