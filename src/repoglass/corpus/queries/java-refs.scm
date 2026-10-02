; Reference captures for Java, beyond java-tags.scm's method calls,
; `new` and supertypes.

; A field read: `count` in `w.count`.
(field_access
  field: (identifier) @name.reference.member) @reference.member

; A class named before a dot: `Widget` in `Widget.make()` and
; `Widget.LIMIT`. A lowercase receiver is a value, which names no
; definition.
((method_invocation
  object: (identifier) @name.reference.type) @reference.type
  (#match? @name.reference.type "^[A-Z]"))

((field_access
  object: (identifier) @name.reference.type) @reference.type
  (#match? @name.reference.type "^[A-Z]"))
