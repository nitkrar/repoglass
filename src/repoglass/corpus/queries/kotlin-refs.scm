; Reference captures for Kotlin, beyond kotlin-tags.scm's calls and
; supertypes. navigation_expression has no field names: the receiver is
; its first child.

; A member, called or read: `render` in `w.render()`, `count` in `w.count`.
(navigation_expression
  (navigation_suffix
    (simple_identifier) @name.reference.member)) @reference.member

; A type named before a dot: `Widget` in `Widget.make()`. A lowercase
; receiver is a value such as `w`, which names no definition.
((navigation_expression
  . (simple_identifier) @name.reference.type) @reference.type
  (#match? @name.reference.type "^[A-Z]"))
