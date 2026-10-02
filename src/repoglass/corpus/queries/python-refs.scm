; Reference captures for Python, beyond python-tags.scm's calls and
; annotations.

; A member, called or read: `render` in `w.render()`, `count` in `w.count`.
(attribute
  attribute: (identifier) @name.reference.member) @reference.member

; A type named before a dot: `Widget` in `Widget.make()`. A lowercase
; receiver is a value such as `w`, `self` or a module, which names no
; definition.
((attribute
  object: (identifier) @name.reference.type) @reference.type
  (#match? @name.reference.type "^[A-Z]"))
