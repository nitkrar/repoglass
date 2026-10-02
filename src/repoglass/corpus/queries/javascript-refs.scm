; Reference captures for JavaScript, beyond javascript-tags.scm's calls
; and `new`.

; A member, called or read: `render` in `w.render()`, `count` in `w.count`.
(member_expression
  property: (property_identifier) @name.reference.member) @reference.member

; A class named before a dot: `Widget` in `Widget.make()`. A lowercase
; receiver is a value, which names no definition.
((member_expression
  object: (identifier) @name.reference.type) @reference.type
  (#match? @name.reference.type "^[A-Z]"))
