; Reference captures for Swift. Upstream swift-tags.scm provides definitions only.
; Generic invocations foo<T>(x) parse as constructor_expression, not call_expression.
(call_expression
  (simple_identifier) @name.reference.call) @reference.call

(constructor_expression
  (user_type (type_identifier) @name.reference.call)) @reference.call

(user_type (type_identifier) @name.reference.type) @reference.type

; A member, called or read: `render` in `w.render()`, `count` in `w.count`.
(navigation_expression
  suffix: (navigation_suffix
    suffix: (simple_identifier) @name.reference.member)) @reference.member

; A type named before a dot: `Widget` in `Widget.make()`. A lowercase
; receiver is a value such as `w` or `self`, which names no definition.
((navigation_expression
  target: (simple_identifier) @name.reference.type) @reference.type
  (#match? @name.reference.type "^[A-Z]"))
