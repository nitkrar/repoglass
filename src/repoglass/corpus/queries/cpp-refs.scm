; Reference captures for C++. Upstream cpp-tags.scm provides definitions only.
; Bare (type_identifier) is omitted: it matches definition sites too.
(call_expression
  function: (identifier) @name.reference.call) @reference.call

; A member, called or read: `count` in `w.count` and `p->count`.
(field_expression
  field: (field_identifier) @name.reference.member) @reference.member

; Either side of `::`: `Widget` and `make` in `Widget::make()`. The
; scope is always a type or a namespace.
(qualified_identifier
  scope: (namespace_identifier) @name.reference.type) @reference.type

(qualified_identifier
  name: (identifier) @name.reference.member) @reference.member
