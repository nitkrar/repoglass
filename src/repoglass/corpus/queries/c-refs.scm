; Reference captures for C. Upstream c-tags.scm provides definitions only.
; Only call sites: a bare (type_identifier) also matches struct/typedef
; definition sites and manufactures self-edges.
(call_expression
  function: (identifier) @name.reference.call) @reference.call

; A field, read or called through: `count` in `s.count` and `p->count`.
(field_expression
  field: (field_identifier) @name.reference.member) @reference.member
