; Reference captures for C++. Upstream cpp-tags.scm provides definitions only.
; Bare (type_identifier) is omitted: it matches definition sites too.
(call_expression
  function: (identifier) @name.reference.call) @reference.call

(call_expression
  function: (field_expression field: (field_identifier) @name.reference.call)) @reference.call

(call_expression
  function: (qualified_identifier name: (identifier) @name.reference.call)) @reference.call
