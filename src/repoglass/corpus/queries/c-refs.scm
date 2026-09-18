; Reference captures for C. Upstream c-tags.scm provides definitions only.
; Only call sites: a bare (type_identifier) also matches struct/typedef
; definition sites and manufactures self-edges.
(call_expression
  function: (identifier) @name.reference.call) @reference.call
