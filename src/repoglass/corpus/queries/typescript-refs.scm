; Reference captures for TypeScript. Upstream typescript-tags.scm
; captures type annotations and `new`, and has no call pattern at all.
; Both patterns below are javascript-tags.scm's, unchanged: the
; TypeScript grammar shares call_expression and member_expression, so
; a separate spelling would be a second copy of one thing.
(call_expression
  function: (identifier) @name.reference.call) @reference.call

(call_expression
  function: (member_expression
    property: (property_identifier) @name.reference.call)) @reference.call
