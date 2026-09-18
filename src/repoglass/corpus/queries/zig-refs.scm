; Reference captures for Zig. Upstream zig-tags.scm provides
; definitions only.
; A call is a suffix on an expression rather than a node of its own:
; `crypto.open(x)` is a FieldOrFnCall carrying `function_call`, and a
; bare `open(x)` is a SuffixExpr whose identifier is followed by
; FnCallArguments.
(FieldOrFnCall
  function_call: (IDENTIFIER) @name.reference.call) @reference.call

(SuffixExpr
  variable_type_function: (IDENTIFIER) @name.reference.call
  (FnCallArguments)) @reference.call
