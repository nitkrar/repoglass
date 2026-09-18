; Reference captures for PHP. Upstream php-tags.scm already has a
; function_call_expression pattern, but it accepts only
; (qualified_name (name)) and (variable_name (name)). The loaded
; grammar parses an unqualified call as a bare (name), so a plain
; function call matched neither branch.
(function_call_expression
  function: (name) @name.reference.call) @reference.call
