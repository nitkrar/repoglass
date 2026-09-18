; Reference captures for Rust. Upstream rust-tags.scm captures a call
; through (identifier) and through a field_expression, which leaves the
; path call -- crypto::open(..) -- uncaptured, and that is the ordinary
; form for anything not imported bare.
(call_expression
  function: (scoped_identifier
    name: (identifier) @name.reference.call)) @reference.call
