; Reference captures for Haskell. Upstream haskell-tags.scm provides
; definitions only.
; Application nests to the left, so `f x y` parses as apply(apply(f,
; x), y) and the function name sits at the innermost `function:`.
; Matching every `function:` would also capture the intermediate
; apply node, which names nothing.
(apply
  function: (variable) @name.reference.call) @reference.call
