; Reference captures for Scala. Upstream scala-tags.scm provides
; definitions only.
; Calls only. A bare (type_identifier) would also match the annotation
; on every parameter and return, which is a type reference rather than
; a use of a definition, and the symptom being fixed is that "who
; calls this" answers nothing.
(call_expression
  function: (identifier) @name.reference.call) @reference.call

(call_expression
  function: (field_expression
    field: (identifier) @name.reference.call)) @reference.call

; A member read: `count` in `w.count`. Calls are matched above.
(field_expression
  field: (identifier) @name.reference.member) @reference.member

; An object named before a dot: `Widget` in `Widget.make()`. A
; lowercase receiver is a value, which names no definition.
((field_expression
  value: (identifier) @name.reference.type) @reference.type
  (#match? @name.reference.type "^[A-Z]"))
