; Reference captures for Go, beyond go-tags.scm's calls, types and
; imports. The name before a dot is a package or a value, never a type,
; and every type is already a reference, so receivers are not captured.

; A field read: `Count` in `w.Count`. Calls are matched upstream.
(selector_expression
  field: (field_identifier) @name.reference.member) @reference.member

; go-tags.scm captures every type_identifier, predeclared ones included;
; `int` names no definition in the repository.
((type_identifier) @ignore
  (#match? @ignore "^(any|bool|byte|comparable|complex64|complex128|error|float32|float64|int|int8|int16|int32|int64|rune|string|uint|uint8|uint16|uint32|uint64|uintptr)$"))
