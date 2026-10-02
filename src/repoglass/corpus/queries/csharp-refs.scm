; Reference captures for C#. Upstream csharp-tags.scm records a call
; only through a member, `w.Render()`, not a bare `Render()`.
(invocation_expression
  function: (identifier) @name.reference.call) @reference.call

; A member, called or read: `Count` in `w.Count`.
(member_access_expression
  name: (identifier) @name.reference.member) @reference.member

; A type named before a dot: `Widget` in `Widget.Make()`. A lowercase
; receiver is a local value, which names no definition.
((member_access_expression
  expression: (identifier) @name.reference.type) @reference.type
  (#match? @name.reference.type "^[A-Z]"))
