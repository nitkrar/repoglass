; Reference captures for PHP. Upstream php-tags.scm already has a
; function_call_expression pattern, but it accepts only
; (qualified_name (name)) and (variable_name (name)). The loaded
; grammar parses an unqualified call as a bare (name), so a plain
; function call matched neither branch.
(function_call_expression
  function: (name) @name.reference.call) @reference.call

; The same gap in object_creation_expression: `new Widget()`.
(object_creation_expression
  (name) @name.reference.class) @reference.class

; A member, called or read: `count` in `$w->count`.
(member_access_expression
  name: (name) @name.reference.member) @reference.member

(nullsafe_member_access_expression
  name: (name) @name.reference.member) @reference.member

; A class named before `::`: `Widget` in `Widget::make()` and
; `Widget::LIMIT`, and the constant after it. `self::` and `static::`
; parse as relative_scope, not name.
(scoped_call_expression
  scope: (name) @name.reference.type) @reference.type

(class_constant_access_expression
  . (name) @name.reference.type) @reference.type

(class_constant_access_expression
  (name) @name.reference.member .) @reference.member
