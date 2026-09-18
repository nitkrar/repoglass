; Reference captures for Swift. Upstream swift-tags.scm provides definitions only.
; Generic invocations foo<T>(x) parse as constructor_expression, not call_expression.
(call_expression
  (simple_identifier) @name.reference.call) @reference.call

(call_expression
  (navigation_expression
    (navigation_suffix (simple_identifier) @name.reference.call))) @reference.call

(constructor_expression
  (user_type (type_identifier) @name.reference.call)) @reference.call

(user_type (type_identifier) @name.reference.type) @reference.type
