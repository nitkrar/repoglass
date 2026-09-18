(module (expression_statement (assignment left: (identifier) @name.definition.constant) @definition.constant))

(class_definition
  name: (identifier) @name.definition.class) @definition.class

(function_definition
  name: (identifier) @name.definition.function) @definition.function

(call
  function: [
      (identifier) @name.reference.call
      (attribute
        attribute: (identifier) @name.reference.call)
  ]) @reference.call

; Annotations. For a type that is never instantiated, these are the
; only uses it has, and without them `references` reports none at all.
; Every annotation position -- parameter, return, variable, attribute
; -- wraps its type in a `type` node, and a generic nests another
; `type` inside its parameter list, so this one pattern reaches
; `Widget` in `list[Widget]` as well as in `w: Widget`.
(type (identifier) @name.reference.type) @reference.type

; The outer name of a generic: `Sequence` in `Sequence[Chunk]`.
(type
  (generic_type (identifier) @name.reference.type)) @reference.type

; A forward reference, written as a string because the name is not
; bound yet. `string_content` is the name without its quotes.
(type
  (string (string_content) @name.reference.type)) @reference.type
