; Written against tree-sitter-scala as bundled in
; tree-sitter-language-pack; Aider ships no scala query.
; `case class` is a class_definition, so it needs no separate pattern.
(class_definition
  name: (identifier) @name.definition.class) @definition.class

(object_definition
  name: (identifier) @name.definition.class) @definition.class

(trait_definition
  name: (identifier) @name.definition.interface) @definition.interface

(function_definition
  name: (identifier) @name.definition.function) @definition.function

(val_definition
  pattern: (identifier) @name.definition.constant) @definition.constant

(type_definition
  name: (type_identifier) @name.definition.type) @definition.type
